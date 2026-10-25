"""NB dispersion estimation via Cox-Reid adjusted profile likelihood.

This is the main "repeated GLM calls" workload of edgeR: for every gene and
every trial dispersion we refit the NB GLM and evaluate

    APL(phi) = l(beta_hat(phi); phi) - 1/2 log det(X^T W X)

(edgeR ``adjustedProfileLik``).  Two stages, both batched over genes:

1. ``apl_grid`` walks a coarse dispersion grid with ``lax.scan``, warm-starting
   each phi from the previous coefficients and taking a fixed number of
   Fisher-scoring steps (predictable cost, no gene waits for a straggler).
2. ``estimate_dispersion_grid`` polishes the grid argmax with a few safeguarded
   Newton steps in theta = log(phi), differentiating the APL through those
   fixed steps with ``jax.grad`` (so d beta_hat / d phi is included).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .families import NegativeBinomial
from .irls import _SMALL_P, _broadcast_inputs, _fit_single, _small_chol, _wls, _working, fit_glm

_NB = NegativeBinomial()
_FIT_DEFAULTS = dict(max_iter=50, tol=1e-8, beta_tol=1e-8, ridge=1e-10, max_halving=30)


def _logdet_spd(A):
    if A.shape[0] <= _SMALL_P:
        L = _small_chol(A)
        return 2.0 * sum(jnp.log(L[j][j]) for j in range(A.shape[0]))
    return 2.0 * jnp.sum(jnp.log(jnp.diagonal(jnp.linalg.cholesky(A))))


def _apl_single(y, X, offset, w, phi, beta):
    """Cox-Reid APL of one feature at its fitted coefficients."""
    eta = X @ beta + offset
    W, _, mu = _working(_NB, y, eta, w, phi)  # NB-log: W = w mu / (1 + phi mu)
    ll = jnp.sum(w * _NB.loglik(y, mu, phi))
    return ll - 0.5 * _logdet_spd((X.T * W) @ X)


def _fixed_steps(y, X, offset, w, phi, beta, steps, ridge):
    """``steps`` Fisher-scoring steps from ``beta`` (no convergence test)."""
    def body(_, b):
        W, z, _ = _working(_NB, y, X @ b + offset, w, phi)
        b_new = _wls(X, W, z - offset, ridge)
        return jnp.where(jnp.all(jnp.isfinite(b_new)), b_new, b)
    return lax.fori_loop(0, steps, body, beta)


def adjusted_profile_loglik(Y, X, phi, *, offset=None, weights=None, fit=None, **fit_kwargs):
    """Cox-Reid adjusted profile log-likelihood per feature. Returns (G,)."""
    Y = jnp.atleast_2d(jnp.asarray(Y))
    if fit is None:
        fit = fit_glm(Y, X, family=_NB, offset=offset, weights=weights, phi=phi, **fit_kwargs)
    Y, X, offset, weights, phi, _, _ = _broadcast_inputs(Y, X, offset, weights, phi, None)
    return jax.vmap(_apl_single, in_axes=(0, None, 0, 0, 0, 0))(Y, X, offset, weights, phi, fit.coef)


@partial(jax.jit, static_argnames=("steps",) + tuple(_FIT_DEFAULTS))
def _grid(Y, X, phi_grid, offset, weights, steps, **fit_kwargs):
    """APL (G, K) and coefficients (G, K, p) along the grid."""
    kw = {**_FIT_DEFAULTS, **fit_kwargs}
    Y, X, offset, weights, _, beta0, _ = _broadcast_inputs(Y, X, offset, weights, None, None)
    phi_grid = jnp.asarray(phi_grid, Y.dtype)

    def fit(phi, beta0, use_beta0):
        kern = partial(_fit_single, family=_NB, **kw)
        return jax.vmap(lambda y, o, w, b: kern(y, X, o, w, phi, b, use_beta0))(
            Y, offset, weights, beta0).coef

    def fixed(phi, beta):
        return jax.vmap(lambda y, o, w, b: _fixed_steps(y, X, o, w, phi, b, steps, kw["ridge"]))(
            Y, offset, weights, beta)

    apl = jax.vmap(_apl_single, in_axes=(0, None, 0, 0, None, 0))

    beta = fit(phi_grid[0], beta0, False)
    apl0 = apl(Y, X, offset, weights, phi_grid[0], beta)

    def step(beta, phi):
        beta = fit(phi, beta, True) if steps is None else fixed(phi, beta)
        return beta, (apl(Y, X, offset, weights, phi, beta), beta)

    _, (apls, betas) = lax.scan(step, beta, phi_grid[1:])
    apls = jnp.concatenate([apl0[None], apls]).T
    betas = jnp.concatenate([beta[None], betas]).transpose(1, 0, 2)
    return apls, betas


def apl_grid(Y, X, phi_grid, *, offset=None, weights=None, steps=3, **fit_kwargs):
    """APL for every (feature, phi) pair. Returns (G, K).

    The grid is walked with ``lax.scan``, each phi warm-started from the
    previous coefficients.  ``steps`` fixed Fisher-scoring steps per grid
    point, with no convergence test: every gene costs the same, so nobody waits
    for a separated gene drifting to -inf (under ``vmap`` the slowest gene sets
    the pace; those took 10-30 iterations while the median needed 3).  Three
    steps put the APL within ~1e-2 of the converged value, 100x below the gap
    between neighbouring grid points, with identical argmax on 5000 simulated
    genes.  ``steps=None`` runs the fully converging IRLS at every grid point.
    """
    Y = jnp.atleast_2d(jnp.asarray(Y))
    return _grid(Y, X, phi_grid, offset, weights, steps, **fit_kwargs)[0]


@partial(jax.jit, static_argnames=("steps", "newton", "common", "ridge"))
def _polish(Y, X, offset, weights, theta, beta, lo, hi, max_step, steps, newton, common, ridge):
    """Safeguarded Newton ascent of the APL in theta = log(phi).

    ``theta`` is (G,) for tagwise or () for common (then gradients and Hessians
    are summed over genes).  Each step is clipped to ``max_step`` (one grid
    spacing), kept in [lo, hi], and falls back to a gradient step when the
    Hessian is not negative.
    """
    Y, X, offset, weights, _, _, _ = _broadcast_inputs(Y, X, offset, weights, None, None)

    def apl_theta(th, y, o, w, b):
        phi = jnp.exp(th)
        return _apl_single(y, X, o, w, phi, _fixed_steps(y, X, o, w, phi, b, steps, ridge))

    g_fn = jax.grad(apl_theta)
    h_fn = jax.jacfwd(g_fn)
    gh = jax.vmap(lambda th, y, o, w, b: (g_fn(th, y, o, w, b), h_fn(th, y, o, w, b)),
                  in_axes=(None if common else 0, 0, 0, 0, 0))

    for _ in range(newton):
        g, h = gh(theta, Y, offset, weights, beta)
        if common:
            g, h = jnp.sum(g), jnp.sum(h)
        step = jnp.where(h < 0, -g / h, jnp.sign(g) * max_step)
        step = jnp.clip(jnp.nan_to_num(step), -max_step, max_step)
        theta = jnp.clip(theta + step, lo, hi)
    return theta


def estimate_dispersion_grid(Y, X, *, offset=None, weights=None, phi_grid=None,
                             steps=3, newton=2, **fit_kwargs):
    """Grid search plus Newton polish in log(phi).

    Returns a dict with
      * ``common``  : argmax_phi sum_g APL_g(phi)   (edgeR common dispersion)
      * ``tagwise`` : argmax_phi APL_g(phi)         (no EB shrinkage yet)
      * ``apl``     : (G, K) matrix and ``phi_grid``
    ``newton`` Newton steps refine both from the grid argmax (0 = grid only).
    """
    if phi_grid is None:
        phi_grid = jnp.logspace(-4, 1, 60)
    Y = jnp.atleast_2d(jnp.asarray(Y))
    phi_grid = jnp.asarray(phi_grid)
    apl, betas = _grid(Y, X, phi_grid, offset, weights, steps, **fit_kwargs)
    k_tag = jnp.argmax(apl, axis=1)
    k_common = jnp.argmax(jnp.sum(apl, axis=0))
    theta_grid = jnp.log(phi_grid)
    tagwise, common = phi_grid[k_tag], phi_grid[k_common]
    if newton:
        lo, hi = theta_grid[0], theta_grid[-1]
        max_step = jnp.max(jnp.diff(theta_grid))
        ridge = {**_FIT_DEFAULTS, **fit_kwargs}["ridge"]
        tagwise = jnp.exp(_polish(Y, X, offset, weights, theta_grid[k_tag], betas[jnp.arange(Y.shape[0]), k_tag],
                                  lo, hi, max_step, steps, newton, False, ridge))
        common = jnp.exp(_polish(Y, X, offset, weights, theta_grid[k_common], betas[:, k_common],
                                 lo, hi, max_step, steps, newton, True, ridge))
    return {"common": common, "tagwise": tagwise, "apl": apl, "phi_grid": phi_grid}
