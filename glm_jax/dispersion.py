"""NB dispersion estimation via Cox-Reid adjusted profile likelihood.

This is the main "repeated GLM calls" workload of edgeR: for every gene and
every trial dispersion we refit the NB GLM and evaluate

    APL(phi) = l(beta_hat(phi); phi) - 1/2 log det(X^T W X)

(edgeR ``adjustedProfileLik``).  Two stages, both batched over genes:

1. ``apl_grid`` fits the first grid point to convergence, then walks the rest
   of the grid with ``lax.scan``, warm-starting each phi from the previous
   coefficients and taking a fixed number of Fisher-scoring steps (predictable
   cost, no gene waits for a straggler).  The APL kernel hoists the phi-free
   lgamma(y+1) term out of the loop, uses a shifted Stirling series for
   lgamma(y + 1/phi), and an unrolled sqrt-free LDL^T for the log-determinant
   (ideas found by the evolve/ loop, see evolve/log.jsonl).
2. ``estimate_dispersion_grid`` polishes the grid argmax with a few safeguarded
   Newton steps in theta = log(phi), differentiating the APL through those
   fixed steps with ``jax.grad`` (so d beta_hat / d phi is included).
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammaln

from .families import NegativeBinomial
from .irls import _SMALL_P, _broadcast_inputs, _fit_single, _wls, _working, fit_glm

_NB = NegativeBinomial()
_FIT_DEFAULTS = dict(max_iter=50, tol=1e-8, beta_tol=1e-8, ridge=1e-10, max_halving=30)
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def _lgamma_pos(x):
    """log Gamma(x) for x > 0: Stirling at x + 6 minus the log of the shift product.

    Abs error < 1e-10 (series truncation ~ 1/(1188 z^9) at z >= 6); cheaper
    than XLA's general gammaln, which carries a reflection branch.  float64 only.
    """
    z = x + 6.0
    iz = 1.0 / z
    iz2 = iz * iz
    series = iz * (1.0 / 12 - iz2 * (1.0 / 360 - iz2 * (1.0 / 1260 - iz2 / 1680)))
    p = x * (x + 1.0) * (x + 2.0) * (x + 3.0) * (x + 4.0) * (x + 5.0)
    return (z - 0.5) * jnp.log(z) - z + _HALF_LOG_2PI + series - jnp.log(p)


def _logdet_spd(A):
    """log det of a small SPD matrix: unrolled sqrt-free LDL^T, sum of log pivots."""
    p = A.shape[0]
    if p > _SMALL_P:
        return 2.0 * jnp.sum(jnp.log(jnp.diagonal(jnp.linalg.cholesky(A))))
    L = [[None] * p for _ in range(p)]
    d = []
    for j in range(p):
        d.append(A[j, j] - sum(L[j][k] * L[j][k] * d[k] for k in range(j)))
        inv = 1.0 / d[j]
        for i in range(j + 1, p):
            L[i][j] = (A[i, j] - sum(L[i][k] * L[j][k] * d[k] for k in range(j))) * inv
    # sum of logs, not log of the product: pivots of separated genes can be ~1e-14
    return sum(jnp.log(dj) for dj in d)


def _apl_single(y, X, offset, w, phi, beta):
    """Cox-Reid APL of one feature at its fitted coefficients (general form)."""
    eta = X @ beta + offset
    W, _, mu = _working(_NB, y, eta, w, phi)  # NB-log: W = w mu / (1 + phi mu)
    ll = jnp.sum(w * _NB.loglik(y, mu, phi))
    return ll - 0.5 * _logdet_spd((X.T * W) @ X)


def _apl_grid_single(y, X, offset, w, phi, beta, c):
    """``_apl_single`` with the phi-free term hoisted: ``c = sum(w lgamma(y+1))``.

    With r = 1/phi and log mu = eta, the NB log-likelihood per observation is
    lgamma(y+r) - lgamma(r) - lgamma(y+1) + y (eta + log phi) - (y+r) log1p(phi mu).
    """
    eta = X @ beta + offset
    mu = jnp.exp(eta)
    pm = phi * mu
    r = 1.0 / phi
    ll = (jnp.sum(w * (_lgamma_pos(y + r) + y * (eta + jnp.log(phi)) - (y + r) * jnp.log1p(pm)))
          - gammaln(r) * jnp.sum(w) - c)
    W = w * mu / (1.0 + pm)
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
    c = jnp.sum(weights * gammaln(Y + 1.0), axis=-1)  # phi-free, computed once

    def fit(phi, beta0, use_beta0):
        kern = partial(_fit_single, family=_NB, **kw)
        return jax.vmap(lambda y, o, w, b: kern(y, X, o, w, phi, b, use_beta0))(
            Y, offset, weights, beta0).coef

    def fixed(phi, beta):
        return jax.vmap(lambda y, o, w, b: _fixed_steps(y, X, o, w, phi, b, steps, kw["ridge"]))(
            Y, offset, weights, beta)

    apl_v = jax.vmap(_apl_grid_single, in_axes=(0, None, 0, 0, None, 0, 0))

    def apl(phi, beta):
        return apl_v(Y, X, offset, weights, phi, beta, c)

    beta = fit(phi_grid[0], beta0, False)  # first grid point: converged fit
    apl0 = apl(phi_grid[0], beta)

    def step(beta, phi):
        beta = fit(phi, beta, True) if steps is None else fixed(phi, beta)
        return beta, (apl(phi, beta), beta)

    _, (apls, betas) = lax.scan(step, beta, phi_grid[1:])
    apls = jnp.concatenate([apl0[None], apls]).T
    betas = jnp.concatenate([beta[None], betas]).transpose(1, 0, 2)
    return apls, betas


def apl_grid(Y, X, phi_grid, *, offset=None, weights=None, steps=2, **fit_kwargs):
    """APL for every (feature, phi) pair. Returns (G, K).

    The first grid point is fit to convergence; the rest are walked with
    ``lax.scan``, each phi warm-started from the previous coefficients with
    ``steps`` fixed Fisher-scoring steps and no convergence test: every gene
    costs the same, so nobody waits for a separated gene drifting to -inf
    (under ``vmap`` the slowest gene sets the pace; those took 10-30 iterations
    while the median needed 3).  Two steps put the APL within ~2e-2 of the
    converged value, ~50x below the gap between neighbouring grid points, with
    identical argmax on 5000 simulated genes.  ``steps=None`` runs the fully
    converging IRLS at every grid point.
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
    # theta is a scalar, so one JVP of the gradient returns (g, h) at once:
    # half the cost of evaluating grad and jacfwd(grad) separately.
    gh = jax.vmap(lambda th, y, o, w, b: jax.jvp(lambda t: g_fn(t, y, o, w, b), (th,), (jnp.ones_like(th),)),
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
                             steps=2, newton=2, **fit_kwargs):
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
