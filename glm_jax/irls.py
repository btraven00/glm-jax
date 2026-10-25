"""Batched IRLS (Fisher scoring) GLM kernel in JAX.

Core idea: write a *single-feature* solver with ``lax.while_loop`` and then
``vmap`` it over features (genes).  Everything is jit-compiled, so fitting
tens of thousands of genes is one fused XLA program, which is what typical
omics pipelines (edgeR, DESeq2, limma-voom ...) need.

Shapes (batched API)
--------------------
Y       : (G, n)    responses, one row per feature
X       : (n, p)    design matrix shared across features
offset  : (n,) or (G, n)
weights : (n,) or (G, n)  prior observation weights
phi     : () or (G,)      family parameter (NB dispersion, Gaussian variance)
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Optional

import os

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve

from .families import Family, NegativeBinomial


class GLMResult(NamedTuple):
    coef: jax.Array          # (..., p)
    deviance: jax.Array      # (...,)
    mu: jax.Array            # (..., n) fitted means
    cov_unscaled: jax.Array  # (..., p, p) (X^T W X)^{-1} at convergence
    n_iter: jax.Array        # (...,)
    converged: jax.Array     # (...,) bool


# --------------------------------------------------------------------------- #
# Single-feature kernel
# --------------------------------------------------------------------------- #
def _working(family: Family, y, eta, w, phi):
    """IRLS working weights W and working response z (without offset)."""
    mu = family.clip_mu(family.link.inverse(eta))
    d = family.link.mu_eta(eta)
    var = family.variance(mu, phi)
    W = w * d**2 / var
    z = eta + (y - mu) / d
    return W, z, mu


def _small_chol(A):
    """Unrolled Cholesky for small static p.

    Under ``vmap`` this becomes pure elementwise arithmetic over the feature
    axis, which is far faster than G separate tiny LAPACK calls on CPU and
    fuses well on GPU.  Returns L as a nested list of scalars.
    """
    p = A.shape[0]
    L = [[None] * p for _ in range(p)]
    for j in range(p):
        s = A[j, j] - sum(L[j][k] ** 2 for k in range(j))
        L[j][j] = jnp.sqrt(s)
        for i in range(j + 1, p):
            s = A[i, j] - sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = s / L[j][j]
    return L


def _small_chol_solve(L, b):
    p = len(L)
    y = [None] * p
    for i in range(p):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [None] * p
    for i in reversed(range(p)):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, p))) / L[i][i]
    return jnp.stack(x)


_SMALL_P = 12
# ponytail: fixed straggler policy; expose if a dataset needs a different split
_FIRST_PASS_ITER = 12
_STRAGGLER_BUCKET = 256
_CPU_CHUNK_ELEMS = 320_000


def _n_cores():
    """Cores this process may run on (honours taskset / SLURM cpusets)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _spd_solve(A, b):
    if A.shape[0] <= _SMALL_P:
        return _small_chol_solve(_small_chol(A), b)
    return cho_solve(cho_factor(A, lower=True), b)


def _spd_inv(A):
    p = A.shape[0]
    if p <= _SMALL_P:
        L = _small_chol(A)
        I = jnp.eye(p, dtype=A.dtype)
        return jnp.stack([_small_chol_solve(L, I[:, j]) for j in range(p)], axis=1)
    return cho_solve(cho_factor(A, lower=True), jnp.eye(p, dtype=A.dtype))


def _wls(X, W, z, ridge):
    p = X.shape[1]
    XtW = X.T * W
    A = XtW @ X + ridge * jnp.eye(p, dtype=X.dtype)
    return _spd_solve(A, XtW @ z)


def _deviance(family, y, mu, w, phi):
    return jnp.sum(w * family.unit_deviance(y, mu, phi))


def _fit_single(y, X, offset, w, phi, beta0, use_beta0, *, family, max_iter, beta_tol,
                tol, ridge, max_halving):
    dtype = X.dtype

    # ---- initial step: from mu0 = init_mu(y) or from a supplied beta0 -----
    eta_init_y = family.link.link(family.init_mu(y))
    eta_init_b = X @ beta0 + offset
    eta0 = jnp.where(use_beta0, eta_init_b, eta_init_y)
    W, z, _ = _working(family, y, eta0, w, phi)
    beta_from_y = _wls(X, W, z - offset, ridge)
    beta = jnp.where(use_beta0, beta0, beta_from_y)

    def dev_of(b):
        eta = X @ b + offset
        mu = family.clip_mu(family.link.inverse(eta))
        return _deviance(family, y, mu, w, phi)

    dev = dev_of(beta)

    # ---- main IRLS loop with step halving ----------------------------------
    def cond(state):
        _, _, it, conv = state
        return jnp.logical_and(it < max_iter, jnp.logical_not(conv))

    def body(state):
        beta, dev, it, _ = state
        eta = X @ beta + offset
        W, z, _ = _working(family, y, eta, w, phi)
        beta_new = _wls(X, W, z - offset, ridge)
        dev_new = dev_of(beta_new)

        # Halve the step while the deviance increases (or is non-finite).
        # Tolerance scales with the dtype: f32 deviance sums carry ~1e-5
        # relative rounding noise, which at a fixed 1e-10 read as increases
        # and triggered max_halving halvings every iteration (10x slower).
        htol = max(1e-10, 1e3 * float(jnp.finfo(dtype).eps))

        def h_cond(s):
            _, d, k = s
            bad = jnp.logical_or(~jnp.isfinite(d), d > dev * (1 + htol) + htol)
            return jnp.logical_and(bad, k < max_halving)

        def h_body(s):
            b, _, k = s
            b = 0.5 * (b + beta)
            return b, dev_of(b), k + 1

        beta_new, dev_new, _ = lax.while_loop(h_cond, h_body, (beta_new, dev_new, 0))
        # Converge on the deviance (as R/edgeR) *and* on the coefficient step,
        # since Fisher scoring for non-canonical links (NB-log) is only
        # linearly convergent and a deviance-only test stops at |dbeta|~sqrt(tol).
        # Fits whose deviance is ~0 (e.g. all-zero rows, beta -> -inf) are
        # accepted on the deviance criterion alone.
        dev_conv = jnp.abs(dev - dev_new) / (jnp.abs(dev_new) + 0.1) < tol
        step = jnp.max(jnp.abs(beta_new - beta))
        step_conv = step < beta_tol * (1.0 + jnp.max(jnp.abs(beta_new)))
        conv = dev_conv & (step_conv | (dev_new < 1e-6))
        return beta_new, dev_new, it + 1, conv

    beta, dev, n_iter, conv = lax.while_loop(
        cond, body, (beta, dev, jnp.asarray(0), jnp.asarray(False))
    )

    # ---- final quantities ---------------------------------------------------
    eta = X @ beta + offset
    W, _, mu = _working(family, y, eta, w, phi)
    p = X.shape[1]
    A = (X.T * W) @ X + ridge * jnp.eye(p, dtype=dtype)
    cov = _spd_inv(A)
    return GLMResult(beta, dev, mu, cov, n_iter, conv)


# --------------------------------------------------------------------------- #
# Batched public API
# --------------------------------------------------------------------------- #
def _broadcast_inputs(Y, X, offset, weights, phi, beta0):
    Y = jnp.asarray(Y)
    X = jnp.asarray(X)
    # Floating Y/X keep their precision (f32 in -> f32 fit); integer counts get
    # float64 when jax_enable_x64 is on (recommended), float32 otherwise.
    floats = [a.dtype for a in (Y, X) if jnp.issubdtype(a.dtype, jnp.floating)]
    dtype = jnp.result_type(*floats) if floats else jax.dtypes.canonicalize_dtype(jnp.float64)
    Y = Y.astype(dtype)
    X = X.astype(dtype)
    G, n = Y.shape
    p = X.shape[1]
    offset = jnp.broadcast_to(jnp.zeros((), dtype) if offset is None else jnp.asarray(offset, dtype), (G, n))
    weights = jnp.broadcast_to(jnp.ones((), dtype) if weights is None else jnp.asarray(weights, dtype), (G, n))
    phi = jnp.broadcast_to(jnp.zeros((), dtype) if phi is None else jnp.asarray(phi, dtype), (G,))
    use_beta0 = beta0 is not None
    beta0 = jnp.broadcast_to(jnp.zeros((), dtype) if beta0 is None else jnp.asarray(beta0, dtype), (G, p))
    return Y, X, offset, weights, phi, beta0, use_beta0


@partial(jax.jit, static_argnames=("family", "max_iter", "tol", "beta_tol", "ridge", "max_halving", "use_beta0",
                                   "batch_size"))
def _fit_batched(Y, X, offset, weights, phi, beta0, use_beta0, family, max_iter, beta_tol,
                 tol, ridge, max_halving, batch_size=None):
    def run(Y, offset, weights, phi, beta0, use_beta0, max_iter, chunk):
        kernel = partial(_fit_single, family=family, max_iter=max_iter, tol=tol, beta_tol=beta_tol,
                         ridge=ridge, max_halving=max_halving)
        if not chunk or batch_size is None:
            return jax.vmap(kernel, in_axes=(0, None, 0, 0, 0, 0, None))(
                Y, X, offset, weights, phi, beta0, use_beta0
            )
        # vmap over chunks of batch_size features: peak memory ~ batch_size * n * p
        return lax.map(lambda a: kernel(a[0], X, a[1], a[2], a[3], a[4], use_beta0),
                       (Y, offset, weights, phi, beta0), batch_size=batch_size)

    # Under vmap every feature waits for the slowest: on real data the median
    # gene needs 6 iterations and one in a thousand needs 49.  So: a capped
    # first pass for everyone, then the remaining iterations for a fixed-size
    # bucket of the least-converged features (static shapes, still one program).
    bucket = max(_STRAGGLER_BUCKET, Y.shape[0] // 8)  # pass 2 costs at most ~40% of pass 1
    if Y.shape[0] <= bucket or max_iter <= _FIRST_PASS_ITER:
        return run(Y, offset, weights, phi, beta0, use_beta0, max_iter, True)
    res = run(Y, offset, weights, phi, beta0, use_beta0, _FIRST_PASS_ITER, True)
    idx = jnp.argsort(res.converged)[:bucket]  # unconverged first
    res2 = run(Y[idx], offset[idx], weights[idx], phi[idx], res.coef[idx], True,
               max_iter - _FIRST_PASS_ITER, False)
    res2 = res2._replace(n_iter=res2.n_iter + res.n_iter[idx])
    redo = ~res.converged[idx]
    return jax.tree_util.tree_map(
        lambda a, b: a.at[idx].set(jnp.where(redo.reshape((-1,) + (1,) * (b.ndim - 1)), b, a[idx])),
        res, res2)


def fit_glm(
    Y,
    X,
    *,
    family: Family = NegativeBinomial(),
    offset=None,
    weights=None,
    phi=None,
    beta0=None,
    max_iter: int = 50,
    tol: float = 1e-8,
    beta_tol: float = 1e-8,
    ridge: float = 1e-10,
    max_halving: int = 30,
    batch_size: int | None = None,
) -> GLMResult:
    """Fit one GLM per row of ``Y`` with a shared design ``X``.

    Parameters
    ----------
    Y : (G, n) array. A 1-D ``(n,)`` input is treated as a single feature.
    X : (n, p) design matrix.
    family : a :class:`~glm_jax.families.Family` instance (static).
    offset : (n,) or (G, n), e.g. ``log(lib_size * norm_factors)``.
    weights : prior weights, (n,) or (G, n).
    phi : family parameter per feature, e.g. NB dispersion, () or (G,).
    beta0 : optional warm start (p,) or (G, p).
    batch_size : fit this many features at a time (caps memory at large n);
        ``None`` fits all features in one batch (CPU: cache-sized chunks).

    Notes
    -----
    ``phi=None`` means phi = 0, which with the default NB family is a Poisson fit.
    Features are fit in two passes: 12 iterations for all, then the remaining
    iterations for the max(256, G/8) least-converged ones. On very hard data
    (low depth, many separated features) more than that may come back with
    ``converged=False`` after 12 iterations; check ``converged.mean()``.
    """
    squeeze = jnp.ndim(Y) == 1
    if squeeze:
        Y = jnp.atleast_2d(Y)
    Y, X, offset, weights, phi, beta0, use_beta0 = _broadcast_inputs(
        Y, X, offset, weights, phi, beta0
    )
    if batch_size is None and jax.default_backend() == "cpu" and _n_cores() == 1:
        # ponytail: cache blocking for a single core. One gene chunk ~ 2.5 MB of f64
        # keeps the per-iteration intermediates in L2/L3 (1.4x on 10k cells);
        # bulk-sized n gets one chunk. Chunks run sequentially, so with several
        # cores XLA's parallelism over genes is worth more; GPUs want the whole batch.
        chunk = max(1, _CPU_CHUNK_ELEMS // Y.shape[1])
        batch_size = chunk if chunk < Y.shape[0] else None
    res = _fit_batched(Y, X, offset, weights, phi, beta0, use_beta0, family,
                       max_iter, beta_tol, tol, ridge, max_halving, batch_size)
    if squeeze:
        res = jax.tree_util.tree_map(lambda a: a[0], res)
    return res
