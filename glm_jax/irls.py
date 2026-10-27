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

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve
from jax.scipy.special import xlogy

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


def _newton_terms(family: Family, y, eta, w, phi):
    """Observed-information weights and score terms per observation.

    With g(eta) = mu_eta / V(mu), the score is s = w (y - mu) g and the
    observed information is w (d g - (y - mu) g').  For canonical links
    g' = 0, so this is the same as Fisher scoring; for NB-log it gives
    w mu (1 + phi y) / (1 + phi mu)^2.  Where the observed weight is not
    positive we use the Fisher weight there, so A stays SPD.
    """
    def g_of(e):
        mu = family.clip_mu(family.link.inverse(e))
        return family.link.mu_eta(e) / family.variance(mu, phi), mu

    (g, mu), (dg, _) = jax.jvp(g_of, (eta,), (jnp.ones_like(eta),))
    d = family.link.mu_eta(eta)
    r = w * (y - mu)
    W_fisher = w * d * g
    W_obs = W_fisher - r * dg
    W = jnp.where(W_obs > 0, W_obs, W_fisher)
    return W, r * g


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
_FIRST_PASS_ITER = 6
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


def _xtwx(X, W, ridge):
    """X^T diag(W) X + ridge I from the p(p+1)/2 unique column products.

    X is shared, so the products are batch-invariant under vmap and the
    batched information matrix is one (G, n) @ (n, q) GEMM that reads W
    once, instead of a (G, p, n) X^T*W intermediate and p^2 batched dots.
    """
    p = X.shape[1]
    i, j = np.triu_indices(p)
    flat = W @ (X[:, i] * X[:, j])
    A = jnp.zeros((p, p), X.dtype).at[i, j].set(flat).at[j, i].set(flat)
    return A + ridge * jnp.eye(p, dtype=X.dtype)


def _wls(X, W, z, ridge):
    return _spd_solve(_xtwx(X, W, ridge), X.T @ (W * z))


def _deviance(family, y, mu, w, phi):
    return jnp.sum(w * family.unit_deviance(y, mu, phi))


def _nb_split(y, mu, lm, phi):
    """NB unit deviance / 2 as (y-only part, mu-dependent part); phi = 0 is Poisson."""
    pos = phi > 0
    inv_phi = 1.0 / jnp.where(pos, phi, 1.0)
    const = xlogy(y, y) - jnp.where(pos, (y + inv_phi) * jnp.log1p(phi * y), y)
    var = jnp.where(pos, (y + inv_phi) * jnp.log1p(phi * mu), mu) - y * lm
    return const, var


def _nb_log_bounds(family, dtype):
    """(log mu_min, log mu_max) if the family's deviance matches the NB-log split, else None.

    Checked numerically against family.unit_deviance at trace time, so a
    family whose formula or clipping differs just takes the generic path.
    f64 only: the split subtracts per-gene sums, too noisy for f32 step halving.
    """
    if dtype != jnp.float64 or not isinstance(family, NegativeBinomial):
        return None
    try:
        with jax.ensure_compile_time_eval():
            f64 = jnp.float64
            e = jnp.array([-20.0, -3.0, 0.0, 2.0, 8.0], f64)
            if not np.allclose(family.link.inverse(e), np.exp(np.asarray(e)), rtol=1e-13, atol=0):
                return None
            lo = float(family.clip_mu(jnp.asarray(0.0, f64)))
            hi = float(family.clip_mu(jnp.asarray(1e300, f64)))
            loglo = float(np.log(lo)) if lo > 0 else -np.inf
            loghi = float(np.log(hi)) if np.isfinite(hi) else np.inf
            mu = family.clip_mu(jnp.exp(e))
            lm = jnp.clip(e, loglo, loghi)
            if not np.allclose(np.log(np.asarray(mu)), np.asarray(lm), rtol=1e-13, atol=1e-13):
                return None
            y = jnp.array([0.0, 1.0, 3.0, 50.0, 7.0], f64)
            for ph in (0.0, 0.1, 2.0):
                ph = jnp.asarray(ph, f64)
                c, v = _nb_split(y, mu, lm, ph)
                ref = family.unit_deviance(y, mu, ph)
                if not np.allclose(np.asarray(ref), 2 * np.asarray(c + v), rtol=1e-10, atol=1e-10):
                    return None
    except Exception:
        return None
    return loglo, loghi


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

    bounds = _nb_log_bounds(family, dtype)
    if bounds is None:
        def dev_of(b):
            eta = X @ b + offset
            mu = family.clip_mu(family.link.inverse(eta))
            return _deviance(family, y, mu, w, phi)
    else:
        # y-only deviance terms once per fit; each evaluation then needs only
        # exp(eta) and log1p(phi mu) per cell (log mu = eta, clipped as mu is).
        loglo, loghi = bounds
        c_y, _ = _nb_split(y, y, jnp.zeros_like(y), phi)
        dev_const = jnp.sum(w * c_y)

        def dev_of(b):
            eta = X @ b + offset
            mu = family.clip_mu(jnp.exp(eta))
            _, v = _nb_split(y, mu, jnp.clip(eta, loglo, loghi), phi)
            return 2.0 * (dev_const + jnp.sum(w * v))

    dev = dev_of(beta)

    # ---- main Newton loop with step halving --------------------------------
    def cond(state):
        _, _, it, conv = state
        return jnp.logical_and(it < max_iter, jnp.logical_not(conv))

    def body(state):
        beta, dev, it, _ = state
        eta = X @ beta + offset
        W, s = _newton_terms(family, y, eta, w, phi)
        # Newton step as one weighted solve: A beta_new = X^T (W (eta - offset) + s).
        # No working response z, so zero weights cannot produce 0/0.
        A = _xtwx(X, W, ridge)
        beta_new = _spd_solve(A, X.T @ (W * (eta - offset) + s))
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
        # Converge on the deviance (as R/edgeR) *and* on the coefficient step.
        # Fits whose deviance is ~0 (e.g. all-zero rows, beta -> -inf) are
        # accepted on the deviance criterion alone.
        dev_conv = jnp.abs(dev - dev_new) / (jnp.abs(dev_new) + 0.1) < tol
        step = jnp.max(jnp.abs(beta_new - beta))
        # Same dtype floor on the step: in f32 a beta_tol of 1e-5 sits under the
        # rounding jitter of beta, so ~3% of genes ran to max_iter (Hao-like 10k,
        # 0.10 vs 0.06 s on GPU) with no gain in accuracy.
        step_conv = step < max(beta_tol, htol) * (1.0 + jnp.max(jnp.abs(beta_new)))
        conv = dev_conv & (step_conv | (dev_new < 1e-6))
        return beta_new, dev_new, it + 1, conv

    beta, dev, n_iter, conv = lax.while_loop(
        cond, body, (beta, dev, jnp.asarray(0), jnp.asarray(False))
    )

    # ---- final quantities (Fisher information for cov_unscaled) -------------
    eta = X @ beta + offset
    W, _, mu = _working(family, y, eta, w, phi)
    cov = _spd_inv(_xtwx(X, W, ridge))
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
    # gene needs ~5 iterations and one in a thousand needs ~50.  So: a short
    # first pass for everyone, then two re-fit stages on the unconverged ones.
    # Each stage picks, at run time, the smallest bucket from a static ladder
    # that holds every unconverged feature (static shapes, still one program).
    G = Y.shape[0]
    K = _FIRST_PASS_ITER
    if G <= _STRAGGLER_BUCKET or max_iter <= 2 * K:
        return run(Y, offset, weights, phi, beta0, use_beta0, max_iter, True)
    # ponytail: ladder tops out at G/2. A full-G rung gathers ~5 (G, n) copies (Y, offset,
    # weights, mu, scatter) and OOMs a 24 GB GPU at 161k cells; past G/2 unconverged the
    # extras come back converged=False. Refit the top rung in place if that ever matters.
    sizes = sorted({0} | {max(_STRAGGLER_BUCKET, G >> k) for k in (1, 2, 3)})

    def norm(r):  # switch branches must agree on (weak) dtypes
        return r._replace(n_iter=r.n_iter.astype(jnp.int32))

    def stage(res, n_it):
        idx = jnp.argsort(res.converged)  # unconverged first

        def branch(b):
            if b == 0:
                return lambda r: r

            def f(r):
                i = idx[:b]
                # chunk=True: re-fits honour batch_size too (unbatched they OOM at large n)
                r2 = run(Y[i], offset[i], weights[i], phi[i], r.coef[i], True, n_it, True)
                r2 = norm(r2._replace(n_iter=r2.n_iter + r.n_iter[i]))
                redo = ~r.converged[i]
                return jax.tree_util.tree_map(
                    lambda a, c: a.at[i].set(jnp.where(redo.reshape((-1,) + (1,) * (c.ndim - 1)), c, a[i])),
                    r, r2)
            return f

        k = jnp.searchsorted(jnp.asarray(sizes), jnp.sum(~res.converged))
        return lax.switch(k, [branch(b) for b in sizes], res)

    res = norm(run(Y, offset, weights, phi, beta0, use_beta0, K, True))
    res = stage(res, K)
    return stage(res, max_iter - 2 * K)


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
    Features are fit in stages: 6 iterations for all, then 6 more and then the
    remaining iterations for the unconverged ones (re-started from their
    current coefficients, in the smallest bucket of G/8, G/4, G/2 that fits).
    If more than G/2 are unconverged after the first pass, the rest come back
    with ``converged=False``; check ``converged.mean()`` on very hard data.
    NB coefficients use Newton steps with the observed information (falling
    back to Fisher weights where those are not positive); ``cov_unscaled`` is
    the Fisher (X^T W X)^-1 at convergence, as before.
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
