"""NB dispersion estimation via Cox-Reid adjusted profile likelihood.

This is the main "repeated GLM calls" workload of edgeR: for every gene and
every trial dispersion we refit the NB GLM and evaluate

    APL(phi) = l(beta_hat(phi); phi) - 1/2 log det(X^T W X)

(edgeR ``adjustedProfileLik``).  Here all (gene, phi) pairs are flattened into
a single batched kernel call.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .families import NegativeBinomial
from .irls import fit_glm

_NB = NegativeBinomial()


def adjusted_profile_loglik(Y, X, phi, *, offset=None, weights=None, fit=None, **fit_kwargs):
    """Cox-Reid adjusted profile log-likelihood per feature. Returns (G,)."""
    Y = jnp.atleast_2d(jnp.asarray(Y))
    G, n = Y.shape
    if fit is None:
        fit = fit_glm(Y, X, family=_NB, offset=offset, weights=weights, phi=phi, **fit_kwargs)
    dtype = fit.mu.dtype
    w = jnp.ones((G, n), dtype) if weights is None else jnp.broadcast_to(jnp.asarray(weights, dtype), (G, n))
    phi_b = jnp.broadcast_to(jnp.asarray(phi, dtype), (G,))[:, None]
    Y = Y.astype(dtype)
    ll = jnp.sum(w * _NB.loglik(Y, fit.mu, phi_b), axis=-1)
    # log-link NB working weights: mu / (1 + phi mu)
    W = w * fit.mu / (1.0 + phi_b * fit.mu)
    X = jnp.asarray(X, dtype)
    A = jnp.einsum("ni,gn,nj->gij", X, W, X)
    _, logdet = jnp.linalg.slogdet(A)
    return ll - 0.5 * logdet


def apl_grid(Y, X, phi_grid, *, offset=None, weights=None, **fit_kwargs):
    """APL for every (feature, phi) pair. Returns (G, K)."""
    Y = jnp.atleast_2d(jnp.asarray(Y))
    G, n = Y.shape
    phi_grid = jnp.asarray(phi_grid)
    K = phi_grid.shape[0]

    def tile(a):
        if a is None:
            return None
        a = jnp.broadcast_to(jnp.asarray(a), (G, n))
        return jnp.repeat(a, K, axis=0)

    Yt = jnp.repeat(Y, K, axis=0)
    phit = jnp.tile(phi_grid, G)
    apl = adjusted_profile_loglik(Yt, X, phit, offset=tile(offset), weights=tile(weights), **fit_kwargs)
    return apl.reshape(G, K)


def estimate_dispersion_grid(Y, X, *, offset=None, weights=None,
                             phi_grid=None, **fit_kwargs):
    """Simple grid-search estimates.

    Returns a dict with
      * ``common``  : argmax_phi sum_g APL_g(phi)   (edgeR common dispersion)
      * ``tagwise`` : argmax_phi APL_g(phi)         (no EB shrinkage yet)
      * ``apl``     : (G, K) matrix and ``phi_grid``
    """
    if phi_grid is None:
        phi_grid = jnp.logspace(-4, 1, 60)
    apl = apl_grid(Y, X, phi_grid, offset=offset, weights=weights, **fit_kwargs)
    common = phi_grid[jnp.argmax(jnp.sum(apl, axis=0))]
    tagwise = phi_grid[jnp.argmax(apl, axis=1)]
    return {"common": common, "tagwise": tagwise, "apl": apl, "phi_grid": phi_grid}
