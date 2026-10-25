"""Hypothesis tests built on top of the batched GLM kernel."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.special import gammaincc, ndtr

from .families import Family, NegativeBinomial
from .irls import GLMResult, fit_glm


class TestResult(NamedTuple):
    stat: jnp.ndarray
    df: int
    pvalue: jnp.ndarray


def chi2_sf(x, df):
    return gammaincc(df / 2.0, jnp.maximum(x, 0.0) / 2.0)


def glm_lrt(
    Y,
    X_full,
    X_reduced,
    *,
    family: Family = NegativeBinomial(),
    full_fit: GLMResult | None = None,
    **fit_kwargs,
) -> tuple[TestResult, GLMResult, GLMResult]:
    """Likelihood-ratio test per feature (edgeR ``glmLRT`` analogue).

    ``fit_kwargs`` (offset, weights, phi, ...) are forwarded to both fits.
    Returns ``(test, full_fit, reduced_fit)``.
    """
    if full_fit is None:
        full_fit = fit_glm(Y, X_full, family=family, **fit_kwargs)
    reduced_fit = fit_glm(Y, X_reduced, family=family, **fit_kwargs)
    df = X_full.shape[1] - X_reduced.shape[1]
    stat = jnp.maximum(reduced_fit.deviance - full_fit.deviance, 0.0)
    return TestResult(stat, df, chi2_sf(stat, df)), full_fit, reduced_fit


def drop_coef(X, coef):
    """Reduced design obtained by dropping column(s) ``coef``."""
    coef = jnp.atleast_1d(jnp.asarray(coef))
    keep = [j for j in range(X.shape[1]) if j not in set(coef.tolist())]
    return jnp.asarray(X)[:, keep]


def wald_test(fit: GLMResult, contrast, scale=1.0) -> TestResult:
    """Wald z-test of ``contrast @ beta == 0`` (DESeq2-style).

    ``scale`` is the dispersion scale for quasi/Gaussian families
    (keep 1.0 for NB/Poisson where the variance is fully specified).
    """
    c = jnp.asarray(contrast, dtype=fit.coef.dtype)
    est = fit.coef @ c
    var = jnp.einsum("i,...ij,j->...", c, fit.cov_unscaled, c) * scale
    z = est / jnp.sqrt(var)
    return TestResult(z, 1, 2.0 * ndtr(-jnp.abs(z)))
