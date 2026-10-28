"""Exponential-family distributions and link functions for GLM fitting.

Families and links are frozen (hashable) dataclasses so they can be passed as
*static* arguments to ``jax.jit``.  Any per-feature distribution parameter
(e.g. the negative-binomial dispersion) is passed separately as an array
``phi`` so it can be batched with ``vmap``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, xlogy


def _lgamma_ratio(y, r):
    """lgamma(y + r) - lgamma(r) - y log r, for y >= 0, r > 0.

    The direct form subtracts two lgammas of size ~ r log r: in f32 at r = 1/phi
    ~ 1e3 that noise flipped the sign of the dispersion Newton gradient on a
    high-count gene (50k cells). For r >= 10 use the Stirling difference
    r h(y/r) - log1p(y/r)/2 + s(y + r) - s(r), h(u) = (1 + u) log1p(u) - u
    (series below u = 0.1), s the Stirling correction (error < 1e-10 at r >= 10).
    float64 keeps the direct form: accurate enough there, and the Stirling one
    costs ~1.8x in the dispersion polish (two branches + series per cell).
    """
    if jnp.result_type(y, r) == jnp.float64:
        return gammaln(y + r) - gammaln(r) - y * jnp.log(r)
    big = r >= 10.0
    rb = jnp.where(big, r, 10.0)  # keep the unused branch finite (and its gradient)
    u = y / rb
    us = jnp.minimum(u, 0.1)
    series = sum((-1.0) ** k * us**k / (k * (k - 1)) for k in range(2, 11))
    h = jnp.where(u < 0.1, series, (1.0 + u) * jnp.log1p(u) - u)

    def s(x):
        ix2 = 1.0 / (x * x)
        return (1.0 / 12 - ix2 * (1.0 / 360 - ix2 / 1260)) / x

    stirling = rb * h - 0.5 * jnp.log1p(u) + s(y + rb) - s(rb)
    return jnp.where(big, stirling, gammaln(y + r) - gammaln(r) - y * jnp.log(r))


# --------------------------------------------------------------------------- #
# Links
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Link:
    name: str = "base"

    def link(self, mu):  # g(mu)
        raise NotImplementedError

    def inverse(self, eta):  # g^{-1}(eta)
        raise NotImplementedError

    def mu_eta(self, eta):  # d mu / d eta
        raise NotImplementedError


@dataclass(frozen=True)
class LogLink(Link):
    name: str = "log"

    def link(self, mu):
        return jnp.log(mu)

    def inverse(self, eta):
        return jnp.exp(eta)

    def mu_eta(self, eta):
        return jnp.exp(eta)


@dataclass(frozen=True)
class IdentityLink(Link):
    name: str = "identity"

    def link(self, mu):
        return mu

    def inverse(self, eta):
        return eta

    def mu_eta(self, eta):
        return jnp.ones_like(eta)


@dataclass(frozen=True)
class LogitLink(Link):
    name: str = "logit"

    def link(self, mu):
        return jnp.log(mu) - jnp.log1p(-mu)

    def inverse(self, eta):
        return jax.nn.sigmoid(eta)  # stable for large |eta|

    def mu_eta(self, eta):
        p = jax.nn.sigmoid(eta)
        return p * (1.0 - p)


# --------------------------------------------------------------------------- #
# Families
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Family:
    """Base exponential family.

    All methods take ``phi`` (scalar per feature) even if unused, so every
    family has the same call signature inside the IRLS kernel.
    """

    link: Link = LogLink()

    def variance(self, mu, phi):
        raise NotImplementedError

    def unit_deviance(self, y, mu, phi):
        raise NotImplementedError

    def loglik(self, y, mu, phi):
        """Per-observation log-likelihood (including constants)."""
        raise NotImplementedError

    def init_mu(self, y):
        """Starting value for mu (must lie in the valid domain)."""
        raise NotImplementedError

    def clip_mu(self, mu):
        return mu


@dataclass(frozen=True)
class Gaussian(Family):
    link: Link = IdentityLink()

    def variance(self, mu, phi):
        return jnp.ones_like(mu)

    def unit_deviance(self, y, mu, phi):
        return (y - mu) ** 2

    def loglik(self, y, mu, phi):
        # phi is the residual variance sigma^2 here
        return -0.5 * (jnp.log(2 * jnp.pi * phi) + (y - mu) ** 2 / phi)

    def init_mu(self, y):
        return y


@dataclass(frozen=True)
class Poisson(Family):
    link: Link = LogLink()

    def variance(self, mu, phi):
        return mu

    def unit_deviance(self, y, mu, phi):
        return 2.0 * (xlogy(y, y) - xlogy(y, mu) - (y - mu))

    def loglik(self, y, mu, phi):
        return xlogy(y, mu) - mu - gammaln(y + 1.0)

    def init_mu(self, y):
        return y + 0.1

    def clip_mu(self, mu):
        return jnp.maximum(mu, 1e-300)


@dataclass(frozen=True)
class NegativeBinomial(Family):
    """NB2 parametrisation used by edgeR / DESeq2: Var = mu + phi * mu^2.

    ``phi`` is the dispersion (= 1/size).  ``phi -> 0`` recovers Poisson.
    """

    link: Link = LogLink()

    def variance(self, mu, phi):
        return mu + phi * mu**2

    def unit_deviance(self, y, mu, phi):
        pois = 2.0 * (xlogy(y, y) - xlogy(y, mu) - (y - mu))
        # Guard phi == 0 (Poisson limit) to keep gradients finite.
        phi_s = jnp.where(phi > 0, phi, 1.0)
        nb = 2.0 * (
            xlogy(y, y) - xlogy(y, mu)
            - (y + 1.0 / phi_s) * (jnp.log1p(phi_s * y) - jnp.log1p(phi_s * mu))
        )
        return jnp.where(phi > 0, nb, pois)

    def loglik(self, y, mu, phi):
        pois = xlogy(y, mu) - mu - gammaln(y + 1.0)
        phi_s = jnp.where(phi > 0, phi, 1.0)
        r = 1.0 / phi_s
        nb = (
            _lgamma_ratio(y, r) - gammaln(y + 1.0)
            + xlogy(y, mu) - (y + r) * jnp.log1p(mu * phi_s)
        )
        return jnp.where(phi > 0, nb, pois)

    def init_mu(self, y):
        return y + 0.1

    def clip_mu(self, mu):
        return jnp.maximum(mu, 1e-300)


@dataclass(frozen=True)
class Binomial(Family):
    """Binomial on proportions ``y in [0,1]``; pass trial counts as weights."""

    link: Link = LogitLink()

    def variance(self, mu, phi):
        return mu * (1.0 - mu)

    def unit_deviance(self, y, mu, phi):
        return 2.0 * (
            xlogy(y, y) - xlogy(y, mu)
            + xlogy(1.0 - y, 1.0 - y) - xlogy(1.0 - y, 1.0 - mu)
        )

    def loglik(self, y, mu, phi):
        # Up to the binomial coefficient (requires the trial count).
        return xlogy(y, mu) + xlogy(1.0 - y, 1.0 - mu)

    def init_mu(self, y):
        return (y + 0.5) / 2.0

    def clip_mu(self, mu):
        eps = 1e-12
        return jnp.clip(mu, eps, 1.0 - eps)
