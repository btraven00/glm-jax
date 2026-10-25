"""glm_jax: batched, jit-compiled GLM fitting for omics pipelines.

Tip: enable double precision before importing for edgeR-level accuracy::

    import jax
    jax.config.update("jax_enable_x64", True)
"""

from .families import (
    Binomial,
    Family,
    Gaussian,
    IdentityLink,
    Link,
    LogitLink,
    LogLink,
    NegativeBinomial,
    Poisson,
)
from .irls import GLMResult, fit_glm
from .inference import TestResult, drop_coef, glm_lrt, wald_test
from .dispersion import adjusted_profile_loglik, apl_grid, estimate_dispersion_grid

__all__ = [
    "Binomial", "Family", "Gaussian", "IdentityLink", "Link", "LogitLink",
    "LogLink", "NegativeBinomial", "Poisson", "GLMResult", "fit_glm",
    "TestResult", "drop_coef", "glm_lrt", "wald_test",
    "adjusted_profile_loglik", "apl_grid", "estimate_dispersion_grid",
]
