# glm-jax

Fast GLM fitting for omics count data, in JAX: one GLM per gene with a shared
design matrix, on CPU or GPU.

It covers the repeated-GLM work in edgeR and DESeq2-style pipelines: fitting at
a given dispersion, estimating negative-binomial dispersions (Cox-Reid adjusted
profile likelihood), and likelihood-ratio and Wald tests.

## Install

```
pip install git+https://github.com/btraven00/glm-jax
```

For development, clone the repository, then `pip install -e ".[test]"` and
`pytest -q` (or `pixi run test`).

## Use

```python
import jax
jax.config.update("jax_enable_x64", True)   # double precision, recommended
import glm_jax as gj

# Y: genes x cells, X: cells x covariates
fit = gj.fit_glm(Y, X, offset=log_size_factors, phi=0.1)
disp = gj.estimate_dispersion_grid(Y, X, offset=log_size_factors)
test, full, reduced = gj.glm_lrt(Y, X, gj.drop_coef(X, 1),
                                 offset=log_size_factors, phi=disp["tagwise"])
```

`phi` is the dispersion: the variance is `mu + phi * mu^2`, and `phi=0` is
Poisson. The default family is negative binomial; Poisson, Gaussian and
binomial are also available.

For data that does not fit on the device, `fit_glm_streamed` and
`estimate_dispersion_streamed` keep `Y` in host memory and send genes over in
chunks. See [docs/streaming.md](docs/streaming.md).

## From R

[`r/`](r/) holds `glmjax`, an R package that runs the same fit kernel with no
Python at runtime. Install and usage are in [r/README.md](r/README.md).

## Repository

| Path | What |
|---|---|
| `glm_jax/` | the library |
| `tests/` | `pytest -q` |
| `examples/` | benchmarks against glmGamPoi, edgeR, DEVIL and delnx |
| `evolve/` | the LLM-driven search used to tune the dispersion kernel |
| `r/` | the `glmjax` R package |

MIT licence.
