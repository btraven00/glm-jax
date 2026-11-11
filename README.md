# glm-jax

Batched, jit-compiled GLM fitting in JAX for omics count data: one GLM per gene
with a shared design, on CPU or GPU. It covers the repeated-GLM workloads of
edgeR and DESeq2-style pipelines: fitting at a given dispersion, and
negative-binomial dispersion estimation by Cox-Reid adjusted profile likelihood.

```python
import jax
jax.config.update("jax_enable_x64", True)   # double precision, recommended
import glm_jax as gj

fit = gj.fit_glm(Y, X, offset=log_size_factors, phi=0.1)        # Y: genes x cells
disp = gj.estimate_dispersion_grid(Y, X, offset=log_size_factors)
test, full, reduced = gj.glm_lrt(Y, X, gj.drop_coef(X, 1), offset=log_size_factors, phi=disp["tagwise"])
```

Families: negative binomial (default), Poisson, Gaussian, binomial.
`fit_glm_streamed` and `estimate_dispersion_streamed` keep `Y` on the host and
size gene chunks to the device (see [docs/streaming.md](docs/streaming.md)).

## Install

```
pip install -e .            # or: pixi run test
```

## From R

[`r/`](r/) is the `glmjax` R package: the same fit kernel, exported as StableHLO
and run through [`xlaport`](https://github.com/btraven00/xlaport) and `pjrt`,
with no Python at runtime.

## Layout

- `glm_jax/`: the library (`irls.py`, `dispersion.py`, `families.py`, `inference.py`, `stream.py`)
- `tests/`: `pytest -q`
- `examples/`: benchmarks against glmGamPoi, edgeR, DEVIL and delnx
- `evolve/`: the LLM-driven search used to tune the dispersion kernel
- `r/`: the R package

MIT licence.
