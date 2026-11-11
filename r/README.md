# glmjax

The glm-jax negative-binomial fit as an R function, with no Python at runtime.
One GLM per gene at a dispersion you supply: a fast fitter to put underneath
differential-expression code, not a DE tool.

```r
fit <- glmjax::fit_glm(Y, X, offset = log(size_factors), dispersion = 0.1)
fit$Beta        # genes x coefficients
fit$deviance; fit$converged; fit$iterations
```

`Y` is genes x cells (dense, or anything `as.matrix()` handles per chunk, such as
a sparse `Matrix`); `dispersion` is one value or one per gene, and 0 is Poisson.

## Install

```r
options(repos = c("https://btraven00.r-universe.dev", "https://r-xla.r-universe.dev", getOption("repos")))
install.packages("glmjax")
pjrt::install_pjrt()   # once: downloads the XLA CPU plugin (64 MB), plus CUDA if an NVIDIA GPU is found
```

## How it works

The kernel ([`glm_jax.fit_glm`](https://github.com/btraven00/glm-jax)) is exported from JAX as StableHLO, one program
per cell-count bucket, and shipped in `inst/xla/`. [`xlaport`](https://github.com/btraven00/xlaport)
compiles the right one through [`pjrt`](https://github.com/r-xla/pjrt) on first
call (about a second on CPU) and keeps it for the session. Your data is padded
to the program's shape, cells with weight 0 and the design with zero columns,
which leaves the result unchanged.

## Limits

- At most 12 design columns and 3,145,728 cells.
- float64 on CPU is what the tests cover. `PJRT_PLATFORM=cuda` is untested from R.
- Per-cell offsets and weights only (no gene x cell matrices).
- No dispersion estimation yet: supply it (edgeR, glmGamPoi) or fit at a fixed value.
- Genes are fitted 16 to 256 at a time and a chunk runs until its slowest gene
  converges, so hard genes cost more here than in the Python package.

## Checked

- `tests/testthat`: coefficients match `glm.fit` (Poisson, and NB via `MASS`) to 1e-7,
  with cell, column and gene padding all in play.
- Hao PBMC, 2,000 genes x 10,000 cells, 10 covariates: coefficients match the
  Python package to 2.8e-14; 2.6 s against 1.6 s in Python on the same 16 cores.

## Rebuilding the kernels

```
pip install "git+https://github.com/btraven00/xlaport#subdirectory=python"
JAX_PLATFORMS=cpu python r/data-raw/export_kernels.py
```

Run it after any change to `glm_jax/irls.py` or `families.py`, then run the tests.
