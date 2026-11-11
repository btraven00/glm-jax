# glmjax

The glm-jax negative-binomial fit as an R function, with no Python at runtime.
One GLM per gene at a dispersion you supply: a fast fitter to put underneath
differential-expression code, not a DE tool.

```r
fit <- glmjax::fit_glm(Y, X, offset = log(size_factors), dispersion = 0.1)
fit$Beta        # genes x coefficients
fit$deviance; fit$converged; fit$iterations
fit <- glmjax::fit_glm(Y, X, dispersion = 0.1, precision = "single")  # float32 kernels, for GPUs
```

`Y` is genes x cells (dense, or anything `as.matrix()` handles per chunk, such as
a sparse `Matrix`); `dispersion` is one value or one per gene, and 0 is Poisson.

## Install

```r
install.packages("pjrt", repos = c("https://r-xla.r-universe.dev", getOption("repos")))  # not on CRAN
install.packages("remotes")
remotes::install_github("btraven00/glm-jax", subdir = "r")   # also installs xlaport from GitHub
pjrt::install_pjrt()   # once: downloads the XLA CPU plugin (64 MB), plus CUDA if an NVIDIA GPU is found
```

`pjrt` builds from source on Linux and needs the protobuf compiler
(`protobuf-compiler` and `libprotobuf-dev` on Debian/Ubuntu).

## How it works

The kernel ([`glm_jax.fit_glm`](https://github.com/btraven00/glm-jax)) is exported from JAX as StableHLO, one program
per cell-count bucket, and shipped in `inst/xla/`. [`xlaport`](https://github.com/btraven00/xlaport)
compiles the right one through [`pjrt`](https://github.com/r-xla/pjrt) on first
call (about a second on CPU) and keeps it for the session. Your data is padded
to the program's shape, cells with weight 0 and the design with zero columns,
which leaves the result unchanged.

## Limits

- At most 12 design columns and 3,145,728 cells.
- The tests run on CPU and CUDA (`PJRT_PLATFORM=cuda`), in float64 and float32
  (`precision = "single"`, coefficients to about 1e-6 relative).
- Per-cell offsets and weights only (no gene x cell matrices).
- No dispersion estimation yet: supply it (edgeR, glmGamPoi) or fit at a fixed value.
- Genes are fitted 16 to 256 at a time and a chunk runs until its slowest gene
  converges, so hard genes cost more here than in the Python package.

## Checked

- `tests/testthat`: coefficients match `glm.fit` (Poisson, and NB via `MASS`) to 1e-7,
  with cell, column and gene padding all in play.
- Hao PBMC, 2,000 genes x 10,000 cells, 10 covariates: coefficients match the
  Python package to 2.8e-14; 2.6 s against 1.6 s in Python on the same 16 cores.
- 10x Visium HD lung, 256 genes x 443,765 bins, Poisson with an offset: genes per
  second 6.6 (CPU, double), 10.2 (CPU, single), 17.6 (RTX 2000 Ada laptop GPU,
  double), 35 (same GPU, single); intercepts match the closed form to 1e-9 in
  double and 1e-5 in single.

## Rebuilding the kernels

```
pip install "git+https://github.com/btraven00/xlaport#subdirectory=python"
JAX_PLATFORMS=cpu python r/data-raw/export_kernels.py
```

Run it after any change to `glm_jax/irls.py` or `families.py`, then run the tests.
