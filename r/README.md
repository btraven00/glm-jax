# glmjax

Fit one negative-binomial GLM per gene from R, with the
[glm-jax](https://github.com/btraven00/glm-jax) kernel and no Python at runtime.

It is a fast fitter to put underneath differential-expression code, not a
differential-expression tool: you supply the dispersion.

## Install

```r
install.packages("pjrt", repos = c("https://r-xla.r-universe.dev", getOption("repos")))  # not on CRAN
install.packages("remotes")
remotes::install_github("btraven00/glm-jax", subdir = "r")   # also installs xlaport from GitHub
pjrt::install_pjrt()   # once: downloads the XLA CPU plugin (64 MB), plus CUDA if an NVIDIA GPU is found
```

On Linux `pjrt` builds from source and needs the protobuf compiler
(`protobuf-compiler` and `libprotobuf-dev` on Debian/Ubuntu).

## Use

```r
fit <- glmjax::fit_glm(Y, X, offset = log(size_factors), dispersion = 0.1)

fit$Beta         # genes x coefficients
fit$deviance     # one per gene
fit$converged    # check this on hard data
fit$iterations
```

| Argument | What |
|---|---|
| `Y` | counts, genes in rows and cells in columns; a dense matrix or a sparse `Matrix` |
| `X` | design matrix, one row per cell, full rank |
| `dispersion` | one value, or one per gene; 0 (the default) is a Poisson fit |
| `offset`, `weights` | one value per cell |
| `precision` | `"double"` (default) or `"single"` |

The first call compiles the kernel, which takes about a second on CPU.

### GPU

```r
Sys.setenv(PJRT_PLATFORM = "cuda")    # before the first fit
fit <- glmjax::fit_glm(Y, X, dispersion = 0.1, precision = "single")
```

Use `precision = "single"` on a GPU: consumer cards run float64 at a fraction
of their float32 rate. Coefficients then carry float32 rounding, about 1e-6
relative.

The [vignette](vignettes/spatial-rds.Rmd) fits all 18,082 genes of a Visium HD
`SpatialExperiment` (443,765 bins) on a laptop GPU, reading the `.rds` in
blocks.

## Limits

- At most 12 design columns and 3,145,728 cells.
- No dispersion estimation. Take it from edgeR or glmGamPoi, or fix a value.
- Offsets and weights are per cell, not per gene and cell.
- Genes are fitted in chunks of 16 to 256, and a chunk runs until its slowest
  gene converges. All-zero genes need about 20 iterations where a typical gene
  needs 5, so drop them first.

## How it works

The kernel is exported from JAX as StableHLO, one program per cell-count
bucket and precision, and shipped in `inst/xla/`.
[`xlaport`](https://github.com/btraven00/xlaport) compiles the right one
through [`pjrt`](https://github.com/r-xla/pjrt). Your data is padded to the
program's shape (extra cells get weight 0, the design gets zero columns), which
does not change the result.

## Accuracy and speed

- **Tests:** coefficients match `glm.fit` to 1e-7 for Poisson and for negative
  binomial (via `MASS`). They run on CPU and CUDA, in both precisions.
- **Against Python:** on Hao PBMC (2,000 genes x 10,000 cells, 10 covariates)
  coefficients match the Python package to 2.8e-14. The fit takes 2.6 s in R
  and 1.6 s in Python on the same 16 cores.
- **Visium HD lung** (256 genes x 443,765 bins, Poisson with an offset).
  Intercepts match the closed form to 1e-9 in double and 1e-5 in single.

  | Device | Precision | Genes per second |
  |---|---|---|
  | CPU | double | 6.6 |
  | CPU | single | 10.2 |
  | RTX 2000 Ada laptop GPU | double | 17.6 |
  | RTX 2000 Ada laptop GPU | single | 35 |

## Rebuilding the kernels

After any change to `glm_jax/irls.py` or `families.py`, from the repository root:

```
pip install -e . "git+https://github.com/btraven00/xlaport#subdirectory=python"
JAX_PLATFORMS=cpu python r/data-raw/export_kernels.py
```

Then run the R tests.
