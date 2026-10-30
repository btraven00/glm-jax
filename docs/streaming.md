# Streaming large datasets

`fit_glm` and `estimate_dispersion_grid` take the whole genes × cells matrix on
the device. At 1M cells × 2k genes that is 8 GB in f32 before any
intermediates. The streamed versions keep `Y` in host memory (dense numpy or
scipy.sparse, genes in rows) and send it to the device in chunks of genes:

```python
import glm_jax as gj

fit, chunk = gj.fit_glm_streamed(Y, X, offset=log_umi, phi=0.1)
disp, chunk = gj.estimate_dispersion_streamed(Y, X, offset=log_umi)
# disp["tagwise"]: per-gene dispersion; disp["apl"], disp["phi_grid"]: the grid
```

Both return the chunk size they used. Passing `chunk=` skips planning.
`estimate_dispersion_streamed` returns no polished common dispersion, because
that needs every gene at once. The grid one is
`phi_grid[argmax(apl.sum(0))]`.

## Choosing the chunk size

The chunk is the largest number of genes whose compiled program fits in 80% of
the device's free memory (`device_budget()`). Sizes come from XLA's own
accounting (`memory_analysis`), not from a hand-tuned constant:

1. **Calibration.** Three compiles at 4,096 cells and chunks of 32, 256 and
   1024 genes give bytes per cell at each chunk size. Program memory is linear
   in the number of cells (200k cells used 9.9× the bytes of 20k), so this
   table extrapolates to any cell count.
2. **Interpolation.** The chunk that fits the budget is read off the table and
   rounded down to a multiple of 256 genes when it exceeds 256.
3. **Check.** One compile at full size confirms the chunk fits, and shrinks it
   if XLA disagrees. That compiled program is the one that runs.

## Caches

Both caches live in `$XDG_CACHE_HOME/glm_jax/`, which defaults to
`~/.cache/glm_jax/`.

| Cache | Holds | Keyed on | Size |
|---|---|---|---|
| `plan.json` | calibration tables (bytes per cell per chunk size) | kernel source hash (`irls.py` + `dispersion.py`), device kind, JAX version, dtype, number of covariates, kernel settings | a few KB |
| `xla/` | compiled XLA programs (JAX persistent compilation cache) | the program and its compile options, by JAX | ~10 MB per data shape |

- The XLA cache is switched on at the first streamed call, unless
  `JAX_COMPILATION_CACHE_DIR` (or `jax.config.jax_compilation_cache_dir`) is
  already set. In that case your setting is used as is.
- A change to `irls.py` or `dispersion.py` invalidates the matching
  `plan.json` entries. Old entries stay in the file but are never read.
- Both caches are safe to delete (`rm -rf ~/.cache/glm_jax`). The next run
  recalibrates and recompiles.
- On a shared cluster home directory, one cache serves every node with the
  same GPU model; entries are keyed on the device kind.

## What compiling costs

A streamed program compiles in 13–145 s on a GPU, depending on the program,
and the time hardly depends on the number of cells. Most of it is program
size; about a third is XLA's GPU autotuning.

Measured on Hao PBMC 100k cells × 2,000 genes, 10 covariates, laptop RTX 2000
Ada, f32, streamed dispersion:

| Call | Time |
|---|---|
| First call ever, or after a kernel change or new data shape (planning: 3 calibration compiles plus 1 full-size compile) | 379 s |
| Same data shape, new process (calibration table and programs cached) | 12 s |
| Second call in the same process | 7.4 s |
| Steady-state compute, all 2,000 genes | 6.0 s |

The 6–12 s on cached calls is JAX tracing the program. The cache cannot skip it.

Turning autotuning off (`XLA_FLAGS=--xla_gpu_autotune_level=0`) cuts compiles
by 15–40 s but makes f32 runs 13–27% slower, so it is left on.

## Dispersion defaults

`estimate_dispersion_grid` walks a 30-point grid of φ from 1e-4 to 10
(`dispersion.PHI_GRID`). It then takes 2 Newton steps in log φ from each
gene's best grid point.

- **Grid size.** Against a 60-point grid, the 30-point grid gave up at most
  0.008 in adjusted profile log-likelihood per gene on three 100k-cell datasets
  (Hao PBMC, aifi, splatter). At 100k cells the profiles are sharply peaked:
  20 points gave up as much as 0.77, because 2 Newton steps from a coarser grid
  fall short of the peak.
- **Newton derivatives.** The Newton step uses forward-over-forward automatic
  differentiation, since log φ is one scalar per gene. Reverse mode stored
  every intermediate array. In f32 the forward form uses 78 bytes per
  gene × cell against 248, and in f64 154 against 560. That is what lets the
  planner choose larger chunks.

Steady-state streamed dispersion on Hao PBMC 100k, laptop GPU, before and
after these changes:

| | f32 | f64 |
|---|---|---|
| before | 11.2 s (chunk 159) | 81.0 s (chunk 69) |
| after | 6.0 s (chunk 256) | 55.1 s (chunk 132) |
