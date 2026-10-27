# evolve/ — LLM-driven search over `glm_jax/dispersion.py` or `glm_jax/irls.py`

A propose → evaluate → keep loop: Claude writes whole-file variants of one kernel file
(`EVOLVE_TARGET=dispersion`, the default, or `EVOLVE_TARGET=irls`), a mechanical evaluator decides. The evaluator is the part to trust; the loop is
80 lines. Runs on the Claude Code subscription (`agent-box claude -p`), no API key.

## Run

```bash
pixi run -e test python evolve/run.py 4            # 4 rounds x 3 parallel proposals, Sonnet
EVOLVE_MODEL=opus pixi run -e test python evolve/run.py 2
EVOLVE_PARALLEL=1 EVOLVE_CORE=3 pixi run -e test python evolve/run.py 4
EVOLVE_TARGET=irls pixi run -e test python evolve/run.py 4   # evolve irls.py instead
EVOLVE_TARGET=irls EVOLVE_CORE=2 pixi run -e test python evolve/evaluate.py glm_jax/irls.py  # score one file
```

One evaluation ≈ 2 min (dispersion: Hao `apl_grid` ≈ 12 s per call; irls: full test file),
so a round ≈ 8 min. Needs `scratch/data/hao_10000.npz` (from `examples/real_data.py`).
Nothing else may run on `EVOLVE_CORE` (default 0) during a run. Check `/usage` for the
subscription window; a round costs 3 model calls.

## What the evaluator checks (`evaluate.py`)

Per-target settings (brief, tests, timed call, reference, gate) live in `TARGETS`; the two
workloads in `WORKLOADS`:

- `synth`: 5000 genes × 12 samples, p = 3, simulated NB;
- `hao`: first 500 genes of `scratch/data/hao_10000.npz` (10 000 cells, p = 10). Missing file
  is an error, not a fallback.

1. Static ban: the candidate may not touch other modules or do I/O (`irls.x = …` /
   `dispersion.x = …` for the module not being evolved, `setattr`, `importlib`, `os.` other
   than `os.sched_getaffinity`/`os.cpu_count`, `subprocess`, `open(`, `sys.modules`).
2. Tests in a scratch copy with the candidate swapped in: dispersion
   `tests/test_glm.py -k "apl or dispersion"`, irls the whole `tests/test_glm.py`.
3. Gate on each timed workload against a reference built once from the repo code:
   - dispersion: `apl_grid` on 20 dispersions finite, within 0.05 (synth) / 0.2 (hao, |APL| ~1e4,
     grid gap ~20) of `apl_grid(steps=None)`, identical per-gene argmax;
   - irls: `fit_glm(…, phi=0.1)` must converge every gene the `max_iter=200` reference
     converged, with max|Δβ| < 1e-6 on those.
4. Fitness: geometric mean over the two workloads of the median of 5 runs, one core, minimal
   environment, timeouts caught.

The incumbent is re-timed every round; a candidate is kept only if it is ≥ 2 % faster than
that fresh measurement.

## Outputs

| | dispersion | irls |
|---|---|---|
| current best (not the library file) | `best.py` | `best_irls.py` |
| one JSON line per attempt | `log.jsonl` | `log_irls.jsonl` |
| every proposed file | `candidates/dispersion-<run_id>/` | `candidates/irls-<run_id>/` |
| references (rebuilt if missing) | `reference_dispersion_{synth,hao}.npz` | `reference_irls_{synth,hao}.npz` |

Log keys: `run_id, ts, gen, model, baseline_time, idea, time` (geomean)`, times`
(per workload)`, target, ok, error, kept, file`. The model sees the last 10 entries. Entries
before 2026-10-27 are synthetic-only times. All of these are git-ignored.

## Promotion

The loop never edits `glm_jax/`. Afterwards: `diff glm_jax/dispersion.py evolve/best.py`,
read each accepted idea for assumptions the evaluator cannot see (dtype, integer-only inputs,
under/overflow), port by hand, run the full suite, commit naming the ideas.

## Results (2026-10-25, laptop, one core)

| change | proposed by | `apl_grid` time |
|---|---|---|
| committed 3-step warm-started scan | — | 0.150 s |
| hoist φ-free Σ lgamma(y+1) out of the grid loop | Opus | 0.117 s |
| 2 Fisher steps on warm-started grid points | Sonnet | ~0.108 s |
| shifted Stirling series for lgamma(y + 1/φ) | Opus | 0.093 s |
| sqrt-free LDLᵀ log-determinant | Opus | 0.085 s |

All four were ported into `glm_jax/dispersion.py` (LDLᵀ with sum-of-log pivots rather than
`log(prod)`, so separated genes cannot underflow). Rejected by the gate: capping the cold
first fit, GEMM-built Gram matrices, several slower micro-rewrites. The Sonnet keeper had
passed an earlier 40-gene-only gate; the full-workload check (3) exists because of it.

The generic recipe — when a loop is worth running, evaluator rules, gotchas — is the `evolve`
skill in `~/lab/skills-btraven/evolve/SKILL.md`.
