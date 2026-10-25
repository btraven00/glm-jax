# evolve/ — LLM-driven search over `glm_jax/dispersion.py`

A propose → evaluate → keep loop: Claude writes whole-file variants of the dispersion
kernel, a mechanical evaluator decides. The evaluator is the part to trust; the loop is
80 lines. Runs on the Claude Code subscription (`agent-box claude -p`), no API key.

## Run

```bash
pixi run -e test python evolve/run.py 4            # 4 rounds x 3 parallel proposals, Sonnet
EVOLVE_MODEL=opus pixi run -e test python evolve/run.py 2
EVOLVE_PARALLEL=1 EVOLVE_CORE=3 pixi run -e test python evolve/run.py 4
```

One round ≈ 3 min (proposals ~90 s in parallel, then ~40 s per evaluation on the pinned core).
Nothing else may run on `EVOLVE_CORE` (default 0) during a run. Check `/usage` for the
subscription window; a round costs 3 model calls.

## What the evaluator checks (`evaluate.py`)

1. Static ban: the candidate may not touch other modules or do I/O
   (`irls.x = …`, `setattr`, `importlib`, `os.`, `subprocess`, `open(`, `sys.modules`).
2. The dispersion tests (`tests/test_glm.py -k "apl or dispersion"`) in a scratch copy of
   the package with the candidate swapped in.
3. On the timed workload itself (5000 genes × 12 samples × 20 dispersions, float64) the
   candidate's APL must be finite, within 0.05 of the fully converged reference
   (`apl_grid(steps=None)`, cached in `reference.npz`) and give identical per-gene argmax.
4. Fitness: median of 5 runs of `apl_grid`, one core, minimal environment, timeouts caught.

The incumbent is re-timed every round; a candidate is kept only if it is ≥ 2 % faster than
that fresh measurement.

## Outputs

- `best.py` — current best candidate (written atomically). **Not** the library file.
- `log.jsonl` — one line per attempt: `run_id, ts, gen, model, baseline_time, idea, time,
  ok, error, kept, file`. The model sees the last 10 entries so it does not repeat ideas.
- `candidates/<run_id>/gen_NNN.py` — every proposed file (git-ignored).
- `reference.npz` — converged reference APL (git-ignored; rebuilt if missing).

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
