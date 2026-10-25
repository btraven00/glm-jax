"""Propose -> evaluate -> keep loop over glm_jax/dispersion.py, Claude Code as the mutator.

    pixi run -e test python evolve/run.py N_ROUNDS

Uses `agent-box claude -p` (the Claude Code CLI on the subscription; no API key),
run from a temporary directory with extra flags from EVOLVE_CLAUDE_ARGS.  Each
round re-times the current best, then dispatches EVOLVE_PARALLEL proposals
(default 3) concurrently with EVOLVE_MODEL (default sonnet) and evaluates them
one at a time on the pinned core.  Nothing under glm_jax/ is modified: the best
candidate lands in evolve/best.py (atomically), every candidate is saved under
evolve/candidates/<run_id>/, every attempt is one JSON line in evolve/log.jsonl,
and the history shown to the model is seeded from that log.  Promote best.py by
hand after reading the diff.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from evaluate import REPO, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, "glm_jax", "dispersion.py")
BEST = os.path.join(HERE, "best.py")
LOG = os.path.join(HERE, "log.jsonl")
CLAUDE = ["agent-box", "claude"] + os.environ.get(
    "EVOLVE_CLAUDE_ARGS", "--disallowedTools Bash Edit Write Read Glob Grep WebFetch WebSearch Agent").split()
MODEL = os.environ.get("EVOLVE_MODEL", "sonnet")
PARALLEL = int(os.environ.get("EVOLVE_PARALLEL", "3"))

BRIEF = """You optimise one file of a small JAX library for CPU speed.

The file is glm_jax/dispersion.py: negative-binomial dispersion estimation by
Cox-Reid adjusted profile likelihood on a grid, plus a Newton polish.  It may
import these private helpers from glm_jax.irls, whose code you cannot change:
_fit_single(y, X, offset, w, phi, beta0, use_beta0, *, family, max_iter, tol,
beta_tol, ridge, max_halving) -> GLMResult(coef, deviance, mu, cov_unscaled,
n_iter, converged); _working(family, y, eta, w, phi) -> (W, z, mu);
_wls(X, W, z, ridge); _small_chol(A) -> lower-triangular L as nested lists;
_broadcast_inputs(Y, X, offset, weights, phi, beta0) -> (Y, X, offset, weights,
phi, beta0, use_beta0); _SMALL_P; fit_glm.

Hard constraints (an automatic evaluator enforces them and discards violators):
- keep the public functions and signatures: adjusted_profile_loglik, apl_grid,
  estimate_dispersion_grid, with the same return shapes and dict keys;
- on 5000 genes x 20 dispersions apl_grid must stay within 0.05 of the fully
  converged APL and give the same per-gene argmax; estimate_dispersion_grid's
  Newton polish must still beat the coarse grid; the jitted apl_grid must lower
  to StableHLO with no custom_call (so no jnp.linalg.* on batched matrices);
- float64 numerics, pure JAX, no new dependencies, no file/OS access, no
  monkeypatching of other modules (statically rejected).

Fitness is the median wall time of apl_grid on 5000 genes x 12 samples x 20
dispersions on one CPU core.  Propose ONE idea per attempt.  Replace the
previous idea comment with yours: exactly one line starting with "# Idea:" at
the top of the file.  Return the complete new file in a single ```python block
and nothing else.  Prefer changes that reduce total floating-point work or
memory traffic; micro-rewrites that XLA already fuses are wasted attempts."""


def propose(code, history, seed):
    prompt = (f"{BRIEF}\n\nCurrent best file:\n```python\n{code}\n```\n\n"
              f"Recent attempts (newest last):\n{history}\n\n"
              f"Attempt #{seed}: pick an idea not in the list above. "
              "Return the complete new file in one ```python block.")
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}  # allow a nested session
    try:
        r = subprocess.run(CLAUDE + ["-p", "--output-format", "text", "--model", MODEL], input=prompt,
                           env=env, cwd=tempfile.gettempdir(), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return None, "claude timeout after 900 s"
    if r.returncode != 0:
        return None, f"claude exit {r.returncode}: {(r.stdout + r.stderr)[-400:]}"
    blocks = re.findall(r"```python\n(.*?)```", r.stdout, re.S)
    return (max(blocks, key=len) if blocks else None), r.stdout[-400:]


def idea_of(code):
    return next((l.lstrip("# ").strip() for l in code.splitlines() if l.startswith("#")), "")[:160]


def history_from_log(k=10):
    if not os.path.exists(LOG):
        return []
    return [json.loads(l) for l in open(LOG).read().splitlines() if l.strip()][-k:]


def main(rounds):
    if not shutil.which(CLAUDE[0]):
        sys.exit(f"{CLAUDE[0]} not on PATH")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    cand_dir = os.path.join(HERE, "candidates", run_id)
    os.makedirs(cand_dir, exist_ok=True)
    best_path = BEST if os.path.exists(BEST) else SRC
    best_code = open(best_path).read()
    base = evaluate(best_path)
    assert base["ok"], base["error"]
    print(f"run {run_id}: baseline {base['time']:.3f}s  model={MODEL} parallel={PARALLEL}", flush=True)
    attempts, n = history_from_log(), 0
    for rnd in range(rounds):
        # re-time the incumbent every round so a lucky measurement cannot ratchet the bar
        best_time = evaluate(best_path, tests=False)["time"] or base["time"]
        history = "\n".join(f"- {a['idea']}: {'%.3fs' % a['time'] if a['time'] is not None else 'FAILED ' + a['error'][-200:]}"
                            for a in attempts[-10:]) or "(none yet)"
        with ThreadPoolExecutor(PARALLEL) as ex:
            proposals = list(ex.map(lambda s: propose(best_code, history, s), range(n, n + PARALLEL)))
        for code, raw in proposals:
            rec = {"run_id": run_id, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "gen": n, "round": rnd,
                   "model": MODEL, "baseline_time": best_time, "idea": "", "time": None, "ok": False,
                   "error": "", "kept": False, "file": ""}
            n += 1
            if code is None:
                rec["error"] = raw
            else:
                rec["idea"] = idea_of(code)
                rec["file"] = os.path.join(cand_dir, f"gen_{rec['gen']:03d}.py")
                open(rec["file"], "w").write(code)
                res = evaluate(rec["file"])
                rec.update(ok=res["ok"], time=res["time"], error=res["error"])
                if res["ok"] and res["time"] < 0.98 * best_time:
                    best_time, best_code = res["time"], code
                    tmp = BEST + ".tmp"
                    open(tmp, "w").write(code)
                    os.replace(tmp, BEST)
                    best_path = BEST
                    rec["kept"] = True
            attempts.append(rec)
            with open(LOG, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"gen {rec['gen']}: {'%.3fs' % rec['time'] if rec['time'] is not None else 'FAILED'}  "
                  f"best {best_time:.3f}s  {'KEPT ' if rec['kept'] else ''}{rec['idea'][:90]}", flush=True)
    print(f"done: best {best_time:.3f}s (baseline {base['time']:.3f}s) -> {BEST}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
