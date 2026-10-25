"""Propose -> evaluate -> keep loop over glm_jax/dispersion.py, Claude Code as the mutator.

    pixi run -e test python evolve/run.py N_ROUNDS

Uses `agent-box claude -p` (the Claude Code CLI on the subscription; no API key).
Each round dispatches EVOLVE_PARALLEL proposals (default 3) concurrently with
EVOLVE_MODEL (default sonnet), then evaluates them one at a time on the pinned
core.  Nothing under glm_jax/ is modified: the best candidate lands in
evolve/best.py, every attempt is one JSON line in evolve/log.jsonl.  Promote
best.py by hand after reading the diff.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from evaluate import REPO, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, "glm_jax", "dispersion.py")
BEST = os.path.join(HERE, "best.py")
LOG = os.path.join(HERE, "log.jsonl")
CLAUDE = ["agent-box", "claude"]
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
- apl_grid must stay within 0.05 of the fully converged APL and give the same
  per-gene argmax; estimate_dispersion_grid's Newton polish must still beat the
  coarse grid; the jitted apl_grid must lower to StableHLO with no custom_call
  (so no jnp.linalg.* on batched matrices);
- float64 numerics, pure JAX, no new dependencies, no file or tool access needed.

Fitness is the median wall time of apl_grid on 5000 genes x 12 samples x 20
dispersions on one CPU core.  Propose ONE idea per attempt, state it in a
one-line comment at the top of the file, and return the complete new file in a
single ```python block and nothing else.  Prefer changes that reduce total
floating-point work or memory traffic; micro-rewrites that XLA already fuses
are wasted attempts."""


def propose(code, history, seed):
    prompt = (f"{BRIEF}\n\nCurrent best file:\n```python\n{code}\n```\n\n"
              f"Recent attempts (newest last):\n{history}\n\n"
              f"Attempt #{seed}: pick an idea not in the list above. "
              "Return the complete new file in one ```python block.")
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}  # allow a nested session
    r = subprocess.run(CLAUDE + ["-p", "--output-format", "text", "--model", MODEL],
                       input=prompt, env=env, capture_output=True, text=True, timeout=900)
    m = re.search(r"```python\n(.*?)```", r.stdout, re.S)
    return (m.group(1) if m else None), r.stdout[-500:] + r.stderr[-500:]


def main(rounds):
    best_code = open(BEST).read() if os.path.exists(BEST) else open(SRC).read()
    base = evaluate(BEST if os.path.exists(BEST) else SRC)
    assert base["ok"], base["error"]
    best_time = base["time"]
    print(f"baseline {best_time:.3f}s  model={MODEL} parallel={PARALLEL}", flush=True)
    attempts, n = [], 0
    for rnd in range(rounds):
        history = "\n".join(f"- {a['idea']}: {'%.3fs' % a['time'] if a['time'] else 'FAILED ' + a['error'][-200:]}"
                            for a in attempts[-10:]) or "(none yet)"
        with ThreadPoolExecutor(PARALLEL) as ex:
            proposals = list(ex.map(lambda s: propose(best_code, history, s), range(n, n + PARALLEL)))
        for code, raw in proposals:
            rec = {"gen": n, "round": rnd, "model": MODEL, "idea": "", "time": None, "ok": False,
                   "error": "", "kept": False}
            n += 1
            if code is None:
                rec["error"] = "no code block: " + raw[-300:]
            else:
                rec["idea"] = next((l.strip("# ").strip() for l in code.splitlines() if l.strip()), "")[:160]
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
                    fh.write(code)
                res = evaluate(fh.name)
                os.unlink(fh.name)
                rec.update(ok=res["ok"], time=res["time"], error=res["error"])
                if res["ok"] and res["time"] < 0.98 * best_time:
                    best_time, best_code = res["time"], code
                    open(BEST, "w").write(code)
                    rec["kept"] = True
            attempts.append(rec)
            with open(LOG, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"gen {rec['gen']}: {'%.3fs' % rec['time'] if rec['time'] else 'FAILED'}  best {best_time:.3f}s  "
                  f"{'KEPT ' if rec['kept'] else ''}{rec['idea'][:90]}", flush=True)
    print(f"done: best {best_time:.3f}s (baseline {base['time']:.3f}s) -> {BEST}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
