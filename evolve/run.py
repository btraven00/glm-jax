"""Propose -> evaluate -> keep loop over glm_jax/dispersion.py, Claude as the mutator.

    pixi run python evolve/run.py N_GENERATIONS

Credentials: ANTHROPIC_API_KEY or an `ant auth login` profile.  Nothing under
glm_jax/ is modified: the best candidate lands in evolve/best.py, every attempt
is one JSON line in evolve/log.jsonl.  Promote best.py by hand after reading
the diff.  Rough cost: ~$0.2 per generation on claude-opus-5-5.
"""

import json
import os
import random
import re
import sys
import tempfile
import time

import anthropic

from evaluate import REPO, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, "glm_jax", "dispersion.py")
BEST = os.path.join(HERE, "best.py")
LOG = os.path.join(HERE, "log.jsonl")

SYSTEM = """You optimise one file of a small JAX library for CPU speed.

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
- float64 numerics, pure JAX, no new dependencies.

Fitness is the median wall time of apl_grid on 5000 genes x 12 samples x 20
dispersions on one CPU core.  Propose ONE idea per attempt, state it in a
one-line comment at the top of the file, and return the complete new file in a
single ```python block.  Prefer changes that reduce total floating-point work
or memory traffic; micro-rewrites that XLA already fuses are wasted attempts."""


def propose(client, code, history):
    prompt = (f"Current best file:\n```python\n{code}\n```\n\nRecent attempts (newest last):\n{history}\n\n"
              "Return the complete new file in one ```python block.")
    msg = client.beta.messages.create(
        model="claude-opus-5-5", max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        thinking={"type": "adaptive"}, output_config={"effort": "high"},
        system=SYSTEM, messages=[{"role": "user", "content": prompt}])
    if msg.stop_reason == "refusal":
        return None, msg.usage
    text = "".join(b.text for b in msg.content if b.type == "text")
    m = re.search(r"```python\n(.*?)```", text, re.S)
    return (m.group(1) if m else None), msg.usage


def main(generations):
    client = anthropic.Anthropic()
    best_code = open(BEST).read() if os.path.exists(BEST) else open(SRC).read()
    base = evaluate(BEST if os.path.exists(BEST) else SRC)
    assert base["ok"], base["error"]
    best_time = base["time"]
    print(f"baseline {best_time:.3f}s")
    attempts = []
    for g in range(generations):
        history = "\n".join(f"- {a['idea']}: {'%.3fs' % a['time'] if a['time'] else 'FAILED ' + a['error'][-200:]}"
                            for a in attempts[-8:]) or "(none yet)"
        code, usage = propose(client, best_code, history)
        rec = {"gen": g, "idea": "", "time": None, "ok": False, "error": "",
               "in_tokens": usage.input_tokens, "out_tokens": usage.output_tokens}
        if code is None:
            rec["error"] = "no code block / refusal"
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
        print(f"gen {g}: {'%.3fs' % rec['time'] if rec['time'] else 'FAILED'}  best {best_time:.3f}s  {rec['idea'][:80]}")
    print(f"done: best {best_time:.3f}s (baseline {base['time']:.3f}s) -> {BEST}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
