"""Propose -> evaluate -> keep loop over glm_jax/<EVOLVE_TARGET>.py, Claude Code as the mutator.

    [EVOLVE_TARGET=dispersion|irls] pixi run -e test python evolve/run.py N_ROUNDS

Uses `agent-box claude -p` (the Claude Code CLI on the subscription; no API key),
run from a temporary directory with extra flags from EVOLVE_CLAUDE_ARGS.  Each
round re-times the current best, then dispatches EVOLVE_PARALLEL proposals
(default 3) concurrently with EVOLVE_MODEL (default sonnet) and evaluates them
one at a time on the pinned core.  Nothing under glm_jax/ is modified: the best
candidate lands in evolve/best.py (best_irls.py for irls; atomically), every candidate is saved under
evolve/candidates/<target>-<run_id>/, every attempt is one JSON line in evolve/log.jsonl
(log_irls.jsonl),
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

from evaluate import FILE, REPO, TARGET, T, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, "glm_jax", f"{FILE}.py")
SFX = "" if TARGET == "dispersion" else f"_{TARGET}"  # dispersion keeps the original file names
BEST = os.path.join(HERE, f"best{SFX}.py")
LOG = os.path.join(HERE, f"log{SFX}.jsonl")
CLAUDE = ["agent-box", "claude"] + os.environ.get(
    "EVOLVE_CLAUDE_ARGS", "--disallowedTools Bash Edit Write Read Glob Grep WebFetch WebSearch Agent").split()
MODEL = os.environ.get("EVOLVE_MODEL", "sonnet")
PARALLEL = int(os.environ.get("EVOLVE_PARALLEL", "3"))

BRIEF = T["brief"]


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
    cand_dir = os.path.join(HERE, "candidates", f"{TARGET}-{run_id}")
    os.makedirs(cand_dir, exist_ok=True)
    best_path = BEST if os.path.exists(BEST) else SRC
    best_code = open(best_path).read()
    base = evaluate(best_path)
    assert base["ok"], base["error"]
    print(f"run {run_id} [{TARGET}]: baseline {base['time']:.3f}s (geomean of {base['times']})  model={MODEL} parallel={PARALLEL}", flush=True)
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
                   "error": "", "kept": False, "file": "", "target": TARGET, "times": {}}
            n += 1
            if code is None:
                rec["error"] = raw
            else:
                rec["idea"] = idea_of(code)
                rec["file"] = os.path.join(cand_dir, f"gen_{rec['gen']:03d}.py")
                open(rec["file"], "w").write(code)
                res = evaluate(rec["file"])
                rec.update(ok=res["ok"], time=res["time"], times=res["times"], error=res["error"])
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
                  f"{rec['times']}  best {best_time:.3f}s  {'KEPT ' if rec['kept'] else ''}{rec['idea'][:90]}", flush=True)
    print(f"done: best {best_time:.3f}s (baseline {base['time']:.3f}s) -> {BEST}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
