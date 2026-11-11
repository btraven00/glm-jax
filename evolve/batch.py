"""Queued evolution: propose a batch while online, evaluate it unattended on the GPU host.

    python evolve/batch.py propose N        # laptop: N proposals from the current best -> evolve/batch/<run_id>/
    python evolve/batch.py evaluate DIR     # GPU host: every gen_*.py in DIR -> DIR/results.jsonl (resumable)

All proposals start from the same incumbent (one wide generation, no compounding);
each wave of EVOLVE_PARALLEL sees the ideas already queued so they do not repeat.
evaluate re-times the incumbent before every candidate and judges each against
that fresh time (keep if < 0.98x).  Nothing is promoted: read results.jsonl, copy
the winner to best*.py, and the next batch starts from it.
"""

import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from run import BEST, PARALLEL, SRC, history_from_log, idea_of, propose
from evaluate import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))


def fmt(a):
    return f"- {a['idea']}: " + ("%.3fs" % a["time"] if a.get("time") is not None else
                                 "QUEUED, not yet evaluated" if a.get("queued") else "FAILED " + a["error"][-200:])


def propose_batch(n):
    best = BEST if os.path.exists(BEST) else SRC
    code = open(best).read()
    out = os.path.join(HERE, "batch", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out)
    shutil.copy(best, os.path.join(out, "incumbent.py"))
    attempts, k = history_from_log(), 0
    while k < n:
        hist = "\n".join(fmt(a) for a in attempts[-12:]) or "(none yet)"
        with ThreadPoolExecutor(PARALLEL) as ex:
            got = list(ex.map(lambda s: propose(code, hist, s), range(k, k + min(PARALLEL, n - k))))
        for c, raw in got:
            if c is None:
                print(f"proposal failed: {raw[-200:]}", flush=True)
                continue
            open(os.path.join(out, f"gen_{k:03d}.py"), "w").write(c)
            attempts.append({"idea": idea_of(c), "queued": True})
            print(f"gen_{k:03d}: {idea_of(c)[:100]}", flush=True)
            k += 1
    print(out)


def evaluate_dir(d):
    res = os.path.join(d, "results.jsonl")
    done = {json.loads(l)["file"] for l in open(res)} if os.path.exists(res) else set()
    inc = os.path.join(d, "incumbent.py")
    base = evaluate(inc)
    assert base["ok"], base["error"]
    for f in sorted(x for x in os.listdir(d) if x.startswith("gen_")):
        if f in done:
            continue
        t_inc = evaluate(inc, tests=False)["time"] or base["time"]  # fresh bar: the GPU is shared
        r = evaluate(os.path.join(d, f))
        rec = {"file": f, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "idea": idea_of(open(os.path.join(d, f)).read()),
               "incumbent_time": t_inc, **r, "kept": bool(r["ok"] and r["time"] < 0.98 * t_inc)}
        with open(res, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{f}: {'%.3fs' % r['time'] if r['ok'] else 'FAILED'} vs incumbent {t_inc:.3f}s "
              f"{'KEPT ' if rec['kept'] else ''}{rec['idea'][:90]}", flush=True)
    print("done")


if __name__ == "__main__":
    {"propose": lambda a: propose_batch(int(a)), "evaluate": evaluate_dir}[sys.argv[1]](sys.argv[2])
