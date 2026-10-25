"""Score one candidate glm_jax/dispersion.py: static ban, correctness gate, pinned timing.

    pixi run -e test python evolve/evaluate.py CANDIDATE.py [--no-tests]   -> one JSON line

1. Regex ban on monkeypatching and I/O (``irls.x = ...``, setattr, importlib, os., subprocess,
   open, sys.modules) - the brief says the candidate may only change this file.
2. The dispersion tests (tests/test_glm.py -k "apl or dispersion") in a scratch copy of the
   package with the candidate swapped in.
3. On the timing workload itself (5000 genes x 12 samples x 20 dispersions) the candidate's
   APL must be finite, within 0.05 of a fully converged reference (steps=None, computed once
   from the repo code into evolve/reference.npz) and give identical per-gene argmax.
Fitness: median of 5 timed runs of apl_grid, f64, one core (EVOLVE_CORE, default 0), in a
minimal environment (no credentials). Timeouts and crashes are reported, never raised.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REF = os.path.join(HERE, "reference.npz")
BANNED = re.compile(r"\b(irls|families|glm_jax)\.\w+\s*=[^=]|\bsetattr\(|\bimportlib\b|\bos\.|"
                    r"\bsubprocess\b|\bopen\(|\b__import__\b|\bsys\.modules\b")

DATA = """
import sys, time, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import glm_jax as gj
G, n, K = 5000, 12, 20
rng = np.random.default_rng(0)
X = np.column_stack([np.ones(n), np.repeat([0, 1], n // 2), rng.normal(size=n)])
offset = np.log(rng.uniform(0.5, 2.0, n) * 1e6)
beta = np.column_stack([rng.normal(-11, 1, G), rng.normal(0, 1, G), rng.normal(0, 0.3, G)])
mu = np.exp(beta @ X.T + offset)
Y = rng.negative_binomial(5.0, 5.0 / (5.0 + mu)).astype(float)
grid = np.logspace(-3, 0.5, K)
"""
MAKE_REF = DATA + """
apl = np.asarray(gj.apl_grid(Y, X, grid, offset=offset, steps=None))
np.savez(sys.argv[1], apl=apl, argmax=apl.argmax(1))
"""
TIMER = DATA + """
ref = np.load(sys.argv[1])
f = jax.jit(lambda Y, X, g, o: gj.apl_grid(Y, X, g, offset=o))
apl = np.asarray(jax.block_until_ready(f(Y, X, grid, offset)))
err = float(np.max(np.abs(apl - ref["apl"]))); agree = float(np.mean(apl.argmax(1) == ref["argmax"]))
if not (apl.shape == ref["apl"].shape and np.all(np.isfinite(apl)) and err < 0.05 and agree == 1.0):
    print(f"REJECT shape={apl.shape} max_abs_diff={err:.3g} argmax_agree={agree:.4f}"); sys.exit(3)
ts = []
for _ in range(5):
    t = time.perf_counter(); jax.block_until_ready(f(Y, X, grid, offset)); ts.append(time.perf_counter() - t)
print(sorted(ts)[2])
"""


def _run(args, cwd, pythonpath, timeout=900):
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""), "PYTHONPATH": pythonpath,
           "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1", "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false"}
    pin = ["taskset", "-c", os.environ.get("EVOLVE_CORE", "0")]
    try:
        return subprocess.run(pin + args, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", f"timeout after {timeout} s")
    except OSError as e:
        return subprocess.CompletedProcess(args, 125, "", str(e))


def ensure_reference():
    if not os.path.exists(REF):
        r = _run([sys.executable, "-c", MAKE_REF, REF], REPO, REPO)
        if r.returncode != 0:
            raise RuntimeError("reference build failed: " + r.stderr[-2000:])


def evaluate(candidate, tests=True):
    m = BANNED.search(open(candidate).read())
    if m:
        return {"ok": False, "time": None, "error": f"banned construct: {m.group(0)!r}"}
    ensure_reference()
    work = tempfile.mkdtemp(prefix="glmjax-eval-")
    try:
        for d in ("glm_jax", "tests"):
            shutil.copytree(os.path.join(REPO, d), os.path.join(work, d),
                            ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(candidate, os.path.join(work, "glm_jax", "dispersion.py"))
        if tests:
            t = _run([sys.executable, "-m", "pytest", "-q", "-x", "tests/test_glm.py", "-k", "apl or dispersion"],
                     work, work)
            if t.returncode != 0:
                return {"ok": False, "time": None, "error": (t.stdout + t.stderr)[-3000:]}
        r = _run([sys.executable, "-c", TIMER, REF], work, work)
        if r.returncode != 0:
            return {"ok": False, "time": None, "error": (r.stdout + r.stderr)[-3000:]}
        return {"ok": True, "time": float(r.stdout.strip().splitlines()[-1]), "error": ""}
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1], tests="--no-tests" not in sys.argv)))
