"""Score one candidate glm_jax/dispersion.py: correctness gate, then pinned 1-core time.

    pixi run python evolve/evaluate.py CANDIDATE.py      -> one JSON line

Gate: the dispersion tests (tests/test_glm.py -k "apl or dispersion") in a scratch
copy of the package with the candidate swapped in; they check APL against
independent converged refits, argmax agreement, Newton polish quality, and that
the StableHLO has no custom calls.  Fitness: median of 3 runs of apl_grid on
5000 genes x 12 samples x 20 dispersions, f64, one core (EVOLVE_CORE, default 0).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMER = """
import time, numpy as np, jax, jax.numpy as jnp
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
f = jax.jit(lambda Y, X, g, o: gj.apl_grid(Y, X, g, offset=o))
jax.block_until_ready(f(Y, X, grid, offset))
ts = []
for _ in range(3):
    t = time.perf_counter(); jax.block_until_ready(f(Y, X, grid, offset)); ts.append(time.perf_counter() - t)
print(sorted(ts)[1])
"""


def evaluate(candidate):
    work = tempfile.mkdtemp(prefix="glmjax-eval-")
    try:
        for d in ("glm_jax", "tests"):
            shutil.copytree(os.path.join(REPO, d), os.path.join(work, d),
                            ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(candidate, os.path.join(work, "glm_jax", "dispersion.py"))
        env = {**os.environ, "PYTHONPATH": work, "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1",
               "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"}
        pin = ["taskset", "-c", os.environ.get("EVOLVE_CORE", "0")]
        t = subprocess.run(pin + [sys.executable, "-m", "pytest", "-q", "-x", "tests/test_glm.py",
                                  "-k", "apl or dispersion"],
                           cwd=work, env=env, capture_output=True, text=True, timeout=900)
        if t.returncode != 0:
            return {"ok": False, "time": None, "error": (t.stdout + t.stderr)[-3000:]}
        r = subprocess.run(pin + [sys.executable, "-c", TIMER],
                           cwd=work, env=env, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            return {"ok": False, "time": None, "error": r.stderr[-3000:]}
        return {"ok": True, "time": float(r.stdout.strip().splitlines()[-1]), "error": ""}
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1])))
