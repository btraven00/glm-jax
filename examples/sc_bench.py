"""Single-cell NB fit benchmark, JAX side: simulate, time, export for R.

    python examples/sc_bench.py OUT_DIR N_CELLS N_GENES [BATCH_SIZE]
    GLMJAX_DATA=real.npz python examples/sc_bench.py OUT_DIR 0 0 [BATCH_SIZE]

Writes Y/X/offset/coef as little-endian f64 in R (column-major) order, dims.txt
(N G p) and fit.mlir (StableHLO text) for examples/sc_bench.R. Fixed
dispersion, so it compares only the fitting kernel. GLMJAX_DATA points at an
.npz from examples/real_data.py (Y, X, offset); N_CELLS/N_GENES are then ignored.
"""

import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import glm_jax as gj

out, N, G = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
B = int(sys.argv[4]) if len(sys.argv) > 4 else None
PHI = 0.1
# GLMJAX_DTYPE=f32: fit in float32 (tolerance 1e-5, 100% converged on Hao 10k) and report the error vs the f64 fit
F32 = os.environ.get("GLMJAX_DTYPE", "f64") == "f32"
TOL = 1e-5 if F32 else 1e-8
DATA = os.environ.get("GLMJAX_DATA")

if DATA:
    d = np.load(DATA)
    Y, X, offset = d["Y"], d["X"], d["offset"]
    G, N = Y.shape
else:
    rng = np.random.default_rng(0)
    # intercept + condition + 3 cell-type indicators; log library-size offset
    cond = rng.integers(0, 2, N)
    ctype = rng.integers(0, 4, N)
    X = np.column_stack([np.ones(N), cond] + [ctype == k for k in (1, 2, 3)]).astype(float)
    offset = np.log(rng.lognormal(np.log(5000), 0.5, N))
    beta = np.column_stack([rng.normal(-8, 1.5, G), rng.normal(0, 0.5, (G, 4))])
    mu = np.exp(beta @ X.T + offset)
    Y = rng.negative_binomial(1 / PHI, 1 / (1 + PHI * mu)).astype(float)
phi = np.full(G, PHI)
open(f"{out}/dims.txt", "w").write(f"{N} {G} {X.shape[1]}\n")


# ponytail: return only (G,)/(G,p) outputs; mu is G*N and won't fit at 100k cells
def fit(Y, X, offset, phi):
    r = gj.fit_glm(Y, X, offset=offset, phi=phi, batch_size=B, tol=TOL, beta_tol=TOL)
    return r.coef, r.deviance, r.converged, r.n_iter


f = jax.jit(fit)
args = [jnp.asarray(a, jnp.float32 if F32 else jnp.float64) for a in (Y, X, offset, phi)]
t = time.perf_counter()
lowered = f.lower(*args)
compiled = lowered.compile()
t_compile = time.perf_counter() - t
open(f"{out}/fit.mlir", "w").write(lowered.as_text())
times = []
for _ in range(3):
    t = time.perf_counter()
    res = jax.block_until_ready(compiled(*args))
    times.append(time.perf_counter() - t)
coef, dev, conv, n_iter = (np.asarray(a) for a in res)

for k, a in [("Y", Y), ("X", X), ("offset", offset), ("coef", coef)]:
    np.asarray(a, dtype="<f8").T.copy(order="C").tofile(f"{out}/{k}.bin")

print(f"jax_{jax.default_backend()}_{'f32' if F32 else 'f64'},{N},{G},{t_compile:.3f},{np.median(times):.3f},{conv.mean():.5f},{np.median(n_iter):.0f}")

if F32:
    ref = gj.fit_glm(Y, X, offset=offset, phi=phi, batch_size=B).coef
    print(f"f32_vs_f64_max_abs_coef_diff,{N},{G},{float(jnp.max(jnp.abs(coef - ref))):.2e}")
