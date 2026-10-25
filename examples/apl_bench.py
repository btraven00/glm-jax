"""Time the APL dispersion grid: per-phi refits (old path) vs apl_grid.

    python examples/apl_bench.py [N_GENES] [N_CELLS] [K]

Default is the edgeR-sized problem of examples/benchmark.py (20k genes, 12
samples, 20 dispersions). Prints CSV: backend,method,G,n,K,compile_s,run_s,max_abs_diff[,argmax_agree].
"""

import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import glm_jax as gj

G = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
K = int(sys.argv[3]) if len(sys.argv) > 3 else 20

rng = np.random.default_rng(0)
group = np.repeat([0, 1], n // 2)
X = np.column_stack([np.ones(n), group, rng.normal(size=n)])
offset = np.log(rng.uniform(0.5, 2.0, n) * 1e6)
beta = np.column_stack([rng.normal(-11, 1, G), rng.normal(0, 1, G), rng.normal(0, 0.3, G)])
mu = np.exp(beta @ X.T + offset)
Y = rng.negative_binomial(5.0, 5.0 / (5.0 + mu)).astype(float)
grid = np.logspace(-3, 0.5, K)


def timed(f, *a):
    t = time.perf_counter()
    out = jax.block_until_ready(f(*a))
    t_first = time.perf_counter() - t
    t = time.perf_counter()
    out = jax.block_until_ready(f(*a))
    return out, t_first - (time.perf_counter() - t), time.perf_counter() - t


# old path: independent refit per phi (what apl_grid did before the warm-started scan)
per_phi = jax.jit(lambda Y, X, grid, offset: jnp.stack(
    [gj.adjusted_profile_loglik(Y, X, grid[k], offset=offset) for k in range(K)], axis=1))
ref, c0, r0 = timed(per_phi, Y, X, grid, offset)
print(f"{jax.default_backend()},per_phi_refit,{G},{n},{K},{c0:.2f},{r0:.3f},0")

for name, steps in (("scan_converged", None), ("scan_3steps", 3)):
    f = jax.jit(lambda Y, X, grid, offset: gj.apl_grid(Y, X, grid, offset=offset, steps=steps))
    apl, c1, r1 = timed(f, Y, X, grid, offset)
    agree = float(jnp.mean(jnp.argmax(apl, 1) == jnp.argmax(ref, 1)))
    print(f"{jax.default_backend()},{name},{G},{n},{K},{c1:.2f},{r1:.3f},{float(jnp.max(jnp.abs(apl - ref))):.2e},argmax_agree={agree:.4f}")

# full estimator: 3-step grid walk + Newton polish in log(phi) (tagwise and common)
for newton in (0, 2):
    f = lambda Y, X, grid, offset: gj.estimate_dispersion_grid(Y, X, offset=offset, phi_grid=grid, newton=newton)["tagwise"]
    tag, c2, r2 = timed(f, Y, X, grid, offset)
    print(f"{jax.default_backend()},estimate_newton{newton},{G},{n},{K},{c2:.2f},{r2:.3f},nonfinite={int(jnp.sum(~jnp.isfinite(tag)))}")
