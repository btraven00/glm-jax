"""Rough timing of the batched kernel on an RNA-seq-sized problem.

    python examples/benchmark.py
"""

import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

import glm_jax as gj

rng = np.random.default_rng(0)
G, n = 20_000, 12
group = np.repeat([0, 1], n // 2)
batch = np.tile([0, 1, 2], n // 3)
X = np.column_stack([np.ones(n), group, batch == 1, batch == 2]).astype(float)
offset = np.log(rng.uniform(0.5, 2, n) * 2e7)
beta = np.column_stack([rng.normal(-12, 2, G), rng.normal(0, 0.7, G),
                        rng.normal(0, 0.2, (G, 2))])
mu = np.exp(beta @ X.T + offset)
phi = 0.1
Y = rng.negative_binomial(1 / phi, 1 / (1 + phi * mu)).astype(float)


def timeit(label, f, reps=3):
    jax.block_until_ready(f())  # compile
    t = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(f())
    print(f"{label:40s} {(time.perf_counter() - t) / reps * 1e3:8.1f} ms")


print(f"G={G}, n={n}, p={X.shape[1]}, backend={jax.default_backend()}")
timeit("NB fit (all genes)", lambda: gj.fit_glm(Y, X, offset=offset, phi=phi).coef)
timeit("NB LRT (group effect)",
       lambda: gj.glm_lrt(Y, X, gj.drop_coef(X, 1), offset=offset, phi=phi)[0].pvalue)
grid = np.logspace(-3, 0.5, 20)
timeit("APL grid (20 dispersions x all genes)",
       lambda: gj.apl_grid(Y, X, grid, offset=offset), reps=1)

fit = gj.fit_glm(Y, X, offset=offset, phi=phi)
print("converged:", float(np.mean(fit.converged)), "median iters:", float(np.median(fit.n_iter)))
