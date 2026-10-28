"""Streamed fit + dispersion at ~1M cells, f64 vs f32, Y kept on the host.

    python examples/million.py [N_CELLS] [N_GENES]

Synthetic NB counts (sc_bench.py's design: intercept + condition + 3 cell
types, lognormal library sizes) with per-gene phi in [1e-3, 2]. Y is a dense
float32 host array (G x N; 8 GB at 1M x 2k), streamed to the device in gene
chunks sized by glm_jax.device_budget. Wall times include the sizing compiles.
f32 is judged against f64 in standard-error units (coefficients) and by the
f64 APL it gives up (dispersion; 1.92 = edge of the 95% profile interval).
"""

import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

import glm_jax as gj
from glm_jax.stream import stream_genes

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
G = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

t = time.perf_counter()
rng = np.random.default_rng(0)
cond = rng.integers(0, 2, N)
ctype = rng.integers(0, 4, N)
X = np.column_stack([np.ones(N), cond] + [ctype == k for k in (1, 2, 3)]).astype(float)
offset = np.log(rng.lognormal(np.log(5000), 0.5, N))
beta = np.column_stack([rng.normal(-8, 1.5, G), rng.normal(0, 0.5, (G, 4))])
phi = np.exp(rng.uniform(np.log(1e-3), np.log(2.0), G))
Y = np.empty((G, N), np.float32)
for i in range(0, G, 50):
    mu = np.exp(beta[i:i + 50] @ X.T + offset)
    Y[i:i + 50] = rng.negative_binomial(1 / phi[i:i + 50, None], 1 / (1 + phi[i:i + 50, None] * mu))
print(f"data N={N} G={G} nnz={np.count_nonzero(Y) / Y.size:.3f} gen={time.perf_counter() - t:.0f}s "
      f"device={jax.devices()[0].device_kind} budget={gj.device_budget() / 2**30:.1f}GiB", flush=True)

fit, disp = {}, {}
for dt, tol in ((np.float64, 1e-8), (np.float32, 1e-5)):
    name = np.dtype(dt).name
    Xd, od = X.astype(dt), offset.astype(dt)
    t = time.perf_counter()
    fit[name], c = gj.fit_glm_streamed(Y, Xd, offset=od, phi=0.1, tol=tol, beta_tol=tol)
    print(f"fit_{name}: {time.perf_counter() - t:.1f}s chunk={c} converged={fit[name].converged.mean():.5f} "
          f"median_iter={np.median(fit[name].n_iter):.0f}", flush=True)
    t = time.perf_counter()
    disp[name], c = gj.estimate_dispersion_streamed(Y, Xd, offset=od)
    print(f"disp_{name}: {time.perf_counter() - t:.1f}s chunk={c}", flush=True)

se = np.sqrt(np.diagonal(fit["float64"].cov_unscaled, axis1=1, axis2=2))
z = np.abs(fit["float32"].coef - fit["float64"].coef) / se
print(f"coef f32 vs f64: max|dβ|/SE {z.max():.2e} median {np.median(z):.2e}", flush=True)


def apl(p):  # f64 APL at given tagwise phi, streamed
    def make(c):
        return lambda Yc, pc, X, o: gj.adjusted_profile_loglik(Yc, X, pc, offset=o)
    return stream_genes(make, (Y, p), (jax.numpy.asarray(X), jax.numpy.asarray(offset)))[0]


p64, p32 = disp["float64"]["tagwise"], disp["float32"]["tagwise"].astype(float)
loss = apl(p64) - apl(p32)
dl = np.abs(np.log(p64) - np.log(p32))
print(f"disp f32 vs f64: APL loss median {np.median(loss):.2e} max {loss.max():.3f} n>0.1 {(loss > 0.1).sum()} "
      f"n>1.92 {(loss > 1.92).sum()}; |dlog phi| p99 {np.quantile(dl, .99):.2e} max {dl.max():.2e}; "
      f"f64 |log phi - truth| median {np.median(np.abs(np.log(p64) - np.log(phi))):.3f}")
