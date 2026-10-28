"""Streamed fit + dispersion at ~1M cells, f64 vs f32, Y kept on the host.

    python examples/million.py [N_CELLS] [N_GENES]
    python examples/million.py DATA.h5ad DESIGN.npz   # real counts, read with picklerick

Synthetic NB counts (sc_bench.py's design: intercept + condition + 3 cell
types, lognormal library sizes) with per-gene phi in [1e-3, 2]. Y is a dense
float32 host array (G x N; 8 GB at 1M x 2k), streamed to the device in gene
chunks sized by glm_jax.device_budget. Wall times include the sizing compiles.
f32 is judged against f64 in standard-error units (coefficients) and by the
f64 APL it gives up (dispersion; 1.92 = edge of the 95% profile interval).
With an h5ad, Y is streamed off disk in cell blocks by picklerick and collected
into a dense gene-major f32 array on the host (smaller than CSR at the >30% density
of top genes once nnz needs int64 indices; ponytail: build CSR for sparse panels) (each GLM needs all cells of a gene, so the
fit cannot start before the last block); DESIGN.npz holds X and offset.
"""

import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

import glm_jax as gj
from glm_jax.stream import stream_genes

if len(sys.argv) > 1 and sys.argv[1].endswith(".h5ad"):
    import picklerick as pk
    import scipy.sparse as sparse

    t = time.perf_counter()
    Y = None
    for blk in pk.open_stream(sys.argv[1], chunk_size=20000):
        if Y is None:
            n_obs = pk.inspect(sys.argv[1])["n_obs"]
            Y = np.empty((blk.n_vars, n_obs), np.float32)
        Y[:, blk.row_offset:blk.row_offset + blk.nrows] = sparse.csr_matrix(
            (blk.data, blk.indices, blk.indptr), shape=(blk.nrows, blk.n_vars)).toarray().T
    t_read = time.perf_counter() - t
    G, N = Y.shape
    d = np.load(sys.argv[2])
    X, offset, phi = d["X"], d["offset"], None
    print(f"data {sys.argv[1]} N={N} G={G} read+to_gene_major={t_read:.1f}s ({Y.nbytes / 2**30:.1f}GiB dense) "
          f"device={jax.devices()[0].device_kind} budget={gj.device_budget() / 2**30:.1f}GiB", flush=True)
else:
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
# Is the f32 error per-gene conditioning? cond(X'WX) = cond of its inverse (cov_unscaled) at the f64 fit
kappa = np.linalg.cond(fit["float64"].cov_unscaled)
zg = z.max(1)
rank = lambda a: np.argsort(np.argsort(a))
print(f"cond(X'WX) median {np.median(kappa):.2e} p99 {np.quantile(kappa, .99):.2e} max {kappa.max():.2e}; "
      f"spearman(cond, max|dβ|/SE) {np.corrcoef(rank(kappa), rank(zg))[0, 1]:.2f}", flush=True)
for g in np.argsort(zg)[::-1][:10]:
    j = z[g].argmax()
    print(f"  gene {g}: |dβ|/SE {zg[g]:.2e} at coef {j} (β64 {fit['float64'].coef[g, j]:+.2f}) cond {kappa[g]:.2e} "
          f"iters {fit['float64'].n_iter[g]}/{fit['float32'].n_iter[g]}", flush=True)


def apl(p):  # f64 APL at given tagwise phi, streamed
    def make(c):
        return lambda Yc, pc, X, o: gj.adjusted_profile_loglik(Yc, X, pc, offset=o)
    return stream_genes(make, (Y, p), (jax.numpy.asarray(X), jax.numpy.asarray(offset)))[0]


p64, p32 = disp["float64"]["tagwise"], disp["float32"]["tagwise"].astype(float)
loss = apl(p64) - apl(p32)
dl = np.abs(np.log(p64) - np.log(p32))
print(f"disp f32 vs f64: APL loss median {np.median(loss):.2e} max {loss.max():.3f} n>0.1 {(loss > 0.1).sum()} "
      f"n>1.92 {(loss > 1.92).sum()}; |dlog phi| p99 {np.quantile(dl, .99):.2e} max {dl.max():.2e}"
      + ("" if phi is None else f"; f64 |log phi - truth| median {np.median(np.abs(np.log(p64) - np.log(phi))):.3f}"))
