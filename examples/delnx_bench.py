"""Head-to-head: glm_jax.fit_glm vs delnx's fit_beta_newton_batch, same inputs, same device.

    DELNX_GLM_DIR=DIR python examples/delnx_bench.py N_CELLS N_GENES [P] [BATCH]
    GLMJAX_DATA=real.npz DELNX_GLM_DIR=DIR python examples/delnx_bench.py 0 0 0 512

BATCH (0 = none) is the gene batch given to both: fit_glm's batch_size and, for
delnx, a Python loop over gene chunks exactly as its nb_fit driver does
(default 512 there).  Needed on GPU above ~50k cells, where the unbatched
(G, N) intermediates of either kernel exceed 24 GB.

DIR holds a package ``delnx_glm/`` made from the delnx checkout (pure JAX, no
scanpy/pydeseq2 needed)::

    mkdir delnx_glm && cp SRC/src/delnx/models/{_glm_gp,_utils}.py delnx_glm/ && touch delnx_glm/__init__.py

Both kernels get identical Y, X, offset and a fixed per-gene dispersion, f64,
so this times only the coefficient fit.  delnx: Newton with observed Hessian,
jnp.linalg.solve per gene, |dbeta| < tol stop, OLS init on log(y/sf+1), no
straggler handling (fit_glm defaults as in sc_bench.py).  Reports compile
time, median of 3 runs, converged fraction and max |dbeta| between the two.
Fixed-dispersion fitting is one stage of either tool; the dispersion stage is
not compared here.
"""

import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import glm_jax as gj

sys.path.insert(0, os.environ["DELNX_GLM_DIR"])
from delnx_glm._glm_gp import fit_beta_newton_batch  # noqa: E402

N, G = int(sys.argv[1]), int(sys.argv[2])
P = int(sys.argv[3]) if len(sys.argv) > 3 and int(sys.argv[3]) else 5
B = int(sys.argv[4]) if len(sys.argv) > 4 and int(sys.argv[4]) else None
PHI = 0.1
TOL = 1e-8

if os.environ.get("GLMJAX_DATA"):
    d = np.load(os.environ["GLMJAX_DATA"])
    Y, X, offset = d["Y"], d["X"], d["offset"]
    G, N = Y.shape
else:  # same generator as sc_bench.py (intercept + condition + P-2 indicators)
    rng = np.random.default_rng(0)
    cond = rng.integers(0, 2, N)
    ctype = rng.integers(0, P - 1, N)
    X = np.column_stack([np.ones(N), cond] + [ctype == k for k in range(1, P - 1)]).astype(float)
    offset = np.log(rng.lognormal(np.log(5000), 0.5, N))
    beta = np.column_stack([rng.normal(-8, 1.5, G), rng.normal(0, 0.5, (G, X.shape[1] - 1))])
    mu = np.exp(beta @ X.T + offset)
    Y = rng.negative_binomial(1 / PHI, 1 / (1 + PHI * mu)).astype(float)
phi = np.full(G, PHI)

# delnx's own initialiser (tl/_glm_gp.py): OLS of log(y/sf + 1) on the design
Q, R = np.linalg.qr(X)
init = np.linalg.solve(R, Q.T @ np.log(Y.T / np.exp(offset)[:, None] + 1.0)).T  # (G, p)


def bench(name, f, *args):
    t = time.perf_counter()
    out = jax.block_until_ready(f(*args))
    t_compile = time.perf_counter() - t
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        out = jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t)
    return name, t_compile, float(np.median(ts)), out


def ours(Y, X, offset, phi):
    r = gj.fit_glm(Y, X, offset=offset, phi=phi, tol=TOL, beta_tol=TOL, batch_size=B)
    return r.coef, r.converged, r.n_iter


def theirs(Y, X, offset, phi, init):
    beta, _, conv = fit_beta_newton_batch(Y.T, X, offset, phi, init, 100, TOL)
    return beta, conv


def theirs_chunked(Y, X, offset, phi, init):
    out = [theirs(Y[i:i + B], X, offset, phi[i:i + B], init[i:i + B]) for i in range(0, Y.shape[0], B)]
    return tuple(jnp.concatenate(parts) for parts in zip(*out))


A = [jnp.asarray(a) for a in (Y, X, offset, phi)]
rows = [bench("glm_jax", jax.jit(ours), *A),
        bench("delnx", theirs_chunked if B else jax.jit(theirs), *A, jnp.asarray(init))]
coef = {r[0]: np.asarray(r[3][0]) for r in rows}
both = np.asarray(rows[0][3][1]) & np.asarray(rows[1][3][1])  # genes both report converged
d = np.max(np.abs(coef["glm_jax"] - coef["delnx"]), axis=1)
dev = jax.default_backend()
print("kernel,device,N,G,p,compile_s,fit_s,converged,max_abs_dbeta_all,max_abs_dbeta_both_converged")
for name, tc, tf, out in rows:
    print(f"{name},{dev},{N},{G},{X.shape[1]},{tc:.2f},{tf:.3f},{float(np.mean(np.asarray(out[1]))):.4f},"
          f"{float(d.max()):.2e},{float(d[both].max()):.2e}")
