"""Export the glm-jax kernels the R package ships: JAX_PLATFORMS=cpu python r/data-raw/export_kernels.py

Needs glm_jax and the xlaport exporter (pip install "git+https://github.com/btraven00/xlaport#subdirectory=python").
Writes r/inst/xla/nb_fit_n<N>.mlir and manifest.json. The R side reads every shape
from the manifest, so the ladders below can change without touching R.
"""
import os
import shutil

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from xlaport_export import export

import glm_jax as gj

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inst", "xla")
P = 12  # = irls._SMALL_P: larger designs need LAPACK custom calls, which are not portable
# Cell buckets, two per octave from 8 to 3.1M: data is padded up by at most 1.5x.
CELLS = sorted({2 ** k for k in range(3, 22)} | {3 * 2 ** k for k in range(2, 21)})


def genes(n):
    """Gene chunk for a cell bucket: ~4M elements per array, and never above 256 so the
    program has no straggler ladder (irls._STRAGGLER_BUCKET), which makes it 5x larger."""
    return min(256, max(16, 1 << ((1 << 22) // n).bit_length() - 1))


def nb_fit(Y, X, offset, weights, phi):
    # batch_size given explicitly: None picks a chunk from this machine's core count
    r = gj.fit_glm(Y, X, offset=offset, weights=weights, phi=phi, batch_size=Y.shape[0])
    return r.coef, r.deviance, r.converged, r.n_iter.astype(jnp.int32)  # R has no native i64


if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True)
    S = lambda *shape: jax.ShapeDtypeStruct(shape, jnp.float64)
    for n in CELLS:
        g = genes(n)
        export(nb_fit, [S(g, n), S(n, P), S(n), S(n), S(g)], name=f"nb_fit_n{n}", out=OUT,
               arg_names=["Y", "X", "offset", "weights", "phi"],
               out_names=["coef", "deviance", "converged", "n_iter"])
    print(f"{len(CELLS)} kernels, cells {CELLS[0]}..{CELLS[-1]}")
