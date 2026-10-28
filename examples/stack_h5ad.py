"""Tile a real-data npz (Y genes x cells, X, offset) K times into a CSR h5ad + design npz.

    python examples/stack_h5ad.py hao_0.npz 6 OUT_PREFIX   ->  OUT_PREFIX.h5ad, OUT_PREFIX.npz

Real sparsity, design and conditioning at ~1M cells. Duplicated cells shrink SEs by sqrt(K),
so judge f32 against f64 in SE units, not against the original fit.
"""

import sys

import anndata as ad
import numpy as np
import scipy.sparse as sparse

z = np.load(sys.argv[1])
K, out = int(sys.argv[2]), sys.argv[3]
Yc = sparse.csr_matrix(z["Y"].T.astype(np.float32))  # cells x genes
ad.AnnData(sparse.vstack([Yc] * K, format="csr")).write_h5ad(out + ".h5ad")
np.savez(out + ".npz", X=np.tile(z["X"], (K, 1)), offset=np.tile(z["offset"], K))
print(f"{out}.h5ad: {Yc.shape[0] * K} cells x {Yc.shape[1]} genes, nnz {Yc.nnz * K}")
