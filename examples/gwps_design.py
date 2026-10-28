"""Design npz for Replogle 2022 K562 genome-wide Perturb-seq (K562_gwps_raw_singlecell_01.h5ad).

    python examples/gwps_design.py K562_gwps_raw_singlecell_01.h5ad OUT.npz

1.99M cells. X = intercept, non-targeting indicator, standardised mito fraction and
the 267 gem groups binned into 10 contiguous blocks (9 dummies): p = 12, close to Hao's
10. offset = log total UMI. Reads obs only (old AnnData layout, __categories).
"""

import sys

import h5py
import numpy as np

with h5py.File(sys.argv[1]) as f:
    o = f["obs"]
    cats = f["obs/__categories/gene"][:].astype(str)
    ctrl = cats[o["gene"][:]] == "non-targeting"
    mito = o["mitopercent"][:]
    gem = o["gem_group"][:]
    umi = o["UMI_count"][:]
ug = np.unique(gem)
block = np.searchsorted(ug, gem) * 10 // len(ug)
X = np.column_stack([np.ones(len(gem)), ctrl, (mito - mito.mean()) / mito.std()] + [block == k for k in range(1, 10)]).astype(float)
np.savez(sys.argv[2], X=X, offset=np.log(umi))
print(f"{sys.argv[2]}: n={len(gem)} p={X.shape[1]} controls={ctrl.sum()} cond(X)={np.linalg.cond(X):.1f}")
