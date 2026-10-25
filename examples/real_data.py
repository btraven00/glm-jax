"""Build a real-data benchmark input (Y, X, offset) as .npz for sc_bench.py.

    python examples/real_data.py kang IR_CTRL IR_STIM OUT.npz [MAX_CELLS ...]
    python examples/real_data.py hao RAW_TAR META_CSV_GZ OUT.npz [MAX_CELLS ...]

kang: two ``scx snapshot`` dirs of the 10x HDF5 files (CSR NPY + index txt);
      design = intercept + stim (p = 2).
hao : GSE164378 RNA 3' MEX streamed out of the RAW tar + Seurat metadata;
      design = intercept + celltype.l1 + time (p = 10, within _SMALL_P).
Genes: detected in >= 1% of cells, top 2000 by variance of log1p(CP10k).
MAX_CELLS subsamples cells (seed 0) to build an N ladder on real data; several
values write OUT_<n>.npz each from the one loaded matrix (0 = all cells).
"""

import csv
import gzip
import io
import subprocess
import sys

import numpy as np
import scipy.sparse as sp
from scipy.io import mmread

G_HVG = 2000


def read_lines(path):
    return [l for l in open(path).read().split("\n") if l]


def load_ir(d):
    """scx snapshot dir -> (cells x genes CSR, gene names)."""
    genes = read_lines(f"{d}/var_index.txt")
    n_obs = len(read_lines(f"{d}/obs_index.txt"))
    # explicit shape: trailing all-zero genes would otherwise shrink the width
    C = sp.csr_matrix((np.load(f"{d}/X/data.npy"),
                       np.load(f"{d}/X/indices.npy").astype(np.int64),
                       np.load(f"{d}/X/indptr.npy").astype(np.int64)),
                      shape=(n_obs, len(genes)))
    return C, genes


def kang(ir_ctrl, ir_stim):
    A, g1 = load_ir(ir_ctrl)
    B, g2 = load_ir(ir_stim)
    assert g1 == g2, "ctrl/stim gene order differs"
    C = sp.vstack([A, B]).tocsr()
    stim = np.r_[np.zeros(A.shape[0]), np.ones(B.shape[0])]
    return C, np.column_stack([np.ones(C.shape[0]), stim]), ["intercept", "stim"]


def tar_member(tar, name):
    return subprocess.run(["tar", "-xOf", tar, name], capture_output=True, check=True).stdout


def hao(tar, meta_gz):
    pre = "GSM5008737_RNA_3P-"
    M = mmread(gzip.GzipFile(fileobj=io.BytesIO(tar_member(tar, pre + "matrix.mtx.gz"))))
    C = sp.csr_matrix(M.T)  # cells x genes
    barcodes = gzip.decompress(tar_member(tar, pre + "barcodes.tsv.gz")).decode().split()
    assert len(barcodes) == C.shape[0]
    meta = {}
    with gzip.open(meta_gz, "rt") as fh:
        for row in csv.DictReader(fh):
            meta[row[""]] = (row["celltype.l1"], row["time"])
    keep = np.array([b in meta for b in barcodes])
    print(f"hao: {keep.sum()}/{len(barcodes)} barcodes have metadata", file=sys.stderr)
    C = C[keep]
    ct = np.array([meta[b][0] for b in np.array(barcodes)[keep]])
    tm = np.array([meta[b][1] for b in np.array(barcodes)[keep]])
    cols, names = [np.ones(C.shape[0])], ["intercept"]
    for lab, arr in (("celltype", ct), ("time", tm)):
        for lev in sorted(set(arr))[1:]:  # drop first level
            cols.append((arr == lev).astype(float))
            names.append(f"{lab}={lev}")
    return C, np.column_stack(cols), names


def finish(C, X, names, out, max_cells):
    if max_cells and C.shape[0] > max_cells:
        idx = np.sort(np.random.default_rng(0).choice(C.shape[0], max_cells, replace=False))
        C, X = C[idx], X[idx]
    lib = np.asarray(C.sum(axis=1)).ravel()
    detected = np.asarray((C > 0).mean(axis=0)).ravel() >= 0.01
    L = C.multiply(1e4 / lib[:, None]).log1p().tocsc()
    mean = np.asarray(L.mean(axis=0)).ravel()
    var = np.asarray(L.multiply(L).mean(axis=0)).ravel() - mean**2
    var[~detected] = -1
    hvg = np.sort(np.argsort(var)[::-1][:G_HVG])
    Y = C[:, hvg].T.toarray().astype(np.float64)  # genes x cells
    rank = np.linalg.matrix_rank(X)
    print(f"N={Y.shape[1]} G={Y.shape[0]} p={X.shape[1]} rank(X)={rank} "
          f"zero-frac={np.mean(Y == 0):.3f} median-lib={np.median(lib):.0f}", file=sys.stderr)
    assert rank == X.shape[1], "design is rank deficient"
    np.savez(out, Y=Y, X=X, offset=np.log(lib), colnames=np.array(names))


if __name__ == "__main__":
    kind, a, b, out = sys.argv[1:5]
    sizes = [int(s) for s in sys.argv[5:]] or [0]
    C, X, names = kang(a, b) if kind == "kang" else hao(a, b)
    for n in sizes:
        finish(C, X, names, out if len(sizes) == 1 else out.replace(".npz", f"_{n}.npz"), n)
