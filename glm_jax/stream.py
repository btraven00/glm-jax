"""Stream genes from host memory through the device in chunks sized to fit it.

``fit_glm`` and ``estimate_dispersion_grid`` take the whole (G, n) matrix on the
device; at 1M cells x 2k genes that is 8 GB in f32 before any intermediates.
Here Y stays on the host (dense or scipy.sparse, genes in rows) and goes over
in chunks of genes. The chunk size comes from XLA's own accounting of the
compiled program (``memory_analysis``: arguments + temporaries + outputs), so it
follows the kernel when that changes rather than a hand-tuned bytes-per-cell.
"""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from .dispersion import estimate_dispersion_grid
from .irls import GLMResult, fit_glm

_DEVICE_BATCH = 256  # fit_glm batch_size cap inside a chunk (GPU time is flat in it)


def device_budget(device=None, frac=0.8):
    """Bytes a program may use on ``device``: ``frac`` of what is free.

    GPU/TPU: the allocator's limit minus bytes in use (JAX preallocates 75% of
    the card by default, and that limit is what counts). CPU: available RAM.
    """
    d = device or jax.devices()[0]
    s = d.memory_stats() if hasattr(d, "memory_stats") else None
    if s and "bytes_limit" in s:
        free = s["bytes_limit"] - s.get("bytes_in_use", 0)
    else:
        free = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return int(frac * free)


def _program_bytes(compiled):
    m = compiled.memory_analysis()
    return m.argument_size_in_bytes + m.temp_size_in_bytes + m.output_size_in_bytes - m.alias_size_in_bytes


_CAL_N = 4096                 # cells in the calibration compiles (cheap: ~5 s each, no remat)
_CAL_CHUNKS = (32, 256, 1024)  # table points; bytes are piecewise linear in chunk between them
_CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "glm_jax", "plan.json")


def _at(shape, n, n0):
    return tuple(n0 if d == n else d for d in shape)


def _table(make, row_specs, shared, n, key):
    """Program bytes per cell at each _CAL_CHUNKS size, from compiles at _CAL_N cells; cached on disk.

    Every large buffer is (chunk or batch) x cells, so bytes scale linearly in the
    cell count (checked: 200k cells uses 9.9x the bytes of 20k) and a table at a few
    thousand cells extrapolates to any n without compiling at full size, which is
    slow near the device limit (XLA tries to rematerialize).
    """
    try:
        with open(_CACHE) as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    if key not in cache:
        sh = [jax.ShapeDtypeStruct(_at(a.shape, n, _CAL_N), a.dtype) for a in shared]
        fixed = sum(np.prod(a.shape) * a.dtype.itemsize for a in sh)
        cache[key] = []
        for c in _CAL_CHUNKS:
            args = [jax.ShapeDtypeStruct((c,) + _at(s_, n, _CAL_N), dt) for s_, dt in row_specs] + sh
            used = _program_bytes(jax.jit(make(c)).lower(*args).compile())
            cache[key].append((used - fixed) / _CAL_N)
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        with open(_CACHE, "w") as f:
            json.dump(cache, f)
    return np.array(_CAL_CHUNKS, float), np.array(cache[key])


def plan_chunk(make, row_specs, shared, G, budget, key):
    """Largest gene chunk whose compiled program fits ``budget``; returns (chunk, compiled).

    ``make(chunk)`` returns the per-chunk function f(*rows, *shared); ``row_specs``
    are (shape-after-gene-axis, dtype) of the per-gene inputs; ``key`` names the
    kernel and its settings for the calibration cache. The chunk comes from the
    calibrated table (see ``_table``); one compile at that size checks it, and
    shrinks it in the rare case XLA's accounting disagrees.
    """
    n = shared[0].shape[0]
    key = f"{key}|{jax.devices()[0].device_kind}|{jax.__version__}|{shared[0].dtype}|{shared[0].shape[1:]}"
    cs, per_cell = _table(make, row_specs, shared, n, key)
    fixed = sum(a.nbytes for a in shared)
    room = (budget - fixed) / n
    if room < per_cell[0] / cs[0]:
        raise MemoryError(f"one gene needs ~{per_cell[0] / cs[0] * n / 2**30:.2f} GiB > budget "
                          f"{budget / 2**30:.2f} GiB; use fewer cells per call or a larger device")
    # piecewise-linear interpolation, extended through the origin and past the last point
    xs, ys = np.r_[0.0, cs], np.r_[0.0, per_cell]
    c = float(np.interp(room, ys, xs)) if room <= ys[-1] else xs[-1] + (room - ys[-1]) * (xs[-1] - xs[-2]) / (ys[-1] - ys[-2])
    c = int(min(G, c))
    if c > _DEVICE_BATCH:
        c -= c % _DEVICE_BATCH  # whole batches: a ragged tail compiles a second kernel copy
    while True:
        args = [jax.ShapeDtypeStruct((c,) + sh, dt) for sh, dt in row_specs]
        args += [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in shared]
        compiled = jax.jit(make(c)).lower(*args).compile()
        used = _program_bytes(compiled)
        if used <= budget or c == 1:
            return c, compiled
        c = max(1, int(c * budget / used * 0.95))


def stream_genes(make, rows, shared, *, budget=None, chunk=None, key=""):
    """Apply ``make(chunk)(*row_chunks, *shared)`` over gene chunks; concatenate on the host.

    ``rows``: host arrays with genes on axis 0 (numpy or scipy.sparse; densified a
    chunk at a time). ``shared``: device arrays every chunk uses (X, offset). The
    function must return a pytree of per-gene arrays (leading axis = chunk). The
    last chunk is zero-padded, so the program compiles once.
    """
    G = rows[0].shape[0]
    dtype = shared[0].dtype
    specs = [(r.shape[1:], dtype) for r in rows]
    if chunk is None:
        chunk, run = plan_chunk(make, specs, shared, G, device_budget() if budget is None else budget, key)
    else:
        run = jax.jit(make(chunk))

    def piece(r, i):
        a = r[i:i + chunk]
        a = np.asarray(a.toarray() if hasattr(a, "toarray") else a, dtype=dtype)
        if a.shape[0] < chunk:
            a = np.concatenate([a, np.zeros((chunk - a.shape[0],) + a.shape[1:], dtype)])
        return a

    # One chunk on the device at a time (the budget is per program), but the host
    # densifies the next chunk while the device runs the current one.
    outs, pending = [], None
    for i in range(0, G, chunk):
        host = [piece(r, i) for r in rows]
        if pending is not None:
            outs.append(jax.tree_util.tree_map(np.asarray, pending))  # blocks, frees the device
        pending = run(*[jax.device_put(h) for h in host], *shared)
    outs.append(jax.tree_util.tree_map(np.asarray, pending))
    return jax.tree_util.tree_map(lambda *a: np.concatenate(a)[:G], *outs), chunk


def fit_glm_streamed(Y, X, *, offset=None, phi=None, budget=None, chunk=None, **fit_kwargs):
    """``fit_glm`` with Y on the host, gene chunks sized to the device. ``mu`` is not returned (None).

    ``offset``: (n,) or None; ``phi``: scalar or (G,). Returns (GLMResult, chunk).
    """
    X = jnp.asarray(X)
    dtype = X.dtype
    G, n = Y.shape
    offset = jnp.zeros(n, dtype) if offset is None else jnp.asarray(offset, dtype)
    phi = np.broadcast_to(np.asarray(0.0 if phi is None else phi, dtype), (G,))

    def make(c):
        def f(Yc, phic, X, offset):
            r = fit_glm(Yc, X, offset=offset, phi=phic, batch_size=min(c, _DEVICE_BATCH), **fit_kwargs)
            return r._replace(mu=None)
        return f

    res, chunk = stream_genes(make, (Y, phi), (X, offset), budget=budget, chunk=chunk,
                              key=f"fit|{sorted(fit_kwargs.items())}")
    return GLMResult(*res), chunk


def estimate_dispersion_streamed(Y, X, *, offset=None, budget=None, chunk=None, **kwargs):
    """Tagwise ``estimate_dispersion_grid`` with Y on the host, gene chunks sized to the device.

    Returns ({"tagwise", "apl", "phi_grid"}, chunk). No polished common
    dispersion (it needs every gene at once); the grid one is
    ``phi_grid[argmax(apl.sum(0))]``.
    """
    X = jnp.asarray(X)
    n = Y.shape[1]
    offset = jnp.zeros(n, X.dtype) if offset is None else jnp.asarray(offset, X.dtype)

    def make(c):
        def f(Yc, X, offset):
            r = estimate_dispersion_grid(Yc, X, offset=offset, **kwargs)
            return {"tagwise": r["tagwise"], "apl": r["apl"]}
        return f

    res, chunk = stream_genes(make, (Y,), (X, offset), budget=budget, chunk=chunk,
                              key=f"disp|{sorted(kwargs.items())}")
    grid = kwargs.get("phi_grid")
    res["phi_grid"] = np.asarray(jnp.logspace(-4, 1, 60) if grid is None else grid, X.dtype)
    return res, chunk
