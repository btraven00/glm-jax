"""Stream genes from host memory through the device in chunks sized to fit it.

``fit_glm`` and ``estimate_dispersion_grid`` take the whole (G, n) matrix on the
device; at 1M cells x 2k genes that is 8 GB in f32 before any intermediates.
Here Y stays on the host (dense or scipy.sparse, genes in rows) and goes over
in chunks of genes. The chunk size comes from XLA's own accounting of the
compiled program (``memory_analysis``: arguments + temporaries + outputs), so it
follows the kernel when that changes rather than a hand-tuned bytes-per-cell.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from .dispersion import estimate_dispersion_grid
from .irls import GLMResult, fit_glm

_PROBE = 64          # genes in the sizing compile
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


def plan_chunk(make, row_specs, shared, G, budget):
    """Largest gene chunk whose compiled program fits ``budget``; returns (chunk, compiled).

    ``make(chunk)`` returns the per-chunk function f(*rows, *shared); ``row_specs``
    are (shape-after-gene-axis, dtype) of the per-gene inputs. Compiles once at
    ``_PROBE`` genes to get bytes per gene, then at the predicted size, shrinking
    by the overshoot ratio until it fits (usually 2 compiles in total).
    """
    def compile_at(c):
        args = [jax.ShapeDtypeStruct((c,) + sh, dt) for sh, dt in row_specs]
        args += [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in shared]
        compiled = jax.jit(make(c)).lower(*args).compile()
        return compiled, _program_bytes(compiled)

    c0 = min(G, _PROBE)
    compiled, used = compile_at(c0)
    while used > budget:
        if c0 == 1:
            raise MemoryError(f"one gene needs {used / 2**30:.2f} GiB > budget {budget / 2**30:.2f} GiB; "
                              "use fewer cells per call or a larger device")
        c0 = max(1, int(c0 * budget / used * 0.9))
        compiled, used = compile_at(c0)
    fixed = sum(a.nbytes for a in shared)
    per_gene = max(1.0, (used - fixed) / c0)
    c = min(G, int((budget - fixed) / per_gene))
    while c > c0:
        compiled_c, used_c = compile_at(c)
        if used_c <= budget:
            return c, compiled_c
        c = int(c * budget / used_c * 0.95)
    return c0, compiled


def stream_genes(make, rows, shared, *, budget=None, chunk=None):
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
        chunk, run = plan_chunk(make, specs, shared, G, device_budget() if budget is None else budget)
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

    res, chunk = stream_genes(make, (Y, phi), (X, offset), budget=budget, chunk=chunk)
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

    res, chunk = stream_genes(make, (Y,), (X, offset), budget=budget, chunk=chunk)
    grid = kwargs.get("phi_grid")
    res["phi_grid"] = np.asarray(jnp.logspace(-4, 1, 60) if grid is None else grid, X.dtype)
    return res, chunk
