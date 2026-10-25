"""Compiled temp/argument memory of fit_glm vs gene batch size, on the default backend.

    python examples/mem_probe.py
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import glm_jax as gj

G, p = 2000, 5
s = lambda *sh: jax.ShapeDtypeStruct(sh, jnp.float64)
for N in (10_000, 50_000):
    for B in (None, 400, 100):
        f = jax.jit(lambda Y, X, o, phi: gj.fit_glm(Y, X, offset=o, phi=phi, batch_size=B).coef)
        m = f.lower(s(G, N), s(N, p), s(N), s(G)).compile().memory_analysis()
        print(f"{jax.default_backend()} N={N} B={B}: temp={m.temp_size_in_bytes / 2**30:.2f} GiB "
              f"args={m.argument_size_in_bytes / 2**30:.2f} GiB")
