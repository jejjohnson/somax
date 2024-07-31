from jaxtyping import Array
import jax.numpy as jnp
import math


def poisson_periodic_wavenumber(k: int, n: int) -> Array:
    if k <= int(n / 2):
        wn = 2.0 * jnp.sin(math.pi * k / float(n))
    else:
        wn = 2.0 * jnp.sin(math.pi * (n - k) / float(n))
    return - wn * wn


def poisson_dirichlet_wavenumber(k: int, n: int) -> Array:
    return 2.0 * (jnp.cos(math.pi * (k + 1) / float(n + 1)) - 1.0)
    # return - 2.0 * jnp.sin( (jnp.pi / 2.0) * (k + 1) / (n + 1) )


def poisson_neumann_wavenumber(k: int, n: int) -> Array:
    return 2.0 * (jnp.cos(math.pi * k / float(n)) - 1.0)
