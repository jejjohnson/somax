from typing import Iterable, Union
from jaxtyping import Array
import jax.numpy as jnp


PAD_WIDTH = {"right": (0, 1), "left": (1, 0), "both": (1, 1), None: (0, 0)}

def padding(u: Array, axes: Iterable[Union[str | None]], *args, **kwargs):

    # check axes length is the same as u
    num_axes = len(axes)
    if num_axes > 1:
        assert num_axes == u.ndim
    # get padding arguments
    pad_width = [PAD_WIDTH[iaxes] for iaxes in axes]

    # pad axis
    u = jnp.pad(u, pad_width=pad_width, *args, **kwargs)

    return u


def zero_boundaries(u, pad_width):
    return jnp.pad(array=u, pad_width=pad_width, mode="constant", constant_values=0.0)

def zero_gradient_boundaries(u, pad_width):
    return jnp.pad(array=u, pad_width=pad_width, mode="edge")

def periodic_boundaries(u, pad_width):
    return jnp.pad(array=u, pad_width=pad_width, mode="wrap")

no_flow_boundaries = zero_boundaries
no_energy_boundaries = zero_boundaries
no_flux_boundaries = zero_boundaries
no_slip_boundaries = zero_boundaries
free_slip_boundaries = zero_gradient_boundaries
