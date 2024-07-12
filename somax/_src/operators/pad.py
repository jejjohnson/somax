import functools as ft

import jax.numpy as jnp
from jaxtyping import Array

from somax.domain import Domain
from somax._src.field.base import Field

PADDING = {
    "both": (1, 1),
    "right": (0, 1),
    "left": (1, 0),
    None: (0, 0),
}


def pad_array(u: Array, pad_width, **kwargs):
    # check lengths
    assert len(pad_width) <= u.ndim

    # convert to tuple if necessary
    pad_width = [
        PADDING[ipad] if isinstance(ipad, (str, type(None))) else ipad
        for ipad in pad_width
    ]

    return jnp.pad(u, pad_width=pad_width, **kwargs)


def pad_domain(domain: Domain, pad_width):
    # check lengths
    assert len(pad_width) <= len(domain.Nx)

    # convert to tuple if necessary
    pad_width = [
        PADDING[ipad] if isinstance(ipad, (str, type(None))) else ipad
        for ipad in pad_width
    ]

    # check if axis is used
    axis = [1 if sum(ipad) > 0 else 0 for ipad in pad_width]

    xmin = [
        xmin - pad_width[i][0] * domain.dx[i] if axis else xmin
        for i, xmin in enumerate(domain.xmin)
    ]
    xmax = [
        xmax + pad_width[i][1] * domain.dx[i] if axis else xmax
        for i, xmax in enumerate(domain.xmax)
    ]

    Nx = [iNx + sum(ipad) for ipad, iNx in zip(pad_width, domain.Nx)]

    Lx = [
        iLx + (pad_width[i][0] * domain.dx[i] + pad_width[i][1] * domain.dx[i])
        for i, (ipad, iLx) in enumerate(zip(pad_width, domain.Lx))
    ]

    domains = [
        Domain(xmin=ixmin, xmax=ixmax, dx=idx, Lx=iLx, Nx=iNx)
        for ixmin, ixmax, idx, iLx, iNx in zip(xmin, xmax, domain.dx, Lx, Nx)
    ]

    domain = ft.reduce(lambda a, b: a * b, domains)

    return domain


def pad_field(u: Field, pad_width, **kwargs) -> Field:
    # values
    u_values = pad_array(u.values, pad_width=pad_width, **kwargs)
    # domain
    domain = pad_domain(u.domain, pad_width=pad_width)
    return Field(values=u_values, domain=domain)
