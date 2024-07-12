from functools import reduce
from operator import mul
import typing as tp

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np

from somax._src.domain.utils import (
    bounds_and_points_to_step,
    bounds_and_step_to_points,
    bounds_to_length,
    make_coords,
    make_grid_coords,
    make_grid_from_coords,
)


def check_inputs_types(x, name: str):
    if isinstance(x, (int, float)):
        return tuple([x])
    elif isinstance(x, list):
        return tuple(x)
    elif isinstance(x, tuple):
        return x
    elif isinstance(x, (np.ndarray, jnp.ndarray)):
        return tuple(map(lambda x: float(x), x))
    raise ValueError(f"Unexpected type for {name}, got {x}.")


def _check_inputs(xmin, xmax, Nx, Lx, dx):
    # check (xmax - xmin) == Lx
    assert bounds_to_length(xmin=xmin, xmax=xmax) == Lx
    # check (xmax - xmin) / (Nx - 1)
    assert bounds_and_points_to_step(xmin=xmin, xmax=xmax, Nx=Nx) == dx


def _batch_check_inputs(xmin, xmax, Nx, Lx, dx):
    pass


class DomainLonLat(eqx.Module):
    """Domain class for a rectangular domain

    Attributes:
        size (Tuple[int]): The size of the domain
        xmin: (Iterable[float]): The min bounds for the input domain
        xmax: (Iterable[float]): The max bounds for the input domain
        coord (List[Array]): The coordinates of the domain
        grid (Array): A grid of the domain
        ndim (int): The number of dimenions of the domain
        size (Tuple[int]): The size of each dimenions of the domain
        cell_volume (float): The total volume of a grid cell
    """

    lat_min: float = eqx.static_field()
    lat_max: float = eqx.static_field()
    d_lat: float = eqx.static_field()
    N_lat: float = eqx.static_field()
    L_lat: float = eqx.static_field()
    lat_coords: Array = eqx.static_field()

    lon_min: float = eqx.static_field()
    lon_min: float = eqx.static_field()
    d_lon: float = eqx.static_field()
    N_lon: int = eqx.static_field()
    L_lat: float = eqx.static_field()
    lon_coords: Array = eqx.static_field()


# def init_domain_from_bounds_and_numpoints(
#     xmin: float = 0.0, xmax: float = 1.0, Nx: int = 50
# ):
#     """initialize 1d domain from bounds and number of points
#     Eqs:
#         dx = (x_max - x_min) / (Nx - 1)
#         Lx = (x1 - x0)

#     Args:
#         xmin (float): x min value
#         xmax (float): maximum value
#         Nx (int): the number of points for domain
#     Returns:
#         domain (Domain): the full domain
#     """

#     # calculate Nx
#     dx = bounds_and_points_to_step(xmin=xmin, xmax=xmax, Nx=Nx)

#     # calculate Lx
#     Lx = bounds_to_length(xmin=xmin, xmax=xmax)

#     return Domain(xmin=xmin, xmax=xmax, dx=dx, Nx=Nx, Lx=Lx)


# def init_domain_from_bounds_and_step(xmin: float, xmax: float, dx: float) -> int:
#     """initialize 1d domain from bounds and step_size
#     Eqs:
#          Nx = 1 + ceil((x_max - x_min) / dx)
#         Lx = (x1 - x0)

#     Args:
#         xmin (float): x min value
#         xmax (float): maximum value
#         Nx (int): the number of points for domain
#     Returns:
#         domain (Domain): the full domain
#     """

#     # calculate Nx
#     Nx = bounds_and_step_to_points(xmin=xmin, xmax=xmax, dx=dx)

#     # calculate Lx
#     Lx = bounds_to_length(xmin=xmin, xmax=xmax)

#     return Domain(xmin=xmin, xmax=xmax, dx=dx, Nx=Nx, Lx=Lx)
