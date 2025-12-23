import math
import typing as tp

import einops
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np


def make_coords(xmin, xmax, nx):
    return jnp.linspace(start=xmin, stop=xmax, num=nx, endpoint=True)


def make_grid_from_coords(coords: tp.Iterable) -> tp.List[Array]:
    if isinstance(coords, tp.Iterable):
        return jnp.meshgrid(*coords, indexing="ij")
    elif isinstance(coords, (jnp.ndarray, np.ndarray)):
        return jnp.meshgrid(coords, indexing="ij")
    else:
        raise ValueError("Unrecognized dtype for inputs")


def make_grid_coords(coords: tp.Iterable) -> Array:
    grid = make_grid_from_coords(coords)

    grid = jnp.stack(grid, axis=0)

    grid = einops.rearrange(grid, "N ... -> (...) N")

    return grid


def create_meshgrid_coordinates(shape):
    meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in shape], indexing="ij")
    # create indices
    indices = jnp.concatenate([jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1)

    return indices


def bounds_to_length(xmin: float, xmax: float) -> float:
    """Calculates the Lx from the minmax
    Eq:
        Lx = abs(xmax - xmin)

    Args:
        xmin (Array | float): the input start point
        xmax (Array | float): the input end point

    Returns:
        Lx (Array | float): the distance between the min and max
            points
    """
    return abs(float(xmax) - float(xmin))


def bounds_and_points_to_step(xmin: float, xmax: float, Nx: float) -> float:
    """Calculates the dx from the minmax
    Eq:
        Lx = abs(xmax - xmin)
        dx = Lx / (Nx - 1)

    Args:
        xmin (Array | float): the input start point
        xmax (Array | float): the input end point
        Nx (int | float): the number of points

    Returns:
        dx (Array | float): the distance between each of the
            steps.
    """
    Lx = bounds_to_length(xmin=xmin, xmax=xmax)
    return length_and_points_to_step(Lx=Lx, Nx=Nx)


def length_and_points_to_step(Lx: float, Nx: float) -> float:
    """Calculates the dx from the number of points
    and distance between the endpoints

    Eq:
        dx = Lx / (Nx - 1)

    Args:
        Lx (Array | float): distance between min and max points
        Nx (int | float): the number of points

    Returns:
        dx (Array | float): stepsize between each point.
    """
    return float(Lx) / (float(Nx) - 1.0)


def length_and_step_to_points(Lx: float, dx: float) -> int:
    """Calculates the number of points from the
    stepsize and distance between the endpoints.
    This assumes we go from (0,Lx)

    Eq:
        Nx = 1 + Lx / dx

    Args:
        Lx (Array | float): distance between min and max points
        dx (Array | float): stepsize between each point.

    Returns:
        Nx (Array | int): the number of points
    """
    return int(math.floor(1.0 + float(Lx) / float(dx)))


def bounds_and_step_to_points(xmin: float, xmax: float, dx: float) -> int:
    """Calculates the number of points from the
    endpoints and the stepsize

    Eq:
        Nx = 1 + floor((xmax-xmin)) / dx)

    Args:
        xmin (Array | float): the input start point
        xmax (Array | float): the input end point
        dx (Array | float): stepsize between each point.

    Returns:
        Nx (Array | int): the number of points
    """
    Lx = bounds_to_length(xmin=xmin, xmax=xmax)
    return length_and_step_to_points(Lx=Lx, dx=dx)


def check_stagger(dx: tp.Tuple, stagger: tp.Tuple[str] = None):
    """Creates stagger values based on semantic names.
    Useful for C-Grid operations

    Args:
    -----
        dx (Iterable): the step sizes
        stagger (Iterable): the stagger direction

    Returns:
    --------
        stagger (Iterable): the stagger values (as a fraction
            of dx).
    """
    if stagger is None:
        stagger = (None,) * len(dx)

    msg = "Length of stagger and dx is off"
    msg += f"\ndx: {len(dx)}"
    msg += f"\nstagger: {len(stagger)}"
    assert len(dx) == len(stagger), msg

    stagger_values = list()
    for istagger in stagger:
        if istagger is None:
            stagger_values.append(0.0)
        elif istagger == "right":
            stagger_values.append(0.5)
        elif istagger == "left":
            stagger_values.append(-0.5)
        else:
            raise ValueError("Unrecognized command")

    return stagger_values


def check_tuple_inputs(x) -> tp.Tuple:
    if isinstance(x, tuple):
        return x
    elif isinstance(x, (int, float)):
        return tuple(x)
    elif isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    elif x is None:
        return None
    else:
        raise ValueError(f"Unrecognized type: {x} | {type(x)}")
