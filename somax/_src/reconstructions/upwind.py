import functools as ft
import typing as tp

import jax
from jax.nn import relu
import jax.numpy as jnp
from jaxtyping import Array

from somax._src.reconstructions.linear import (
    linear_2pts,
    linear_3pts_left,
    linear_5pts_left,
)
from somax._src.reconstructions.weno import (
    weno_3pts,
    weno_3pts_improved,
    weno_5pts,
    weno_5pts_improved,
)


def plusminus(u: Array, way: int = 1) -> tp.Tuple[Array, Array]:
    msg = "Way should be 1 or -1."
    msg += f"\nWay: {way}"
    assert way in [1, -1], msg
    u_pos = float(way) * relu(float(way) * u)
    u_neg = u - u_pos
    return u_pos, u_neg


def upwind_1pt(q: Array, dim: int) -> tp.Tuple[Array, Array]:
    """creates the stencils for the upwind scheme
    - 1 pts inside domain & boundary
    Args:
        q (Array): input tracer
            shape[dim] = N

    Return:
        qi_left (Array): output tracer left size
            shape[dim] = N-1
        qi_right (Array): output tracer left size
            shape[dim] = N-1
    """
    # get number of points
    num_pts = q.shape[dim]

    # define slicers
    dyn_slicer = ft.partial(jax.lax.dynamic_slice_in_dim, axis=dim)

    qi_left = dyn_slicer(q, 0, num_pts - 1)
    qi_right = dyn_slicer(q, 1, num_pts - 1)

    return qi_left, qi_right


def upwind_2pt_bnds(
    q: Array, dim: int, method: str = "linear"
) -> tp.Tuple[Array, Array]:
    """creates the stencils for the upwind scheme
    - 3 pts inside domain
    Args:
        q (Array): the input array to be spliced
            shape[dim] = N
    Returns:
        qi_left (Array): the spliced array on the left side
            shape[dim] = N-2
        qi_right (Array): the spliced array on the left side
            shape[dim] = N-2
    """

    # define slicers
    dyn_slicer = ft.partial(jax.lax.dynamic_slice_in_dim, axis=dim)

    # interior slices
    qleft_0 = dyn_slicer(q, 0, 1)
    qleft_1 = dyn_slicer(q, 1, 1)
    qright_0 = dyn_slicer(q, -1, 1)
    qright_1 = dyn_slicer(q, -2, 1)

    # DO WENO Interpolation
    if method == "linear":
        qi_left_interior = linear_2pts(qleft_0, qleft_1)
        qi_right_interior = linear_2pts(qright_0, qright_1)
    else:
        msg = f"Unrecognized method: {method}"
        msg += "\nNeeds to be 'linear', 'weno', or 'wenoz'."
        raise ValueError(msg)

    return qi_left_interior, qi_right_interior


def upwind_3pt(q: Array, dim: int, method: str = "weno") -> tp.Tuple[Array, Array]:
    """creates the stencils for the upwind scheme
    - 3 pts inside domain
    Args:
        q (Array): the input array to be spliced
            shape[dim] = N
    Returns:
        qi_left (Array): the spliced array on the left side
            shape[dim] = N-2
        qi_right (Array): the spliced array on the left side
            shape[dim] = N-2
    """

    # get number of points
    num_pts = q.shape[dim]

    # define slicers
    dyn_slicer = ft.partial(jax.lax.dynamic_slice_in_dim, axis=dim)

    # interior slices
    q0 = dyn_slicer(q, 0, num_pts - 2)
    q1 = dyn_slicer(q, 1, num_pts - 2)
    q2 = dyn_slicer(q, 2, num_pts - 2)

    # DO WENO Interpolation
    if method == "linear":
        qi_left_interior = linear_3pts_left(q0, q1, q2)
        qi_right_interior = linear_3pts_left(q2, q1, q0)
    elif method == "weno":
        qi_left_interior = weno_3pts(q0, q1, q2)
        qi_right_interior = weno_3pts(q2, q1, q0)
    elif method == "wenoz":
        qi_left_interior = weno_3pts_improved(q0, q1, q2)
        qi_right_interior = weno_3pts_improved(q2, q1, q0)
    else:
        msg = f"Unrecognized method: {method}"
        msg += "\nNeeds to be 'linear', 'weno', or 'wenoz'."
        raise ValueError(msg)

    return qi_left_interior, qi_right_interior


def upwind_3pt_bnds(q: Array, dim: int, method: str = "weno") -> tp.Tuple[Array, Array]:
    """creates the stencils for the upwind scheme
    - 3 pts inside domain
    Args:
        q (Array): the input array to be spliced
            shape[dim] = N
    Returns:
        qi_left (Array): the spliced array on the left side
            shape[dim] = N-2
        qi_right (Array): the spliced array on the left side
            shape[dim] = N-2
    """

    # define slicers
    dyn_slicer = ft.partial(jax.lax.dynamic_slice_in_dim, axis=dim)

    # interior slices
    q0 = jnp.concatenate([dyn_slicer(q, 0, 1), dyn_slicer(q, -3, 1)], axis=dim)
    q1 = jnp.concatenate([dyn_slicer(q, 1, 1), dyn_slicer(q, -2, 1)], axis=dim)
    q2 = jnp.concatenate([dyn_slicer(q, 2, 1), dyn_slicer(q, -1, 1)], axis=dim)

    # DO WENO Interpolation
    if method == "linear":
        qi_left_interior = linear_3pts_left(q0, q1, q2)
        qi_right_interior = linear_3pts_left(q2, q1, q0)
    elif method == "weno":
        qi_left_interior = weno_3pts(q0, q1, q2)
        qi_right_interior = weno_3pts(q2, q1, q0)
    elif method == "wenoz":
        qi_left_interior = weno_3pts_improved(q0, q1, q2)
        qi_right_interior = weno_3pts_improved(q2, q1, q0)
    else:
        msg = f"Unrecognized method: {method}"
        msg += "\nNeeds to be 'linear', 'weno', or 'wenoz'."
        raise ValueError(msg)

    return qi_left_interior, qi_right_interior


def upwind_5pt(q: Array, dim: int, method: str = "weno") -> tp.Tuple[Array, Array]:
    """creates the stencils for the upwind scheme
    - 5 pts inside domain
    Args:
        q (Array): the input array to be spliced
            shape[dim] = N
    Returns:
        qi_left (Array): the spliced array on the left side
            shape[dim] = N-4
        qi_right (Array): the spliced array on the left side
            shape[dim] = N-4
    """

    # get number of points
    num_pts = q.shape[dim]

    # define slicers
    dyn_slicer = ft.partial(jax.lax.dynamic_slice_in_dim, axis=dim)

    # interior slices
    q0 = dyn_slicer(q, 0, num_pts - 4)
    q1 = dyn_slicer(q, 1, num_pts - 4)
    q2 = dyn_slicer(q, 2, num_pts - 4)
    q3 = dyn_slicer(q, 3, num_pts - 4)
    q4 = dyn_slicer(q, 4, num_pts - 4)

    # DO WENO Interpolation
    if method == "linear":
        qi_left_interior = linear_5pts_left(q0, q1, q2, q3, q4)
        qi_right_interior = linear_5pts_left(q4, q3, q2, q1, q0)
    elif method == "weno":
        qi_left_interior = weno_5pts(q0, q1, q2, q3, q4)
        qi_right_interior = weno_5pts(q4, q3, q2, q1, q0)
    elif method == "wenoz":
        qi_left_interior = weno_5pts_improved(q0, q1, q2, q3, q4)
        qi_right_interior = weno_5pts_improved(q4, q3, q2, q1, q0)
    else:
        msg = f"Unrecognized method: {method}"
        msg += "\nNeeds to be 'linear', 'weno', or 'wenoz'."
        raise ValueError(msg)

    return qi_left_interior, qi_right_interior
