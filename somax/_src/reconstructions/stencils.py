from typing import Tuple
from jax.lax import dynamic_slice_in_dim
from functools import partial
from jaxtyping import Array


def stencil_2pt(u: Array, dim: int) -> Tuple[Array, Array]:
    """
    Compute the 2-point stencil of an array along a given dimension.

    Args:
        u (Array): The input array.
        dim (int): The dimension along which to compute the stencil.

    Returns:
        Tuple[Array, Array]: A tuple containing two arrays, q0 and q1, which represent
        the 2-point stencil of the input array u along the specified dimension.
    """
    # get number of points
    num_pts = u.shape[dim]

    # define slicers
    dyn_slicer = partial(dynamic_slice_in_dim, axis=dim)

    q0 = dyn_slicer(u, 0, num_pts - 1)
    q1 = dyn_slicer(u, 1, num_pts - 1)

    return q0, q1


def stencil_2pt_bounds(
    u: Array, dim: int
) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
    """
    Compute the 2-point stencil bounds for a given array along a specified dimension.

    Args:
        u (Array): The input array.
        dim (int): The dimension along which to compute the stencil bounds.

    Returns:
        Tuple[Tuple[Array, Array], Tuple[Array, Array]]: A tuple containing two tuples,
        each representing the stencil bounds. The first tuple represents the left stencil
        bounds, and the second tuple represents the right stencil bounds.
    """

    # define slicers
    dyn_slicer = partial(dynamic_slice_in_dim, axis=dim)

    # interior slices
    qleft_0 = dyn_slicer(u, 0, 1)
    qleft_1 = dyn_slicer(u, 1, 1)
    qright_0 = dyn_slicer(u, -1, 1)
    qright_1 = dyn_slicer(u, -2, 1)

    return (qleft_0, qleft_1), (qright_0, qright_1)


def stencil_3pt(
        u: Array, dim: int
        ) -> Tuple[Array, Array, Array]:
    """
    Compute the 3-point stencil for a given array along a specified dimension.

    Args:
        u (Array): The input array.
        dim (int): The dimension along which to compute the stencil.

    Returns:
        Tuple[Array, Array, Array]: A tuple containing three arrays representing the 3-point stencil.
    """
    # get number of points
    num_pts = u.shape[dim]

    # define slicers
    dyn_slicer = partial(dynamic_slice_in_dim, axis=dim)

    # interior slices
    u0 = dyn_slicer(u, 0, num_pts - 2)
    u1 = dyn_slicer(u, 1, num_pts - 2)
    u2 = dyn_slicer(u, 2, num_pts - 2)

    return u0, u1, u2


def stencil_3pt_bounds(
    u: Array, dim: int
) -> Tuple[Tuple[Array, Array, Array], Tuple[Array, Array, Array]]:
    """
    Compute the 3-point stencil bounds for a given array along a specified dimension.

    Args:
        u (Array): The input array.
        dim (int): The dimension along which to compute the stencil bounds.

    Returns:
        Tuple[Tuple[Array, Array, Array], Tuple[Array, Array, Array]]: A tuple of two tuples,
        where each inner tuple contains three slices representing the stencil bounds.
        The first tuple represents the left stencil bounds, and the second tuple represents
        the right stencil bounds.
    """

    # define slicers
    dyn_slicer = partial(dynamic_slice_in_dim, axis=dim)

    # interior slices
    qleft_0 = dyn_slicer(u, 0, 1)
    qleft_1 = dyn_slicer(u, 1, 1)
    qleft_2 = dyn_slicer(u, 2, 1)
    qright_0 = dyn_slicer(u, -1, 1)
    qright_1 = dyn_slicer(u, -2, 1)
    qright_2 = dyn_slicer(u, -3, 1)

    return (qleft_0, qleft_1, qleft_2), (qright_0, qright_1, qright_2)


def stencil_5pt(
        u: Array, dim: int
        ) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Compute the 5-point stencil for a given array along a specified dimension.

    Args:
        u (Array): The input array.
        dim (int): The dimension along which to compute the stencil.

    Returns:
        Tuple[Array, Array, Array, Array, Array]: A tuple of five arrays representing the 5-point stencil.
    """

    # get number of points
    num_pts = u.shape[dim]

    # define slicers
    dyn_slicer = partial(dynamic_slice_in_dim, axis=dim)

    # interior slices
    u0 = dyn_slicer(u, 0, num_pts - 4)
    u1 = dyn_slicer(u, 1, num_pts - 4)
    u2 = dyn_slicer(u, 2, num_pts - 4)
    u3 = dyn_slicer(u, 3, num_pts - 4)
    u4 = dyn_slicer(u, 4, num_pts - 4)

    return u0, u1, u2, u3, u4
