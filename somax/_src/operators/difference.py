from typing import (
    List,
    Tuple,
)
from functools import partial
import jax
import equinox as eqx
import finitediffx as fdx
import jax.numpy as jnp
from jaxtyping import (
    Array,
    ArrayLike,
    Float,
)
import kernex as kex
from plum import dispatch
# from somax._src.field.base import Field

generate_finitediff_coeffs = fdx._src.utils.generate_finitediff_coeffs
generate_backward_offsets = fdx._src.utils._generate_backward_offsets
generate_central_offsets = fdx._src.utils._generate_central_offsets
generate_forward_offsets = fdx._src.utils._generate_forward_offsets


class FDParams(eqx.Module):
    method: str
    derivative: int
    accuracy: int
    offsets: Tuple[int, ...]
    coeffs: Tuple[int, ...]
    padding: Tuple[str, ...]

    def __init__(
        self, method: str = "backward", derivative: int = 1, accuracy: int = 1
    ):
        self.method = method
        self.derivative = derivative
        self.accuracy = accuracy
        # generate offsets
        self.offsets = tuple(
            generate_offsets(self.method, self.derivative, self.accuracy)
        )
        # generate coefficients
        self.coeffs = generate_finitediff_coeffs(self.offsets, self.derivative)
        # generate padding
        self.padding = tuple(
            generate_padding(self.method, self.derivative, self.accuracy)
        )

    @property
    def coeffs_dim(self):
        return len(self.coeffs)


def difference_1D(
    u: ArrayLike,
    step_size: float | ArrayLike = 1.0,
    method: str = "backward",
    derivative: int = 1,
    accuracy: int = 1,
    padding: str = "valid",
) -> ArrayLike:
    """
    Compute the numerical difference of an array along a specified axis using finite differences.

    Parameters:
    u (Array): The input array.
    axis (int, optional): The axis along which to compute the difference. Default is -2.
    step_size (float, optional): The step size used in the finite difference approximation. Default is 1.0.
    method (str, optional): The method used for differencing. Must be either "backward" or "forward". Default is "backward".

    Returns:
    Array: The differenced array.

    Raises:
    AssertionError: If the method is not "backward" or "forward".

    """
    # check method
    assert method in ["backward", "forward"]
    num_dims = 1
    # generate offsets
    offsets = tuple(
        generate_offsets(method, derivative=derivative, accuracy=accuracy)
    )
    # generate coefficients
    coeffs = generate_finitediff_coeffs(offsets, derivative=derivative)
    # kernel size
    kernel_size = (len(coeffs),)
    # initialize strides
    strides = (1,) * num_dims
    # initialize weight
    weight = jnp.array(coeffs).reshape(kernel_size)

    return apply_conv(
        x=u,
        weight=weight / (step_size**derivative),
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
    )


# @partial(jax.jit, static_argnames=("axis", "step_size", "method", "derivative", "accuracy", "padding"))
def difference_2D(
    u: ArrayLike,
    axis: int = 0,
    step_size: float | ArrayLike = 1.0,
    method: str = "backward",
    derivative: int = 1,
    accuracy: int = 1,
    padding: str = "valid",
) -> ArrayLike:
    """
    Compute the numerical difference of an array along a specified axis using finite differences.

    Parameters:
    u (Array): The input array.
    axis (int, optional): The axis along which to compute the difference. Default is -2.
    step_size (float, optional): The step size used in the finite difference approximation. Default is 1.0.
    method (str, optional): The method used for differencing. Must be either "backward" or "forward". Default is "backward".

    Returns:
    Array: The differenced array.

    Raises:
    AssertionError: If the method is not "backward" or "forward".

    """
    # check method
    num_dims = 2
    # assert method in ["backward", "forward"]
    # generate offsets
    offsets = tuple(
        generate_offsets(method, derivative=derivative, accuracy=accuracy)
    )
    # generate coefficients
    coeffs = generate_finitediff_coeffs(offsets, derivative=derivative)
    # kernel size
    kernel_size = tuple(
        calculate_kernel_size(len(coeffs), num_dims, axis=axis)
    )
    # initialize strides
    strides = (1,) * num_dims
    # initialize weight
    weight = jnp.array(coeffs).reshape(kernel_size)

    return apply_conv(
        x=u,
        weight=weight / (step_size**derivative),
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
    )


def difference_3D(
    u: ArrayLike,
    axis: int = 0,
    step_size: float | ArrayLike = 1.0,
    method: str = "backward",
    derivative: int = 1,
    accuracy: int = 1,
    padding: str = "valid",
) -> ArrayLike:
    """
    Compute the numerical difference of an array along a specified axis using finite differences.

    Parameters:
    u (Array): The input array.
    axis (int, optional): The axis along which to compute the difference. Default is -2.
    step_size (float, optional): The step size used in the finite difference approximation. Default is 1.0.
    method (str, optional): The method used for differencing. Must be either "backward" or "forward". Default is "backward".

    Returns:
    Array: The differenced array.

    Raises:
    AssertionError: If the method is not "backward" or "forward".

    """
    # check method
    num_dims = 3
    assert method in ["backward", "forward"]
    # generate offsets
    offsets = tuple(
        generate_offsets(method, derivative=derivative, accuracy=accuracy)
    )
    # generate coefficients
    coeffs = generate_finitediff_coeffs(offsets, derivative=derivative)
    # kernel size
    kernel_size = tuple(
        calculate_kernel_size(len(coeffs), num_dims, axis=axis)
    )
    # initialize strides
    strides = (1,) * 2
    # initialize weight
    weight = jnp.array(coeffs).reshape(kernel_size)

    return apply_conv(
        x=num_dims,
        weight=weight / (step_size**derivative),
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
    )



# @dispatch
# def difference_x(
#     u: Field,
#     method: str = "backward",
#     derivative: int = 1,
#     accuracy: int = 1,
#     padding: str = "valid",
# ) -> Field:
#     """
#     Compute the numerical difference of an array along a specified axis using finite differences.

#     Parameters:
#     u (Array): The input array.
#     axis (int, optional): The axis along which to compute the difference. Default is -2.
#     step_size (float, optional): The step size used in the finite difference approximation. Default is 1.0.
#     method (str, optional): The method used for differencing. Must be either "backward" or "forward". Default is "backward".

#     Returns:
#     Array: The differenced array.

#     Raises:
#     AssertionError: If the method is not "backward" or "forward".

#     """
#     num_dims = len(u.shape)
#     # check method
#     assert method in ["backward", "forward"]
#     # generate offsets
#     offsets = tuple(
#         generate_offsets(method, derivative=derivative, accuracy=accuracy)
#     )
#     # generate coefficients
#     coeffs = generate_finitediff_coeffs(offsets, derivative=derivative)
#     # kernel size
#     kernel_size = tuple(
#         calculate_kernel_size(len(coeffs), num_dims, axis=num_dims + -2)
#     )
#     # initialize strides
#     strides = (1,) * num_dims
#     # initialize weight
#     weight = jnp.array(coeffs).reshape(kernel_size)

#     # apply convolution on values
#     u_values = apply_conv(
#         x=u.values,
#         weight=weight / (u.domain.resolution[0]**derivative),
#         kernel_size=kernel_size,
#         padding=padding,
#         strides=strides,
#     )

#     # stagger domain
#     u_domain = u.domain.stagger_x(direction="inner", stagger=True)

#     return Field(u_values, u_domain)


# @dispatch
# def difference_y(
#     u: Field,
#     method: str = "backward",
#     derivative: int = 1,
#     accuracy: int = 1,
#     padding: str = "valid",
# ) -> Field:
#     """
#     Compute the numerical difference of an array along a specified axis using finite differences.

#     Parameters:
#     u (Array): The input array.
#     axis (int, optional): The axis along which to compute the difference. Default is -2.
#     step_size (float, optional): The step size used in the finite difference approximation. Default is 1.0.
#     method (str, optional): The method used for differencing. Must be either "backward" or "forward". Default is "backward".

#     Returns:
#     Array: The differenced array.

#     Raises:
#     AssertionError: If the method is not "backward" or "forward".

#     """
#     num_dims = len(u.shape)
#     # check method
#     assert method in ["backward", "forward"]
#     # generate offsets
#     offsets = tuple(
#         generate_offsets(method, derivative=derivative, accuracy=accuracy)
#     )
#     # generate coefficients
#     coeffs = generate_finitediff_coeffs(offsets, derivative=derivative)
#     # kernel size
#     kernel_size = tuple(
#         calculate_kernel_size(len(coeffs), num_dims, axis=num_dims + -1)
#     )
#     # initialize strides
#     strides = (1,) * num_dims
#     # initialize weight
#     weight = jnp.array(coeffs).reshape(kernel_size)

#     # apply convolution on values
#     u_values = apply_conv(
#         x=u.values,
#         weight=weight / (u.domain.resolution[1]**derivative),
#         kernel_size=kernel_size,
#         padding=padding,
#         strides=strides,
#     )

#     # stagger domain
#     u_domain = u.domain.stagger_y(direction="inner", stagger=True)

#     return Field(u_values, u_domain)


def calculate_kernel_size(kernel_dims, num_dims, axis):
    return (1,) * axis + (kernel_dims,) + (1,) * (num_dims - axis - 1)


def generate_forward_padding(
    derivative: int, accuracy: int
) -> Tuple[int, int]:
    return (0, derivative + accuracy - 1)


def generate_central_padding(
    derivative: int, accuracy: int
) -> Tuple[int, int]:
    left_offset = -((derivative + accuracy - 1) // 2)
    right_offset = (derivative + accuracy - 1) // 2 + 1
    return (abs(left_offset), abs(right_offset - 1))


def generate_backward_padding(
    derivative: int, accuracy: int
) -> Tuple[int, int]:
    return (abs(-(derivative + accuracy - 1)), 0)


def generate_padding(
    method: str, derivative: int, accuracy: int
) -> Tuple[int, int]:
    if method == "central":
        return generate_forward_padding(
            derivative=derivative, accuracy=accuracy
        )
    elif method == "forward":
        return generate_central_padding(
            derivative=derivative, accuracy=accuracy
        )
    elif method == "backward":
        return generate_backward_padding(
            derivative=derivative, accuracy=accuracy
        )
    else:
        raise ValueError("Unrecognized method.")


def generate_offsets(method: str, derivative: int, accuracy: int) -> List[int]:
    if method == "central":
        offsets = generate_central_offsets(
            derivative=derivative, accuracy=accuracy + 1
        )
    elif method == "backward":
        offsets = generate_backward_offsets(
            derivative=derivative, accuracy=accuracy
        )
    elif method == "forward":
        offsets = generate_forward_offsets(
            derivative=derivative, accuracy=accuracy
        )
    else:
        raise ValueError(f"Unrecognized offsets method: {method}")
    return offsets

def apply_conv(
    x: ArrayLike,
    weight: ArrayLike,
    kernel_size: Tuple[int, ...],
    padding: Tuple[int, ...],
    strides: Tuple[str, ...] | str,
) -> ArrayLike:
    """
    Applies convolution operation on the input array `x` using the given `weight` and parameters.

    Args:
        x (Array): The input array on which convolution is applied.
        weight (Array): The weight array used for convolution.
        kernel_size (Tuple[int, ...]): The size of the convolution kernel.
        padding (Tuple[int, ...]): The padding applied to the input array.
        strides (Tuple[str, ...] | str): The stride used for convolution.

    Returns:
        Array: The result of applying convolution on the input array.
    """

    @kex.kmap(padding=padding, kernel_size=kernel_size, strides=strides)
    def kernel_fn(x):
        return jnp.sum(x * weight)

    return kernel_fn(x)


def x_diff_1D(
    u: Float[ArrayLike, "... Dx"],
    step_size: ArrayLike | Float[ArrayLike, "Dx-1"] = 1.0,
    derivative: int = 1,
    method: str = "backward",
) -> Float[ArrayLike, "... Dx-1"]:
    """
    Compute the first derivative of a 1D array `u` along the x-axis.

    Parameters:
        u (Float[Array, "... Dx"]): The input array.
        step_size (float, optional): The step size between adjacent points along the x-axis. Defaults to 1.0.
        method (str, optional): The method used for differencing. Defaults to "backward".

    Returns:
        Float[Array, "... Dx-1"]: The computed first derivative array.

    Examples:
        >>> u = [1, 2, 3, 4, 5]
        >>> x_diff_1D(u)
        array([1., 1., 1., 1.])

    """
    assert derivative in [1, 2]
    return difference_1D(
        u=u,
        step_size=step_size,
        method=method,
        derivative=derivative,
        accuracy=1,
        padding="valid",
    )


# @partial(jax.jit, static_argnames=("step_size", "derivative", "method",))
def x_diff_2D(
    u: Float[ArrayLike, "... Dx Dy"],
    step_size: ArrayLike | Float[ArrayLike, "Dx-1"] = 1.0,
    derivative: int = 1,
    method: str = "backward",
) -> Float[ArrayLike, "... Dx-1 Dy"]:
    """
    Compute the second-order partial derivative of a 2D array `u` with respect to the x-axis.

    Parameters:
        u (Float[Array, "... Dx Dy"]): The input 2D array.
        step_size (float | Float[Array, "Dx-1"]): The step size for the finite difference approximation.
            Default is 1.0.
        method (str): The method used for the finite difference approximation.
            Default is "backward".

    Returns:
        Float[Array, "... Dx-1 Dy]": The computed second-order partial derivative of `u` with respect to the x-axis.

    """
    assert derivative in [1, 2]
    return difference_2D(
        u=u,
        axis=0,
        step_size=step_size,
        method=method,
        derivative=derivative,
        accuracy=1,
        padding="valid",
    )


# @partial(jax.jit, static_argnames=("step_size", "derivative", "method",))
def y_diff_2D(
    u: Float[ArrayLike, "... Dx Dy"],
    step_size: float | Float[ArrayLike, "Dy-1"] = 1.0,
    derivative: int = 1,
    method: str = "backward",
) -> Float[ArrayLike, "... Dx Dy-1"]:
    """
    Compute the second-order derivative of a 2D array `u` along the y-axis.

    Parameters:
        u (Float[Array, "... Dx Dy"]): The input array of shape (..., Dx, Dy).
        step_size (float | Float[Array, "Dy-1"], optional): The step size for the finite difference approximation.
            Defaults to 1.0.
        method (str, optional): The method used for the finite difference approximation.
            Defaults to "backward".

    Returns:
        Float[Array, "... Dx Dy-1"]: The second-order derivative of `u` along the y-axis.

    """
    assert derivative in [1, 2]
    return difference_2D(
        u=u,
        axis=1,
        step_size=step_size,
        method=method,
        derivative=derivative,
        accuracy=1,
        padding="valid",
    )


def laplacian_2D(
    u: Float[Array, "... Dx Dy"],
    step_size_x: float | Float[Array, "Dx-1"] = 1.0,
    step_size_y: float | Float[Array, "Dy-1"] = 1.0,
) -> Float[Array, "... Dx-2 Dy-2"]:
    """
    Compute the 2D Laplacian of the given scalar field.

    Parameters:
        u (Float[Array, "... Dx Dy"]): The scalar field to compute the Laplacian of.
        step_size_x (float | Float[Array, "Dx-1"]): The step size in the x-direction. Default is 1.0.
        step_size_y (float | Float[Array, "Dy-1"]): The step size in the y-direction. Default is 1.0.

    Returns:
        Float[Array, "... Dx-2 Dy-2"]: The computed 2D Laplacian of the scalar field.
    """
    d2u_dx2 = x_diff_2D(
        u=u,
        step_size=step_size_x,
        derivative=2,
    )
    d2u_dy2 = y_diff_2D(
        u=u,
        step_size=step_size_y,
        derivative=2,
    )
    return d2u_dx2[..., 1:-1] + d2u_dy2[..., 1:-1, :]


def perpendicular_gradient_2D(
    u: Float[Array, "Nx Ny"],
    step_size_x: float | Float[Array, "Dx-1"] = 1.0,
    step_size_y: float | Float[Array, "Dy-1"] = 1.0,
) -> tuple[Float[Array, "Nx Ny-1"], Float[Array, "Nx-1 Ny"]]:
    """
    Calculates the geostrophic gradient for a staggered grid.

    The geostrophic gradient is calculated using the following equations:
        u velocity = -∂yΨ
        v velocity = ∂xΨ

    Args:
        u (Array): The input variable.
            Size = [Nx,Ny]
        step_size_x (float | Array): The step size for the x-direction.
        step_size_y (float | Array): The step size for the y-direction.

    Returns:
        tuple[Float[Array, "Nx Ny-1"], Float[Array, "Nx-1 Ny"]]: A tuple containing the geostrophic velocities.
            - du_dy (Array): The geostrophic velocity in the y-direction.
                Size = [Nx,Ny-1]
            - du_dx (Array): The geostrophic velocity in the x-direction.
                Size = [Nx-1,Ny]

    Note:
        For the geostrophic velocity, we need to multiply the derivative in the x-direction by -1.
    """

    du_dy = y_diff_2D(u=u, step_size=step_size_y, derivative=1)
    du_dx = x_diff_2D(u=u, step_size=step_size_x, derivative=1)
    return -du_dy, du_dx


def divergence_2D(
    u: Float[Array, "Dx+1 Dy"],
    v: Float[Array, "Dx Dy+1"],
    step_size_x: float | Float[Array, "Dx"] = 1.0,
    step_size_y: float | Float[Array, "Dy"] = 1.0,
    method: str = "backward",
) -> Float[Array, "Dx Dy"]:
    """Calculates the divergence for a staggered grid

    This function calculates the divergence of a vector field on a staggered grid.
    The divergence represents the net flow of a vector field out of a given point.
    It is calculated as the sum of the partial derivatives of the vector field with respect to each coordinate.

    Equation:
        ∇⋅u̅  = ∂x(u) + ∂y(v)

    Args:
        u (Array): The input array for the u direction. Size = [Nx+1, Ny]
        v (Array): The input array for the v direction. Size = [Nx, Ny-1]
        step_size_x (float | Array, optional): The step size for the x-direction. Defaults to 1.0.
        step_size_y (float | Array, optional): The step size for the y-direction. Defaults to 1.0.

    Returns:
        div (Array): The divergence of the vector field. Size = [Nx-1, Ny-1]
    """
    # ∂xu
    dudx = x_diff_2D(u=u, step_size=step_size_x, derivative=1, method=method)
    # ∂yv
    dvdx = y_diff_2D(u=v, step_size=step_size_y, derivative=1, method=method)

    return dudx + dvdx


def curl_2D(
    u: Float[Array, "Dx+1 Dy"],
    v: Float[Array, "Dx Dy+1"],
    step_size_x: float | Float[Array, "Dx"] = 1.0,
    step_size_y: float | Float[Array, "Dy"] = 1.0,
    method: str = "backward",
) -> Float[Array, "Nx-1 Ny-1"]:
    """
    Calculates the curl by using finite difference in the y and x direction for the u and v velocities respectively.

    Eqn:
        ζ = ∂v/∂x - ∂u/∂y

    Args:
        u (Array): The input array for the u direction. Size = [Nx, Ny-1].
        v (Array): The input array for the v direction. Size = [Nx-1, Ny].
        step_size_x (float | Array, optional): The step size for the x-direction. Defaults to 1.0.
        step_size_y (float | Array, optional): The step size for the y-direction. Defaults to 1.0.
        method (str, optional): The method to use for finite difference calculation. Defaults to "backward".

    Returns:
        zeta (Array): The relative vorticity. Size = [Nx-1, Ny-1].
    """
    # ∂xv
    dv_dx: Float[Array, "Dx-1 Dy+1"] = x_diff_2D(u=v, step_size=step_size_x, derivative=1, method=method)
    # ∂yu
    du_dy: Float[Array, "Dx+1 Dy-1"] = y_diff_2D(u=u, step_size=step_size_y, derivative=1, method=method)

    # slice appropriate axes
    dv_dx: Float[Array, "Nx-1 Ny-1"] = dv_dx[:, 1:-1]
    du_dy: Float[Array, "Nx-1 Ny-1"] = du_dy[1:-1, :]

    return dv_dx - du_dy
