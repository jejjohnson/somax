from typing import Tuple, Callable
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from somax._src.field.base import Field
from somax._src.operators.difference import calculate_kernel_size, FDParams, apply_conv


class DifferenceLayer(eqx.Module):
    weight: Array
    axis: int
    kernel_size: Tuple[int, ...]
    strides: Tuple[int, ...]
    padding: Tuple[str, ...] | str

    @classmethod
    def init_from_fd_params(
        cls, fd_params, num_dims: int, axis: int = 0, padding: str = "valid"
    ):
        return cls(
            num_dims=num_dims,
            derivative=fd_params.derivative,
            accuracy=fd_params.accuracy,
            method=fd_params.method,
            padding=padding,
            axis=axis,
        )

    def __init__(
        self,
        num_dims: int,
        derivative: int = 1,
        accuracy: int = 1,
        method: str = "backward",
        padding: str = "valid",
        axis: int = 0,
    ):
        # initialize kernel parameters
        fd_params = FDParams(method, derivative, accuracy)

        self.kernel_size = tuple(
            calculate_kernel_size(fd_params.coeffs_dim, num_dims, axis)
        )

        # initialize strides
        self.strides = (1,) * num_dims

        # initialize weight
        self.weight = jnp.array(fd_params.coeffs).reshape(self.kernel_size)
        self.axis = axis
        self.padding = padding

    def __call__(self, x: Array, dx: ArrayLike = 1.0) -> Array:
        return apply_conv(
            x=x,
            weight=self.weight / dx,
            kernel_size=self.kernel_size,
            padding=self.padding,
            strides=self.strides,
        )


class FuncOperator(eqx.Module):
    fn: Callable

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, u: Field, *args, **kwargs) -> Field:
        # operate on values
        u_values = self.fn(u.values, *args, **kwargs)

        # replace values
        u = u.replace_values(u_values)

        return u


