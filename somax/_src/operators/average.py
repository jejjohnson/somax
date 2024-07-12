from typing import (
    Callable,
    Optional,
)

import jax.numpy as jnp
from jaxtyping import (
    Array,
)
import kernex as kex


def avg_pool(
    u: Array,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: Optional = None,
    mean_fn: str = "arithmetic",
    **kwargs,
) -> Array:
    # get mean function
    mean_fn = get_mean_function(mean_fn=mean_fn)

    # create mean kernel
    @kex.kmap(
        kernel_size=kernel_size, strides=stride, padding=padding, **kwargs
    )
    def kernel_fn(x):
        return mean_fn(x)

    # apply kernel function
    return kernel_fn(u)


def x_avg_1D(u: Array, mean_fn: str = "arithmetic") -> Array:
    assert u.ndim == 1
    return avg_pool(
        u, kernel_size=(2,), stride=(1,), padding="VALID", mean_fn=mean_fn
    )


def x_avg_2D(u: Array, mean_fn: str = "arithmetic") -> Array:
    assert u.ndim == 2
    return avg_pool(
        u, kernel_size=(2, 1), stride=(1, 1), padding="VALID", mean_fn=mean_fn
    )


def y_avg_2D(u: Array, mean_fn: str = "arithmetic") -> Array:
    assert u.ndim == 2
    return avg_pool(
        u, kernel_size=(1, 2), stride=(1, 1), padding="VALID", mean_fn=mean_fn
    )


def center_avg_2D(u: Array, mean_fn: str = "arithmetic") -> Array:
    assert u.ndim == 2
    return avg_pool(
        u, kernel_size=(2, 2), stride=(1, 1), padding="VALID", mean_fn=mean_fn
    )


def get_mean_function(mean_fn: str = "arithmetic") -> Callable:
    if mean_fn.lower() == "arithmetic":
        fn = avg_arithmetic
        return fn
    elif mean_fn.lower() == "geometric":
        fn = avg_geometric
        return fn
    elif mean_fn.lower() == "harmonic":
        fn = avg_harmonic
        return fn
    elif mean_fn.lower() == "quadratic":
        fn = avg_quadratic
        return fn
    else:
        msg = "Unrecognized function"
        msg += f"\n{mean_fn}"
        raise ValueError(msg)


def avg_arithmetic(x: Array) -> Array:
    """
    Calculates the arithmetic average of an Array.

    Parameters:
    x [Array]: A list of numbers.

    Returns:
    float: The arithmetic average of the numbers in the list.
    """
    return jnp.mean(x)


def avg_harmonic(x: Array) -> Array:
    """
    Calculates the harmonic average of a number.

    Parameters:
    x (Array): The number.

    Returns:
    float: The harmonic average of x.
    """
    return jnp.reciprocal(avg_arithmetic(jnp.reciprocal(x)))


def avg_geometric(x: Array) -> Array:
    """
    Calculates the geometric average of an array of numbers.

    Parameters:
    - x: An array of numbers.

    Returns:
    The geometric average of the numbers in the array.
    """
    return jnp.exp(avg_arithmetic(jnp.log(x)))


def avg_quadratic(x: Array) -> Array:
    """
    Calculates the average of two values using the quadratic mean.

    Parameters:
    - x: The value to calculate the average for.

    Returns:
    The average of x using the quadratic mean.
    """
    return jnp.sqrt(avg_arithmetic(jnp.square(x)))
