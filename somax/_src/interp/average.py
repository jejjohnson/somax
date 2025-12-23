from typing import (
    Callable,
    Optional,
)

import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)
import kernex as kex

from somax.domain import Domain


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
    @kex.kmap(kernel_size=kernel_size, strides=stride, padding=padding, **kwargs)
    def kernel_fn(x):
        return mean_fn(x)

    # apply kernel function
    return kernel_fn(u)


def x_avg_1D(u: Array, mean_fn: str = "arithmetic") -> Array:
    assert u.ndim == 1
    return avg_pool(u, kernel_size=(2,), stride=(1,), padding="VALID", mean_fn=mean_fn)


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
        fn = lambda x: jnp.mean(x)
        return fn
    elif mean_fn.lower() == "geometric":
        fn = lambda x: jnp.exp(jnp.mean(jnp.log(x)))
        return fn
    elif mean_fn.lower() == "harmonic":
        fn = lambda x: jnp.reciprocal(jnp.mean(jnp.reciprocal(x)))
        return fn
    elif mean_fn.lower() == "quadratic":
        fn = lambda x: jnp.sqrt(jnp.mean(jnp.square(x)))
        return fn
    else:
        msg = "Unrecognized function"
        msg += f"\n{mean_fn}"
        raise ValueError(msg)


def avg_arithmetic(x, y):
    """
    Calculates the arithmetic average of two numbers.

    Parameters:
    x (float): The first number.
    y (float): The second number.

    Returns:
    float: The arithmetic average of x and y.
    """
    return 0.5 * (x + y)


def avg_harmonic(x, y):
    """
    Calculates the harmonic average of two numbers.

    Parameters:
    x (float): The first number.
    y (float): The second number.

    Returns:
    float: The harmonic average of x and y.
    """
    x_ = jnp.reciprocal(x)
    y_ = jnp.reciprocal(y)
    return jnp.reciprocal(avg_arithmetic(x_, y_))


def avg_geometric(x, y):
    """
    Calculates the geometric average of two numbers.

    Parameters:
    - x: The first number.
    - y: The second number.

    Returns:
    The geometric average of x and y.
    """
    x_ = jnp.log(x)
    y_ = jnp.log(y)
    return jnp.exp(avg_arithmetic(x_, y_))


def avg_quadratic(x, y):
    """
    Calculates the average of two values using the quadratic mean.

    Parameters:
    - x: The first value.
    - y: The second value.

    Returns:
    The average of x and y using the quadratic mean.
    """
    x_ = jnp.square(x)
    y_ = jnp.square(y)
    return jnp.sqrt(avg_arithmetic(x_, y_))


