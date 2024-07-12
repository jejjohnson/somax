from abc import (
    ABC,
)
from typing import (
    Callable,
    Tuple,
)

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import (
    Array,
    ArrayLike,
    Float,
)
from plum import dispatch

from somax._src.domain.cartesian import Domain

from .utils import check_discretization


class Field(eqx.Module, ABC):
    """A Field for a discrete domain

    Attributes:
        values (Array): An arbitrary sized array
        domain (Domain): the domain for the array
    """

    values: Float
    domain: Domain

    def __init__(self, values: Array, domain: Domain):
        """
        Args:
            values (Array): An arbitrary sized array
            domain (Domain): the domain for the array
        """
        values = jnp.atleast_1d(values)
        msg = "Incorrect dimensions\n"
        msg += f"Values: {values.shape} | Grid: {domain.shape}"
        assert values.shape == domain.shape, msg

        self.values = values
        self.domain = domain

    @classmethod
    def init_from_fn(cls, domain: Domain, fn: Callable, *args, **kwargs):
        """
        Initialize a field by evaluating a function over the given domain.

        Parameters:
        - domain (Domain): The domain over which the function is evaluated.
        - fn (Callable): The function to be evaluated.
        - *args, **kwargs: Additional arguments to be passed to the function.

        Returns:
        - cls: An instance of the field class with the evaluated values.

        """
        # vectorize coordinate values

        values = fn(*args, **kwargs)

        return cls(values=values, domain=domain)

    @classmethod
    def init_from_ones(cls, domain: Domain):
        """
        Initialize a field with all values set to 1.

        Parameters:
        - domain (Domain): The domain of the field.

        Returns:
        - Field: The initialized field.
        """
        return cls(values=jnp.ones(domain.shape), domain=domain)

    def replace_values(self, values):
        """
        Replaces the values of the field with the given values.

        Args:
            values (numpy.ndarray): The new values to replace the field values with.

        Returns:
            Field: The field object with the updated values.
        """
        assert values.shape == self.shape
        return eqx.tree_at(lambda x: x.values, self, values)

    def replace_domain(self, domain):
        """
        Replaces the domain of the field with the given domain.

        Parameters:
        - domain: The new domain to replace with. It should have the same shape as the current domain.

        Returns:
        - The field with the updated domain.
        """
        assert self.shape == domain.shape
        return eqx.tree_at(lambda x: x.domain, self, domain)

    @property
    def shape(self) -> Tuple[int]:
        return self.values.shape

    def isel_x(self, sel):
        values = self.values[..., sel, :]
        domain = self.domain.isel_x(sel)
        return Field(values=values, domain=domain)

    def isel_y(self, sel):
        values = self.values[..., sel]
        domain = self.domain.isel_y(sel)
        return Field(values=values, domain=domain)

    def pad_x(self, pad_width, **kwargs):
        return pad_x_field(self, pad_width, **kwargs)

    def pad_y(self, pad_width, **kwargs):
        return pad_y_field(self, pad_width, **kwargs)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return radd(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return rmul(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return rsub(self, other)

    def __truediv__(self, other):
        return truediv(self, other)

    def __rtruediv__(self, other):
        return rtruediv(self, other)

    def __neg__(self):
        return neg(self)

    def __pow__(self, other):
        return pow(self, other)

    def __rpow__(self, other):
        return rpow(self, other)

    def min(self):
        return self.values.min()

    def max(self):
        return self.values.max()


def field_to_field_op(x: Field, y: Field, fn: Callable):
    """
    Perform a field operation on two fields.

    Args:
        x (Field): The first field.
        y (Field): The second field.
        fn (Callable): The function to apply to the field values.

    Returns:
        Field: The resulting field after applying the function to the values.
    """
    # check discretization
    check_discretization(x.domain, y.domain)
    # apply function values
    values = fn(x.values, y.values)
    # initialize field with values (of the same field)
    x = eqx.tree_at(lambda x: x.values, x, values)
    return x


def field_to_array_op(x: Field, y: ArrayLike, fn: Callable):
    """
    Perform a field operation on two fields.

    Args:
        x (Field): The first field.
        y (Field): The second field.
        fn (Callable): The function to apply to the field values.

    Returns:
        Field: The resulting field after applying the function to the values.
    """
    # initialize field with values (of the same field)
    x = eqx.tree_at(lambda x: x.values, x, fn(x.values, y))
    return x


# ADDITION


@dispatch
def add(x: Field, y: Field):
    return field_to_field_op(x, y, lambda x, y: x + y)


@dispatch
def add(x: Field, y: ArrayLike):
    return field_to_array_op(x, y, lambda x, y: x + y)


@dispatch
def radd(x: Field, y: Field):
    return field_to_field_op(x, y, lambda x, y: y + x)


@dispatch
def radd(x: Field, y: ArrayLike):
    return field_to_array_op(x, y, lambda x, y: y + x)


# SUBTRACTION
@dispatch
def sub(x: Field, y: Field):
    return field_to_field_op(x, y, lambda x, y: x - y)


@dispatch
def sub(x: Field, y: ArrayLike):
    return field_to_array_op(x, y, lambda x, y: x - y)


@dispatch
def rsub(x: Field, y: Field):
    return field_to_field_op(x, y, lambda x, y: y - x)


@dispatch
def rsub(x: Field, y: ArrayLike):
    return field_to_array_op(x, y, lambda x, y: y - x)


# MULTIPLICATION
@dispatch
def mul(x: Field, y: Field):
    return field_to_field_op(x, y, lambda x, y: x * y)


@dispatch
def mul(x: Field, y: ArrayLike):
    return field_to_array_op(x, y, lambda x, y: x * y)


@dispatch
def rmul(x: Field, y: Field):
    return field_to_field_op(x, y, lambda x, y: y * x)


@dispatch
def rmul(x: Field, y: ArrayLike):
    return field_to_array_op(x, y, lambda x, y: y * x)


# DIVISTION
@dispatch
def truediv(x: Field, y: Field):
    return field_to_field_op(x, y, lambda x, y: x / y)


@dispatch
def truediv(x: Field, y: ArrayLike):
    return field_to_array_op(x, y, lambda x, y: x / y)


@dispatch
def rtruediv(x: Field, y: Field):
    return field_to_field_op(x, y, lambda x, y: y / x)


@dispatch
def rtruediv(x: Field, y: ArrayLike):
    return field_to_array_op(x, y, lambda x, y: y / x)


# POWER
@dispatch
def pow(x: Field, y: Field):
    return field_to_field_op(x, y, lambda x, y: x**y)


@dispatch
def pow(x: Field, y: ArrayLike):
    return field_to_array_op(x, y, lambda x, y: x**y)


@dispatch
def rpow(x: Field, y: Field):
    return field_to_field_op(x, y, lambda x, y: y**x)


@dispatch
def rpow(x: Field, y: ArrayLike):
    return field_to_array_op(x, y, lambda x, y: y**x)


# NEGATION
def neg(x: Field):
    return eqx.tree_at(lambda x: x.values, x, -x.values)


def pad_x_field(u: Field, pad_width, **kwargs) -> Field:
    """
    Pads the x-axis of a field with the specified pad width.

    Args:
        u (Field): The input field to be padded.
        pad_width (int or sequence of ints): The pad width to be applied to the x-axis.
        **kwargs: Additional keyword arguments to be passed to the `jnp.pad` function.

    Returns:
        Field: The padded field.

    """
    # pad domain
    domain = u.domain.pad_x(pad_width=pad_width)

    # increase pad-width for extra dimensions
    num_dims = len(u.shape)
    pad_width = ((0, 0),) * (num_dims - 2) + (pad_width, (0, 0))

    # pad values
    u_values = jnp.pad(u.values, pad_width=pad_width, **kwargs)

    u = eqx.tree_at(lambda x: x.values, u, u_values)
    u = eqx.tree_at(lambda x: x.domain, u, domain)

    return u


def pad_y_field(u: Field, pad_width, **kwargs) -> Field:
    """
    Pads the y-axis of a field with the specified pad width.

    Args:
        u (Field): The input field to be padded.
        pad_width: The pad width to be applied to the y-axis.
        **kwargs: Additional keyword arguments to be passed to `jnp.pad`.

    Returns:
        Field: The padded field.

    """
    # pad domain
    domain = u.domain.pad_y(pad_width=pad_width)

    # increase pad-width for extra dimensions
    num_dims = len(u.shape)
    pad_width = ((0, 0),) * (num_dims - 1) + (pad_width,)

    # pad values
    u_values = jnp.pad(u.values, pad_width=pad_width, **kwargs)

    u = eqx.tree_at(lambda x: x.values, u, u_values)
    u = eqx.tree_at(lambda x: x.domain, u, domain)

    return u
