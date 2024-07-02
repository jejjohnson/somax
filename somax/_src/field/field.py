import typing as tp

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)

from somax.domain import Domain

from .utils import check_discretization

# TODO: add power operation
# TODO: add negative operation


class Field(eqx.Module):
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
        msg = "Incorrect dimentions\n"
        msg += f"Values: {values.shape} | Grid: {domain.grid_axis[0].shape}"
        assert values.shape == domain.grid_axis[0].shape, msg

        self.values = values
        self.domain = domain

    @classmethod
    def init_from_fn(cls, domain: Domain, fn: tp.Callable, *args, **kwargs):
        # vectorize coordinate values

        values = fn(*args, **kwargs)

        return cls(values=values, domain=domain)

    # @classmethod
    # def init_from_array(cls, values: Array):
    #     return cls(values=values, domain=Domain.from_array(u=values))

    @classmethod
    def init_from_ones(cls, domain: Domain):
        return cls(values=jnp.ones(domain.Nx), domain=domain)

    def replace_values(self, values):
        return Field(values=values, domain=self.domain)

    @property
    def shape(self) -> tp.Tuple[int]:
        return self.values.shape

    def binop(self, other, fn: tp.Callable):
        # check discretization
        check_discretization(self.domain, other.domain)
        values = fn(self.values, other.values)
        return Field(values, self.domain)

    def __getitem__(self, idx):
        values = self.values[idx]
        domain = self.domain[idx]
        return Field(values=values, domain=domain)

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self.binop(other, lambda x, y: y + x)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self.binop(other, lambda x, y: y * x)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)

    # def __bool__(self):
    #     return eqx.tree_at(lambda x: x.values, self, self.values.__bool__())

    def __neg__(self):
        return eqx.tree_at(lambda x: x.values, self, -self.values)

    def __pow__(self, power):
        return eqx.tree_at(lambda x: x.values, self, self.values**power)

    def __rpow__(self, power):
        return eqx.tree_at(lambda x: x.values, self, power**self.values)

    def __truediv__(self, other):
        return self.binop(other, lambda x, y: x / y)

    def __rtruediv__(self, other):
        return self.binop(other, lambda x, y: y / x)

    def inverse(self):
        return eqx.tree_at(lambda x: x.values, self, 1 / self.values)

    def min(self):
        return self.values.min()

    def max(self):
        return self.values.max()
