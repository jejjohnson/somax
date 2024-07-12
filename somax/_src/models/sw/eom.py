from typing import NamedTuple
import equinox as eqx
from jaxtyping import Array


class State(eqx.Module):
    h: Array
    u: Array
    v: Array


def enforce_boundaries(u, grid: str = "u"):

    return u


def vector_field():

    return None
