"""``OperatorTerm`` — use any pipekit ``Operator`` / ``Graph`` as a :class:`Term`.

A :class:`~somax._src.core.terms.Term` maps ``(t, state, args) ->
tendency`` with ``tendency`` shaped like ``state``. A pipekit operator maps
fields to fields. ``OperatorTerm`` is the adapter::

    state = {h, u, v}                 (a State module or a dict)
        │  select `inputs`            (fields, by name)
        ▼
    op(*fields)  or  graph(**fields)  (pipekit Operator / Graph / JaxModelOp)
        │  result: array | tuple | dict
        ▼
    tendency = 0 everywhere, the op's result(s) written into `outputs`

so a learned closure or a finitevolX Tier-B graph is a summand of an
ordinary term tree::

    rhs = gravity + coriolis + OperatorTerm(op=closure, inputs=("u", "v"),
                                            outputs=("u", "v"))

The operator is a field of the term, so when it is a pytree
(``pipekit_jax.JaxOperator``, ``JaxModelOp``, or a ``Graph`` /
``Sequential`` of them — ``pipekit_jax`` registers those containers) its
array leaves are leaves of the term tree and ``eqx.filter_grad`` reaches
them together with the physics coefficients. A plain ``pipekit.Operator``
still works, but it is opaque to ``jax.grad``: its internals are not
leaves, so its parameters are constants.

This lives on the pipekit side of somax (``somax.operators``); the term
algebra in :mod:`somax._src.core.terms` stays pipekit-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pipekit_jax  # noqa: F401  (registers Graph / Sequential as pytrees)
from jaxtyping import PyTree
from pipekit import Graph

from somax._src.core.terms import EXPLICIT, IMPLICIT, Kind, Term


def _get(state: Any, name: str) -> Any:
    """Field ``name`` of a State module (attribute) or a dict (key)."""
    if isinstance(state, Mapping):
        return state[name]
    return getattr(state, name)


def _write(tendency: Any, values: Mapping[str, Any]) -> Any:
    """``tendency`` with the named fields replaced by ``values``."""
    if isinstance(tendency, Mapping):
        return {**tendency, **values}
    names = tuple(values)
    return eqx.tree_at(
        lambda s: tuple(getattr(s, n) for n in names),
        tendency,
        tuple(values[n] for n in names),
    )


class OperatorTerm(Term):
    """Lift a pipekit operator into the :class:`Term` algebra.

    ``tendency = zeros_like(state)`` with ``op``'s result(s) written into
    ``outputs``; fields not listed get zero tendency.

    Calling convention: a pipekit ``Graph`` is called with keywords
    (``graph(**{name: field})``, so ``inputs`` must name its inputs); any
    other operator positionally (``op(*fields)``, in ``inputs`` order).

    Result → fields:

    - an array: written to the single entry of ``outputs``;
    - a tuple / list: written to ``outputs`` in order (lengths must match);
    - a dict (e.g. a ``Graph``): written by key; ``outputs`` may be
      ``None`` (every key is a state field) or a subset to keep.

    Args:
        op: The operator (``JaxOperator``, ``Graph``, ``JaxModelOp``, …).
        inputs: State field names fed to ``op``.
        outputs: State field names the result is written to (see above).
        kind_: IMEX label (``"explicit"`` or ``"implicit"``), so
            :func:`~somax._src.core.terms.partition` routes the term.
    """

    op: Any
    inputs: tuple[str, ...] = eqx.field(static=True, converter=tuple)
    outputs: tuple[str, ...] | None = eqx.field(
        static=True,
        default=None,
        converter=lambda x: None if x is None else tuple(x),
    )
    kind_: Kind = eqx.field(static=True, default=EXPLICIT)

    def __check_init__(self) -> None:
        if self.kind_ not in (EXPLICIT, IMPLICIT):
            raise ValueError(
                f"kind_ must be {EXPLICIT!r} or {IMPLICIT!r}, got {self.kind_!r}"
            )
        if not self.inputs:
            raise ValueError("OperatorTerm needs at least one input field")

    @property
    def kind(self) -> Kind:
        """IMEX label of this summand."""
        return self.kind_

    def __call__(self, t: float, state: PyTree, args: PyTree | None = None) -> PyTree:
        fields = [_get(state, name) for name in self.inputs]
        if isinstance(self.op, Graph):
            result = self.op(**dict(zip(self.inputs, fields, strict=True)))
        else:
            result = self.op(*fields)
        return _write(jax.tree.map(jnp.zeros_like, state), self._by_name(result))

    def _by_name(self, result: Any) -> dict[str, Any]:
        """Map the operator's result to ``{state field: value}``."""
        if isinstance(result, Mapping):
            keys = tuple(result) if self.outputs is None else self.outputs
            missing = [k for k in keys if k not in result]
            if missing:
                raise ValueError(
                    f"OperatorTerm outputs {missing} are not keys of the "
                    f"operator's result {sorted(result)}"
                )
            return {k: result[k] for k in keys}
        if self.outputs is None:
            raise ValueError(
                "OperatorTerm needs `outputs` unless the operator returns a dict"
            )
        values = tuple(result) if isinstance(result, tuple | list) else (result,)
        if len(values) != len(self.outputs):
            raise ValueError(
                f"OperatorTerm has outputs {self.outputs} but the operator "
                f"returned {len(values)} value(s)"
            )
        return dict(zip(self.outputs, values, strict=True))
