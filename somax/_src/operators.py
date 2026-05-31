"""pipekit ``Operator`` bridge for somax forward models.

somax's core stays pipekit-free — models satisfy ``pipekit_cycle``'s
``ForwardModel`` *structurally* (via ``SomaxModel.step``) without importing
pipekit. This module is the opposite end: an opt-in integration layer
(available under the ``somax[sim]`` extra, which carries pipekit) that
exposes somax models as first-class :class:`pipekit.Operator` s.

Why a companion Operator instead of making the model itself an Operator?
A somax model is an ``eqx.Module`` whose fields are grids, finitevolx
operators, and assembled term trees — none of them JSON primitives, so
the model can't round-trip through ``pipekit.serial``. The Operator here
instead carries the model's *flat primitive construction recipe*
(``nx``, ``nu``, ``dt``, …). Because every config value is a primitive,
:func:`pipekit.dumps` / :func:`pipekit.loads` round-trips the Operator
faithfully — it rebuilds the model on construction. This is the
serializable counterpart to the eqx.Module numeric core.

Each Operator is also a stepping stage: ``op(state) -> next_state``
advances the wrapped model by one ``dt``, so models compose with the
rest of pipekit (``op | op``, graphs) and drive ``pipekit_cycle.Cycle``.
The Operator additionally satisfies ``ForwardModel`` via :meth:`step`.

This module imports pipekit at module load, so it must not be imported
from ``somax``'s core import path — keep it behind ``import
somax.operators`` (or the ``somax[sim]`` extra).
"""

from __future__ import annotations

from typing import Any

from pipekit import Operator

from somax._src.models.pde2d.burgers_terms import Burgers2DTermModel


class SomaxModelOp(Operator):
    """Base pipekit Operator wrapping a somax forward model.

    Subclasses set two instance attributes in ``__init__``:

    - ``_model``: the built somax model (an ``eqx.Module`` exposing
      ``step(state, dt)``), and
    - ``dt``: the default step size used by :meth:`_apply`.

    Subclasses store their constructor kwargs as same-named attributes
    (the ConfigMixin convention) so :meth:`pipekit.ConfigMixin.get_config`
    auto-derives the flat primitive config that ``pipekit.serial``
    round-trips. ``_model`` is intentionally not a constructor parameter,
    so it is excluded from the config and rebuilt on ``loads``.

    The Operator is a one-step stage (``op(state) -> next_state``) and
    satisfies the ``pipekit_cycle.ForwardModel`` protocol via
    :meth:`step`.
    """

    _model: Any
    dt: float

    def _apply(self, state: Any) -> Any:
        """Advance ``state`` by one default ``dt`` step (pipeline stage)."""
        return self.step(state, self.dt)

    def step(self, state: Any, dt: float) -> Any:
        """Advance ``state`` by ``dt`` (``pipekit_cycle.ForwardModel``)."""
        return self._model.step(state, dt)

    @property
    def state_signature(self) -> Any:
        """No named-dimension signature — somax states are bare pytrees.

        Part of the ``pipekit_cycle.ForwardModel`` protocol (which
        requires ``step``, ``dt``, and ``state_signature``); ``None``
        means the model doesn't advertise a shape/dtype signature.
        """
        return None


class Burgers2DOp(SomaxModelOp):
    """Serializable pipekit Operator for the term-based 2D Burgers model.

    Wraps :class:`~somax._src.models.pde2d.burgers_terms.Burgers2DTermModel`.
    All constructor arguments are JSON primitives, so::

        op = Burgers2DOp(nx=32, ny=32, nu=0.05, dt=1e-3)
        assert pipekit.loads(pipekit.dumps(op)).get_config() == op.get_config()

    round-trips the build recipe. ``op(state)`` advances the Burgers
    state by one ``dt`` step; the Operator drives ``pipekit_cycle.Cycle``
    and composes with the rest of pipekit.

    Args:
        nx: Interior cells in x.
        ny: Interior cells in y.
        Lx: Domain length in x.
        Ly: Domain length in y.
        nu: Kinematic viscosity (diffusion coefficient).
        method: Advection reconstruction method.
        imex: Tag diffusion implicit for IMEX integration (see
            :meth:`Burgers2DTermModel.create`).
        dt: Default step size for :meth:`_apply`.
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        Lx: float = 2.0,
        Ly: float = 2.0,
        nu: float = 0.01,
        method: str = "upwind1",
        imex: bool = False,
        dt: float = 0.001,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.nu = nu
        self.method = method
        self.imex = imex
        self.dt = dt
        self._model = Burgers2DTermModel.create(
            nx=nx,
            ny=ny,
            Lx=Lx,
            Ly=Ly,
            nu=nu,
            method=method,
            imex=imex,
        )

    @property
    def model(self) -> Burgers2DTermModel:
        """The built term-based Burgers model."""
        return self._model


__all__ = [
    "Burgers2DOp",
    "SomaxModelOp",
]
