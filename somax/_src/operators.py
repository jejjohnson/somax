"""pipekit ``Operator`` bridge for somax forward models.

somax's core stays pipekit-free — models satisfy ``pipekit_cycle``'s
``ForwardModel`` *structurally* (via ``SomaxModel.step``) without importing
pipekit. This module is the opposite end: an opt-in integration layer
(available under the ``somax[sim]`` extra, which carries pipekit) that
exposes somax models as first-class :class:`pipekit.Operator` s.

Why a companion Operator instead of making the model itself an Operator?
A somax model is an ``eqx.Module`` whose fields are grids, finitevolx
operators, and assembled term trees — none of them JSON primitives, so
the model can't round-trip through ``pipekit.serial``. Two forms bridge
the gap:

- :class:`SomaxModelOp` wraps *any* built model
  (``SomaxModelOp(model, dt)`` or :meth:`SomaxModelOp.from_registry`).
  It is a stepping stage / ForwardModel / Cycle driver, but — holding a
  non-primitive ``eqx.Module`` — is **not** serial-round-trippable
  (``forbid_in_yaml = True``).
- Flat-recipe subclasses (e.g. :class:`Burgers2DOp`) instead carry the
  model's *flat primitive construction recipe* (``nx``, ``nu``, ``dt``,
  …). Because every config value is a primitive,
  :func:`pipekit.dumps` / :func:`pipekit.loads` round-trips them
  faithfully — rebuilding the model on construction. This is the
  serializable counterpart to the eqx.Module numeric core.

Every Operator is a stepping stage: ``op(state) -> next_state`` advances
the wrapped model by one ``dt``, so models compose with the rest of
pipekit (``op | op``, graphs) and drive ``pipekit_cycle.Cycle``. All
satisfy ``ForwardModel`` via :meth:`step` / ``dt`` / ``state_signature``.

This module imports pipekit at module load, so it must not be imported
from ``somax``'s core import path — keep it behind ``import
somax.operators`` (or the ``somax[sim]`` extra).
"""

from __future__ import annotations

from typing import Any

from pipekit import Operator

from somax._src.models.pde2d.burgers_terms import Burgers2DTermModel


class SomaxModelOp(Operator):
    """A pipekit Operator wrapping *any* built somax forward model.

    Construct directly from a built model (``SomaxModelOp(model, dt)``)
    or from a scenario x model pair via :meth:`from_registry`. The
    Operator is a one-step stage (``op(state) -> next_state`` advances by
    ``dt``) so models compose with the rest of pipekit (``op | op``,
    graphs) and drive ``pipekit_cycle.Cycle``; it also satisfies the
    ``pipekit_cycle.ForwardModel`` protocol (``step`` / ``dt`` /
    ``state_signature``).

    Serialization: the wrapped model is an ``eqx.Module`` (grids,
    finitevolx operators, term trees — not JSON primitives), so this
    general form is **not** faithfully round-trippable through
    ``pipekit.serial`` — it sets ``forbid_in_yaml = True`` and an empty
    auto-config. Subclasses whose construction is a *flat primitive
    recipe* (e.g. :class:`Burgers2DOp`) re-enable the round-trip by
    rebuilding the model from those primitives.

    Args:
        model: A built somax model exposing ``step(state, dt)``.
        dt: Default step size used by :meth:`_apply`.
    """

    forbid_in_yaml = True
    # The general wrapper holds a non-primitive eqx.Module, so there's no
    # faithful auto-config. Flat-recipe subclasses set this back to True.
    __config_mixin_auto__ = False

    def __init__(self, model: Any, dt: float) -> None:
        self._model = model
        self.dt = dt

    @classmethod
    def from_registry(
        cls,
        scenario_name: str,
        model_name: str,
        *,
        dt: float,
        scenario_params: dict[str, Any] | None = None,
        model_params: dict[str, Any] | None = None,
    ) -> tuple[SomaxModelOp, Any]:
        """Build an Operator for a scenario x model pair via the dispatcher.

        Wraps :func:`somax._src.cli._factories.build`, so any compatible
        registered pair becomes a Cycle-driveable ForwardModel Operator.

        Args:
            scenario_name: Key in the scenarios registry.
            model_name: Key in the models registry.
            dt: Default step size for the Operator.
            scenario_params: YAML ``scenario:`` knobs (grid, consts, …).
            model_params: YAML ``model:`` knobs (params, stratification).

        Returns:
            ``(op, state0)`` — the wrapping Operator and the registry's
            initial state, ready to drive ``pipekit_cycle.Cycle``.
        """
        from somax._src.cli._factories import build

        model, state0 = build(
            scenario_name,
            model_name,
            scenario_params=scenario_params,
            model_params=model_params,
        )
        return cls(model, dt), state0

    def _apply(self, state: Any) -> Any:
        """Advance ``state`` by one default ``dt`` step (pipeline stage)."""
        return self.step(state, self.dt)

    def step(self, state: Any, dt: float) -> Any:
        """Advance ``state`` by ``dt`` (``pipekit_cycle.ForwardModel``)."""
        return self._model.step(state, dt)

    @property
    def model(self) -> Any:
        """The wrapped, built somax model."""
        return self._model

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

    # Flat primitive recipe → faithful serial round-trip (unlike the
    # general SomaxModelOp wrapper).
    forbid_in_yaml = False
    __config_mixin_auto__ = True

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


__all__ = [
    "Burgers2DOp",
    "SomaxModelOp",
]
