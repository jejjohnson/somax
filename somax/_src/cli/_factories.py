"""Scenario x model dispatcher for somax-sim.

Phase 3 (#77) completed the cutover away from the legacy per-testcase
adapter registry. The sole public entry point is :func:`build`, which:

1. Validates that the ``(scenario, model)`` pair is compatible
   (see :mod:`somax._src.cli._compatibility`).
2. Looks up the registered
   :class:`somax._src.cli.scenarios.ScenarioEntry` and builds the
   :class:`~somax._src.cli.scenarios.ScenarioBundle`.
3. Looks up the registered
   :class:`somax._src.cli.models_registry.ModelEntry` and calls its
   ``build`` with the bundle + model-side params.
4. Returns the flat ``(model, state0)`` tuple the runner expects.

The two-block YAML schema (``scenario:`` + ``model:``) funnels through
this dispatcher; the legacy single-block ``testcase:`` schema was
removed in Phase 3.
"""

from __future__ import annotations

from typing import Any


def build(
    scenario_name: str,
    model_name: str,
    *,
    scenario_params: dict[str, Any] | None = None,
    model_params: dict[str, Any] | None = None,
    nondim: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Build ``(model, state0)`` from a scenario x model pair.

    Args:
        scenario_name: Key in the scenarios registry.
        model_name: Key in the models registry.
        scenario_params: YAML ``scenario:`` side knobs (grid, consts,
            forcing, initial_condition). Defaults to ``{}``.
        model_params: YAML ``model:`` side knobs (stratification,
            differentiable params). Defaults to ``{}``.
        nondim: YAML ``scenario.nondim`` block. When non-empty the
            model is built from dimensionless numbers through its
            ``from_nondimensional`` entry instead of from SI constants.

    Returns:
        ``(model, state0)`` — the pytree model and its initial state.

    Raises:
        KeyError: Unknown scenario or model name.
        IncompatiblePairError: Pair rejected by the compatibility rules.
        NotImplementedError: Registered but stubbed (Phase 4/5) entry.
        ValueError: A ``nondim`` block was given for a model that has no
            nondimensional factory.
    """
    from somax._src.cli._compatibility import check_compatible
    from somax._src.cli.models_registry import get_model
    from somax._src.cli.scenarios import get_scenario

    check_compatible(scenario_name, model_name)
    scenario_entry = get_scenario(scenario_name)
    model_entry = get_model(model_name)
    bundle = scenario_entry.build(scenario_params or {})
    if nondim:
        if model_entry.from_nondimensional is None:
            raise ValueError(
                f"model {model_name!r} has no nondimensional factory, so it "
                "cannot be built from a scenario.nondim block. Remove the "
                "block and give scenario.consts instead."
            )
        _reject_dimensional_overrides(model_name, model_params, bundle)
        built = model_entry.from_nondimensional(bundle, nondim)
    else:
        built = model_entry.build(bundle, model_params or {})
    return built.model, built.state0


#: ``model.params`` keys a nondimensional build derives for itself, and
#: ``scenario.forcing`` keys it would otherwise discard.
_DIMENSIONAL_PARAM_KEYS = frozenset(
    {"lateral_viscosity", "bottom_drag", "wind_amplitude", "nu", "kappa"}
)
_DIMENSIONAL_FORCING_KEYS = frozenset({"wind_amplitude", "tau0"})


def _reject_dimensional_overrides(
    model_name: str,
    model_params: dict[str, Any] | None,
    bundle: Any,
) -> None:
    """Refuse a config that mixes dimensionless and dimensional knobs.

    The nondimensional path never forwards ``model_params``, and the
    adapters read only ``wind_profile`` from the forcing block — so a
    config could carry ``scenario.nondim`` alongside
    ``model.params.lateral_viscosity`` or
    ``scenario.forcing.wind_amplitude``, pass validation, and then run
    with coefficients that are not the ones it asks for. Since the
    dimensionless numbers *derive* those coefficients, the two cannot
    both be honoured; saying so beats silently picking one.

    Args:
        model_name: For the error message.
        model_params: The ``model`` block, if any.
        bundle: The built scenario, whose ``forcing_params`` is checked.

    Raises:
        ValueError: If a conflicting dimensional key is present.
    """
    clashes = []
    params = dict((model_params or {}).get("params", {}))
    clashes += [
        f"model.params.{key}"
        for key in sorted(params)
        if key in _DIMENSIONAL_PARAM_KEYS
    ]
    forcing = dict(getattr(bundle, "forcing_params", {}) or {})
    clashes += [
        f"scenario.forcing.{key}"
        for key in sorted(forcing)
        if key in _DIMENSIONAL_FORCING_KEYS
    ]
    if clashes:
        raise ValueError(
            f"{model_name}: {clashes} cannot be combined with a "
            f"scenario.nondim block — the dimensionless numbers derive those "
            f"coefficients, so the run would not use the values given here. "
            f"Drop them, or drop scenario.nondim and give scenario.consts."
        )
