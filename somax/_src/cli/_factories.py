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
) -> tuple[Any, Any]:
    """Build ``(model, state0)`` from a scenario x model pair.

    Args:
        scenario_name: Key in the scenarios registry.
        model_name: Key in the models registry.
        scenario_params: YAML ``scenario:`` side knobs (grid, consts,
            forcing, initial_condition). Defaults to ``{}``.
        model_params: YAML ``model:`` side knobs (stratification,
            differentiable params). Defaults to ``{}``.

    Returns:
        ``(model, state0)`` — the pytree model and its initial state.

    Raises:
        KeyError: Unknown scenario or model name.
        IncompatiblePairError: Pair rejected by the compatibility rules.
        NotImplementedError: Registered but stubbed (Phase 4/5) entry.
    """
    from somax._src.cli._compatibility import check_compatible
    from somax._src.cli.models_registry import get_model
    from somax._src.cli.scenarios import get_scenario

    check_compatible(scenario_name, model_name)
    scenario_entry = get_scenario(scenario_name)
    model_entry = get_model(model_name)
    bundle = scenario_entry.build(scenario_params or {})
    built = model_entry.build(bundle, model_params or {})
    return built.model, built.state0
