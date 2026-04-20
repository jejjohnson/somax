"""Scenario x model compatibility checks.

Per decision 3 of epic #72: compatibility is **inferred** from
``Geometry.kind`` x model ``coordinates`` plus ``SupportFlags.masks``,
with explicit overrides for nonsense pairs held in
:data:`INCOMPATIBLE_PAIRS`.

The checker operates on registry entries rather than on a built
:class:`ScenarioBundle`, so a stubbed entry (Phase-2 state) is enough
for compatibility dispatch to work.
"""

from __future__ import annotations

from somax._src.cli.models_registry import ModelEntry, get_model
from somax._src.cli.scenarios import ScenarioEntry, get_scenario


# ---------------------------------------------------------------------------
# Explicit incompatibility overrides
# ---------------------------------------------------------------------------
#
# Each entry is ``(scenario_name, model_name)``. Use this set only when
# the inferred compatibility (geometry x coordinates + masks) gives the
# wrong answer for a genuinely nonsensical combination — prefer fixing
# the metadata on the registry entry first.
INCOMPATIBLE_PAIRS: set[tuple[str, str]] = set()


class IncompatiblePairError(ValueError):
    """Raised by :func:`check_compatible` when a pair is rejected.

    The message explains the reason so the CLI can surface a
    user-readable error — for example, a masked scenario paired with
    a model whose ``SupportFlags.masks`` is ``False``.
    """


def _geometry_matches(scenario: ScenarioEntry, model: ModelEntry) -> bool:
    """Geometry x coordinates rule.

    - ``rectangular``  → ``cartesian`` model
    - ``real_basin``   → ``cartesian`` model (mask support checked separately)
    - ``spherical_cap``→ ``spherical`` model
    """
    if scenario.geometry_kind == "rectangular":
        return model.coordinates == "cartesian"
    if scenario.geometry_kind == "real_basin":
        return model.coordinates == "cartesian"
    if scenario.geometry_kind == "spherical_cap":
        return model.coordinates == "spherical"
    return False


def _requires_masks(scenario: ScenarioEntry) -> bool:
    """Which geometries carry a land/coastline mask the model must handle."""
    return scenario.geometry_kind in ("real_basin", "spherical_cap")


def _reason_for_rejection(scenario: ScenarioEntry, model: ModelEntry) -> str | None:
    """Return ``None`` if the pair is compatible, else an explanation string."""
    if (scenario.name, model.name) in INCOMPATIBLE_PAIRS:
        return (
            f"{scenario.name!r} x {model.name!r} is explicitly marked "
            "incompatible in INCOMPATIBLE_PAIRS."
        )
    if not _geometry_matches(scenario, model):
        return (
            f"geometry kind {scenario.geometry_kind!r} is incompatible "
            f"with model coordinates {model.coordinates!r}."
        )
    if _requires_masks(scenario) and not model.supports.masks:
        return (
            f"scenario {scenario.name!r} has a land mask but model "
            f"{model.name!r} does not set SupportFlags.masks=True."
        )
    return None


def is_compatible(scenario_name: str, model_name: str) -> bool:
    """Return ``True`` iff the pair is accepted by the compatibility rules."""
    return (
        _reason_for_rejection(get_scenario(scenario_name), get_model(model_name))
        is None
    )


def check_compatible(scenario_name: str, model_name: str) -> None:
    """Raise :class:`IncompatiblePairError` if the pair is rejected.

    Args:
        scenario_name: Key in the scenarios registry.
        model_name: Key in the models registry.

    Raises:
        KeyError: If either name is not registered.
        IncompatiblePairError: If the pair fails the geometry / mask /
            explicit-override checks.
    """
    scenario = get_scenario(scenario_name)
    model = get_model(model_name)
    reason = _reason_for_rejection(scenario, model)
    if reason is not None:
        raise IncompatiblePairError(
            f"scenario {scenario_name!r} and model {model_name!r} are "
            f"incompatible: {reason}"
        )


def compatibility_matrix() -> dict[str, dict[str, bool]]:
    """Compute the full compatibility matrix over registered pairs.

    Returns:
        Nested dict ``{scenario_name: {model_name: bool}}`` with one
        entry per registered scenario x model. Useful for rendering
        the ``list-pairs`` subcommand.
    """
    from somax._src.cli.models_registry import list_models as _list_models
    from somax._src.cli.scenarios import list_scenarios as _list_scenarios

    return {
        s: {m: is_compatible(s, m) for m in _list_models()} for s in _list_scenarios()
    }
