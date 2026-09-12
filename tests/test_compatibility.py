"""Pure-logic tests for the scenario x model compatibility checker (#76).

Tests operate against the stubbed Phase-2 registries; they don't build
models. The compatibility layer is decoupled from ``build``, which is
what lets Phase 2 ship a compatibility matrix before the models are
populated (Phases 3-5).
"""

from __future__ import annotations

import pytest

from somax._src.cli._compatibility import (
    INCOMPATIBLE_PAIRS,
    IncompatiblePairError,
    check_compatible,
    compatibility_matrix,
    is_compatible,
)
from somax._src.cli.models_registry import list_models
from somax._src.cli.scenarios import list_scenarios


# ---------------------------------------------------------------------------
# Geometry x coordinates rule
# ---------------------------------------------------------------------------


class TestRectangularScenario:
    """``double_gyre`` is unmasked rectangular → every Cartesian model."""

    @pytest.mark.parametrize(
        "model",
        [
            "linear_swm",
            "nonlinear_swm",
            "barotropic_qg",
            "multilayer_nonlinear_swm",
            "multilayer_qg",
            "reparam_multilayer_qg",
        ],
    )
    def test_double_gyre_accepts_cartesian_models(self, model):
        assert is_compatible("double_gyre", model)

    @pytest.mark.parametrize("model", ["spherical_swm", "spherical_qg"])
    def test_double_gyre_rejects_spherical_models(self, model):
        assert not is_compatible("double_gyre", model)
        with pytest.raises(IncompatiblePairError, match="coordinates"):
            check_compatible("double_gyre", model)


class TestRealBasinScenario:
    """Masked Cartesian basins need a Cartesian mask-aware model."""

    @pytest.mark.parametrize("scenario", ["north_atlantic", "med_sea", "gulf_stream"])
    def test_linear_swm_rejected_because_no_masks(self, scenario):
        # linear_swm is cartesian but supports.masks=False.
        assert not is_compatible(scenario, "linear_swm")
        with pytest.raises(IncompatiblePairError, match="mask"):
            check_compatible(scenario, "linear_swm")

    @pytest.mark.parametrize("scenario", ["north_atlantic", "med_sea", "gulf_stream"])
    @pytest.mark.parametrize(
        "model",
        [
            "nonlinear_swm",
            "barotropic_qg",
            "multilayer_nonlinear_swm",
            "multilayer_qg",
            "reparam_multilayer_qg",
        ],
    )
    def test_mask_aware_cartesian_models_accepted(self, scenario, model):
        assert is_compatible(scenario, model)

    @pytest.mark.parametrize("scenario", ["north_atlantic", "med_sea", "gulf_stream"])
    @pytest.mark.parametrize("model", ["spherical_swm", "spherical_qg"])
    def test_spherical_models_rejected_by_geometry(self, scenario, model):
        assert not is_compatible(scenario, model)
        with pytest.raises(IncompatiblePairError, match="coordinates"):
            check_compatible(scenario, model)


class TestSphericalCapScenario:
    """Masked spherical scenarios → spherical mask-aware models only."""

    @pytest.mark.parametrize("model", ["spherical_swm", "spherical_qg"])
    def test_spherical_models_accepted(self, model):
        assert is_compatible("southern_ocean", model)

    @pytest.mark.parametrize("model", ["spherical_swm", "spherical_qg"])
    def test_the_full_sphere_scenario_is_rejected(self, model):
        """``global_ocean`` reaches both poles, which the models refuse.

        Their polar rows carry a solid-wall condition — right for a
        latitude band, wrong at a true pole — so the pairing cannot be
        built and should not be advertised.
        """
        assert not is_compatible("global_ocean", model)

    @pytest.mark.parametrize("scenario", ["southern_ocean", "global_ocean"])
    @pytest.mark.parametrize(
        "model",
        [
            "linear_swm",
            "nonlinear_swm",
            "barotropic_qg",
            "multilayer_nonlinear_swm",
            "multilayer_qg",
            "reparam_multilayer_qg",
        ],
    )
    def test_cartesian_models_rejected_by_geometry(self, scenario, model):
        assert not is_compatible(scenario, model)


# ---------------------------------------------------------------------------
# Compatibility matrix shape matches the epic (#72)
# ---------------------------------------------------------------------------


class TestCompatibilityMatrix:
    def test_matrix_covers_all_pairs(self):
        matrix = compatibility_matrix()
        scenarios = list_scenarios()
        models = list_models()
        assert set(matrix) == set(scenarios)
        for s, row in matrix.items():
            assert set(row) == set(models), (
                f"compatibility_matrix row {s!r} is missing models"
            )

    def test_expected_pair_counts(self):
        """Sanity-check the matrix against the scenario/model taxonomy.

        - ``double_gyre`` (unmasked rectangular): 6 Cartesian ✓, 2 spherical ✗
        - 3 real-basin Cartesian x 5 mask-aware Cartesian = 15 ✓
          (``linear_swm`` excluded because it doesn't support masks)
        - ``southern_ocean`` (spherical cap): 2 spherical ✓ = 2
        - ``global_ocean`` (full sphere): 0 — both spherical models
          reject a latitude range that reaches a pole, so the pairs are
          explicit overrides in ``INCOMPATIBLE_PAIRS``
        Total ✓ pairs = 6 + 15 + 2 = 23.
        """
        matrix = compatibility_matrix()
        true_count = sum(v for row in matrix.values() for v in row.values())
        assert true_count == 23, (
            f"compatibility matrix has {true_count} compatible pairs; "
            "expected 23 (6 + 15 + 2) from the scenario/model taxonomy"
        )


# ---------------------------------------------------------------------------
# Explicit-override set shape
# ---------------------------------------------------------------------------


class TestIncompatiblePairs:
    def test_is_a_set(self):
        assert isinstance(INCOMPATIBLE_PAIRS, set)

    def test_holds_only_the_full_sphere_overrides(self):
        """The one case the geometry x coordinates rule cannot see.

        ``global_ocean`` and the spherical models are both "spherical",
        so the inferred answer is compatible — but the models reject a
        latitude range that reaches a pole, because their polar rows
        carry a solid-wall condition rather than spherical regularity.
        These come out again when pole coupling lands.
        """
        assert {
            ("global_ocean", "spherical_swm"),
            ("global_ocean", "spherical_qg"),
        } == INCOMPATIBLE_PAIRS

    def test_the_reason_is_surfaced(self):
        from somax._src.cli._compatibility import (
            IncompatiblePairError,
            check_compatible,
        )

        with pytest.raises(IncompatiblePairError, match="INCOMPATIBLE_PAIRS"):
            check_compatible("global_ocean", "spherical_swm")

    def test_the_spherical_cap_scenario_is_unaffected(self):
        assert is_compatible("southern_ocean", "spherical_swm")


# ---------------------------------------------------------------------------
# Unknown names surface a helpful error
# ---------------------------------------------------------------------------


class TestUnknownNames:
    def test_unknown_scenario_raises_keyerror(self):
        with pytest.raises(KeyError):
            check_compatible("no_such_scenario", "linear_swm")

    def test_unknown_model_raises_keyerror(self):
        with pytest.raises(KeyError):
            check_compatible("double_gyre", "no_such_model")
