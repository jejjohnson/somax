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

    @pytest.mark.parametrize("scenario", ["southern_ocean", "global_ocean"])
    @pytest.mark.parametrize("model", ["spherical_swm", "spherical_qg"])
    def test_spherical_models_accepted(self, scenario, model):
        assert is_compatible(scenario, model)

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
        - ``southern_ocean`` + ``global_ocean`` (spherical-cap):
          2 x 2 spherical ✓ = 4
        Total ✓ pairs = 6 + 15 + 4 = 25.
        """
        matrix = compatibility_matrix()
        true_count = sum(v for row in matrix.values() for v in row.values())
        assert true_count == 25, (
            f"compatibility matrix has {true_count} compatible pairs; "
            "expected 25 (6 + 15 + 4) from the scenario/model taxonomy"
        )


# ---------------------------------------------------------------------------
# Explicit-override set shape
# ---------------------------------------------------------------------------


class TestIncompatiblePairs:
    def test_is_a_set(self):
        assert isinstance(INCOMPATIBLE_PAIRS, set)

    def test_empty_by_default(self):
        """Phase 2 doesn't need any explicit overrides yet."""
        assert set() == INCOMPATIBLE_PAIRS


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
