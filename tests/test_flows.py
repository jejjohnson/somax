"""Tests for the optional flowjax bridge.

Skipped entirely when the ``flows`` extra is not installed, which is
itself part of the contract: importing somax must not require flowjax.
"""

from __future__ import annotations

import importlib.util

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import paramax
import pytest
from jax.flatten_util import ravel_pytree

from somax._src.core.scales import Scales
from somax._src.core.transforms import StateAffine
from somax._src.da.flatten import state_to_vector
from somax._src.models.swm.nonlinear_2d import NonlinearSW2DState


HAS_FLOWJAX = importlib.util.find_spec("flowjax") is not None

pytestmark = pytest.mark.skipif(
    not HAS_FLOWJAX, reason="requires the optional 'flows' extra (flowjax)"
)

NY, NX = 4, 5


@pytest.fixture
def scales():
    return Scales.advective(L=1e6, U=0.1, f0=1e-4, H=1000.0)


@pytest.fixture
def state(scales):
    rng = np.random.RandomState(0)
    return NonlinearSW2DState(
        h=jnp.asarray(scales.H + rng.randn(NY, NX)),
        u=jnp.asarray(0.1 * rng.randn(NY, NX)),
        v=jnp.asarray(0.1 * rng.randn(NY, NX)),
    )


@pytest.fixture
def transform(scales):
    return StateAffine.from_scales(NonlinearSW2DState, scales)


@pytest.fixture
def bijection(transform, state):
    from somax.flows import to_flowjax

    return to_flowjax(transform, state)


class TestAdapter:
    def test_shape_is_the_flat_state_size(self, bijection, state):
        flat, _ = ravel_pytree(state)
        assert bijection.shape == (flat.size,)

    def test_is_unconditional(self, bijection):
        assert bijection.cond_shape is None

    def test_forward_matches_the_pytree_transform(self, bijection, transform, state):
        flat, _ = ravel_pytree(state)
        got, _ = bijection.transform_and_log_det(flat)
        expected, _ = ravel_pytree(transform.forward(state))
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_log_det_matches_the_pytree_transform(self, bijection, transform, state):
        flat, _ = ravel_pytree(state)
        _, got = bijection.transform_and_log_det(flat)
        assert float(got) == pytest.approx(float(transform.log_det(state)), rel=1e-6)

    def test_round_trip(self, bijection, state):
        flat, _ = ravel_pytree(state)
        forward, _ = bijection.transform_and_log_det(flat)
        back, _ = bijection.inverse_and_log_det(forward)
        np.testing.assert_allclose(back, flat, rtol=1e-5)

    def test_inverse_log_det_negates_forward(self, bijection, state):
        flat, _ = ravel_pytree(state)
        _, forward = bijection.transform_and_log_det(flat)
        _, inverse = bijection.inverse_and_log_det(flat)
        assert float(forward) == pytest.approx(-float(inverse), rel=1e-6)

    def test_layout_matches_the_da_bridge(self, bijection, transform, state):
        """One vector must feed both a DA filter and a flow."""
        from_da, _ = state_to_vector(state, transform)
        flat, _ = ravel_pytree(state)
        from_flow, _ = bijection.transform_and_log_det(flat)
        np.testing.assert_allclose(from_flow, from_da, rtol=1e-6)

    def test_is_a_flowjax_bijection(self, bijection):
        from flowjax.bijections import AbstractBijection

        assert isinstance(bijection, AbstractBijection)

    def test_flowjax_convenience_methods_work(self, bijection, transform, state):
        """``transform`` / ``inverse`` come from the base class."""
        flat, _ = ravel_pytree(state)
        expected, _ = ravel_pytree(transform.forward(state))
        np.testing.assert_allclose(bijection.transform(flat), expected, rtol=1e-6)

    def test_usable_under_jit(self, bijection, state):
        flat, _ = ravel_pytree(state)
        jitted = jax.jit(bijection.transform_and_log_det)
        got, _ = jitted(flat)
        expected, _ = bijection.transform_and_log_det(flat)
        np.testing.assert_allclose(got, expected, rtol=1e-6)


class TestChaining:
    def test_chaining_with_an_identity_flow_is_the_affine_alone(
        self, bijection, transform, state
    ):
        from flowjax.bijections import Chain, Identity

        flat, _ = ravel_pytree(state)
        chained = Chain([bijection, Identity(bijection.shape)])
        got, got_ld = chained.transform_and_log_det(flat)
        expected, expected_ld = bijection.transform_and_log_det(flat)
        np.testing.assert_allclose(got, expected, rtol=1e-6)
        assert float(got_ld) == pytest.approx(float(expected_ld), rel=1e-6)

    def test_chaining_accumulates_log_dets(self, bijection, state):
        from flowjax.bijections import Affine, Chain

        flat, _ = ravel_pytree(state)
        doubling = Affine(
            loc=jnp.zeros(bijection.shape), scale=jnp.full(bijection.shape, 2.0)
        )
        chained = Chain([bijection, doubling])
        _, chained_ld = chained.transform_and_log_det(flat)
        _, affine_ld = bijection.transform_and_log_det(flat)
        _, doubling_ld = doubling.transform_and_log_det(flat)
        assert float(chained_ld) == pytest.approx(
            float(affine_ld) + float(doubling_ld), rel=1e-5
        )


class TestLearnedPrior:
    def test_density_accounts_for_the_affine_jacobian(self, transform, state):
        """log p_X(x) = log p_Y(phi(x)) + log|det J|."""
        from flowjax.bijections import Identity
        from flowjax.distributions import Normal

        from somax.flows import learned_prior, to_flowjax

        flat, _ = ravel_pytree(state)
        base = Normal(jnp.zeros(flat.size))
        prior = learned_prior(transform, state, Identity((flat.size,)), base)

        bijection = to_flowjax(transform, state)
        transformed, log_det = bijection.transform_and_log_det(flat)
        expected = float(base.log_prob(transformed)) + float(log_det)
        assert float(prior.log_prob(flat)) == pytest.approx(expected, rel=1e-4)

    def test_prior_is_a_flowjax_distribution(self, transform, state):
        from flowjax.bijections import Identity
        from flowjax.distributions import AbstractDistribution, Normal

        from somax.flows import learned_prior

        flat, _ = ravel_pytree(state)
        prior = learned_prior(
            transform, state, Identity((flat.size,)), Normal(jnp.zeros(flat.size))
        )
        assert isinstance(prior, AbstractDistribution)

    def test_sampling_lands_in_physical_magnitudes(self, transform, state):
        """A standard-normal base maps back to states of the right size."""
        from flowjax.bijections import Identity
        from flowjax.distributions import Normal

        from somax.flows import learned_prior

        flat, unravel = ravel_pytree(state)
        prior = learned_prior(
            transform, state, Identity((flat.size,)), Normal(jnp.zeros(flat.size))
        )
        # The prior is over *physical* states already: the bijection
        # maps the base through the flow and then back out of
        # standardised coordinates, so a draw only needs unravelling.
        draws = prior.sample(jax.random.key(0), (1024,))
        physical = jax.vmap(unravel)(draws)
        # h is centred on H with a spread of order the geostrophic scale.
        assert float(jnp.mean(physical.h)) == pytest.approx(transform.loc.h, rel=0.2)
        assert float(jnp.std(physical.u)) == pytest.approx(
            float(transform.scale.u), rel=0.2
        )


class TestPackaging:
    def test_importing_somax_does_not_require_flowjax(self):
        """The core import path must stay free of the optional dep."""
        import subprocess
        import sys

        code = (
            "import sys\n"
            "sys.modules['flowjax'] = None\n"
            "import somax\n"
            "assert 'somax' in sys.modules\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_public_module_exports_the_bridge(self):
        import somax.flows as flows

        assert set(flows.__all__) == {
            "StateAffineBijection",
            "learned_prior",
            "to_flowjax",
        }

    def test_flows_extra_is_declared(self):
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        config = tomllib.loads(pyproject.read_text())
        extras = config["project"]["optional-dependencies"]
        assert any("flowjax" in dep for dep in extras["flows"])

    def test_flowjax_is_not_a_core_dependency(self):
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        config = tomllib.loads(pyproject.read_text())
        assert not any("flowjax" in dep for dep in config["project"]["dependencies"])


class TestTheAffineIsNotTrainable:
    """The scaling must survive training the flow on top of it.

    ``loc`` and ``scale`` are arrays whenever they come from
    ``from_samples`` or an array-valued override. Left as ordinary
    fields they are trainable leaves, so fitting the ``Transformed``
    distribution would drift the physical normalisation along with the
    learned flow — silently changing what the prior means.
    """

    def bijection(self):
        from somax._src.core.flows import to_flowjax

        rng = np.random.RandomState(0)
        shape = (4, 4)
        samples = NonlinearSW2DState(
            h=jnp.asarray(100.0 + rng.randn(6, *shape)),
            u=jnp.asarray(rng.randn(6, *shape)),
            v=jnp.asarray(rng.randn(6, *shape)),
        )
        transform = StateAffine.from_samples(samples, per_gridpoint=True)
        example = jax.tree_util.tree_map(lambda leaf: leaf[0], samples)
        return to_flowjax(transform, example)

    def test_the_statistics_really_are_arrays(self):
        """Otherwise there would be nothing for an optimiser to move."""
        bijection = self.bijection()
        loc = paramax.unwrap(bijection.state_transform).loc
        assert jnp.ndim(loc.h) > 0

    def test_every_statistic_leaf_is_wrapped_non_trainable(self):
        """``paramax.non_trainable`` wraps the leaves, not the module."""
        leaves = jax.tree_util.tree_leaves(
            self.bijection().state_transform,
            is_leaf=lambda x: isinstance(x, paramax.NonTrainable),
        )
        assert leaves
        assert all(isinstance(leaf, paramax.NonTrainable) for leaf in leaves)

    def test_its_gradients_are_exactly_zero(self):
        bijection = self.bijection()

        def loss(bij):
            bij = paramax.unwrap(bij)
            return jnp.sum(bij.transform_and_log_det(jnp.ones(bij.shape))[0] ** 2)

        grads = eqx.filter_grad(loss)(bijection)
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
        assert leaves
        assert all(float(jnp.abs(leaf).max()) == 0.0 for leaf in leaves)

    def test_the_map_still_works(self):
        """Wrapping must not change what the bijection computes."""
        bijection = self.bijection()
        x = jnp.linspace(-1.0, 1.0, bijection.shape[0])
        y, _ = bijection.transform_and_log_det(x)
        back, _ = bijection.inverse_and_log_det(y)
        np.testing.assert_allclose(np.asarray(back), np.asarray(x), atol=1e-4)

    def test_the_log_determinant_is_unchanged(self):
        bijection = self.bijection()
        x = jnp.zeros(bijection.shape)
        _, forward = bijection.transform_and_log_det(x)
        y, _ = bijection.transform_and_log_det(x)
        _, inverse = bijection.inverse_and_log_det(y)
        assert float(forward) == pytest.approx(-float(inverse), rel=1e-5)


class TestOptionalExtraIsExercisedInCi:
    """These tests must actually run somewhere.

    The module-level marker skips every one of them when flowjax is
    absent, so a CI job that does not install the extra would stay
    green while the bridge rots.
    """

    def workflows(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / ".github" / "workflows"
        return [root / "ci.yml", root / "full-tests.yml"]

    def test_the_workflows_exist(self):
        for path in self.workflows():
            assert path.exists(), path

    def test_every_test_job_installs_the_flows_extra(self):
        for path in self.workflows():
            text = path.read_text()
            for line in text.splitlines():
                if "uv sync" in line and "--group dev" in line:
                    assert "--extra flows" in line, f"{path.name}: {line.strip()}"
