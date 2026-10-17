"""``OperatorTerm``: pipekit operators as summands of a term tree."""

from __future__ import annotations

from collections.abc import Iterator

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from finitevolx import Coriolis2D, Difference2D
from pipekit import Graph, Input, Lambda
from pipekit_jax import JaxOperator

from somax._src.core.terms import IMPLICIT, partition
from somax._src.models.swm.linear_2d import LinearSW2DState
from somax._src.models.swm.linear_swm_terms import (
    LinearSWM2DTermModel,
    LinearSWMCoriolis,
    LinearSWMDrag,
    LinearSWMGravityWave,
)
from somax.core import Scaled
from somax.operators import OperatorTerm


# ----------------------------------------------------------------------
# Fixture operators (finitevolX Tier A wrapped as pipekit JaxOperators)
# ----------------------------------------------------------------------


class GravityWaveOp(JaxOperator):
    """(h, u, v) ↦ {h: -H₀ ∇·(u, v), u: -g δₓh, v: -g δᵧh}; g, H₀ learnable."""

    diff: Difference2D
    g: jax.Array = eqx.field(converter=jnp.asarray)
    H0: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, h, u, v):
        return {
            "h": -self.H0 * self.diff.divergence(u, v),
            "u": -self.g * self.diff.diff_x_T_to_U(h),
            "v": -self.g * self.diff.diff_y_T_to_V(h),
        }


class CoriolisOp(JaxOperator):
    """(u, v) ↦ (+f v̄, -f ū)."""

    coriolis: Coriolis2D
    f: jax.Array

    def _apply(self, u, v):
        return self.coriolis(u, v, self.f)


class Drag(JaxOperator):
    """(u, v) ↦ (-a u, -a v); ``a`` learnable."""

    a: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, u, v):
        return -self.a * u, -self.a * v


class Neg(JaxOperator):
    """x ↦ -c x."""

    c: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, x):
        return -self.c * x


class Div(JaxOperator):
    diff: Difference2D

    def _apply(self, u, v):
        return self.diff.divergence(u, v)


class GradX(JaxOperator):
    diff: Difference2D

    def _apply(self, h):
        return self.diff.diff_x_T_to_U(h)


class GradY(JaxOperator):
    diff: Difference2D

    def _apply(self, h):
        return self.diff.diff_y_T_to_V(h)


@pytest.fixture
def x64() -> Iterator[None]:
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", False)


def _model(kappa: float = 0.0) -> LinearSWM2DTermModel:
    return LinearSWM2DTermModel.create(
        nx=16,
        ny=16,
        Lx=1e6,
        Ly=1e6,
        f0=1e-4,
        beta=2e-11,
        H0=100.0,
        lateral_viscosity=0.0,
        bottom_drag=kappa,
    )


def _state(model, seed: int = 0) -> LinearSW2DState:
    g = model.grid
    ks = jax.random.split(jax.random.key(seed), 3)
    h, u, v = (0.1 * jax.random.normal(k, (g.Ny, g.Nx)) for k in ks)
    return LinearSW2DState(h=h, u=u, v=v)


def _summands(model):
    terms = model.terms.terms
    gravity = next(t for t in terms if isinstance(t, LinearSWMGravityWave))
    coriolis = next(t for t in terms if isinstance(t, LinearSWMCoriolis))
    return gravity, coriolis


def _assert_state_equal(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


# ----------------------------------------------------------------------
# Equivalence with hand-written kernels
# ----------------------------------------------------------------------


def test_dict_valued_operator_matches_the_gravity_wave_kernel():
    model = _model()
    gravity, _ = _summands(model)
    op = GravityWaveOp(diff=gravity.diff, g=gravity.g, H0=gravity.H0)
    term = OperatorTerm(op=op, inputs=("h", "u", "v"))
    s = _state(model)
    _assert_state_equal(term(0.0, s), gravity(0.0, s))


def test_tuple_valued_operator_matches_the_coriolis_kernel():
    model = _model()
    _, coriolis = _summands(model)
    op = CoriolisOp(coriolis=coriolis.coriolis, f=coriolis.f_field)
    term = OperatorTerm(op=op, inputs=("u", "v"), outputs=("u", "v"))
    s = _state(model)
    _assert_state_equal(term(0.0, s), coriolis(0.0, s))


def test_a_pipekit_graph_is_called_by_input_name():
    model = _model()
    gravity, _ = _summands(model)
    h, u, v = Input("h"), Input("u"), Input("v")
    graph = Graph(
        inputs={"h": h, "u": u, "v": v},
        outputs={
            "h": Neg(c=gravity.H0)(Div(diff=gravity.diff)(u, v)),
            "u": Neg(c=gravity.g)(GradX(diff=gravity.diff)(h)),
            "v": Neg(c=gravity.g)(GradY(diff=gravity.diff)(h)),
        },
    )
    term = OperatorTerm(op=graph, inputs=("h", "u", "v"))
    s = _state(model)
    _assert_state_equal(term(0.0, s), gravity(0.0, s))


def test_swapping_physics_for_operator_terms_keeps_the_model_tendency():
    model = _model(kappa=1e-5)
    gravity, coriolis = _summands(model)
    rhs = (
        OperatorTerm(
            op=GravityWaveOp(diff=gravity.diff, g=gravity.g, H0=gravity.H0),
            inputs=("h", "u", "v"),
        )
        + OperatorTerm(
            op=CoriolisOp(coriolis=coriolis.coriolis, f=coriolis.f_field),
            inputs=("u", "v"),
            outputs=("u", "v"),
        )
        + OperatorTerm(op=Drag(a=1e-5), inputs=("u", "v"), outputs=("u", "v"))
    )
    hybrid = LinearSWM2DTermModel(terms=rhs, grid=model.grid, bc_type=model.bc_type)
    s = model.apply_boundary_conditions(_state(model))
    want = model.vector_field(0.0, s)
    got = hybrid.vector_field(0.0, s)
    for x, y in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-12)


def test_dict_states_are_supported():
    term = OperatorTerm(op=Neg(c=2.0), inputs=("a",), outputs=("b",))
    out = term(0.0, {"a": jnp.ones(3), "b": jnp.full(3, 7.0), "c": jnp.ones(2)})
    np.testing.assert_array_equal(out["a"], 0.0)
    np.testing.assert_array_equal(out["b"], -2.0)
    np.testing.assert_array_equal(out["c"], 0.0)


# ----------------------------------------------------------------------
# IMEX routing
# ----------------------------------------------------------------------


def test_partition_and_diffrax_terms_respect_kind():
    model = _model()
    gravity, _ = _summands(model)
    stiff = OperatorTerm(
        op=Drag(a=1.0), inputs=("u", "v"), outputs=("u", "v"), kind_=IMPLICIT
    )
    explicit_part, implicit_part = partition(gravity + stiff)
    assert implicit_part is stiff
    assert explicit_part is gravity
    hybrid = LinearSWM2DTermModel(terms=gravity + stiff, grid=model.grid)
    assert isinstance(hybrid.build_terms(), dfx.MultiTerm)


def test_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="kind_"):
        OperatorTerm(op=Neg(c=1.0), inputs=("u",), outputs=("u",), kind_="stiff")


# ----------------------------------------------------------------------
# Gradients
# ----------------------------------------------------------------------


def test_gradient_reaches_a_leaf_inside_the_operator(x64):
    model = _model()
    gravity, _ = _summands(model)
    s = _state(model)

    def loss(term):
        return jnp.sum(term(0.0, s).u ** 2)

    def term_with(g):
        op = GravityWaveOp(diff=gravity.diff, g=g, H0=gravity.H0)
        return OperatorTerm(op=op, inputs=("h", "u", "v"))

    grad = eqx.filter_grad(loss)(term_with(9.81)).op.g
    eps = 1e-4
    fd = (loss(term_with(9.81 + eps)) - loss(term_with(9.81 - eps))) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-6)


def test_gradient_flows_through_a_model_step(x64):
    model = _model()
    gravity, coriolis = _summands(model)

    def hybrid(a):
        closure = OperatorTerm(op=Drag(a=a), inputs=("u", "v"), outputs=("u", "v"))
        return LinearSWM2DTermModel(terms=gravity + coriolis + closure, grid=model.grid)

    s0 = _state(model)

    def loss(m):
        return jnp.sum(m.step(s0, 600.0).u ** 2)

    grad = eqx.filter_grad(loss)(hybrid(1e-4))
    a_grad = grad.terms.terms[2].op.a
    eps = 1e-7
    fd = (loss(hybrid(1e-4 + eps)) - loss(hybrid(1e-4 - eps))) / (2 * eps)
    np.testing.assert_allclose(a_grad, fd, rtol=1e-5)
    # the closure is the same physics as the built-in drag kernel
    ref = LinearSWM2DTermModel(
        terms=gravity + coriolis + Scaled(LinearSWMDrag(), -1e-4), grid=model.grid
    )
    _assert_close(hybrid(1e-4).step(s0, 600.0), ref.step(s0, 600.0))


def _assert_close(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-15)


def test_a_plain_operator_works_but_is_opaque_to_grad():
    c = jnp.asarray(3.0)
    plain = Lambda(lambda x: -c * x)
    term = OperatorTerm(op=plain, inputs=("u",), outputs=("u",))
    out = term(0.0, {"u": jnp.ones(4)})
    np.testing.assert_array_equal(out["u"], -3.0)
    leaves = [x for x in jax.tree.leaves(term) if eqx.is_inexact_array(x)]
    assert leaves == []  # nothing for filter_grad to reach


# ----------------------------------------------------------------------
# Output mapping errors
# ----------------------------------------------------------------------


def test_output_mapping_errors_are_explicit():
    s = {"u": jnp.ones(3), "v": jnp.ones(3)}
    with pytest.raises(ValueError, match="needs `outputs`"):
        OperatorTerm(op=Neg(c=1.0), inputs=("u",))(0.0, s)
    with pytest.raises(ValueError, match="returned 2 value"):
        OperatorTerm(op=Drag(a=1.0), inputs=("u", "v"), outputs=("u",))(0.0, s)
    with pytest.raises(ValueError, match="not keys"):
        OperatorTerm(op=Lambda(lambda u: {"w": u}), inputs=("u",), outputs=("u",))(
            0.0, s
        )


def test_a_finitevolx_tier_b_graph_matches_gravity_plus_coriolis():
    """The epic's target: finitevolx.pipekit ops wired as a Graph, one Term."""
    import finitevolx.pipekit as fp

    model = _model()
    gravity, coriolis = _summands(model)
    G = fp.as_grid_spec(model.grid)
    g, H0 = float(gravity.g), float(gravity.H0)
    h, u, v = Input("h"), Input("u"), Input("v")
    cor = CoriolisOp(coriolis=coriolis.coriolis, f=coriolis.f_field)(u, v)
    gx = fp.Difference(grid=G, axis="x", src="T", dst="U")(h)
    gy = fp.Difference(grid=G, axis="y", src="T", dst="V")(h)
    graph = Graph(
        inputs={"h": h, "u": u, "v": v},
        outputs={
            "h": fp.LinComb(coeffs=(-H0,))(fp.Divergence(grid=G)(u, v)),
            "u": fp.LinComb(coeffs=(-g, 1.0))(gx, fp.Pick(index=0)(cor)),
            "v": fp.LinComb(coeffs=(-g, 1.0))(gy, fp.Pick(index=1)(cor)),
        },
    )
    term = OperatorTerm(op=graph, inputs=("h", "u", "v"))
    s = _state(model)
    want = (gravity + coriolis)(0.0, s)
    for x, y in zip(jax.tree.leaves(term(0.0, s)), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-12)
