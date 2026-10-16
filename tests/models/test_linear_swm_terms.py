"""Term-based LinearSWM2D vs the canonical LinearShallowWater2D.

The second term-kernel worked example (after Burgers): a four-term
decomposition — gravity-wave + Coriolis + diffusion + drag. These tests
are the evidence the decomposition is faithful (identical tendencies and
explicit trajectory) and that the richer term set still supports IMEX.
"""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from somax._src.core.model import SomaxModel, TermModel
from somax._src.models.swm.linear_2d import LinearShallowWater2D, LinearSW2DState
from somax._src.models.swm.linear_swm_terms import (
    LinearSWM2DTermModel,
    LinearSWMCoriolis,
    LinearSWMDiffusion,
    LinearSWMDrag,
    LinearSWMGravityWave,
)


def _gaussian_bump(model) -> LinearSW2DState:
    grid = model.grid
    x = jnp.arange(grid.Nx) * grid.dx
    y = jnp.arange(grid.Ny) * grid.dy
    X, Y = jnp.meshgrid(x, y)
    cx = grid.Nx * grid.dx / 2
    cy = grid.Ny * grid.dy / 2
    sigma = grid.Nx * grid.dx / 8
    h = jnp.exp(-0.5 * (((X - cx) / sigma) ** 2 + ((Y - cy) / sigma) ** 2))
    return LinearSW2DState(h=h, u=jnp.zeros_like(h), v=jnp.zeros_like(h))


def _matched_models(*, nu=50.0, kappa=1e-6, imex=False):
    base = LinearShallowWater2D.create(
        nx=16,
        ny=16,
        Lx=1e6,
        Ly=1e6,
        f0=1e-4,
        beta=2e-11,
        H0=100.0,
        lateral_viscosity=nu,
        bottom_drag=kappa,
        bc="periodic",
    )
    term = LinearSWM2DTermModel.from_model(base, imex=imex)
    return base, term


# ----------------------------------------------------------------------
# Type / conformance
# ----------------------------------------------------------------------


def test_is_term_model_and_somax_model():
    model = LinearSWM2DTermModel.create(nx=8, ny=8)
    assert isinstance(model, TermModel)
    assert isinstance(model, SomaxModel)


def test_four_kernels_present():
    _base, term = _matched_models()
    summands = term.terms.terms
    flat = []
    for s in summands:
        flat.append(s)
        if hasattr(s, "term"):  # Scaled wrappers
            flat.append(s.term)
    kinds = {type(t).__name__ for t in flat}
    for cls in (
        LinearSWMGravityWave,
        LinearSWMCoriolis,
        LinearSWMDiffusion,
        LinearSWMDrag,
    ):
        assert cls.__name__ in kinds


def test_nu_and_kappa_read_back():
    _base, term = _matched_models(nu=50.0, kappa=1e-6)
    assert jnp.allclose(term.nu, 50.0)
    assert jnp.allclose(term.kappa, 1e-6)


# ----------------------------------------------------------------------
# Faithfulness to the canonical model
# ----------------------------------------------------------------------


def test_tendency_matches_monolithic():
    base, term = _matched_models()
    state = _gaussian_bump(base)
    bc = base.apply_boundary_conditions(state)

    mono = base.vector_field(0.0, bc)
    got = term.terms(0.0, bc)

    assert jnp.allclose(mono.h, got.h, atol=1e-10)
    assert jnp.allclose(mono.u, got.u, atol=1e-10)
    assert jnp.allclose(mono.v, got.v, atol=1e-10)


def test_explicit_trajectory_matches_monolithic():
    base, term = _matched_models()
    state0 = _gaussian_bump(base)

    mono_sol = base.integrate(state0, t0=0.0, t1=600.0, dt=30.0)
    term_sol = term.integrate(state0, t0=0.0, t1=600.0, dt=30.0)

    assert jnp.allclose(mono_sol.ys.h, term_sol.ys.h, atol=1e-6)
    assert jnp.allclose(mono_sol.ys.u, term_sol.ys.u, atol=1e-6)


def test_step_conforms_to_forward_model():
    model = LinearSWM2DTermModel.create(nx=8, ny=8, lateral_viscosity=50.0)
    state0 = _gaussian_bump(model)
    stepped = model.step(state0, dt=30.0)
    assert stepped.h.shape == state0.h.shape
    assert jnp.all(jnp.isfinite(stepped.h))


def test_wall_bc_matches_monolithic():
    base = LinearShallowWater2D.create(nx=12, ny=12, bc="wall", lateral_viscosity=50.0)
    term = LinearSWM2DTermModel.from_model(base)
    state = _gaussian_bump(base)
    assert jnp.allclose(
        base.apply_boundary_conditions(state).u,
        term.apply_boundary_conditions(state).u,
        atol=1e-12,
    )


# ----------------------------------------------------------------------
# IMEX: the four-term split
# ----------------------------------------------------------------------


def test_diffrax_terms_explicit_is_single_odeterm():
    model = LinearSWM2DTermModel.create(nx=8, ny=8, lateral_viscosity=50.0, imex=False)
    assert isinstance(model.build_terms(), dfx.ODETerm)


def test_diffrax_terms_imex_is_multiterm():
    model = LinearSWM2DTermModel.create(nx=8, ny=8, lateral_viscosity=50.0, imex=True)
    assert isinstance(model.build_terms(), dfx.MultiTerm)


@pytest.mark.slow
def test_imex_integration_matches_explicit():
    explicit_model = LinearSWM2DTermModel.create(
        nx=12, ny=12, lateral_viscosity=200.0, imex=False
    )
    imex_model = LinearSWM2DTermModel.create(
        nx=12, ny=12, lateral_viscosity=200.0, imex=True
    )
    state0 = _gaussian_bump(explicit_model)

    explicit_sol = explicit_model.integrate(state0, t0=0.0, t1=300.0, dt=30.0)
    imex_sol = imex_model.integrate(
        state0,
        t0=0.0,
        t1=300.0,
        dt=30.0,
        solver=dfx.KenCarp3(),
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-7),
        max_steps=10_000,
    )
    assert jnp.allclose(explicit_sol.ys.h, imex_sol.ys.h, atol=1e-2)


# ----------------------------------------------------------------------
# Differentiability of the viscosity coefficient
# ----------------------------------------------------------------------


def test_grad_through_nu():
    model = LinearSWM2DTermModel.create(nx=12, ny=12, lateral_viscosity=100.0)
    state0 = _gaussian_bump(model)

    @eqx.filter_grad
    def grad_fn(m):
        sol = m.integrate(state0, t0=0.0, t1=120.0, dt=30.0)
        return jnp.sum(sol.ys.h**2 + sol.ys.u**2 + sol.ys.v**2)

    grads = grad_fn(model)
    assert jnp.isfinite(grads.nu)
    assert grads.nu != 0.0


# ----------------------------------------------------------------------
# Differentiable physical constants (g, H0, f)
# ----------------------------------------------------------------------


@pytest.fixture
def x64():
    import jax

    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", False)


def _with(model, *, g=None, H0=None, f_scale=None):
    """``model`` with the gravity-wave constants / Coriolis field replaced."""
    coriolis = next(t for t in model.terms.terms if isinstance(t, LinearSWMCoriolis))
    if g is not None:
        model = eqx.tree_at(lambda m: m.terms.terms[0].g, model, jnp.asarray(g))
    if H0 is not None:
        model = eqx.tree_at(lambda m: m.terms.terms[0].H0, model, jnp.asarray(H0))
    if f_scale is not None:
        model = eqx.tree_at(
            lambda m: m.terms.terms[1].f_field, model, coriolis.f_field * f_scale
        )
    return model


def test_constants_are_leaves_and_read_back():
    _, term = _matched_models()
    assert eqx.is_array(term.g) and eqx.is_array(term.H0)
    assert float(term.g) == pytest.approx(9.81)
    assert float(term.H0) == pytest.approx(100.0)
    assert term.f_field.shape == (term.grid.Ny, term.grid.Nx)


@pytest.mark.parametrize("name", ["g", "H0", "f"])
def test_gradient_wrt_physical_constants_matches_finite_differences(x64, name):
    _, model = _matched_models()
    s0 = model.apply_boundary_conditions(_gaussian_bump(model))

    def loss(m):
        s = m.step(s0, 1800.0)
        return jnp.sum(s.h**2) + jnp.sum(s.u**2) + jnp.sum(s.v**2)

    grads = eqx.filter_grad(loss)(model)
    if name == "g":
        x0, ad = 9.81, grads.terms.terms[0].g

        def perturbed(x):
            return _with(model, g=x)
    elif name == "H0":
        x0, ad = 100.0, grads.terms.terms[0].H0

        def perturbed(x):
            return _with(model, H0=x)
    else:  # d/dc J(c f) at c = 1, i.e. <dJ/df, f>
        x0 = 1.0
        ad = jnp.vdot(grads.terms.terms[1].f_field, model.terms.terms[1].f_field)

        def perturbed(x):
            return _with(model, f_scale=x)

    eps = 1e-4 * abs(x0)
    fd = (loss(perturbed(x0 + eps)) - loss(perturbed(x0 - eps))) / (2 * eps)
    assert abs(float(fd)) > 0
    np.testing.assert_allclose(ad, fd, rtol=1e-3)


def test_changing_a_constant_does_not_retrace():
    import jax

    _, model = _matched_models()
    s0 = model.apply_boundary_conditions(_gaussian_bump(model))
    traces = []

    @eqx.filter_jit
    def step(m, s):
        traces.append(1)
        return m.step(s, 600.0)

    jax.block_until_ready(step(model, s0))
    jax.block_until_ready(step(_with(model, g=10.0, H0=80.0), s0))
    assert len(traces) == 1
