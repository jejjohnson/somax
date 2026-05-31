"""Term-based Burgers2D vs the canonical Burgers2D.

These tests are the evidence that the term-kernel decomposition is
*faithful*: the assembled ``advection + nu * diffusion`` tree reproduces
the monolithic model's tendencies and explicit trajectory exactly, and
additionally unlocks IMEX integration (diffusion implicit) that the
monolithic model cannot express.
"""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import pytest

from somax._src.core.model import SomaxModel, TermModel
from somax._src.models.pde2d.burgers import Burgers2D, Burgers2DState
from somax._src.models.pde2d.burgers_terms import (
    Burgers2DAdvection,
    Burgers2DDiffusion,
    Burgers2DTermModel,
)


def _gaussian_2d(x, y, mux, muy, sigma):
    return jnp.exp(-0.5 * (((x - mux) / sigma) ** 2 + ((y - muy) / sigma) ** 2))


def _make_coords(grid):
    x = jnp.arange(grid.Nx) * grid.dx
    y = jnp.arange(grid.Ny) * grid.dy
    return jnp.meshgrid(x, y)


def _gaussian_state(model) -> Burgers2DState:
    X, Y = _make_coords(model.grid)
    g = _gaussian_2d(X, Y, 1.0, 1.0, 0.3)
    return Burgers2DState(u=g, v=g)


# ----------------------------------------------------------------------
# Type / protocol conformance
# ----------------------------------------------------------------------


def test_is_term_model_and_somax_model():
    model = Burgers2DTermModel.create(nx=16, ny=16)
    assert isinstance(model, TermModel)
    assert isinstance(model, SomaxModel)


def test_kernels_are_terms():
    monolith = Burgers2D.create(nx=8, ny=8)
    term = Burgers2DTermModel.from_model(monolith)
    # advection (explicit) + scaled diffusion -> the assembled Sum
    summands = term.terms.terms
    assert any(isinstance(s, Burgers2DAdvection) for s in _leaf_terms(summands))


def _leaf_terms(summands):
    from somax._src.core.terms import Scaled

    leaves = []
    for s in summands:
        leaves.append(s)
        if isinstance(s, Scaled):
            leaves.append(s.term)
    return leaves


# ----------------------------------------------------------------------
# Faithfulness to the canonical model
# ----------------------------------------------------------------------


def test_tendency_matches_monolithic():
    monolith = Burgers2D.create(nx=16, ny=16, nu=0.05)
    term = Burgers2DTermModel.from_model(monolith)

    state = _gaussian_state(monolith)
    bc = monolith.apply_boundary_conditions(state)

    mono_tend = monolith.vector_field(0.0, bc)
    term_tend = term.terms(0.0, bc)

    assert jnp.allclose(mono_tend.u, term_tend.u, atol=1e-12)
    assert jnp.allclose(mono_tend.v, term_tend.v, atol=1e-12)


def test_explicit_trajectory_matches_monolithic():
    monolith = Burgers2D.create(nx=16, ny=16, nu=0.05)
    term = Burgers2DTermModel.from_model(monolith)
    state0 = _gaussian_state(monolith)

    mono_sol = monolith.integrate(state0, t0=0.0, t1=0.02, dt=0.001)
    term_sol = term.integrate(state0, t0=0.0, t1=0.02, dt=0.001)

    assert jnp.allclose(mono_sol.ys.u, term_sol.ys.u, atol=1e-10)
    assert jnp.allclose(mono_sol.ys.v, term_sol.ys.v, atol=1e-10)


def test_step_conforms_to_forward_model():
    model = Burgers2DTermModel.create(nx=16, ny=16, nu=0.05)
    state0 = _gaussian_state(model)
    stepped = model.step(state0, dt=0.001)
    assert stepped.u.shape == state0.u.shape
    assert stepped.v.shape == state0.v.shape
    assert jnp.all(jnp.isfinite(stepped.u))


# ----------------------------------------------------------------------
# IMEX: the payoff of the decomposition
# ----------------------------------------------------------------------


def test_diffrax_terms_explicit_is_single_odeterm():
    model = Burgers2DTermModel.create(nx=8, ny=8, imex=False)
    assert isinstance(model.build_terms(), dfx.ODETerm)


def test_diffrax_terms_imex_is_multiterm():
    model = Burgers2DTermModel.create(nx=8, ny=8, imex=True)
    assert isinstance(model.build_terms(), dfx.MultiTerm)


def test_diffusion_kind_toggles_with_imex():
    explicit_model = Burgers2DTermModel.create(nx=8, ny=8, imex=False)
    imex_model = Burgers2DTermModel.create(nx=8, ny=8, imex=True)
    # Pull the diffusion kernel out of each assembled tree.
    assert _diffusion_kind(explicit_model) == "explicit"
    assert _diffusion_kind(imex_model) == "implicit"


def _diffusion_kind(model):
    # terms = Sum(advection, <scaled or _Kinded> diffusion)
    for summand in model.terms.terms:
        for leaf in _all_leaves(summand):
            if isinstance(leaf, Burgers2DDiffusion):
                return leaf.kind
    raise AssertionError("no diffusion kernel found")


def _all_leaves(term):
    from somax._src.core.terms import Scaled, _Kinded

    out = [term]
    if isinstance(term, Scaled | _Kinded):
        out.extend(_all_leaves(term.term))
    return out


# Marked slow: the implicit KenCarp3 Newton solves dominate runtime
# (~tens of seconds vs <1.5s for every other test here). Excluded from
# the fast PR CI subset; runs in the manual full-suite workflow.
@pytest.mark.slow
def test_imex_integration_matches_explicit():
    # Same physics, two integration strategies: the IMEX solve (diffusion
    # implicit via KenCarp3) should agree with the explicit Tsit5 solve.
    explicit_model = Burgers2DTermModel.create(nx=8, ny=8, nu=0.05, imex=False)
    imex_model = Burgers2DTermModel.create(nx=8, ny=8, nu=0.05, imex=True)
    state0 = _gaussian_state(explicit_model)

    explicit_sol = explicit_model.integrate(state0, t0=0.0, t1=0.005, dt=0.001)
    imex_sol = imex_model.integrate(
        state0,
        t0=0.0,
        t1=0.005,
        dt=0.001,
        solver=dfx.KenCarp3(),
        stepsize_controller=dfx.PIDController(rtol=1e-4, atol=1e-6),
        max_steps=10_000,
    )

    assert jnp.allclose(explicit_sol.ys.u, imex_sol.ys.u, atol=1e-3)
    assert jnp.allclose(explicit_sol.ys.v, imex_sol.ys.v, atol=1e-3)


# ----------------------------------------------------------------------
# Differentiability of the viscosity coefficient (Scaled.coeff)
# ----------------------------------------------------------------------


def test_grad_through_nu():
    model = Burgers2DTermModel.create(nx=16, ny=16, nu=0.05)
    state0 = _gaussian_state(model)

    @eqx.filter_grad
    def grad_fn(m):
        sol = m.integrate(state0, t0=0.0, t1=0.01, dt=0.001)
        return jnp.sum(sol.ys.u**2 + sol.ys.v**2)

    grads = grad_fn(model)
    # nu lives as the Scaled coefficient inside the term tree; the grad
    # must flow back to it.
    assert jnp.isfinite(grads.nu)
    assert grads.nu != 0.0
