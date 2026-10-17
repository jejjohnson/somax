"""``diagnose()`` built on finitevolX Tier-B operators equals the old formulas.

The C-grid models now compute their diagnostics with ``finitevolx.pipekit``
operators. Each test re-states the previous hand-written formula (built on
the model's own mask-aware Tier-A operators) and checks the new
``diagnose()`` against it, unmasked and masked, on states whose ghost ring
holds the boundary condition (the states a run produces).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from finitevolx import Mask2D, kinetic_energy, multilayer

from somax._src.models.qg.barotropic import BarotropicQG, BarotropicQGState
from somax._src.models.swm.linear_2d import LinearShallowWater2D, LinearSW2DState
from somax._src.models.swm.multilayer import (
    MultilayerShallowWater2D,
    MultilayerSW2DState,
)
from somax._src.models.swm.nonlinear_2d import (
    NonlinearShallowWater2D,
    NonlinearSW2DState,
)


N = 16
S = (slice(1, -1), slice(1, -1))


def _mask() -> Mask2D:
    wet = np.ones((N + 2, N + 2), dtype=bool)
    wet[0, :] = wet[-1, :] = wet[:, 0] = wet[:, -1] = False
    wet[6:10, 5:9] = False  # an island
    return Mask2D.from_mask(wet)


def _fields(*shape, seed=0, offset=0.0):
    keys = jax.random.split(jax.random.key(seed), 3)
    return [
        offset * (i == 0) + 0.1 * jax.random.normal(k, shape)
        for i, k in enumerate(keys)
    ]


def _allclose(a, b):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-7)


# ----------------------------------------------------------------------
# Previous formulas (reference)
# ----------------------------------------------------------------------


def _old_linear(m, s):
    g, H0 = m.consts.gravity, m.consts.H0
    area = m.grid.dx * m.grid.dy
    energy = 0.5 * jnp.sum(g * s.h[S] ** 2 + H0 * (s.u[S] ** 2 + s.v[S] ** 2)) * area
    return energy, m.diff.curl(s.u, s.v)


def _old_nonlinear(m, s):
    h, u, v = s.h, s.u, s.v
    ke = kinetic_energy(u, v)
    zeta = m.vorticity.relative_vorticity(u, v)
    q = m.vorticity.potential_vorticity(u, v, h, m.f_field)
    area = m.grid.dx * m.grid.dy
    energy = area * jnp.sum(ke[S] + 0.5 * m.consts.gravity * h[S] ** 2)
    h_X = m.interp.T_to_X(h)
    return {
        "energy": energy,
        "enstrophy": 0.5 * area * jnp.sum(q[S] ** 2 * h_X[S]),
        "mass": area * jnp.sum(h[S]),
        "casimir_q3": area * jnp.sum(q[S] ** 3 * h_X[S]),
        "potential_vorticity": q,
        "relative_vorticity": zeta,
        "kinetic_energy_field": ke,
    }


def _old_multilayer(m, s):
    h, u, v = s.h, s.u, s.v
    sl = (slice(None), *S)
    ke = multilayer(kinetic_energy)(u, v)
    zeta = multilayer(m.vorticity.relative_vorticity)(u, v)
    q = multilayer(m.vorticity.potential_vorticity)(u, v, h, m.f_field_ml)
    h_X = multilayer(m.interp.T_to_X)(h)
    area = m.grid.dx * m.grid.dy
    energy = (
        jnp.sum(ke[sl], axis=(-2, -1)) * area
        + 0.5 * m.strat.g_prime * jnp.sum(h[sl] ** 2, axis=(-2, -1)) * area
    )
    enstrophy = 0.5 * jnp.sum(q[sl] ** 2 * h_X[sl], axis=(-2, -1)) * area
    return {
        "energy": energy,
        "total_energy": jnp.sum(energy),
        "enstrophy": enstrophy,
        "total_enstrophy": jnp.sum(enstrophy),
        "mass": jnp.sum(h[sl], axis=(-2, -1)) * area,
        "casimir_q3": jnp.sum(q[sl] ** 3 * h_X[sl], axis=(-2, -1)) * area,
        "potential_vorticity": q,
        "relative_vorticity": zeta,
        "kinetic_energy_field": ke,
    }


def _old_barotropic(m, s):
    q = s.q
    psi = m._invert_pv(q)
    u = -m.diff.diff_y_T_to_V(psi)
    v = m.diff.diff_x_T_to_U(psi)
    u_T, v_T = m.interp.V_to_T(u), m.interp.U_to_T(v)
    area = m.grid.dx * m.grid.dy
    return {
        "psi": psi,
        "u": u,
        "v": v,
        "kinetic_energy": 0.5 * jnp.sum(u_T[S] ** 2 + v_T[S] ** 2) * area,
        "enstrophy": 0.5 * jnp.sum(q[S] ** 2) * area,
        "relative_vorticity": m.diff.laplacian(psi),
    }


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bc", "masked"), [("periodic", False), ("wall", False), ("wall", True)]
)
def test_linear_sw(bc, masked):
    m = LinearShallowWater2D.create(
        nx=N, ny=N, Lx=1e6, Ly=1e6, bc=bc, mask=_mask() if masked else None
    )
    h, u, v = _fields(N + 2, N + 2)
    s = m.apply_boundary_conditions(LinearSW2DState(h=h, u=u, v=v))
    d = m.diagnose(s)
    energy, zeta = _old_linear(m, s)
    _allclose(d.energy, energy)
    _allclose(d.relative_vorticity, zeta)


@pytest.mark.parametrize(
    ("bc", "masked"), [("periodic", False), ("wall", False), ("wall", True)]
)
def test_nonlinear_sw(bc, masked):
    m = NonlinearShallowWater2D.create(
        nx=N, ny=N, Lx=1e6, Ly=1e6, H0=100.0, bc=bc, mask=_mask() if masked else None
    )
    h, u, v = _fields(N + 2, N + 2, offset=100.0)
    s = m.apply_boundary_conditions(NonlinearSW2DState(h=h, u=u, v=v))
    d = m.diagnose(s)
    for name, want in _old_nonlinear(m, s).items():
        _allclose(getattr(d, name), want)


@pytest.mark.parametrize("masked", [False, True])
def test_multilayer_sw(masked):
    m = MultilayerShallowWater2D.create(
        nx=N,
        ny=N,
        n_layers=2,
        H=(500.0, 1500.0),
        g_prime=(9.81, 0.02),
        bc="wall" if masked else "periodic",
        mask=_mask() if masked else None,
    )
    sh = (2, N + 2, N + 2)
    h, u, v = _fields(*sh)
    h = m.strat.H[:, None, None] + h
    s = m.apply_boundary_conditions(MultilayerSW2DState(h=h, u=u, v=v))
    d = m.diagnose(s)
    for name, want in _old_multilayer(m, s).items():
        _allclose(getattr(d, name), want)


@pytest.mark.parametrize("masked", [False, True])
def test_barotropic_qg(masked):
    m = BarotropicQG.create(
        nx=N, ny=N, Lx=1e6, Ly=1e6, mask=_mask() if masked else None
    )
    q = 1e-5 * jax.random.normal(jax.random.key(3), (N + 2, N + 2))
    s = BarotropicQGState(q=q)
    d = m.diagnose(s)
    for name, want in _old_barotropic(m, s).items():
        _allclose(getattr(d, name), want)


def test_importing_a_model_does_not_import_pipekit():
    import subprocess
    import sys

    code = (
        "import sys; import somax._src.models.swm.nonlinear_2d; "
        "assert not any(m.split('.')[0].startswith('pipekit') for m in sys.modules)"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
