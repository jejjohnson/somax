"""Pre-configured GFD test cases.

Each function returns a ``(model, state0)`` tuple ready for
``model.integrate(state0, ...)``.

Phase 3 (#77) removed the double-gyre per-pair factories
(``barotropic_jet_instability``, ``doublegyre_qg``,
``doublegyre_baroclinic_qg``, ``baroclinic_instability_swm``,
``doublegyre_reparameterized_qg``) — their logic moved into the
``double_gyre`` scenario plus the per-model adapters under
:mod:`somax._src.cli.models_registry`. The 1D / geostrophic adjustment
helpers that are not part of the ``double_gyre x model`` decomposition
stay here because they exercise non-forced, pedagogical cases that
aren't yet represented as scenarios.
"""

from __future__ import annotations

import jax.numpy as jnp

from somax._src.models.swm.linear_1d import LinearShallowWater1D, LinearSW1DState
from somax._src.models.swm.linear_2d import LinearShallowWater2D, LinearSW2DState


def gravity_wave_1d(
    nx: int = 400,
    Lx: float = 1e6,
    g: float = 9.81,
    H0: float = 100.0,
    sigma: float = 5e4,
) -> tuple[LinearShallowWater1D, LinearSW1DState]:
    """1D gravity wave: Gaussian height perturbation, no rotation.

    Phase speed c = sqrt(g*H0) ~ 31.3 m/s for default parameters.

    Args:
        nx: Number of interior grid cells.
        Lx: Domain length (m).
        g: Gravitational acceleration (m/s²).
        H0: Mean layer depth (m).
        sigma: Gaussian width (m).

    Returns:
        ``(model, state0)`` tuple.
    """
    model = LinearShallowWater1D.create(nx=nx, Lx=Lx, g=g, f0=0.0, H0=H0)
    x = jnp.arange(model.grid.Nx) * model.grid.dx
    x0 = Lx / 2.0
    h0 = jnp.exp(-0.5 * ((x - x0) / sigma) ** 2)
    u0 = jnp.zeros_like(h0)
    v0 = jnp.zeros_like(h0)
    state0 = LinearSW1DState(h=h0, u=u0, v=v0)
    return model, state0


def inertial_oscillation_1d(
    nx: int = 50,
    Lx: float = 1e6,
    f0: float = 1e-4,
    u_init: float = 1.0,
) -> tuple[LinearShallowWater1D, LinearSW1DState]:
    """1D inertial oscillation: uniform initial u, period = 2*pi/f0.

    Args:
        nx: Number of interior grid cells.
        Lx: Domain length (m).
        f0: Coriolis parameter (1/s).
        u_init: Initial x-velocity (m/s).

    Returns:
        ``(model, state0)`` tuple.
    """
    model = LinearShallowWater1D.create(nx=nx, Lx=Lx, f0=f0, H0=1000.0)
    h0 = jnp.zeros(model.grid.Nx)
    u0 = jnp.full(model.grid.Nx, u_init)
    v0 = jnp.zeros(model.grid.Nx)
    state0 = LinearSW1DState(h=h0, u=u0, v=v0)
    return model, state0


def geostrophic_adjustment_2d(
    nx: int = 128,
    ny: int = 128,
    Lx: float = 1e6,
    Ly: float = 1e6,
    f0: float = 1e-4,
    H0: float = 100.0,
    eta_max: float = 1.0,
) -> tuple[LinearShallowWater2D, LinearSW2DState]:
    """2D geostrophic adjustment: step-function height perturbation.

    A north-south height step adjusts to geostrophic balance,
    radiating gravity waves.

    Args:
        nx: Interior cells in x.
        ny: Interior cells in y.
        Lx: Domain length in x (m).
        Ly: Domain length in y (m).
        f0: Coriolis parameter (1/s).
        H0: Mean layer depth (m).
        eta_max: Height perturbation amplitude (m).

    Returns:
        ``(model, state0)`` tuple.
    """
    model = LinearShallowWater2D.create(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, f0=f0, H0=H0, bc="periodic"
    )
    x = jnp.arange(model.grid.Nx) * model.grid.dx
    X = jnp.broadcast_to(x[None, :], (model.grid.Ny, model.grid.Nx))
    # Smooth tanh height step: high on left, low on right
    h0 = -eta_max * jnp.tanh((X - Lx / 2.0) / (Lx / 20.0))
    u0 = jnp.zeros_like(h0)
    v0 = jnp.zeros_like(h0)
    state0 = LinearSW2DState(h=h0, u=u0, v=v0)
    return model, state0
