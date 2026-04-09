"""Pre-configured GFD test cases.

Each function returns a ``(model, state0)`` tuple ready for
``model.integrate(state0, ...)``.
"""

from __future__ import annotations

import jax.numpy as jnp

from somax._src.models.qg.baroclinic import BaroclinicQG, BaroclinicQGState
from somax._src.models.qg.barotropic import BarotropicQG, BarotropicQGState
from somax._src.models.qg.reparameterized import ReparameterizedQG
from somax._src.models.swm.linear_1d import LinearShallowWater1D, LinearSW1DState
from somax._src.models.swm.linear_2d import LinearShallowWater2D, LinearSW2DState
from somax._src.models.swm.multilayer import (
    MultilayerShallowWater2D,
    MultilayerSW2DState,
)
from somax._src.models.swm.nonlinear_2d import (
    NonlinearShallowWater2D,
    NonlinearSW2DState,
)


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


def barotropic_jet_instability(
    nx: int = 128,
    ny: int = 128,
    Lx: float = 1e6,
    Ly: float = 1e6,
    f0: float = 1e-4,
    beta: float = 1.6e-11,
    H0: float = 1000.0,
    jet_speed: float = 1.0,
    jet_width: float = 5e4,
    perturbation: float = 0.01,
    lateral_viscosity: float = 100.0,
) -> tuple[NonlinearShallowWater2D, NonlinearSW2DState]:
    """Barotropic jet instability on a β-plane.

    A zonal jet with small perturbations develops
    barotropic instabilities (meanders and vortex shedding).

    Args:
        nx: Interior cells in x.
        ny: Interior cells in y.
        Lx: Domain length in x (m).
        Ly: Domain length in y (m).
        f0: Coriolis parameter (1/s).
        beta: β parameter (1/(m·s)).
        H0: Mean layer depth (m).
        jet_speed: Peak jet velocity (m/s).
        jet_width: Gaussian half-width of jet (m).
        perturbation: Amplitude of initial perturbation.
        lateral_viscosity: Harmonic viscosity (m²/s).

    Returns:
        ``(model, state0)`` tuple.
    """
    model = NonlinearShallowWater2D.create(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        f0=f0,
        beta=beta,
        H0=H0,
        lateral_viscosity=lateral_viscosity,
        bc="periodic",
    )
    grid = model.grid
    x = jnp.arange(grid.Nx) * grid.dx
    y = jnp.arange(grid.Ny) * grid.dy
    X, Y = jnp.meshgrid(x, y)

    y0 = Ly / 2.0
    # Gaussian zonal jet
    u0 = jet_speed * jnp.exp(-0.5 * ((Y - y0) / jet_width) ** 2)
    v0 = (
        perturbation
        * jnp.sin(4.0 * jnp.pi * X / Lx)
        * jnp.exp(-0.5 * ((Y - y0) / jet_width) ** 2)
    )
    h0 = jnp.full_like(u0, H0)
    state0 = NonlinearSW2DState(h=h0, u=u0, v=v0)
    return model, state0


def doublegyre_qg(
    nx: int = 64,
    ny: int = 64,
    Lx: float = 1e6,
    Ly: float = 1e6,
    f0: float = 1e-4,
    beta: float = 1.6e-11,
    lateral_viscosity: float = 500.0,
    bottom_drag: float = 1e-7,
    wind_amplitude: float = 1e-12,
) -> tuple[BarotropicQG, BarotropicQGState]:
    """Double-gyre wind-driven QG circulation.

    A sinusoidal wind stress curl drives a double-gyre
    circulation with western boundary intensification.

    Args:
        nx: Interior cells in x.
        ny: Interior cells in y.
        Lx: Domain length in x (m).
        Ly: Domain length in y (m).
        f0: Coriolis parameter (1/s).
        beta: beta parameter (1/(m*s)).
        lateral_viscosity: Harmonic viscosity (m^2/s).
        bottom_drag: Linear bottom drag (1/s).
        wind_amplitude: Wind stress curl amplitude.

    Returns:
        ``(model, state0)`` tuple.
    """
    model = BarotropicQG.create(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        f0=f0,
        beta=beta,
        lateral_viscosity=lateral_viscosity,
        bottom_drag=bottom_drag,
        wind_amplitude=wind_amplitude,
        wind_profile="doublegyre",
    )
    q0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
    state0 = BarotropicQGState(q=q0)
    return model, state0


def doublegyre_baroclinic_qg(
    nx: int = 128,
    ny: int = 128,
    Lx: float = 4e6,
    Ly: float = 4e6,
    f0: float = 9.375e-5,
    beta: float = 1.754e-11,
    n_layers: int = 3,
    H: tuple[float, ...] = (400.0, 1100.0, 2600.0),
    g_prime: tuple[float, ...] = (9.81, 0.025, 0.0125),
    lateral_viscosity: float = 15.0,
    bottom_drag: float = 1e-7,
    wind_amplitude: float = 1.3e-10,
) -> tuple[BaroclinicQG, BaroclinicQGState]:
    """Double-gyre wind-driven multilayer QG circulation.

    A sinusoidal wind stress curl drives a multilayer double-gyre
    circulation with baroclinic instability.

    Default parameters follow louity/MQGeometry 3-layer config.
    Wind amplitude corresponds to tau0=0.08 Pa scaled by 2*pi/(Ly*rho0).

    Args:
        nx: Interior cells in x.
        ny: Interior cells in y.
        Lx: Domain length in x (m).
        Ly: Domain length in y (m).
        f0: Coriolis parameter (1/s).
        beta: Beta parameter (1/(m*s)).
        n_layers: Number of layers.
        H: Layer thicknesses (m), top to bottom.
        g_prime: Reduced gravities (m/s^2).
        lateral_viscosity: Harmonic viscosity (m^2/s).
        bottom_drag: Linear bottom drag (1/s).
        wind_amplitude: Wind stress curl amplitude.

    Returns:
        ``(model, state0)`` tuple.
    """
    model = BaroclinicQG.create(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        f0=f0,
        beta=beta,
        n_layers=n_layers,
        H=H,
        g_prime=g_prime,
        lateral_viscosity=lateral_viscosity,
        bottom_drag=bottom_drag,
        wind_amplitude=wind_amplitude,
        wind_profile="doublegyre",
    )
    q0 = jnp.zeros((model.consts.n_layers, model.grid.Ny, model.grid.Nx))
    state0 = BaroclinicQGState(q=q0)
    return model, state0


def baroclinic_instability_swm(
    nx: int = 64,
    ny: int = 64,
    Lx: float = 1e6,
    Ly: float = 1e6,
    f0: float = 1e-4,
    beta: float = 1.6e-11,
    H: tuple[float, ...] = (500.0, 4500.0),
    g_prime: tuple[float, ...] = (9.81, 0.025),
    lateral_viscosity: float = 100.0,
    bottom_drag: float = 1e-7,
    jet_speed: float = 0.5,
    jet_width: float = 5e4,
    perturbation: float = 0.01,
) -> tuple[MultilayerShallowWater2D, MultilayerSW2DState]:
    """2-layer baroclinic instability in the multilayer shallow water model.

    A zonal jet with opposite sign in each layer and a small
    perturbation develops baroclinic instability. Hard-coded to
    2 layers (opposite jets require exactly 2 layers).

    Args:
        nx: Interior cells in x.
        ny: Interior cells in y.
        Lx: Domain length in x (m).
        Ly: Domain length in y (m).
        f0: Coriolis parameter (1/s).
        beta: Beta parameter (1/(m*s)).
        H: Layer thicknesses (m), top to bottom (must have length 2).
        g_prime: Reduced gravities (m/s^2) (must have length 2).
        lateral_viscosity: Harmonic viscosity (m^2/s).
        bottom_drag: Linear bottom drag (1/s).
        jet_speed: Peak jet velocity (m/s).
        jet_width: Gaussian half-width of jet (m).
        perturbation: Amplitude of initial perturbation.

    Returns:
        ``(model, state0)`` tuple.
    """
    nl = 2
    model = MultilayerShallowWater2D.create(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        f0=f0,
        beta=beta,
        n_layers=nl,
        H=H,
        g_prime=g_prime,
        lateral_viscosity=lateral_viscosity,
        bottom_drag=bottom_drag,
        bc="periodic",
    )
    grid = model.grid
    x = jnp.arange(grid.Nx) * grid.dx
    y = jnp.arange(grid.Ny) * grid.dy
    X, Y = jnp.meshgrid(x, y)

    y0 = Ly / 2.0
    jet_profile = jnp.exp(-0.5 * ((Y - y0) / jet_width) ** 2)

    # Opposite jets in each layer (baroclinic shear)
    signs = jnp.array([1.0, -1.0])[:nl]
    u0 = signs[:, None, None] * jet_speed * jet_profile[None]
    v0 = jnp.broadcast_to(
        perturbation * jnp.sin(4.0 * jnp.pi * X / Lx)[None] * jet_profile[None],
        (nl, grid.Ny, grid.Nx),
    ).copy()
    h0 = jnp.ones((nl, grid.Ny, grid.Nx)) * model.strat.H[:, None, None]

    state0 = MultilayerSW2DState(h=h0, u=u0, v=v0)
    return model, state0


def doublegyre_reparameterized_qg(
    nx: int = 128,
    ny: int = 128,
    Lx: float = 4e6,
    Ly: float = 4e6,
    f0: float = 9.375e-5,
    beta: float = 1.754e-11,
    n_layers: int = 3,
    H: tuple[float, ...] = (400.0, 1100.0, 2600.0),
    g_prime: tuple[float, ...] = (9.81, 0.025, 0.0125),
    lateral_viscosity: float = 15.0,
    bottom_drag: float = 3.6e-8,
    wind_amplitude: float = 8e-5,
) -> tuple[ReparameterizedQG, MultilayerSW2DState]:
    """Double-gyre reparameterized QG model (QG = SWM + projection).

    A wind-driven multilayer double-gyre using the reparameterized QG
    approach (Thiry et al. 2024). Same physics as
    ``doublegyre_baroclinic_qg`` but solved in (u, v, h) state space
    with geostrophic projection.

    Default parameters follow louity/qgsw-pytorch 3-layer config.
    Wind amplitude is tau0/rho0 = 0.08/1000 = 8e-5 m^2/s^2.

    Args:
        nx: Interior cells in x.
        ny: Interior cells in y.
        Lx: Domain length in x (m).
        Ly: Domain length in y (m).
        f0: Coriolis parameter (1/s).
        beta: Beta parameter (1/(m*s)).
        n_layers: Number of layers.
        H: Layer thicknesses (m), top to bottom.
        g_prime: Reduced gravities (m/s^2).
        lateral_viscosity: Harmonic viscosity (m^2/s).
        bottom_drag: Linear bottom drag (1/s).
        wind_amplitude: Wind stress amplitude (m/s^2).

    Returns:
        ``(model, state0)`` tuple.
    """
    model = ReparameterizedQG.create(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        f0=f0,
        beta=beta,
        n_layers=n_layers,
        H=H,
        g_prime=g_prime,
        lateral_viscosity=lateral_viscosity,
        bottom_drag=bottom_drag,
        wind_amplitude=wind_amplitude,
        wind_profile="doublegyre",
        bc="wall",
    )
    nl = model.consts.n_layers
    Ny, Nx = model.grid.Ny, model.grid.Nx
    h0 = jnp.ones((nl, Ny, Nx)) * model.strat.H[:, None, None]
    u0 = jnp.zeros((nl, Ny, Nx))
    v0 = jnp.zeros((nl, Ny, Nx))
    state0 = MultilayerSW2DState(h=h0, u=u0, v=v0)
    return model, state0
