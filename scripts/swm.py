""" swm.py

2D shallow water model with:

- varying Coriolis force
- nonlinear terms
- no lateral friction
- periodic boundary conditions
- reconstructions, 5pt, improved weno
    * mass term: h --> u,v
    * momentum term: q --> uh,vh
"""
from typing import NamedTuple

import autoroot
from finitevolx._src.domain.domain import Domain
import jax
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)
import matplotlib.pyplot as plt
import numpy as np

from finitevolx import (
    MaskGrid,
    bernoulli_potential,
    center_avg_2D,
    difference,
    kinetic_energy,
    reconstruct,
    relative_vorticity,
    x_avg_2D,
    y_avg_2D,
)

jax.config.update("jax_enable_x64", True)


# ==============================
# DOMAIN
# ==============================

# grid setup, height
Nx, Ny = 200, 104
dx, dy = 5e3, 5e3
Lx, Ly = Nx * dx, Ny * dy

# define domains - Arakawa C-Grid Configuration
h_domain = Domain(xmin=(0.0, 0.0), xmax=(Lx, Ly), Lx=(Lx, Ly), Nx=(Nx, Ny), dx=(dx, dy))
u_domain = Domain(
    xmin=(-0.5, 0.0),
    xmax=(Lx + 0.5 * dx, Ly),
    Lx=(Lx + dx, Ly),
    Nx=(Nx + 1, Ny),
    dx=(dx, dy),
)
v_domain = Domain(
    xmin=(0.0, -0.5),
    xmax=(Lx, Ly + 0.5 * dy),
    Lx=(Lx, Ly + 0.5 * dy),
    Nx=(Nx, Ny + 1),
    dx=(dx, dy),
)
q_domain = Domain(
    xmin=(-0.5, -0.5),
    xmax=(Lx + 0.5 * dx, Ly + 0.5 * dy),
    Lx=(Lx + dx, Ly + dy),
    Nx=(Nx + 1, Ny + 1),
    dx=(dx, dy),
)
Nx, Ny = h_domain.Nx
x, y = h_domain.coords_axis
# ==============================
# PARAMETERS
# ==============================
# physical parameters
gravity = 9.81
depth = 100.0
coriolis_f = 2e-4
coriolis_beta = 2e-11
coriolis_param: Float[Array, "Nx Ny"] = (
    coriolis_f + h_domain.grid_axis[1] * coriolis_beta
)
lateral_viscosity = 1e-3 * coriolis_f * dx**2

# other parameters
periodic_boundary_x = False
linear_mass = False
linear_momentum = False

adams_bashforth_a = 1.5 + 0.1
adams_bashforth_b = -(0.5 + 0.1)

dt = 0.125 * min(dx, dy) / np.sqrt(gravity * depth)

phase_speed = np.sqrt(gravity * depth)
rossby_radius = np.sqrt(gravity * depth) / coriolis_param.mean()

# plot parameters
plot_range = 10
plot_every = 10
max_quivers = 41


# model params
num_pts = 3
method = "wenoz"


F_0 = 0.12  # Pa
rho_0 = 1e3  # kgm-3
F_x = jnp.cos(2 * jnp.pi * (2 / u_domain.Lx[1] - 0.5))
F_x += 2 * jnp.cos(2 * jnp.pi * (u_domain.grid_axis[1] / u_domain.Lx[1] - 0.5))
F_x *= F_0 / (rho_0 * depth)


# ==============================
# MASK
# ==============================
mask = jnp.ones(h_domain.Nx)
masks = MaskGrid.init_mask(mask, "center")


# ==============================
# INITIAL CONDITIONS
# ==============================
def init_u0(domain):
    Y = domain.grid_axis[1]
    y = domain.coords_axis[1]
    Ny = domain.Nx[1]
    Lx = domain.Lx[0]
    u0 = 10 * jnp.exp(-((Y - y[Ny // 2]) ** 2) / (0.02 * Lx) ** 2)
    return u0


def init_h0_jet(domain, u0):
    dy = domain.dx[1]
    Lx, Ly = domain.Lx
    X, Y = domain.grid_axis

    h_geostrophy = jnp.cumsum(-dy * x_avg_2D(u0) * coriolis_param / gravity, axis=1)

    h0 = (
        depth
        + h_geostrophy
        # make sure h0 is centered around depth
        - h_geostrophy.mean()
        # small perturbation
        + 0.2 * jnp.sin(X / Lx * 10.0 * jnp.pi) * jnp.cos(Y / Ly * 8.0 * jnp.pi)
    )

    return h0


u0 = init_u0(u_domain)
h0 = init_h0_jet(h_domain, u0)
v0 = jnp.zeros(v_domain.Nx)


# ==============================
# STATE
# ==============================
class State(NamedTuple):
    h: Array
    u: Array
    v: Array


# ==============================
# BOUNDARY CONDITIONS
# ==============================


def prepare_plot():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # move velocities (faces) to height (centers)
    u0_on_h = x_avg_2D(u0)
    v0_on_h = y_avg_2D(v0)

    cs = update_plot(0, h0, u0_on_h, v0_on_h, ax)
    plt.colorbar(cs, label="$\\eta$ (m)")
    return fig, ax


def update_plot(t, h, u, v, ax):
    h, u, v = list(map(np.asarray, [h, u, v]))
    eta = h - depth

    quiver_stride = (slice(1, -1, Nx // max_quivers), slice(1, -1, Ny // max_quivers))

    ax.clear()
    cs = ax.pcolormesh(
        x[1:-1] / 1e3,
        y[1:-1] / 1e3,
        eta[1:-1, 1:-1].T,
        vmin=-plot_range,
        vmax=plot_range,
        cmap="RdBu_r",
    )

    if np.any((u[quiver_stride] != 0) | (v[quiver_stride] != 0)):
        ax.quiver(
            x[quiver_stride[0]] / 1e3,
            y[quiver_stride[1]] / 1e3,
            u[quiver_stride].T,
            v[quiver_stride].T,
            clip_on=False,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (km)")
    ax.set_ylabel("$y$ (km)")
    ax.set_xlim(x[1] / 1e3, x[-2] / 1e3)
    ax.set_ylim(y[1] / 1e3, y[-2] / 1e3)
    ax.set_title(
        "t=%5.2f days, R=%5.1f km, c=%5.1f m/s "
        % (t / 86400, rossby_radius / 1e3, phase_speed)
    )
    plt.pause(0.00001)
    return cs


# ####################################
# EQUATION OF MOTION
# ####################################


# ====================================
# Nonlinear Terms
# ====================================
def calculate_uvh_flux(
    h: Float[Array, "Nx+2 Ny+2"], u: Float[Array, "Nx+1 Ny"], v: Float[Array, "Nx Ny+1"]
):
    """
    Eq:
        (uh), (vh)
    """

    # calculate h fluxes
    uh_flux: Float[Array, "Nx+1 Ny"] = reconstruct(
        q=h[:, 1:-1], u=u, u_mask=masks.face_u, dim=0, num_pts=num_pts, method=method
    )
    vh_flux: Float[Array, "Nx Ny+1"] = reconstruct(
        q=h[1:-1, :], u=v, u_mask=masks.face_v, dim=1, num_pts=num_pts, method=method
    )

    # apply masks
    uh_flux *= masks.face_u.values
    vh_flux *= masks.face_v.values

    return uh_flux, vh_flux


def potential_vorticity(
    h: Float[Array, "Nx Ny"], u: Float[Array, "Nx+1 Ny"], v: Float[Array, "Nx Ny+1"]
):
    """
    Eq:
        ζ = ∂v/∂x - ∂u/∂y
        q = (ζ + f) / h
    """

    # planetary vorticity, f
    f_on_q: Float[Array, "Nx+1 Ny+1"] = (
        coriolis_f + q_domain.grid_axis[1] * coriolis_beta
    )
    # f_on_q: Float[Array, "Nx+1 Ny+1"] = coriolis_f + coriolis_beta * (
    #     q_domain.grid_axis[1] - q_domain.Lx[1] / 2.0
    # )

    # relative vorticity, ζ = dv/dx - du/dy
    vort_r: Float[Array, "Nx-1 Ny-1"] = relative_vorticity(
        u=u[1:-1], v=v[:, 1:-1], dx=v_domain.dx[0], dy=u_domain.dx[1]
    )

    # potential vorticity, q = (ζ + f) / h
    h_on_q: Float[Array, "Nx-1 Ny-1"] = center_avg_2D(h)
    q: Float[Array, "Nx-1 Ny-1"] = (vort_r + f_on_q[1:-1, 1:-1]) / h_on_q

    # pad array
    q: Float[Array, "Nx+1 Ny+1"] = jnp.pad(q, pad_width=((1, 1), (1, 1)))

    # apply masks
    q *= masks.node.values

    return q


# ====================================
# HEIGHT
# ====================================
def h_linear_rhs(u, v):
    """
    Eq:
       ∂h/∂t = - H (∂u/∂x + ∂v/∂y)
    """

    # calculate RHS terms
    du_dx: Float[Array, "Nx Ny"] = difference(
        u, step_size=u_domain.dx[0], axis=0, derivative=1
    )
    dv_dy: Float[Array, "Nx Ny"] = difference(
        v, step_size=v_domain.dx[1], axis=1, derivative=1
    )

    # calculate RHS
    h_rhs: Float[Array, "Nx Ny"] = -depth * (du_dx + dv_dy)

    # apply masks
    h_rhs *= masks.center.values

    return h_rhs


def h_nonlinear_rhs(
    uh_flux: Float[Array, "Nx+1 Ny"], vh_flux: Float[Array, "Nx Ny+1"]
) -> Float[Array, "Nx Ny"]:
    """
    Eq:
        ∂h/∂t + ∂/∂x((H+h)u) + ∂/∂y((H+h)v) = 0
    """

    # calculate RHS terms
    dhu_dx: Float[Array, "Nx Ny"] = difference(
        uh_flux, step_size=u_domain.dx[0], axis=0, derivative=1
    )
    dhv_dy: Float[Array, "Nx Ny"] = difference(
        vh_flux, step_size=v_domain.dx[1], axis=1, derivative=1
    )

    # calculate RHS
    h_rhs: Float[Array, "Nx Ny"] = -(dhu_dx + dhv_dy)

    # apply masks
    h_rhs *= masks.center.values

    return h_rhs


# ================================
# ZONAL VELOCITY, u
# ================================
def u_linear_rhs(h: Float[Array, "Nx Ny"], v: Float[Array, "Nx Ny+1"]):
    """
    Eq:
        ∂u/∂t = fv - g ∂h/∂x
    """

    # calculate RHS terms
    v_avg: Float[Array, "Nx-1 Ny"] = center_avg_2D(v)
    dh_dx: Float[Array, "Nx-1 Ny"] = difference(
        h, step_size=h_domain.dx[0], axis=0, derivative=1
    )

    # calculate RHS
    u_rhs: Float[Array, "Nx-1 Ny"] = coriolis_f * v_avg - gravity * dh_dx

    # mask values
    u_rhs *= masks.face_u.values

    return u_rhs


def u_nonlinear_rhs(
    q: Float[Array, "Nx-1 Ny-1"],
    vh_flux: Float[Array, "Nx Ny+1"],
    p: Float[Array, "Nx Ny"],
):
    """
    Eq:
        work = g ∂h/∂x
        ke = 0.5 (u² + v²)
        ∂u/∂t = qhv - work - ke

    Notes:
        - uses reconstruction (5pt, improved weno) of q on vh flux
    """
    vh_flux_on_u: Float[Array, "Nx-1 Ny"] = center_avg_2D(vh_flux)

    qhv_flux_on_u: Float[Array, "Nx-1 Ny-2"] = reconstruct(
        q=q[1:-1, 1:-1],
        u=vh_flux_on_u[:, 1:-1],
        u_mask=masks.face_u[1:-1, 1:-1],
        dim=1,
        method=method,
        num_pts=num_pts,
    )

    qhv_flux_on_u: Float[Array, "Nx+1 Ny"] = jnp.pad(
        qhv_flux_on_u, pad_width=((1, 1), (1, 1)), mode="constant"
    )

    # apply mask
    qhv_flux_on_u *= masks.face_u.values

    # calculate work
    p: Float[Array, "Nx+2 Ny"] = jnp.pad(
        p, pad_width=((1, 1), (0, 0)), mode="constant", constant_values=0.0
    )
    dpdx: Float[Array, "Nx+1 Ny"] = difference(
        p, step_size=h_domain.dx[0], axis=0, derivative=1
    )

    # calculate u RHS
    u_rhs: Float[Array, "Nx+1 Ny"] = qhv_flux_on_u - dpdx

    # apply mask
    u_rhs *= masks.face_u.values

    return u_rhs


# ================================
# MERIDIONAL VELOCITY, v
# ================================


def v_linear_rhs(h: Float[Array, "Nx Ny"], u: Float[Array, "Nx+1 Ny"]):
    """
    Eq:
        ∂v/∂t = - fu - g ∂h/∂y
    """
    # calculate RHS terms
    u_avg: Float[Array, "Nx Ny-1"] = center_avg_2D(u)
    dh_dy: Float[Array, "Nx Ny-1"] = difference(
        h, step_size=h_domain.dx[1], axis=1, derivative=1
    )

    # calculate RHS
    v_rhs: Float[Array, "Nx-1 Ny-1"] = -coriolis_f * u_avg - gravity * dh_dy

    # mask values
    v_rhs *= masks.face_v.values

    return v_rhs


def v_nonlinear_rhs(
    q: Float[Array, "Nx+1 Ny+1"],
    uh_flux: Float[Array, "Nx+1 Ny"],
    p: Float[Array, "Nx Ny"],
):
    """
    Eq:
        ∂v/∂t = - qhu - ∂p/∂y
        p = ke + gh
        ke = 0.5 (u² + v²)

    Notes:
        - uses reconstruction (5pt, improved weno) of q on uh flux
    """

    uh_flux_on_v: Float[Array, "Nx Ny-1"] = center_avg_2D(uh_flux)

    qhu_flux_on_v: Float[Array, "Nx-2 Ny-1"] = reconstruct(
        q=q[1:-1, 1:-1],
        u=uh_flux_on_v[1:-1],
        u_mask=masks.face_v[1:-1, 1:-1],
        dim=0,
        method=method,
        num_pts=num_pts,
    )

    qhu_flux_on_v: Float[Array, "Nx Ny+1"] = jnp.pad(
        qhu_flux_on_v, pad_width=((1, 1), (1, 1)), mode="constant"
    )

    qhu_flux_on_v *= masks.face_v.values

    # calculate kinetic energy
    p: Float[Array, "Nx Ny+2"] = jnp.pad(
        p,
        pad_width=(
            (0, 0),
            (1, 1),
        ),
        mode="constant",
        constant_values=0.0,
    )
    dp_dy: Float[Array, "Nx Ny+1"] = difference(
        p, step_size=h_domain.dx[1], axis=1, derivative=1
    )

    # calculate u RHS
    v_rhs: Float[Array, "Nx-2 Ny-1"] = -dp_dy - qhu_flux_on_v

    # apply masks
    v_rhs *= masks.face_v.values

    return v_rhs


# vector field
def equation_of_motion(h, u, v):
    if not linear_mass or not linear_momentum:
        # zero-gradient boundaries
        h_pad: Float[Array, "Nx+2 Ny+2"] = jnp.pad(
            h, pad_width=((1, 1), (1, 1)), mode="edge"
        )
        # calculate flux
        uh_flux, vh_flux = calculate_uvh_flux(h=h_pad, u=u, v=v)

    # mass equation

    if linear_mass:
        h_rhs = h_linear_rhs(u=u, v=v)
    else:
        h_rhs = h_nonlinear_rhs(uh_flux=uh_flux, vh_flux=vh_flux)

    # momentum equations

    if linear_momentum:
        # zero-derivative
        h_pad: Float[Array, "Nx+2 Ny"] = jnp.pad(
            h, pad_width=((1, 1), (1, 1)), mode="edge"
        )
        # zero-boundaries (no-slip)
        v_pad: Float[Array, "Nx+2 Ny+1"] = jnp.pad(
            v, pad_width=((1, 1), (0, 0)), mode="constant"
        )
        u_pad: Float[Array, "Nx+1 Ny+2"] = jnp.pad(
            u, pad_width=((0, 0), (1, 1)), mode="constant"
        )

        # Zonal Velocity
        u_rhs = u_linear_rhs(h=h_pad[:, 1:-1], v=v_pad)
        u_rhs *= masks.face_u.values

        # Meridional Velocity
        v_rhs = v_linear_rhs(h=h_pad[1:-1, :], u=u_pad)
        v_rhs *= masks.face_v.values
    else:
        ke = kinetic_energy(u=u, v=v)
        ke *= masks.center.values

        p: Float[Array, "Nx Ny"] = bernoulli_potential(h=h, u=u, v=v)
        q: Float[Array, "Nx+1 Ny+1"] = potential_vorticity(h=h, u=u, v=v)

        u_rhs = u_nonlinear_rhs(q=q, vh_flux=vh_flux, p=p)
        v_rhs = v_nonlinear_rhs(q=q, uh_flux=uh_flux, p=p)

    u_rhs += F_x
    return h_rhs, u_rhs, v_rhs


equation_of_motion_jitted = jax.jit(equation_of_motion)


def iterate_shallow_water():
    # allocate arrays
    u, v, h = jnp.empty_like(u0), jnp.empty_like(v0), jnp.empty_like(h0)

    # initial conditions
    h: Float[Array, "Nx Ny"] = h.at[:].set(h0)
    u: Float[Array, "Nx+1 Ny"] = u.at[:].set(u0)
    v: Float[Array, "Nx Ny+1"] = v.at[:].set(v0)

    first_step = True

    # time step equations
    while True:
        # ==================
        # SPATIAL OPERATIONS
        # ==================
        h_rhs, u_rhs, v_rhs = equation_of_motion_jitted(h, u, v)

        # ==================
        # TIME STEPPING
        # ==================
        if first_step:
            u += dt * u_rhs
            v += dt * v_rhs
            h += dt * h_rhs
            first_step = False
        else:
            u += dt * (adams_bashforth_a * u_rhs + adams_bashforth_b * u_rhs_old)
            v += dt * (adams_bashforth_a * v_rhs + adams_bashforth_b * v_rhs_old)
            h += dt * (adams_bashforth_a * h_rhs + adams_bashforth_b * h_rhs_old)
        # #
        # h = enforce_boundaries(h, 'h')
        # u = enforce_boundaries(u, 'u')
        # v = enforce_boundaries(v, 'v')

        # rotate quantities
        h_rhs_old = h_rhs
        v_rhs_old = v_rhs
        u_rhs_old = u_rhs

        yield h, u, v


if __name__ == "__main__":
    fig, ax = prepare_plot()

    # create model generator
    model = iterate_shallow_water()

    # iterate through steps
    for iteration, (h, u, v) in enumerate(model):
        if iteration % plot_every == 0:
            t = iteration * dt

            # move face variables to center
            # u,v --> h
            u_on_h = center_avg_2D(u)
            v_on_h = center_avg_2D(v)



            # update plot
            update_plot(t, h, u_on_h, v_on_h, ax)

        # stop if user closes plot window
        if not plt.fignum_exists(fig.number):
            break
