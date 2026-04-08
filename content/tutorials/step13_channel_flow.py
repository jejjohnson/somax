# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Step 13 -- Channel Flow (Poiseuille) and Taylor-Green Vortex
#
# The final step in the "13 Steps to Navier--Stokes" series validates the full incompressible solver with two problems that have exact analytical solutions.
#
# **Part A -- Poiseuille flow:** We verify that the solver maintains the exact steady-state parabolic velocity profile of pressure-driven channel flow.
#
# **Part B -- Taylor-Green vortex:** We verify that the solver correctly reproduces the exponential decay of a vortex with a known analytical solution.
#
# **What you will learn:**
#
# 1. How to initialise the NS solver with a known analytical vorticity field
# 2. How to validate against exact steady-state and decaying solutions
# 3. How to differentiate through the full Navier--Stokes solver with `jax.grad`

# %% [markdown]
# ## Arakawa C-Grid and Boundary Conditions
#
# Both problems use **Dirichlet BCs** ($\psi = 0$ on all walls) with the cavity model (all walls stationary, `u_lid = 0`).
#
# ```
#              u=0, v=0 (no-slip)
#         +------------------------+
#         |                        |
#  u=0    |    omega, psi          |  u=0
#  v=0    |    (interior)          |  v=0
#         |                        |
#         +------------------------+
#              u=0, v=0 (no-slip)
# ```
#
# Wall vorticity is set via Thom's formula (see Step 12).

# %%
from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt

from somax.models import IncompressibleNS2D, NSVorticityState


# %% [markdown]
# ---
# ## Part A: Poiseuille (Channel) Flow
#
# For flow between two plates driven by a constant pressure gradient $G$, the exact steady-state velocity profile is parabolic:
#
# $$
# u(y) = \frac{G}{2\nu}\,y\,(L_y - y)
# $$
#
# The corresponding vorticity is:
#
# $$
# \omega(y) = -\frac{\partial u}{\partial y} = -\frac{G}{2\nu}(L_y - 2y)
# $$
#
# We initialise the solver with this exact vorticity profile and verify that the solution remains steady (the tendency should be nearly zero).

# %%
nx, ny = 32, 32
Lx, Ly = 2.0, 1.0
nu_channel = 0.1
G = 1.0  # pressure gradient

model_ch = IncompressibleNS2D.create(
    nx=nx,
    ny=ny,
    Lx=Lx,
    Ly=Ly,
    nu=nu_channel,
    problem="cavity",
    u_lid=0.0,
)

# Coordinates (including ghost cells)
x_ch = jnp.arange(model_ch.grid.Nx) * model_ch.grid.dx
y_ch = jnp.arange(model_ch.grid.Ny) * model_ch.grid.dy
X_ch, Y_ch = jnp.meshgrid(x_ch, y_ch)

# Analytical steady-state vorticity: omega = -G/(2*nu) * (Ly - 2y)
omega_poiseuille = -(G / (2 * nu_channel)) * (Ly - 2 * Y_ch)

# Zero vorticity in ghost cells (walls handled by apply_bcs)
omega_poiseuille = omega_poiseuille.at[0, :].set(0.0)
omega_poiseuille = omega_poiseuille.at[-1, :].set(0.0)
omega_poiseuille = omega_poiseuille.at[:, 0].set(0.0)
omega_poiseuille = omega_poiseuille.at[:, -1].set(0.0)

state_ch = NSVorticityState(omega=omega_poiseuille)

print(f"Grid: {model_ch.grid.Ny} x {model_ch.grid.Nx}")
print(f"nu = {nu_channel}, G = {G}")
print(f"Max analytical u = G*Ly^2/(8*nu) = {G * Ly**2 / (8 * nu_channel):.4f}")

# %% [markdown]
# ### Integrate and check steadiness
#
# If the initialisation is correct, the vorticity should remain almost unchanged.

# %%
sol_ch = model_ch.integrate(
    state_ch,
    t0=0.0,
    t1=0.5,
    dt=0.001,
    saveat=dfx.SaveAt(t1=True),
)

omega_final_ch = sol_ch.ys.omega[0]
print(f"All finite: {bool(jnp.all(jnp.isfinite(omega_final_ch)))}")

change = jnp.max(jnp.abs(omega_final_ch[1:-1, 1:-1] - omega_poiseuille[1:-1, 1:-1]))
print(f"Max vorticity change after t=0.5: {float(change):.6e}")

# %% [markdown]
# ### Recover velocity and compare to analytical

# %%
diag_ch = model_ch.diagnose(NSVorticityState(omega=omega_final_ch))

# Extract u along vertical centreline
mid_x = model_ch.grid.Nx // 2
u_centreline = diag_ch.u[1:-1, mid_x]
y_interior = y_ch[1:-1]

# Analytical Poiseuille profile
u_analytical = (G / (2 * nu_channel)) * y_interior * (Ly - y_interior)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(
    u_centreline, y_interior, "o-", color="#2196F3", markersize=4, label="Numerical"
)
axes[0].plot(u_analytical, y_interior, "--", color="#F44336", lw=2, label="Analytical")
axes[0].set_xlabel("u(y)")
axes[0].set_ylabel("y")
axes[0].set_title("Poiseuille velocity profile")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

error_u = jnp.abs(u_centreline - u_analytical)
axes[1].plot(error_u, y_interior, "s-", color="#4CAF50", markersize=4)
axes[1].set_xlabel("|u_num - u_exact|")
axes[1].set_ylabel("y")
axes[1].set_title("Absolute error")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Poiseuille flow verification", fontsize=14)
plt.tight_layout()
plt.show()

rel_error = float(jnp.sqrt(jnp.mean((u_centreline - u_analytical) ** 2)))
print(f"RMS velocity error: {rel_error:.6e}")

# %% [markdown]
# The parabolic profile is well captured. Residual errors come from the finite grid resolution and the wall vorticity approximation (Thom's formula).

# %% [markdown]
# ---
# ## Part B: Taylor-Green Vortex
#
# The [Taylor-Green vortex](https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex) is a decaying flow with an exact solution. The vorticity decays exponentially:
#
# $$
# \omega(x,y,t) = A\,\sin(k_x x)\sin(k_y y)\,e^{-\nu(k_x^2+k_y^2)t}
# $$
#
# where $k_x = \pi/L_x$, $k_y = \pi/L_y$ and $A = k_x + k_y^2/k_x$.

# %%
nx_tg, ny_tg = 64, 64
Lx_tg, Ly_tg = 1.0, 1.0
nu_tg = 0.1

model_tg = IncompressibleNS2D.create(
    nx=nx_tg,
    ny=ny_tg,
    Lx=Lx_tg,
    Ly=Ly_tg,
    nu=nu_tg,
    problem="cavity",
    u_lid=0.0,
)

kx = jnp.pi / Lx_tg
ky = jnp.pi / Ly_tg
decay_rate = nu_tg * (kx**2 + ky**2)

x_tg = jnp.arange(model_tg.grid.Nx) * model_tg.grid.dx
y_tg = jnp.arange(model_tg.grid.Ny) * model_tg.grid.dy
X_tg, Y_tg = jnp.meshgrid(x_tg, y_tg)


def omega_exact(t):
    """Exact Taylor-Green vorticity at time t."""
    A = (kx + ky**2 / kx) * jnp.exp(-decay_rate * t)
    return A * jnp.sin(kx * X_tg) * jnp.sin(ky * Y_tg)


print(f"Decay rate: {decay_rate:.4f}")
print(f"e-folding time: {1 / decay_rate:.4f}")

# %% [markdown]
# ### Integrate and compare

# %%
omega0_tg = omega_exact(0.0)
state0_tg = NSVorticityState(omega=omega0_tg)

t_final = 0.5
ts_tg = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

sol_tg = model_tg.integrate(
    state0_tg,
    t0=0.0,
    t1=t_final,
    dt=0.001,
    saveat=dfx.SaveAt(ts=ts_tg),
)

print(f"All finite: {bool(jnp.all(jnp.isfinite(sol_tg.ys.omega)))}")

# %% [markdown]
# ### Visualize the vortex decay

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
interior = (slice(1, -1), slice(1, -1))
Xi_tg, Yi_tg = X_tg[interior], Y_tg[interior]
vmax = float(jnp.max(jnp.abs(omega0_tg[interior])))

for i, (ax, t_val) in enumerate(zip(axes.flat, ts_tg, strict=False)):
    omega_i = sol_tg.ys.omega[i][interior]
    kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    im = ax.pcolormesh(Xi_tg, Yi_tg, omega_i, **kw)
    ax.set_title(f"t = {float(t_val):.1f}")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("Taylor-Green vortex decay", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Peak vorticity: numerical vs analytical

# %%
peak_num = [
    float(jnp.max(jnp.abs(sol_tg.ys.omega[i][interior]))) for i in range(len(ts_tg))
]
peak_ana = [float(jnp.max(jnp.abs(omega_exact(float(t))[interior]))) for t in ts_tg]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ts_tg, peak_num, "o-", color="#2196F3", label="Numerical")
ax.plot(ts_tg, peak_ana, "--", color="#F44336", label="Analytical")
ax.set_xlabel("Time")
ax.set_ylabel("Peak |omega|")
ax.set_title("Vorticity decay: numerical vs analytical")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

omega_num_final = sol_tg.ys.omega[-1]
omega_ana_final = omega_exact(t_final)
err = omega_num_final[interior] - omega_ana_final[interior]
print(f"L2 error at t={t_final}: {float(jnp.sqrt(jnp.mean(err**2))):.6e}")

# %% [markdown]
# ## Differentiability -- gradient through the NS solver
#
# We compute $\partial \mathcal{E} / \partial \nu$ where $\mathcal{E}$ is the final enstrophy. This tells us how viscosity affects the vortex decay rate.

# %%


@eqx.filter_grad
def grad_enstrophy(m):
    """Gradient of final enstrophy w.r.t. model parameters."""
    s0 = NSVorticityState(omega=omega0_tg)
    sol = m.integrate(
        s0,
        t0=0.0,
        t1=0.2,
        dt=0.001,
        saveat=dfx.SaveAt(t1=True),
    )
    final = NSVorticityState(omega=sol.ys.omega[0])
    return m.diagnose(final).enstrophy


grads = grad_enstrophy(model_tg)
dE_dnu = float(grads.params.nu)
print(f"dEnstrophy/dnu = {dE_dnu:.6f}")
sign = "negative (more viscosity = faster decay)" if dE_dnu < 0 else "positive"
print(f"Sign: {sign}")

# %% [markdown]
# The negative gradient confirms that increasing viscosity reduces the final enstrophy -- the vortex decays faster.

# %% [markdown]
# ## Summary
#
# | Problem | Validation | Key result |
# |---------|------------|------------|
# | Poiseuille flow | Parabolic $u(y)$ profile | Steady state maintained |
# | Taylor-Green vortex | Exponential $\omega$ decay | L2 error $< 10^{-3}$ |
# | Differentiability | $\partial\mathcal{E}/\partial\nu$ | Negative (physical) |
#
# **Congratulations!** You have completed all 13 steps from linear advection to the full incompressible Navier-Stokes equations. The somax model contract -- `vector_field`, `integrate`, `diagnose`, `jax.grad` -- scales from the simplest 1D PDE to a validated 2D fluid solver.
