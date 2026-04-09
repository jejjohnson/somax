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
# # Step 19 --- Baroclinic Quasi-Geostrophic (Multilayer Double Gyre)
#
# The baroclinic (multilayer) QG model extends the barotropic model to
# multiple vertical layers, enabling baroclinic instability and a richer
# representation of mesoscale ocean dynamics.
#
# PV inversion decouples via vertical mode decomposition:
#
# $$q_k^{(m)} = C_{l2m} \cdot q_k^{(l)}$$
# $$(\nabla^2 - f_0^2 \lambda_m)\psi_m = q_m \quad \text{per mode}$$
# $$\psi_k^{(l)} = C_{m2l} \cdot \psi_k^{(m)}$$
#
# This tutorial demonstrates:
# - 3-layer QG with modal PV inversion
# - Wind-driven double gyre with baroclinic structure
# - Per-layer diagnostics (PV, streamfunction, velocity)
# - Rossby deformation radii from vertical mode decomposition

# %%
import diffrax as dfx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter

from somax.core import StratificationProfile
from somax.models import BaroclinicQG, BaroclinicQGState

# %% [markdown]
# ## Model Setup
#
# A 3-layer ocean basin following the louity/MQGeometry configuration:
# - Layer thicknesses: 400 m, 1100 m, 2600 m
# - Reduced gravities: 0.025, 0.0125 m/s^2
# - Beta-plane with mid-latitude parameters
# - Wind forcing on the top layer only

# %%
nx, ny = 64, 64
Lx = Ly = 4e6  # 4000 km basin

model = BaroclinicQG.create(
    nx=nx,
    ny=ny,
    Lx=Lx,
    Ly=Ly,
    f0=9.375e-5,
    beta=1.754e-11,
    n_layers=3,
    H=(400.0, 1100.0, 2600.0),
    g_prime=(9.81, 0.025, 0.0125),
    lateral_viscosity=50.0,
    bottom_drag=1e-7,
    wind_amplitude=8e-5,
    wind_profile="doublegyre",
)

# Print stratification info
print(f"Layers: {model.consts.n_layers}")
print(f"Layer thicknesses (m): {np.asarray(model.strat.H)}")
print(f"Rossby radii (km): {np.asarray(model.modal.rossby_radii) / 1e3}")
print(f"Helmholtz lambdas: {np.asarray(model.helmholtz_lambdas)}")

# %% [markdown]
# ## Initial Condition: Ocean at Rest

# %%
q0 = jnp.zeros((3, model.grid.Ny, model.grid.Nx))
state0 = BaroclinicQGState(q=q0)

# %% [markdown]
# ## Integration
#
# Spin up the model under wind forcing. The wind injects energy into
# the surface layer, which then cascades to deeper layers via baroclinic
# instability.

# %%
t_final = 5e7  # ~1.6 years
n_frames = 200
ts = jnp.linspace(0.0, t_final, n_frames)
dt = 500.0

sol = model.integrate(
    state0,
    t0=0.0,
    t1=t_final,
    dt=dt,
    saveat=dfx.SaveAt(ts=ts),
    max_steps=200_000,
)
print(f"All finite: {bool(jnp.all(jnp.isfinite(sol.ys.q)))}")

# %% [markdown]
# ## Diagnostics

# %%
final_state = BaroclinicQGState(q=sol.ys.q[-1])
diag = model.diagnose(final_state)
print(f"KE per layer: {np.asarray(diag.kinetic_energy)}")
print(f"Total KE: {float(diag.total_kinetic_energy):.4e}")
print(f"Total enstrophy: {float(diag.total_enstrophy):.4e}")
print(f"Rossby radii (km): {np.asarray(diag.rossby_radii) / 1e3}")

# %% [markdown]
# ## Save to xarray

# %%
x_km = np.asarray(jnp.arange(model.grid.Nx) * model.grid.dx)[2:-2] / 1e3
y_km = np.asarray(jnp.arange(model.grid.Ny) * model.grid.dy)[2:-2] / 1e3
layer_names = ["upper", "middle", "lower"]

ds = xr.Dataset(
    {
        "q": (
            ["time", "layer", "y", "x"],
            np.asarray(sol.ys.q[:, :, 2:-2, 2:-2]),
        ),
    },
    coords={
        "time": np.asarray(ts) / 86400.0,
        "layer": layer_names,
        "x": x_km,
        "y": y_km,
    },
)

# Compute psi at each snapshot
psi_all = []
for i in range(n_frames):
    psi_i = model._invert_pv(sol.ys.q[i])
    psi_all.append(np.asarray(psi_i[:, 2:-2, 2:-2]))
ds["psi"] = (["time", "layer", "y", "x"], np.stack(psi_all))
ds

# %% [markdown]
# ## Create GIF: Upper-layer PV, Streamfunction, Speed
#
# Three panels showing the upper-layer dynamics:
# - PV anomaly (the prognostic variable)
# - Streamfunction (from PV inversion)
# - Speed (geostrophic velocity magnitude)

# %%
# Compute speed at each snapshot for the upper layer
speed_all = []
for i in range(n_frames):
    psi_i = model._invert_pv(sol.ys.q[i])
    u_i = -model.diff.diff_y_T_to_V(psi_i[0])
    v_i = model.diff.diff_x_T_to_U(psi_i[0])
    u_T = model.interp.V_to_T(u_i)
    v_T = model.interp.U_to_T(v_i)
    spd = np.asarray(jnp.sqrt(u_T**2 + v_T**2)[2:-2, 2:-2])
    speed_all.append(spd)
ds["speed_upper"] = (["time", "y", "x"], np.stack(speed_all))

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

q_data = ds["q"].sel(layer="upper")
psi_data = ds["psi"].sel(layer="upper")
spd_data = ds["speed_upper"]

q_max = float(np.percentile(np.abs(q_data.values[-1]), 95))
if q_max < 1e-15:
    q_max = 1e-10
psi_max = float(np.percentile(np.abs(psi_data.values[-1]), 95))
if psi_max < 1e-6:
    psi_max = 1.0
spd_max = float(np.percentile(spd_data.values[-1], 95))
if spd_max < 1e-10:
    spd_max = 1e-5

im0 = axes[0].pcolormesh(
    ds.x,
    ds.y,
    q_data.isel(time=0),
    cmap="RdBu_r",
    vmin=-q_max,
    vmax=q_max,
    shading="auto",
)
im1 = axes[1].pcolormesh(
    ds.x,
    ds.y,
    psi_data.isel(time=0),
    cmap="RdBu_r",
    vmin=-psi_max,
    vmax=psi_max,
    shading="auto",
)
im2 = axes[2].pcolormesh(
    ds.x,
    ds.y,
    spd_data.isel(time=0),
    cmap="magma",
    vmin=0,
    vmax=spd_max,
    shading="auto",
)
plt.colorbar(im0, ax=axes[0], label="PV anomaly (1/s)")
plt.colorbar(im1, ax=axes[1], label="streamfunction (m^2/s)")
plt.colorbar(im2, ax=axes[2], label="speed (m/s)")
for ax in axes:
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal")
axes[0].set_title("Upper Layer PV")
axes[1].set_title("Upper Layer Streamfunction")
axes[2].set_title("Upper Layer Speed")
title = fig.suptitle("t = 0 days", fontsize=14)
plt.tight_layout()


def update(frame):
    im0.set_array(q_data.isel(time=frame).values.ravel())
    im1.set_array(psi_data.isel(time=frame).values.ravel())
    im2.set_array(spd_data.isel(time=frame).values.ravel())
    title.set_text(f"t = {ds.time.values[frame]:.0f} days")
    return im0, im1, im2, title


anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
anim.save(
    "step19_baroclinic_qg.gif",
    writer=PillowWriter(fps=20),
    dpi=80,
)
plt.close()

# %%
from IPython.display import Image, display

display(Image(filename="step19_baroclinic_qg.gif"))

# %% [markdown]
# ## Layer Comparison: Final Streamfunction
#
# The upper layer shows strong western boundary currents, while deeper
# layers have weaker, broader circulation patterns.

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, layer in zip(axes, layer_names, strict=True):
    psi_final = ds["psi"].sel(layer=layer).isel(time=-1)
    lim = float(np.percentile(np.abs(psi_final.values), 95))
    if lim < 1e-6:
        lim = 1.0
    cs = ax.contourf(
        ds.x.values,
        ds.y.values,
        psi_final.values,
        levels=20,
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
    )
    plt.colorbar(cs, ax=ax, label="$\\psi$ (m$^2$/s)")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal")
    ax.set_title(f"{layer.capitalize()} layer streamfunction")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Differentiability

# %%
import equinox as eqx


@eqx.filter_grad
def grad_fn(m):
    sol = m.integrate(state0, t0=0.0, t1=1e6, dt=dt)
    return jnp.sum(sol.ys.q**2)


grads = grad_fn(model)
print(f"d(loss)/d(viscosity)      = {float(grads.params.lateral_viscosity):.6e}")
print(f"d(loss)/d(bottom_drag)    = {float(grads.params.bottom_drag):.6e}")
print(f"d(loss)/d(wind_amplitude) = {float(grads.params.wind_amplitude):.6e}")

# %% [markdown]
# ## Alternative: Build from StratificationProfile

# %%
strat = StratificationProfile.from_N2_constant(N2=1e-5, depth=4000.0, n_layers=5)
print("5-layer stratification:")
print(f"  H = {np.asarray(strat.H)}")
print(f"  g_prime = {np.asarray(strat.g_prime)}")
print(f"  rho = {np.asarray(strat.rho)}")

model_5L = BaroclinicQG.create(
    nx=32,
    ny=32,
    Lx=4e6,
    Ly=4e6,
    stratification=strat,
    lateral_viscosity=50.0,
    wind_amplitude=8e-5,
)
print(f"  Rossby radii (km): {np.asarray(model_5L.modal.rossby_radii) / 1e3}")

# %% [markdown]
# ## Summary
#
# - The baroclinic QG model extends the barotropic model with vertical modes
# - PV inversion decouples into independent Helmholtz problems per mode
# - Wind forcing drives upper-layer circulation that cascades to depth
# - The `StratificationProfile` class enables flexible vertical configurations
# - All parameters remain differentiable through the modal inversion
