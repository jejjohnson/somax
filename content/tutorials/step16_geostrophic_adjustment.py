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
# # Step 16 --- 2D Geostrophic Adjustment
#
# When a rotating fluid starts out of balance, it adjusts toward
# geostrophic equilibrium by radiating gravity waves. The final
# balanced state has the pressure gradient force matching the
# Coriolis force: $fv = g \partial h / \partial x$.
#
# This tutorial demonstrates:
# - 2D linear shallow water on a rotating f-plane
# - Geostrophic adjustment from a height step
# - The Rossby deformation radius $L_d = \sqrt{gH_0}/f_0$

# %%
import diffrax as dfx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter

from somax.models import LinearShallowWater2D, LinearSW2DState

# %% [markdown]
# ## Model Setup

# %%
nx, ny = 128, 128
Lx = Ly = 1e6  # 1000 km
g, f0, H0 = 9.81, 1e-4, 100.0

model = LinearShallowWater2D.create(
    nx=nx, ny=ny, Lx=Lx, Ly=Ly, g=g, f0=f0, H0=H0, bc="periodic"
)

Ld = float(jnp.sqrt(g * H0) / f0)
print(f"Rossby deformation radius: Ld = {Ld / 1e3:.0f} km")
print(f"Grid spacing: dx = {model.grid.dx / 1e3:.1f} km")

# %% [markdown]
# ## Initial Condition
#
# A smooth height step (tanh profile) across the centre of the domain.
# No initial velocity --- the flow must develop through adjustment.

# %%
x = jnp.arange(model.grid.Nx) * model.grid.dx
y = jnp.arange(model.grid.Ny) * model.grid.dy
X, Y = jnp.meshgrid(x, y)

eta_max = 0.5  # metres
h0 = -eta_max * jnp.tanh((X - Lx / 2.0) / (Lx / 20.0))
u0 = jnp.zeros_like(h0)
v0 = jnp.zeros_like(h0)
state0 = LinearSW2DState(h=h0, u=u0, v=v0)

# %% [markdown]
# ## Integration

# %%
c = float(jnp.sqrt(g * H0))
t_final = 2e5  # ~55 hours (several inertial periods)
n_frames = 150
ts = jnp.linspace(0.0, t_final, n_frames)
dt = 0.4 * model.grid.dx / c

sol = model.integrate(state0, t0=0.0, t1=t_final, dt=dt, saveat=dfx.SaveAt(ts=ts))
print(f"All finite: {bool(jnp.all(jnp.isfinite(sol.ys.h)))}")

# %% [markdown]
# ## Save to xarray

# %%
x_km = np.asarray(x[2:-2]) / 1e3
y_km = np.asarray(y[2:-2]) / 1e3

ds = xr.Dataset(
    {
        "h": (["time", "y", "x"], np.asarray(sol.ys.h[:, 2:-2, 2:-2])),
        "u": (["time", "y", "x"], np.asarray(sol.ys.u[:, 2:-2, 2:-2])),
        "v": (["time", "y", "x"], np.asarray(sol.ys.v[:, 2:-2, 2:-2])),
    },
    coords={
        "time": np.asarray(ts) / 3600.0,
        "x": x_km,
        "y": y_km,
    },
    attrs={"Ld_km": Ld / 1e3, "f0": f0, "H0": H0},
)
ds

# %% [markdown]
# ## Create GIF Animation

# %%
fig, ax = plt.subplots(figsize=(8, 7))
vmax = float(eta_max)
im = ax.pcolormesh(
    ds.x.values,
    ds.y.values,
    ds["h"].isel(time=0).values,
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax,
    shading="auto",
)
plt.colorbar(im, ax=ax, label="h (m)")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.set_aspect("equal")
title = ax.set_title("t = 0.0 h")


def update(frame):
    im.set_array(ds["h"].isel(time=frame).values.ravel())
    title.set_text(f"t = {ds.time.values[frame]:.1f} h")
    return im, title


anim = FuncAnimation(fig, update, frames=n_frames, interval=67, blit=True)
anim.save(
    "step16_geostrophic_adjustment.gif",
    writer=PillowWriter(fps=15),
    dpi=80,
)
plt.close()

# %%
from IPython.display import Image, display

display(Image(filename="step16_geostrophic_adjustment.gif"))

# %% [markdown]
# ## Balanced State Check
#
# After adjustment, the flow near the centre should be close to
# geostrophic balance: small tendencies.

# %%
final_state = LinearSW2DState(h=sol.ys.h[-1], u=sol.ys.u[-1], v=sol.ys.v[-1])
final_state = model.apply_boundary_conditions(final_state)
tendency = model.vector_field(t_final, final_state)
s = (slice(10, -10), slice(10, -10))
print(f"Max |dh/dt| in interior: {float(jnp.max(jnp.abs(tendency.h[s]))):.2e}")
print(f"Max |du/dt| in interior: {float(jnp.max(jnp.abs(tendency.u[s]))):.2e}")

# %% [markdown]
# ## Summary
#
# - An initial height step adjusts to geostrophic balance on the rotating f-plane
# - Gravity waves radiate away, leaving a balanced jet along the step
# - The width of the balanced region scales with the Rossby deformation radius $L_d$
#
# **Next:** [Step 17 --- 2D Nonlinear Shallow Water](step17_shallow_water_2d.py)
