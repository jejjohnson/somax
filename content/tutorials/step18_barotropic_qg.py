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
# # Step 18 --- Barotropic Quasi-Geostrophic (Double Gyre)
#
# The barotropic QG model filters out fast gravity waves, retaining only
# the slow, large-scale balanced dynamics governed by potential vorticity.
# The classic double-gyre circulation driven by wind stress curl
# produces western boundary intensification (the Gulf Stream analogue).
#
# This tutorial demonstrates:
# - PV advection via the Arakawa Jacobian (energy + enstrophy conserving)
# - Poisson PV inversion (DST spectral solver)
# - Wind-driven double-gyre circulation

# %%
import diffrax as dfx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter

from somax.models import BarotropicQG, BarotropicQGState

# %% [markdown]
# ## Model Setup
#
# A square basin (1000 km) with:
# - Double-gyre wind stress curl: $\text{curl}(\tau) \sim -\sin(2\pi y / L_y)$
# - Lateral viscosity for numerical stability
# - Bottom drag for energy dissipation
# - Beta-plane ($\beta = 1.6 \times 10^{-11}$) for Rossby wave propagation

# %%
nx, ny = 64, 64
Lx = Ly = 1e6

model = BarotropicQG.create(
    nx=nx,
    ny=ny,
    Lx=Lx,
    Ly=Ly,
    f0=1e-4,
    beta=1.6e-11,
    lateral_viscosity=500.0,
    bottom_drag=1e-7,
    wind_amplitude=1e-12,
    wind_profile="doublegyre",
)

# %% [markdown]
# ## Initial Condition: PV at Rest

# %%
q0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
state0 = BarotropicQGState(q=q0)

# %% [markdown]
# ## Integration
#
# We spin up the model for a long time to reach a statistical
# steady state. The wind forcing injects energy, bottom drag and
# viscosity dissipate it.

# %%
t_final = 5e7  # ~1.6 years
n_frames = 200
ts = jnp.linspace(0.0, t_final, n_frames)
dt = 500.0  # time step (seconds)

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
final_state = BarotropicQGState(q=sol.ys.q[-1])
diag = model.diagnose(final_state)
print(f"Final KE: {float(diag.kinetic_energy):.4e}")
print(f"Final enstrophy: {float(diag.enstrophy):.4e}")

# %% [markdown]
# ## Save to xarray

# %%
x_km = np.asarray(jnp.arange(model.grid.Nx) * model.grid.dx)[2:-2] / 1e3
y_km = np.asarray(jnp.arange(model.grid.Ny) * model.grid.dy)[2:-2] / 1e3

ds = xr.Dataset(
    {
        "q": (["time", "y", "x"], np.asarray(sol.ys.q[:, 2:-2, 2:-2])),
    },
    coords={
        "time": np.asarray(ts) / 86400.0,  # days
        "x": x_km,
        "y": y_km,
    },
    attrs={"beta": 1.6e-11, "wind_amplitude": 1e-12},
)

# Compute psi and vorticity at each snapshot
from finitevolx import Difference2D

diff = Difference2D(grid=model.grid)
psi_all, zeta_all = [], []
for i in range(n_frames):
    psi_i = model._invert_pv(sol.ys.q[i])
    zeta_i = diff.curl(
        -diff.diff_y_T_to_V(psi_i),  # u = -dpsi/dy
        diff.diff_x_T_to_U(psi_i),  # v = dpsi/dx
    )
    psi_all.append(np.asarray(psi_i[2:-2, 2:-2]))
    zeta_all.append(np.asarray(zeta_i[2:-2, 2:-2]))
ds["psi"] = (["time", "y", "x"], np.stack(psi_all))
ds["vorticity"] = (["time", "y", "x"], np.stack(zeta_all))
ds

# %% [markdown]
# ## Create GIF: PV, Streamfunction, Vorticity

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Colour ranges from final frame
q_max = float(np.percentile(np.abs(ds["q"].values[-1]), 95))
if q_max < 1e-15:
    q_max = 1e-10
psi_max = float(np.percentile(np.abs(ds["psi"].values[-1]), 95))
if psi_max < 1e-6:
    psi_max = 1.0
zeta_max = float(np.percentile(np.abs(ds["vorticity"].values[-1]), 95))
if zeta_max < 1e-15:
    zeta_max = 1e-10

im0 = axes[0].pcolormesh(
    ds.x,
    ds.y,
    ds["q"].isel(time=0),
    cmap="RdBu_r",
    vmin=-q_max,
    vmax=q_max,
    shading="auto",
)
im1 = axes[1].pcolormesh(
    ds.x,
    ds.y,
    ds["psi"].isel(time=0),
    cmap="RdBu_r",
    vmin=-psi_max,
    vmax=psi_max,
    shading="auto",
)
im2 = axes[2].pcolormesh(
    ds.x,
    ds.y,
    ds["vorticity"].isel(time=0),
    cmap="RdBu_r",
    vmin=-zeta_max,
    vmax=zeta_max,
    shading="auto",
)
plt.colorbar(im0, ax=axes[0], label="PV anomaly (1/s)")
plt.colorbar(im1, ax=axes[1], label="streamfunction (m^2/s)")
plt.colorbar(im2, ax=axes[2], label="vorticity (1/s)")
for ax in axes:
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal")
axes[0].set_title("Potential Vorticity")
axes[1].set_title("Streamfunction")
axes[2].set_title("Relative Vorticity")
title = fig.suptitle("t = 0 days", fontsize=14)
plt.tight_layout()


def update(frame):
    im0.set_array(ds["q"].isel(time=frame).values.ravel())
    im1.set_array(ds["psi"].isel(time=frame).values.ravel())
    im2.set_array(ds["vorticity"].isel(time=frame).values.ravel())
    title.set_text(f"t = {ds.time.values[frame]:.0f} days")
    return im0, im1, im2, title


anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
anim.save(
    "step18_barotropic_qg.gif",
    writer=PillowWriter(fps=20),
    dpi=80,
)
plt.close()

# %%
from IPython.display import Image, display

display(Image(filename="step18_barotropic_qg.gif"))

# %% [markdown]
# ## Final Streamfunction
#
# The double-gyre pattern with western boundary intensification.

# %%
fig, ax = plt.subplots(figsize=(8, 7))
cs = ax.contourf(
    ds.x.values, ds.y.values, ds["psi"].isel(time=-1).values, levels=20, cmap="RdBu_r"
)
plt.colorbar(cs, ax=ax, label="$\\psi$")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.set_aspect("equal")
ax.set_title("Streamfunction (final)")
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
# ## Summary
#
# - The barotropic QG model captures large-scale ocean circulation
# - Wind-driven double-gyre develops western boundary intensification
# - The Arakawa Jacobian conserves energy and enstrophy
# - All parameters (viscosity, drag, wind amplitude) are differentiable
#
# This concludes the **18 Steps to Navier-Stokes** tutorial series.
