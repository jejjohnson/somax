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
# # Lorenz '63 — Your First somax Simulation
#
# The [Lorenz '63 system](https://en.wikipedia.org/wiki/Lorenz_system) is the
# canonical low-dimensional chaotic attractor. It is the simplest model in
# somax and a good starting point for understanding the API.
#
# **What you'll learn:**
#
# 1. How to create a model, state, and run a forward simulation
# 2. How the `SomaxModel` contract works (`vector_field`, `integrate`, `diagnose`)
# 3. How to differentiate through a simulation with `jax.grad`
# 4. How to run an ensemble with `jax.vmap`

# %% [markdown]
# ## Background
#
# The system is defined by three coupled ODEs with parameters
# $\sigma$ (Prandtl number), $\rho$ (Rayleigh number), and
# $\beta$ (geometric factor):
#
# $$
# \frac{dx}{dt} = \sigma(y - x), \qquad
# \frac{dy}{dt} = x(\rho - z) - y, \qquad
# \frac{dz}{dt} = xy - \beta z
# $$
#
# At the standard parameters $(\sigma, \rho, \beta) = (10, 28, 8/3)$ the
# system exhibits deterministic chaos: nearby trajectories diverge
# exponentially, tracing the famous butterfly-shaped strange attractor.

# %%
from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt

from somax.models import L63State, Lorenz63


# %% [markdown]
# ## 1. Create the model
#
# `Lorenz63.create()` builds a model with the standard parameter values.
# The model is an `eqx.Module` — an immutable pytree whose fields
# (including `params`) are visible to `jax.grad` and `jax.jit`.

# %%
model = Lorenz63.create(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
print(model)

# %% [markdown]
# ## 2. Forward simulation
#
# `model.integrate()` wraps `diffrax.diffeqsolve` with automatic
# boundary-condition enforcement. We save the trajectory at dense
# output times to visualize the attractor.

# %%
state0 = L63State(x=jnp.array(1.0), y=jnp.array(1.0), z=jnp.array(1.0))

t0, t1, dt = 0.0, 40.0, 0.01
ts = jnp.arange(t0, t1, dt)

sol = model.integrate(
    state0,
    t0=t0,
    t1=t1,
    dt=dt,
    saveat=dfx.SaveAt(ts=ts),
)

print(f"Trajectory shape: x={sol.ys.x.shape}, y={sol.ys.y.shape}, z={sol.ys.z.shape}")

# %% [markdown]
# ## 3. Visualize the attractor
#
# The classic 3D butterfly and the three time series.

# %%
colors = {
    "trajectory": "#2196F3",
    "start": "#4CAF50",
    "x": "#2196F3",
    "y": "#FF9800",
    "z": "#9C27B0",
}

fig = plt.figure(figsize=(14, 5))

# --- 3D attractor ---
ax3d = fig.add_subplot(121, projection="3d")
ax3d.plot(
    sol.ys.x,
    sol.ys.y,
    sol.ys.z,
    lw=0.4,
    alpha=0.8,
    color=colors["trajectory"],
)
ax3d.scatter(
    [sol.ys.x[0]],
    [sol.ys.y[0]],
    [sol.ys.z[0]],
    s=60,
    c=colors["start"],
    edgecolors="k",
    linewidths=0.5,
    zorder=5,
    label="IC",
)
ax3d.set_xlabel("x")
ax3d.set_ylabel("y")
ax3d.set_zlabel("z")
ax3d.set_title("Lorenz '63 attractor")
ax3d.legend(loc="upper left", fontsize=9)

# --- Time series ---
ax_ts = fig.add_subplot(122)
ax_ts.plot(ts, sol.ys.x, lw=0.8, color=colors["x"], label="x(t)")
ax_ts.plot(ts, sol.ys.y, lw=0.8, color=colors["y"], label="y(t)")
ax_ts.plot(ts, sol.ys.z, lw=0.8, color=colors["z"], label="z(t)")
ax_ts.set_xlabel("Time")
ax_ts.set_ylabel("State")
ax_ts.set_title("Time series")
ax_ts.legend(fontsize=9)
ax_ts.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Diagnostics
#
# `model.diagnose()` computes on-demand quantities from the state.
# For L63 it returns the kinetic energy $E = \tfrac{1}{2}(x^2 + y^2 + z^2)$.

# %%
energy = jax.vmap(model.diagnose)(sol.ys).energy

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(ts, energy, lw=0.8, color="#F44336")
ax.set_xlabel("Time")
ax.set_ylabel("Energy")
ax.set_title("Kinetic energy along the trajectory")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Sensitivity via `jax.grad`
#
# Because the model, state, and solver are all JAX-native, we can
# differentiate a scalar loss through the entire simulation with
# respect to model parameters.
#
# Here we compute $\partial \mathcal{L} / \partial \theta$ where
# $\mathcal{L} = \| x(T) \|^2$ and $\theta = (\sigma, \rho, \beta)$.

# %%
state0_grad = L63State(x=jnp.array(1.0), y=jnp.array(1.0), z=jnp.array(1.0))


@eqx.filter_grad
def compute_grad(model):
    sol = model.integrate(state0_grad, t0=0.0, t1=1.0, dt=0.01)
    return jnp.sum(sol.ys.x**2 + sol.ys.y**2 + sol.ys.z**2)


grads = compute_grad(model)

print(f"dL/d(sigma) = {grads.params.sigma:.4f}")
print(f"dL/d(rho)   = {grads.params.rho:.4f}")
print(f"dL/d(beta)  = {grads.params.beta:.4f}")

# %% [markdown]
# ## 6. Ensemble simulation with `jax.vmap`
#
# Chaotic systems are sensitive to initial conditions. We can explore
# this by running an ensemble of trajectories with small perturbations
# and watching them diverge.

# %%
n_ensemble = 50
key = jrandom.PRNGKey(42)
perturbations = 0.01 * jrandom.normal(key, shape=(n_ensemble, 3))

ensemble_states = L63State(
    x=1.0 + perturbations[:, 0],
    y=1.0 + perturbations[:, 1],
    z=1.0 + perturbations[:, 2],
)

ts_ens = jnp.arange(0.0, 30.0, 0.01)


def integrate_one(state0):
    return model.integrate(
        state0,
        t0=0.0,
        t1=30.0,
        dt=0.01,
        saveat=dfx.SaveAt(ts=ts_ens),
    )


ensemble_sol = eqx.filter_vmap(integrate_one)(ensemble_states)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

for ax, var, name, color in zip(
    axes,
    [ensemble_sol.ys.x, ensemble_sol.ys.y, ensemble_sol.ys.z],
    ["x(t)", "y(t)", "z(t)"],
    [colors["x"], colors["y"], colors["z"]],
    strict=True,
):
    for j in range(n_ensemble):
        ax.plot(ts_ens, var[j], lw=0.3, alpha=0.4, color=color)
    ax.set_xlabel("Time")
    ax.set_ylabel(name)
    ax.set_title(f"Ensemble — {name}")
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f"Ensemble of {n_ensemble} trajectories (perturbation = 0.01)",
    fontsize=13,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# | Concept | somax API |
# |---------|-----------|
# | Create a model | `Lorenz63.create(sigma=10, rho=28, beta=8/3)` |
# | Initial condition | `L63State(x=..., y=..., z=...)` |
# | Forward simulation | `model.integrate(state0, t0, t1, dt, saveat=...)` |
# | Diagnostics | `model.diagnose(state)` |
# | Gradients | `eqx.filter_grad(loss)(model)` |
# | Ensemble | `eqx.filter_vmap(integrate_one)(batch_states)` |
#
# **Next steps:**
#
# - Try the [Lorenz '96](lorenz96_simulation) tutorial for a
#   higher-dimensional chaotic system
# - Explore the [Shallow Water Model](swm_simulation) for your first PDE simulation
