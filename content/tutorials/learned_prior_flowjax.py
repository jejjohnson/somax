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
# # Learned priors — composing a flow onto a physical scaling
#
# A data-assimilation prior has to say which ocean states are plausible. A
# Gaussian says "close to this mean, with this covariance"; a normalising flow
# can say something much richer. But a flow trained directly on a dimensional
# ocean state has to spend its capacity learning that thickness is measured in
# hundreds of metres and velocity in tenths of a metre per second — facts we
# already know.
#
# `somax.flows` lets you hand the flow that knowledge for free. A
# `StateAffine` carries the physical scaling; the flow is composed on top and
# learns only the departure from it.
#
# **Requires the optional extra:** `pip install somax[flows]`, or
# `uv sync --extra flows`.
#
# **What you'll see:**
#
# 1. Building a `StateAffine` from characteristic `Scales`
# 2. Adapting it to a flowjax bijection with `to_flowjax`
# 3. Composing it with a flow into a prior over *physical* states
# 4. Why the log-density picks up a constant from the scaling

# %%
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from somax import Scales, StateAffine
from somax._src.models.swm.nonlinear_2d import NonlinearSW2DState
from somax.flows import learned_prior, to_flowjax


# %% [markdown]
# ## 1. A state and its physical scales
#
# A shallow-water state whose fields sit four orders of magnitude apart:
# thickness around 1000 m, velocities around 0.1 m/s.

# %%
NY, NX = 8, 8
rng = np.random.RandomState(0)

scales = Scales.advective(L=1.0e6, U=0.1, f0=1.0e-4, H=1000.0)
state = NonlinearSW2DState(
    h=jnp.asarray(scales.H + rng.randn(NY, NX)),
    u=jnp.asarray(0.1 * rng.randn(NY, NX)),
    v=jnp.asarray(0.1 * rng.randn(NY, NX)),
)

print(f"h  ~ {float(jnp.mean(state.h)):.1f} m")
print(f"u  ~ {float(jnp.std(state.u)):.3f} m/s")
print(f"geostrophic height scale = {scales.eta:.3f} m")

# %% [markdown]
# ## 2. The transform, as a flowjax bijection
#
# `StateAffine` already exposes `transform_and_log_det`; `to_flowjax` wraps it
# in the flat-array interface flowjax bijections use. The flat layout is the
# same one `somax.da` uses, so a single vector feeds both a filter and a flow.

# %%
transform = StateAffine.from_scales(NonlinearSW2DState, scales)
bijection = to_flowjax(transform, state)

flat, unravel = ravel_pytree(state)
standardised, log_det = bijection.transform_and_log_det(flat)

print(f"flat state size      : {bijection.shape[0]}")
print(f"standardised |h| max : {float(jnp.abs(unravel(standardised).h).max()):.2f}")
print(f"log|det J|           : {float(log_det):.2f}")

# %% [markdown]
# The standardised state is O(1) in every field. The log-determinant is a
# constant — the Jacobian of a fixed affine map does not depend on where you
# evaluate it — and it is exactly what a density has to pick up when it is
# pushed through the scaling.

# %% [markdown]
# ## 3. A prior over physical states
#
# `learned_prior` composes base → flow → *out of* standardised coordinates, so
# the resulting distribution is over physical states while the flow itself
# only ever sees O(1) numbers. Any flowjax bijection works as the flow; a
# trained `masked_autoregressive_flow` is the realistic choice, and `Identity`
# here keeps the arithmetic checkable by hand.

# %%
from flowjax.bijections import Identity
from flowjax.distributions import Normal


base = Normal(jnp.zeros(flat.size))
prior = learned_prior(transform, state, Identity((flat.size,)), base)

draws = jax.vmap(unravel)(prior.sample(jax.random.key(0), (2048,)))
h_mean = float(jnp.mean(draws.h))
u_spread = float(jnp.std(draws.u))
print(f"sampled h mean   : {h_mean:.1f} m   (loc {transform.loc.h:.1f})")
print(f"sampled u spread : {u_spread:.3f} m/s (scale {transform.scale.u:.3f})")

# %% [markdown]
# Samples come out in physical units with the right magnitudes, from a base
# distribution that is standard normal. That is the scaling doing its job.

# %% [markdown]
# ## 4. The density, term by term
#
# For $\phi$ the standardising map,
#
# $$\log p_X(x) = \log p_Y(\phi(x)) + \log|\det J_\phi(x)|.$$
#
# With an identity flow both terms are computable directly, so the
# composition can be checked rather than taken on trust.

# %%
lhs = float(prior.log_prob(flat))
rhs = float(base.log_prob(standardised)) + float(log_det)
print(f"prior.log_prob(x)              = {lhs:.4f}")
print(f"log p_base(phi(x)) + log|det J| = {rhs:.4f}")
assert np.isclose(lhs, rhs, rtol=1e-4)

# %% [markdown]
# ## Why flowjax is optional
#
# flowjax is not a core somax dependency. Its own `Affine` forces the scale
# through a trainable softplus parameterisation, which is the opposite of what
# a *fixed* physical scale wants, and it brings the whole flow stack with it.
# The log-determinant only matters when transforming densities; everywhere
# else in somax — the DA flatten bridge, `ScaledModel`, the nondimensional
# factories — the affine map is used directly and no flow library is involved.
