# somax — Gap Analysis vs. MASSH (VarDyn branch)

**Scope:** models / primitives only. Reference: `leguillf/MASSH@VarDyn`, cores
`mapping/models/model_qg1l/jqgm.py` (QG1L), `mapping/models/model_qgsw/sw.py` (SW),
`mapping/models/model_sw1l/jswm.py` (SW + internal tides). **Date:** 2026-06-05.

Self-contained: math, MASSH reference (`file:line`), proposed `somax` API in the *actual*
`models_registry` style, worked example. Mapped against the live registry
(`somax/_src/cli/models_registry/__init__.py`):
`linear_swm`, `nonlinear_swm`, `barotropic_qg`, `multilayer_nonlinear_swm`, `multilayer_qg`,
`reparam_multilayer_qg`, `spherical_swm`, `spherical_qg`.

```{note}
**Provenance of references.** The somax-side claims (registry mechanism, model contract,
finitevolX operators) in this doc have been **verified against the repo on `main`**
(2026-06-05) and carry concrete API names. The MASSH `file:line` citations are taken from the
upstream `VarDyn` branch and have **not** been re-verified here — treat them as pointers, not
exact line guarantees.
```

-----

## 0. Not a gap

`multilayer_qg`, `reparam_multilayer_qg`, `nonlinear_swm`, `multilayer_nonlinear_swm`,
`barotropic_qg`, `linear_swm` — plus the spherical pair `spherical_swm` / `spherical_qg` —
already cover MASSH's *dynamical* repertoire (QG1L, multilayer QG, single- and multi-layer SW).
MASSH adds nothing on the pure-dynamics axis somax doesn't have. The gaps are
(1) **passive-tracer transport coupled to the flow**, (2) **internal-tide forcing**,
(3) **TGL/ADJ + adjoint test as first-class, packaged primitives**.

```{important}
**Two conventions this doc corrects relative to an earlier draft.** somax has no
`@register_model` decorator and no `model.init()` / `model.rollout()` API. Models are
**`SomaxModel` (`eqx.Module`) subclasses** constructed by a `create()` classmethod, integrated
through **diffrax** (`vector_field` + `integrate`/`step`), and surfaced to the CLI via a
**`ModelEntry` dataclass** registered in the `MODELS` dict. All proposed APIs below follow that
real contract. (The scenario × model decomposition — issue #72 — is what introduced the
`ModelEntry` / `ScenarioBundle` indirection; new models plug into it.)
```

-----

## 1. Passive-tracer advection coupled to the dynamical core  *(largest gap)*

### Why

The single biggest model-level gap, and directly on the roadmap (the land/ice/SST–SSS
ambitions; "differentiable baselines for learned corrections"). MASSH carries SST/SSS as
passive tracers advected by the QG or SW flow, fully AD-compatible, so the tracer fields are
part of the 4DVar control. somax has the dynamical cores but **no tracer-transport model
riding on top** (verified: no model State carries a concentration/tracer field).

### Math

A passive concentration `c` (°C, PSU) obeys `∂_t c = −u·∇c` (advective form). The subtlety
MASSH documents: if you naively use the flux-divergence form `∂_t(hc) = −∇·(huc)` and divide
out, you get

```{math}
\partial_t c = -(u\cdot\nabla c) - c\,(\nabla\cdot u)
```

The second term is ~0 for QG (non-divergent) but **not** for SW, where it injects a spurious
source/sink at velocity gradients (jets, fronts) — e.g. negative salinity in convergence
zones. The fix is to keep the monotone WENO upwind reconstruction of `c` at faces but
explicitly subtract back the `c·(∇·u)` term, recovering pure advection while preserving
spatial constants exactly.

### MASSH reference

`mapping/models/model_qgsw/sw.py:507` — `advection_tracer` (core logic, paraphrased):

```python
c_phys = c_area / self.area
c_flux_y = area_y * self.h_flux_y(c_phys, V_surf[..., 1:-1])     # WENO upwind at faces
c_flux_x = area_x * self.h_flux_x(c_phys, U_surf[..., 1:-1, :])
dt_c_fluxdiv = -div_nofluxbc(c_flux_x, c_flux_y)                 # flux-div form (has spurious term)
vel_div_area = div_nofluxbc(area_x*U_surf[...,1:-1,:], area_y*V_surf[...,1:-1])
return (dt_c_fluxdiv + c_phys*vel_div_area) * self.masks.h       # cancel c·div(u) → pure advection
```

Two more notes MASSH bakes in and somax should replicate:

- **float32 stability**: reconstruct on `c_phys` (O(30)), not area-scaled `c_area` (~3e9), or
  WENO-Z smoothness weights (β²~1e38) overflow float32. **This warning is directly relevant:
  somax runs in single precision by default** — it does *not* call
  `jax.config.update("jax_enable_x64", True)` at import, so `jnp.zeros(1).dtype == float32`.
- **Diffusion** (`sw.py:573`, `add_tracer_diffusion`): `κ∇²c_phys` on the h-grid.

Tracer is advected by **surface (layer-0) velocity** for all tracer layers
(`U_surf = U[:,0:1]`). The integration is `step_with_tracer` (`sw.py:1299`); the AD wrappers
mirror the dynamics ones. For QG1L the analogue is `Qgm_trac`
(`doc/overview/04_dynamical_models.md` §4a), with options `ageo_velocities` (include
ageostrophic velocity) and `forcing_tracer_from_bc` (nudge tracer to BC).

### What somax can reuse (verified)

- **Scalar advection by a velocity field** already exists and is exercised by the SWM mass
  equation. `nonlinear_2d.py:161` does `dh_dt = self.advection(h, u, v, method=self.method)`
  where `self.advection` is `finitevolx.Advection2D(grid, mask)` and `method` selects the
  reconstruction (`"upwind1"`, `"weno5"`, `"wenoz5"`, …). **The same operator advects a tracer
  `c`** — `self.advection(c, u, v, method="weno5")`. (Note: `method=` is a `__call__` arg,
  *not* a constructor arg.) Crucially, this call returns the **flux-form** tendency
  `−∇·(c\,u) = −(u·∇c) − c\,(∇·u)`, so it *already contains* the spurious `c·(∇·u)` term; the
  fix is to **add it back** (`+ c·∇·u`), exactly as MASSH does
  (`dt_c_fluxdiv + c_phys*vel_div_area`).
- **Divergence** for the `c·(∇·u)` correction: `finitevolx.divergence_2d(u, v, dx, dy)` (free
  function) or `finitevolx.Divergence2D(grid)(u, v)` / `Difference2D.divergence` (all real).
- **WENO reconstruction** is real and rich — `method="weno3/5/7/9"`, `"wenoz5"`, plus TVD
  limiters (`"minmod"`, `"van_leer"`, `"superbee"`, `"mc"`) — reached through
  `Advection2D(...).__call__(..., method=...)`, which is the idiomatic route and avoids
  hand-wiring WENO. (The low-level pieces also live at `finitevolx.weno_3pts/5pts/…` and an
  internal `_src/advection/weno.py`, but the public, masking-aware path is `Advection2D`.)

```{warning}
**`rayleigh_relaxation` does not exist in finitevolX.** An earlier draft referenced
"the new `rayleigh_relaxation`" op for `forcing_tracer_from_bc`; `dir(finitevolx)` has no such
symbol (the only "relaxation" string is an unrelated internal in `multigrid.py`). Tracer-to-BC
nudging must be written in somax — a one-line relaxation term `−(c − c_bc)/τ` added in the
tracer tendency — or contributed upstream first. somax's `forcing.py` (`ForcingProtocol`,
`ConstantForcing`, `InterpolatedForcing`) is the right pattern to model the nudging target on.
```

### Proposed somax API (corrected to the real contract)

A `SomaxModel` subclass that *wraps* an existing dynamical core and adds tracer state, plus a
`ModelEntry` in the registry — **not** a `@register_model` decorator, **not** `init/rollout`,
and time-stepping via diffrax (`vector_field`), **not** a hand-rolled SSP-RK3:

```python
# somax/_src/models/swm/tracer.py
class TracerSWMState(State):           # eqx.Module pytree -> jvp/vjp traverse it
    h: Float[Array, "Ny Nx"]
    u: Float[Array, "Ny Nx"]
    v: Float[Array, "Ny Nx"]
    c: Float[Array, "n_trac Ny Nx"]    # passive tracers (e.g. SST, SSS)

class TracerSWM(SomaxModel):
    """nonlinear (or multilayer) SW + N passive tracers advected by surface velocity.

    Tracers advected in pure-advective (constant-preserving) form, optional
    Laplacian diffusion. AD-compatible end to end (the eqx.Module state carries c).
    Mirrors MASSH sw.py::advection_tracer.
    """
    swm: NonlinearShallowWater2D       # the wrapped dynamical core
    advection: FVXAdvection2D
    diff: Difference2D
    diff_coef_trac: float = eqx.field(static=True, default=0.0)

    def vector_field(self, t, state, args=None) -> TracerSWMState:
        dyn = self.swm.vector_field(t, state, args)          # (dh, du, dv)
        # Advection2D returns the FLUX-form tendency -div(c*u) = -(u.grad c) - c*(div u);
        # add back c*(div u) to recover pure advection (constant-preserving), per MASSH.
        div_u = self.diff.divergence(state.u, state.v)
        adv = vmap(lambda c: self.advection(c, state.u, state.v, method="weno5")
                             + c * div_u)(state.c)
        dc = adv + self.diff_coef_trac * vmap(self.diff.laplacian)(state.c)
        return TracerSWMState(h=dyn.h, u=dyn.u, v=dyn.v, c=dc)

    @staticmethod
    def create(..., n_trac=1, diff_coef_trac=0.0) -> "TracerSWM": ...

# somax/_src/cli/models_registry/tracer_swm.py
def _build(scenario, params): ...      # -> BuiltModel(model=TracerSWM.create(...), state0=...)
TRACER_SWM = ModelEntry(
    name="tracer_swm", family="swm", layers=1, coordinates="cartesian",
    supports=SupportFlags(masks=True, spherical=False, forcing=("tau_x", "tau_y")),
    build=_build,
)
# then add "tracer_swm": TRACER_SWM to MODELS in models_registry/__init__.py
```

A QG variant (`tracer_qg`, wrapping `multilayer_qg` / `barotropic_qg`) can **skip the
`c·(∇·u)` correction** (QG is non-divergent) and optionally add ageostrophic velocity.

### Example (real API)

```python
from somax.models import TracerSWM
model = TracerSWM.create(nx=128, ny=128, n_trac=2, diff_coef_trac=100.0)  # SST, SSS
# state0 built via the registry's _build, or constructed directly as TracerSWMState(...)
sol = model.integrate(state0, t0=0.0, t1=5*86400.0, dt=300.0)   # diffrax; AD-differentiable in c0
# or: state1 = model.step(state0, dt)
```

### Validation

Constant tracer field stays constant (the point of the correction); closed-basin tracer
variance non-increasing under pure advection; adjoint test on the tracer path (§3 —
`adjoint_test` traverses the `c` field for free because the state is an `eqx.Module` pytree).

-----

## 2. Single-layer SW with internal-tide forcing

### Why

Lower priority for the stated scope, but a genuine MASSH model somax lacks, relevant if
internal-tide / SSH disentangling enters scope (a live altimetry problem). The
`model_sw1l/jswm.py` core (~1201 LOC) is a single-layer SW with baroclinic-tide generation.

### Math

Linearized single-layer SW with prescribed multi-constituent tidal forcing. Per constituent
`ω` (M2, S2, K1, …), the forcing enters momentum as oscillatory body forcing

```{math}
F_\omega(x,t) = \mathrm{Re}\!\left[\,(a_\omega(x) + i\,b_\omega(x))\,e^{i\omega t}\,\right],
```

and the inverse problem solves for the spatial amplitude fields `(a_ω, b_ω)`. The model is
otherwise linear SW (`∂_t u = −g∇η − f×u + F`, `∂_t η = −H∇·u`), which makes its TGL trivially
itself.

### MASSH reference

`mapping/models/model_sw1l/jswm.py` — the `Swm` class with per-frequency forcing amplitudes as
control parameters. (The 4DVar examples `config_2022a_4DVARSW.py` and `…4DVARQGSW.py` exercise
it.)

### What somax can reuse (verified)

somax already has `linear_swm` (the linear single-layer SW core) **and an oscillatory body
forcing**: `somax/_src/core/forcing.py:43` `SeasonalWindForcing` produces
`tau0 * cos(omega*t + phase)` with a *learnable* amplitude `tau0` and static `omega`. The
internal-tide forcing is the **multi-constituent generalization** of exactly this object:
a sum over `ω` of `Re[(a_ω + i b_ω) e^{iωt}]` with learnable spatial `(a_ω, b_ω)` fields. So
this gap is best framed as **a new `ForcingProtocol` subclass + a thin `linear_swm` variant
that consumes it**, not a new model written from scratch.

### Proposed somax API (corrected)

```python
# somax/_src/core/forcing.py
class TidalForcing(ForcingProtocol):
    """Multi-constituent oscillatory body forcing: Sum_w Re[(a_w + i b_w) e^{i w t}].
    The per-constituent amplitude fields (a_w, b_w) are differentiable controls for 4DVar."""
    omegas: tuple[float, ...] = eqx.field(static=True)     # constituent frequencies
    a: Float[Array, "n_omega Ny Nx"]                       # learnable
    b: Float[Array, "n_omega Ny Nx"]                       # learnable
    def __call__(self, t):
        return jnp.sum(self.a * jnp.cos(...) - self.b * jnp.sin(...), axis=0)

# a registry entry mirroring linear_swm but with forcing=TidalForcing:
INTERNAL_TIDE_SWM = ModelEntry(
    name="internal_tide_swm", family="swm", layers=1, coordinates="cartesian",
    supports=SupportFlags(masks=True, spherical=False, forcing=("tidal",)),
    build=_build,
)
```

Because the dynamics are linear, the tangent-linear is the model itself and the adjoint is
exact — the cheapest possible case for the §3 machinery.

-----

## 3. Tangent-linear / adjoint models + adjoint test as first-class primitives

### Why

somax is JAX + Equinox, so `jvp` / `vjp` are *available* (and every model's state is an
`eqx.Module` pytree, so AD traverses the full state — verified). But MASSH packages TGL/ADJ as
named methods plus an **adjoint test**, and that packaging is itself a primitive for a
4DVar-baseline library. A learned-correction or hybrid user shouldn't hand-roll the adjoint
plumbing or remember to verify `⟨M dx, y⟩ = ⟨dx, M*y⟩`. **Verified: somax has no
`adjoint_test`, no TGL/ADJ helper, no `jvp`/`vjp` wrapper today** — `jax.grad`
interoperability is noted in `model.py` but nothing packages it.

### Math

For a (possibly nonlinear) step `M`, the tangent-linear `M' = ∂M/∂x` and its adjoint `M*`
satisfy the duality identity, for all perturbations `dx` and cotangents `y`:

```{math}
\langle\, M'(x)\,dx ,\; y \,\rangle \;=\; \langle\, dx ,\; M^{*}(x)\,y \,\rangle .
```

The adjoint test checks this to round-off on random masked vectors — the standard correctness
gate before trusting 4DVar gradients.

### MASSH reference

`sw.py:1266` `step_tgl` and `sw.py:1283` `step_adj` (thin `jvp`/`vjp` wrappers around `step`);
`sw.py:1498` `adjoint_test_sw` (the verification harness: small masked random vectors, checks
`⟨M dx, y⟩ == ⟨dx, M* y⟩`).

### Proposed somax API (matches the real `step(state, dt)` contract)

Free functions over any `SomaxModel` (whose `step(state, dt)` already exists), in a new
`somax/_src/models/_adjoint.py`, re-exported from `somax.models`:

```python
# somax/_src/models/_adjoint.py
def step_tgl(model, state, dstate, dt):
    """Tangent-linear: jax.jvp of the model's diffrax step."""
    _, dy = jax.jvp(lambda s: model.step(s, dt), (state,), (dstate,))
    return dy

def step_adj(model, state, cotangent, dt):
    """Adjoint: jax.vjp of the model's diffrax step applied to a cotangent."""
    _, vjp = jax.vjp(lambda s: model.step(s, dt), state)
    return vjp(cotangent)[0]

def adjoint_test(model, state, dt, *, key, scale=1e-4, atol=1e-6) -> float:
    """Verify <M' dx, y> == <dx, M* y> on random pytree perturbations.
    Returns the relative residual; assert < atol in CI. MASSH: sw.py:1498."""
```

Because the states are Equinox pytrees, `jvp` / `vjp` traverse the full state (including the
`c` tracers from §1) automatically — so `adjoint_test` doubles as the **tracer-path adjoint
check** MASSH lists as future work.

```{note}
**Caveat on differentiating `step`.** somax's `step` integrates with diffrax (default
`Tsit5`, adaptive). Differentiating through `diffeqsolve` is supported, but for a packaged
`step_adj` it's worth pinning the adjoint method (`diffrax.RecursiveCheckpointAdjoint` or
`BacksolveAdjoint`) rather than relying on the default, and documenting the cost. For the
linear `internal_tide_swm` (§2) the adjoint is exact and cheap; for the nonlinear SW/QG cores
it is the usual reverse-mode cost.
```

### Example

```python
from somax.models import adjoint_test, step_adj
res = adjoint_test(model, state0, dt, key=jax.random.PRNGKey(0))
assert res < 1e-6                                  # gate before running 4DVar
dstate_T = step_adj(model, state0, cotangent, dt)  # gradient seed
```

-----

## Suggested ordering for somax

1. **`adjoint_test` + `step_tgl` / `step_adj`** (§3) — small, pure-JAX, no finitevolX
   dependency, and it's the correctness gate the 4DVar baselines depend on. Ships as free
   functions over the existing `step` contract.
2. **`tracer_swm` / `tracer_qg`** (§1) — the big one; reuses the existing `FVXAdvection2D`
   scalar-advection path + `Difference2D.divergence`. The constant-preservation correction
   (`+ c·∇·u`) and the in-somax nudging term are the only model-specific logic. (Do **not**
   wait on a finitevolX `rayleigh_relaxation` op — it doesn't exist.)
3. **`internal_tide_swm`** (§2) — only if internal-tide / SSH work enters scope; best done as
   a `TidalForcing(ForcingProtocol)` + a `linear_swm` variant, generalizing the existing
   `SeasonalWindForcing`.

## References

- **somax (verified on `main`, 2026-06-05):** `somax/_src/cli/models_registry/` (`ModelEntry`,
  `MODELS` dict, `_build` pattern), `somax/_src/core/model.py` (`SomaxModel.vector_field` /
  `integrate` / `step`), `somax/_src/models/swm/nonlinear_2d.py:161`
  (`Advection2D` scalar advection of `h`), `somax/_src/core/forcing.py:43`
  (`SeasonalWindForcing`), `finitevolx` (`Advection2D`, `divergence_2d`,
  `Difference2D.divergence`, `weno_5pts` / `Reconstruction2D`).
- **MASSH (`VarDyn`, unverified line numbers):** `model_qgsw/sw.py` (`advection_tracer`,
  `step_with_tracer`, `step_{tgl,adj}`, `adjoint_test_sw`), `model_qg1l/jqgm.py` (`Qgm_trac`),
  `model_sw1l/jswm.py`; `doc/overview/04_dynamical_models.md`, `13_notes_future_work.md`;
  `config_2022a_4DVARSW.py`.
