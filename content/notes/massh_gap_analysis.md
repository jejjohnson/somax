# somax — Gap Analysis vs. MASSH (VarDyn branch)

**Scope:** models / primitives only. Reference: `leguillf/MASSH@VarDyn`, cores
`mapping/models/model_qg1l/jqgm.py` (QG1L), `mapping/models/model_qgsw/sw.py` (SW),
`mapping/models/model_sw1l/jswm.py` (SW + internal tides). **Date:** 2026-06-05,
re-verified 2026-09-15 against somax 0.0.14.

Self-contained: math, MASSH reference (`file:line`), proposed `somax` API in the *actual*
`models_registry` style, worked example. Mapped against the live registry
(`somax/_src/cli/models_registry/__init__.py`):
`linear_swm`, `nonlinear_swm`, `barotropic_qg`, `multilayer_nonlinear_swm`, `multilayer_qg`,
`reparam_multilayer_qg`, `spherical_swm`, `spherical_qg`.

```{note}
**Provenance of references.** The somax-side claims (registry mechanism, model contract,
finitevolX operators) in this doc have been **verified against the repo on `main`**
(2026-06-05, re-checked 2026-09-15 on 0.0.14, with line numbers refreshed) and carry concrete
API names. The MASSH `file:line` citations are taken from the upstream `VarDyn` branch and
have **not** been re-verified here — treat them as pointers, not exact line guarantees.

Since the first draft, two things landed that this note now accounts for: the reduced-order
forcing basis (`somax/_src/core/basis.py`, 0.0.12), which changes the shape of the
internal-tide proposal in §2, and the data-assimilation bridges (`somax.da`), which make the
adjoint test in §3 a gate on a path that is now exercised for real. Issue
[#141](https://github.com/jejjohnson/somax/issues/141) was also filed from a second pass over
MASSH and covers two *models* this note does not: the equivalent-barotropic SSH-keyed QG and
the internal-tide shallow water with open boundaries. §2 here overlaps with the second and
defers to #141 for the model; this note keeps the forcing-level framing.
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
  equation. `nonlinear_2d.py:176` does `dh_dt = self.advection(h, u, v, method=self.method)`
  where `self.advection` is `finitevolx.Advection2D(grid=grid, mask=mask)` (imported as
  `FVXAdvection2D`, with `mask: Mask2D | None`) and `method` selects the reconstruction
  (`"upwind1"`, `"weno5"`, `"wenoz5"`, …). **The same operator advects a tracer
  `c`** — `self.advection(c, u, v, method="weno5")`. (Note: `method=` is a `__call__` arg,
  *not* a constructor arg.) Crucially, this call returns the **flux-form** tendency
  `−∇·(c\,u) = −(u·∇c) − c\,(∇·u)`, so it *already contains* the spurious `c·(∇·u)` term; the
  fix is to **add it back** (`+ c·∇·u`), exactly as MASSH does
  (`dt_c_fluxdiv + c_phys*vel_div_area`).
- **Divergence** for the `c·(∇·u)` correction: `finitevolx.divergence_2d(u, v, dx, dy)` (free
  function) or `finitevolx.Divergence2D(grid)(u, v)` / `Difference2D.divergence` (all real).
- **WENO reconstruction** is real and rich. `Advection2D(...).__call__(..., method=...)`
  accepts `"upwind1"` / `"upwind2"` / `"upwind3"`, `"weno3"` / `"weno5"` / `"weno7"` /
  `"weno9"` and `"wenoz5"` (verified against the 0.0.14 pin), which is the idiomatic route
  and avoids hand-wiring WENO. The TVD limiters (`minmod`, `van_leer`, `superbee`, `mc`) also
  exist, but **not** as `method=` strings: they are reached through
  `Reconstruction2D.tvd_x(h, u, limiter="minmod")` / `tvd_y`, one level below `Advection2D`.
  (The low-level pieces also live at `finitevolx.weno_3pts/5pts/…`, but the public,
  masking-aware path is `Advection2D`.)

```{warning}
**`rayleigh_relaxation` does not exist in finitevolX.** An earlier draft referenced
"the new `rayleigh_relaxation`" op for `forcing_tracer_from_bc`; `dir(finitevolx)` has no such
symbol (the only "relaxation" string is an unrelated internal in `multigrid.py`). Tracer-to-BC
nudging must be written in somax — a one-line relaxation term `−(c − c_bc)/τ` added in the
tracer tendency — or contributed upstream first. somax's `forcing.py` (`ForcingProtocol`,
`ConstantForcing`, `NoForcing`, `SeasonalWindForcing`, `InterpolatedForcing`) is the right
pattern to model the nudging target on, and `basis.py` now has the seam that was missing when
this was first written: `ForcingTerm` / `add_to` lift any `ForcingProtocol` field onto a state
component as a tendency, so the nudging target can enter the tracer equation without a
bespoke hook.
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
    # The two per-state conventions every 0.0.14 model declares, so the
    # nondimensional path (StateAffine.from_scales reads scale_kinds) and the
    # mask threading (mask_locations picks the stagger) know about ``c``:
    scale_kinds: ClassVar[dict[str, str]] = {"c": "tracer"}
    mask_locations: ClassVar[dict[str, str]] = {"c": "h"}

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
# (ModelEntry also takes an optional ``from_nondimensional`` builder since
# 0.0.14; leave it None until the tracer model has a scale set.)
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
`tau0 * cos(omega*t + phase)` with a *learnable* amplitude `tau0` and static `omega`.

**What changed since the first draft.** The multi-constituent generalization this section
originally proposed as a new class now exists as a composition of things in
`somax/_src/core/basis.py` (the reduced-order forcing basis, 0.0.12):

- `FourierInTime(freqs, phases)` is a cosine temporal gate `b_a(t) = cos(ω_a t + φ_a)` over
  any number of atoms — its docstring names the one-mode case as `SeasonalWindForcing`.
- `BasisForcing(coeffs, spatial, temporal)` pairs that gate with a fixed spatial dictionary
  `Phi (Ngrid, m)` and a learnable coefficient vector `coeffs (m,)`, which is the DA control.
- `ForcingTerm` / `add_to` lift the resulting field onto a state component as a tendency, the
  seam that lets a `ForcingProtocol` reach a model RHS at all.

The tidal body forcing `Σ_ω [a_ω(x) cos ωt − b_ω(x) sin ωt]` is exactly a `BasisForcing`: two
atoms per constituent (phases `0` and `π/2`) tiled against the spatial dictionary that holds
the `a_ω` / `b_ω` patterns, with `coeffs` as the learnable amplitudes. So this gap is best
framed as **a builder in `forcing_bank.py` that returns such a `BasisForcing`, plus a
`linear_swm` variant that attaches it through `ForcingTerm`** — not a new forcing class and
not a new model written from scratch.

### Proposed somax API (corrected)

```python
# somax/_src/core/forcing_bank.py — alongside ssh_geostrophic / sss_coastal
def tidal_constituents(
    spatial: SpatialBasis,                 # a_w / b_w patterns as dictionary columns
    omegas: tuple[float, ...],             # M2, S2, K1, ... [rad/s]
) -> BasisForcing:
    """Sum_w Re[(a_w + i b_w) e^{i w t}] as a BasisForcing: FourierInTime with
    phases (0, pi/2) per constituent, tiled against ``spatial`` via tile_in_time."""

# a registry entry mirroring linear_swm, attaching the forcing with ForcingTerm:
INTERNAL_TIDE_SWM = ModelEntry(
    name="internal_tide_swm", family="swm", layers=1, coordinates="cartesian",
    supports=SupportFlags(masks=True, spherical=False, forcing=("tidal",)),
    build=_build,
)
```

Because the dynamics are linear, the tangent-linear is the model itself and the adjoint is
exact — the cheapest possible case for the §3 machinery.

```{note}
**Where #141 goes further.** Issue #141 proposes an `InternalTideSW` model with an equivalent
depth `He`, per-constituent *open-boundary plane-wave* forcing (`compute_IT_2D` /
`_wave_phases` in MASSH) and radiating boundary conditions. The body-forcing framing above is
the part `BasisForcing` already covers; open and relaxation boundary conditions are new
infrastructure and belong to #141. Treat this section as the forcing half of that issue.
```

-----

## 3. Tangent-linear / adjoint models + adjoint test as first-class primitives

### Why

somax is JAX + Equinox, so `jvp` / `vjp` are *available* (and every model's state is an
`eqx.Module` pytree, so AD traverses the full state — verified). But MASSH packages TGL/ADJ as
named methods plus an **adjoint test**, and that packaging is itself a primitive for a
4DVar-baseline library. A learned-correction or hybrid user shouldn't hand-roll the adjoint
plumbing or remember to verify `⟨M dx, y⟩ = ⟨dx, M*y⟩`. **Verified (still true on 0.0.14):
somax has no `adjoint_test`, no TGL/ADJ helper, no `jvp`/`vjp` wrapper** — `jax.grad`
interoperability is noted in `model.py` but nothing packages it.

The stakes are higher than in June. `somax.da` now ships `SomaxForwardModel`, the flat-vector
`ForwardModel` that vardax's `StrongFourDVar` / `IncrementalFourDVar` roll out, and
`tests/da/test_vardax.py` runs a 4DVar fit through it. That is exactly the gradient path an
adjoint test is meant to gate, and it currently has no such gate.

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
`step_adj` it's worth pinning the adjoint method rather than relying on the default, and
documenting the cost. Don't invent a new knob for that: `pipekit_jax.to_diffrax_adjoint`
already maps a pipekit adjoint spec onto `diffrax.RecursiveCheckpointAdjoint` (its
recommended default), `DirectAdjoint`, `BacksolveAdjoint` (flagged there as divergent for
chaotic dynamics — never a default) or `ImplicitAdjoint`, and `truncated_scan` handles the
truncated-BPTT case at the rollout layer. `step_adj` should take that spec. For the linear
`internal_tide_swm` (§2) the adjoint is exact and cheap; for the nonlinear SW/QG cores it is
the usual reverse-mode cost.
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
   a `tidal_constituents` builder returning a `BasisForcing` (`FourierInTime` gate, two atoms
   per constituent) + a `linear_swm` variant that attaches it with `ForcingTerm`. The
   open-boundary and equivalent-depth half of the model is #141's.

## References

- **somax (verified on `main` 2026-06-05, re-verified on 0.0.14 2026-09-15):**
  `somax/_src/cli/models_registry/` (`ModelEntry`, `MODELS` dict, `_build` pattern),
  `somax/_src/core/model.py` (`SomaxModel.vector_field` / `integrate` / `step`),
  `somax/_src/models/swm/nonlinear_2d.py:176` (`Advection2D` scalar advection of `h`),
  `somax/_src/core/forcing.py:43` (`SeasonalWindForcing`), `somax/_src/core/basis.py`
  (`BasisForcing`, `FourierInTime`, `ForcingTerm`, `add_to`), `somax/da.py`
  (`SomaxForwardModel`, `SomaxDynamics`), `finitevolx` (`Advection2D`, `divergence_2d`,
  `Difference2D.divergence`, `weno_5pts` / `Reconstruction2D`, `Mask2D`), `pipekit_jax`
  (`to_diffrax_adjoint`, `truncated_scan`).
- **MASSH (`VarDyn`, unverified line numbers):** `model_qgsw/sw.py` (`advection_tracer`,
  `step_with_tracer`, `step_{tgl,adj}`, `adjoint_test_sw`), `model_qg1l/jqgm.py` (`Qgm_trac`),
  `model_sw1l/jswm.py`; `doc/overview/04_dynamical_models.md`, `13_notes_future_work.md`;
  `config_2022a_4DVARSW.py`.
