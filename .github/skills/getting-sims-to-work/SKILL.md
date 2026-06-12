---
name: getting-sims-to-work
description: >
  Ordered methodology for bringing up a somax ocean simulation (SWM / QG,
  single- and multi-layer) so it runs stably without wasting compute or
  tokens. Use whenever a model is failing to integrate, blowing up
  (NaN/Inf), running too slowly, or when configuring a new double-gyre /
  wind-driven run. Encodes the resolution↔forcing↔viscosity↔dt consistency
  constraints, a fail-fast observability-first workflow, and the
  start-small-then-scale loop.
---

# Getting somax simulations to work

A simulation that blows up or runs for ten minutes before crashing is a
**process** failure, not bad luck. Follow this order. Each step is cheap and
catches a whole class of failure before the expensive step after it. **Do not
reorder** — instrumenting after a blow-up, or scaling resolution before fixing
forcing, is how you burn an afternoon (and a token budget).

The golden rule, stated once:

> For a chosen grid spacing `dx`, the lateral viscosity must satisfy **both**
> `A_h ≥ β·(N·dx)³` (resolve the Munk western-boundary layer with `N≈2–3`
> points) **and** `A_h ≥ |u|_max·dx/2` (grid Reynolds number ≲ 2). Then scale
> the **wind-stress amplitude** so the Sverdrup-driven `|u|_max` keeps the
> second inequality true. Over-forcing at fixed resolution → faster jet →
> under-resolved boundary layer → grid-scale noise → blow-up.

---

## Step 0 — Observability first (instrument BEFORE you run)

Never launch a run you can't diagnose. somax already ships the tooling — use it.

- **Fail-fast guards inside the RHS** (`somax/_src/guards.py`, built on
  `eqx.error_if`, raise *immediately* mid-step, not at the end of the jit
  region):
  - `guard_finite(x, where=...)` — abort on any NaN/Inf.
  - `guard_positive(h, where=...)` — abort on non-positive layer thickness
    (the #1 instability mode in layered/SWM models).
  The multilayer SWM RHS already calls these; keep them on while bringing a
  model up.
- **Post-step finiteness** as the minimum smoke check:
  `bool(jnp.all(jnp.isfinite(sol.ys.<field>)))`. `finite=False` ⇒ it blew up;
  everything else is secondary until that is `True`.
- **Conservation diagnostics** via `model.diagnose(state)` — at least
  `kinetic_energy` and `enstrophy` (QG) / energy, enstrophy, mass per layer
  (SWM). Interpretation:
  - inviscid + unforced test → should be **flat to round-off**; monotone drift
    = a bug or excess numerical dissipation (validate the Arakawa
    energy/enstrophy-conserving Jacobian this way).
  - forced–dissipative run → should reach a **statistical steady balance**
    (input ≈ dissipation). Unbounded growth = under-dissipated/unstable; rapid
    decay = over-diffusive or `A_h` too large.
- **Preflight / postflight assertions** (`somax/_src/cli/_assertions.py`,
  driven by the `assertions:` block in configs):
  `check_cfl`, `check_deformation_radius`, `check_pv_inversion`,
  `check_static_stability`, `check_bounded_metric`. Run `run_preflight(spec,
  model)` before integrating — it catches a bad `dt`, an under-resolved
  deformation radius, or `g'<0` (statically unstable layering) *before* the
  scan.
- **Real-model precedent** (mirror these thresholds for ad-hoc runs): MOM6
  self-stops on `ntrunc > MAXTRUNC` or energy > `max_energy`; NEMO aborts on
  `|ssh|>20 m`, `|U|>10 m/s`, `S∉(0,100)` or NaN and dumps the crashing
  `(i,j,k)`; MITgcm prints `advcfl_*_max` and `cg2d_init_res` per
  `monitorFreq` (`cg2d_res=NaN` is the classic blow-up signature).

## Step 1 — Start small, and probe cost before committing

Compile cost and run cost are different. Measure before you wait.

- **Smallest meaningful grid first.** Use the config `debug:` block (`32×32`,
  `t1=86400`) or `nx=ny=32–64`. A double gyre is qualitatively right at 64²;
  128²+ is for the final figure, not for debugging.
- **Timing probe** — warm up the jit once (block on the result to force
  compilation), then time a short window and extrapolate **before** the long
  run:

  ```python
  import time, jax
  t = time.time()
  sol = model.integrate(s0, t0=0.0, t1=60*86400.0, dt=dt,
                        saveat=dfx.SaveAt(ts=jnp.array([60*86400.0])), max_steps=200_000)
  jax.block_until_ready(sol.ys.q)
  n = int(60*86400.0/dt); el = time.time()-t
  print(f"{el/n*1e3:.2f} ms/step → 1 yr ≈ {el/n*31557600/dt:.0f}s")
  ```

  At 64² a QG step is ~2.5 ms ⇒ 1 yr ≈ 65 s. **128² × 3 yr is minutes** — never
  your first run. If `ms/step × total_steps` is more than a couple of minutes,
  shrink the grid or shorten `t1` until the smoke test is < ~30 s.
- **Scale up only after green:** 32²/64² smoke (finite + sane diagnostics) →
  full-resolution figure run. Optionally spin up coarse, then interpolate the
  balanced state to fine resolution so the fine run only has to grow the eddies.
- **Checkpoint the expensive result BEFORE any post-processing.** Never let a
  plot/diagnostic crash a long integration: the instant the solve returns, write
  the arrays (`np.savez` / zarr) to disk, *then* compute figures from the saved
  arrays in a separate cheap step. A one-character matplotlib/LaTeX typo
  (`\tfrac` is not valid mathtext) at the end of an 8-minute run is otherwise a
  total loss. Corollary: run long jobs **unbuffered** (`python -u` and/or
  `print(..., flush=True)`) so the diagnostics survive a crash, and re-plot from
  the `.npz` rather than re-integrating.
- **Run from the project root** (`cd /home/user/somax`) so `uv run` finds the
  env — "`No module named 'diffrax'`" / "`--no-sync` outside of a project" means
  wrong cwd, not a code bug.

## Step 2 — Make forcing & dissipation consistent with the resolution

This is where most blow-ups are *born* (and where this skill's author burned
tokens by setting `wind_amplitude=1e-9`, 1000× the stable value). Pin these
**before** touching `dt`.

- **Resolve the boundary layer → minimum viscosity.** Munk layer
  `δ_M = (A_h/β)^(1/3)`; Stommel layer `δ_S = r/β` (`r` = bottom drag). Require
  the layer to span `N≈2–3` cells ⇒ `A_h ≥ β·(N·dx)³` and `r ≥ β·N·dx`.
- **Grid Reynolds number ≲ 2** for centered advection: `Re_Δ = |u|_max·dx/A_h`,
  so `A_h ≥ |u|_max·dx/2`. Coarser `dx` or stronger jets force larger `A_h`.
- **Scale the forcing to the grid.** Interior speed is Sverdrup-set,
  `β·V = curl(τ)/ρ`, so `|u|_max` grows ~linearly with wind-stress amplitude.
  For fixed `dx, A_h`, too-large `τ₀` → faster jet → `Re_Δ > 2` and `δ_M`
  under-resolved → grid-scale noise → blow-up. **Find the stable forcing band by
  a short scan** (vary `wind_amplitude` over decades for ~180 days, keep the
  largest value that stays finite with a visible circulation) rather than
  guessing big.
- **Prefer scale-selective dissipation at higher res.** Biharmonic
  `A_4 ≈ U_d·dx³/8` (Griffies–Hallberg; `A_4` scales like `dx³`) damps the grid
  scale while leaving the mesoscale near-inviscid. somax's QG advection already
  uses the energy/enstrophy-conserving **Arakawa Jacobian**; SWM/thickness
  advection uses **upwind/FVX** (implicitly limited). Note: explicit flux
  limiters / biharmonic are **not yet a public knob** in somax — the available
  dissipation is harmonic `lateral_viscosity` + the conserving/upwind advection.
  Use those; if a run needs limiters, that's a model-code change, not a config.

## Step 3 — Time-stepping & integrator (pick `dt` from the binding limit)

Compute *every* limit, take `dt = safety · min(limits)`, `safety ≈ 0.2–0.5`.

- **External gravity wave (SWM, the stiff one):** `c = √(gH)` (~200 m/s for
  `H~4 km`); `dt < CFL·dx/(|u|+c)`. This is almost always binding for a
  free-surface SWM ⇒ tiny `dt` (multilayer SWM: `dt~20 s` at 64²).
- **Internal/baroclinic wave (multilayer):** `c_n = √(g'_n H_n)` (~1–3 m/s) —
  ~100× slower. A **reduced-gravity / rigid-lid / QG** formulation removes the
  external mode, so `dt` can be ~100× larger (QG runs at `dt~600 s`).
- **Advective:** `dt < dx/|u|_max` — binding for QG and strong jets.
- **Viscous:** harmonic `dt < dx²/(2·d·A_h)` (`d`=dims; 2-D ⇒ `dx²/(4A_h)`);
  biharmonic `dt < dx⁴/(π⁴·A_4)` — refining `dx` cuts biharmonic `dt` by 16×.
- **Rossby:** `c_R = β·L_d²` (cm/s) — sets the slowest signal, rarely binding.
- **Integrator:** somax uses `diffrax` (default `Tsit5`, `ConstantStepSize`).
  Keep `ConstantStepSize` for conservation/diagnostics; an adaptive
  `PIDController` is the documented fix if an IMEX/implicit solver is used (see
  `imex_solver` / `imex_stepsize_controller`). The config `cfl` assertion
  (`wave_speed_m_per_s`, `max_cfl=0.5`) is your guardrail — set
  `wave_speed_m_per_s` to the model's fastest mode (`√(gH)` for SWM; the small
  advective/`2.0 m/s` value committed for QG).

## Step 4 — Initialization & spin-up

- **Prefer a balanced (geostrophic) IC.** Starting from rest hits the model with
  a geostrophic-adjustment burst of inertia-gravity waves; if you start at rest,
  ramp the forcing rather than applying full `τ` at `t=0`.
- **Budget the spin-up to the physics:** barotropic adjustment = days–weeks;
  first-baroclinic Rossby basin crossing ≈ **3 yr** (subtropical gyre) + ~5 yr
  for mode-2; eddy/mean equilibration = years–decades. Don't call it "spun up"
  until the diagnostic of interest (often eddy KE) plateaus. For a *figure*, a
  1–2 yr barotropic / a few-yr baroclinic run shows the gyre; full equilibration
  is a different (DVC-pipeline) job.

## Step 5 — When it blows up: bisect via `dt`

- **Halve `dt` and rerun.**
  - Survives / lasts much longer ⇒ **CFL / numerical** problem: find the binding
    limit (Step 3), lower `dt`, or raise `A_h` / switch to a stiffer solver.
  - Still dies at the **same model time** regardless of `dt` ⇒ **not CFL**.
    Suspect forcing too strong (Step 2), a bad IC/BC, `g'<0` layering
    (`check_static_stability`), or an under-resolved boundary layer.
- **Localize** with the guards: the first `where=` that fires, and `max|u|` /
  thickness-positivity, point at the offending field/cell.

## Step 6 — Differentiable (JAX / diffrax) concerns

- **Separate compile vs run cost** (Step 1 probe). Build the solve as one
  jitted program; the time loop is a `lax.scan`, not a Python `for`.
- **Gradients through rollouts:** diffrax's default `RecursiveCheckpointAdjoint`
  (≈O(log n) memory, *exact* grads) is almost always right; reach for
  `BacksolveAdjoint` only for true O(1) memory (approximate grads). Verify
  `eqx.filter_grad` of a short rollout is finite before differentiating a long
  one.
- **Long chaotic rollouts** give exploding/NaN gradients (positive Lyapunov
  exponents). Mitigate with shorter windows (truncated BPTT), checkpointing, and
  gradient clipping.

---

## somax quick reference

**API (all 5 ocean models):**
`Model.create(...)` → `StateClass(...)` → `model.integrate(state0, t0, t1, dt,
saveat=dfx.SaveAt(ts=...), max_steps=...)` → `model.diagnose(state)`. QG
streamfunction for the gyre plot: `model._invert_pv(q)`.

**Known-stable double-gyre starting points** (from `configs/simulation/`,
verified to integrate finite):

| Model | class | grid | key params | dt |
|---|---|---|---|---|
| Barotropic QG | `BarotropicQG` | 64² / 1e6 m | `ν=500, drag=1e-7, wind≤1e-11` (`1e-10` blows up) | 600–1200 s |
| Baroclinic / ML QG | `BaroclinicQG` | 128² / 4e6 m, 3 layers `H=(400,1100,2600)`, `g'=(9.81,.025,.0125)` | `ν=15, drag=1e-7, wind=1.3e-10` | 600 s |
| Multilayer SWM | `MultilayerShallowWater2D` | 64² / 1e6 m, 2 layers `H=(500,4500)`, `g'=(9.81,.025)` | `ν=1000, drag=1e-7` (jet IC) | **20 s** (√gH CFL) |
| Single-layer SWM | `NonlinearShallowWater2D` | 64² | `H0`, `ν`, `wind`, `bc` | √gH CFL |
| Reparam ML QG | `ReparameterizedQG` | like ML QG, **`bc="wall"`** | `ν=15, drag=1e-7` | 600 s |

**The minimal smoke-test, every time:** small grid → timing probe → run short
→ assert `finite=True` and sane `diagnose()` → only then scale up. If you change
`dx`, re-check Step 2 (viscosity/forcing) and Step 3 (`dt`) — they are coupled.

## Sources

CFL / time-stepping / filters: Durran, *Numerical Methods for Fluid Dynamics*
(2010); Vallis, *AOFD* (2017); Williams (2009/2011) RAW filter; MITgcm / ROMS /
MOM6 / NEMO docs. Resolution↔forcing↔viscosity: Munk (1950); Stommel (1948);
Pedlosky; Cushman-Roisin & Beckers; Griffies & Hallberg (2000) biharmonic;
Arakawa & Lamb (1977). Observability / spin-up / autodiff: MITgcm/MOM6/NEMO
monitor packages; Price *A Coriolis Tutorial Pt. 4* (WHOI); diffrax adjoint docs
(Kidger); Kochkov et al. (2021, JAX-CFD); APEBench / "Stabilizing BPTT for
Physics" (2024).
