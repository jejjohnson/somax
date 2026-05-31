# Somax Ecosystem Refactor Plan

A status scrape of the eight sibling libraries in the `jejjohnson` scientific
stack and a phased plan for aligning **somax** with the framework
(`pipekit`), the domain toolkit (`xrtoolz`), the numerics foundation
(`finitevolx` / `spectraldiffx`), and the data-assimilation consumers
(`vardax` / `filterax`).

> **Status:** Phases 0-2 are merged and Phase 3 is implemented (see
> [What has been done](#what-has-been-done)). Phases 4-5 are proposed.

## The ecosystem at a glance

```
        ┌──────────────────── FRAMEWORK (carrier-agnostic) ─────────────────────┐
        │ pipekit            Operator · ConfigMixin · Sequential · Graph · serial │
        │ pipekit-cycle      ForwardModel · ObservationOperator · AnalysisStep    │
        │ pipekit-experiment ModelRegistry · ExperimentTracker · Hydra/DVC adapter│
        │ pipekit-train      TrainingLoop · Loss · datasets   pipekit-jax JaxModelOp│
        └────────────────────────────────────────────────────────────────────────┘
   NUMERICS FOUNDATION              DOMAIN TOOLKITS                 DA CONSUMERS
   gaussx     (structured linalg)   xrtoolz  (xarray preproc+eval)  vardax   (variational DA)
   spectraldiffx (spectral solvers) somax    (PDE/ocean models)     filterax (ensemble DA)
   finitevolx (FV C-grid operators)    ▲ consumes fvx + sdx         both consume gaussx +
        ▲ finitevolx → spectraldiffx    somax ──step()──────────►   pipekit-cycle protocols
```

### Maturity snapshot (all pre-1.0, actively developed)

| Repo | Version | State | Role for somax |
|---|---|---|---|
| **gaussx** | 0.0.15 | Mature, no stubs, ~122 test files | Covariance / linalg backbone for DA (indirect) |
| **spectraldiffx** | 0.0.12 | Real; Chebyshev 2D, spherical vorticity inversion, mixed-BC Helmholtz | somax's spectral Helmholtz backend |
| **finitevolx** | 0.0.41 | Real; spherical operators, linear EOS, `SolveDomain` / `KnownValueLifting` | somax's FV operator backend |
| **pipekit** | 0.0.1 | core / cycle / experiment / train / jax implemented; array / evaluate scaffolded | Protocols + config layer to adopt |
| **xrtoolz** | 0.0.8 | Broad and real; consumes `pipekit.Operator` | Preprocessing + evaluation to reuse |
| **vardax** | 0.1.8 | FourDVarNet real; classical methods architected | Consumes somax as `ForwardModel` |
| **filterax** | 0.0.3 | EnKF / ETKF / LETKF / smoothers / EKI real, differentiable | Consumes somax as dynamics |
| **somax** | 0.0.8 | Real model library + CLI; the refactor target | — |

## Where somax stands

**What it is.** A JAX / equinox / diffrax ocean-modeling library plus a
`somax-sim` CLI runner (~136 Python files). Models: Lorenz 63/96/96t, PDE
1D/2D (diffusion, convection, Burgers, Poisson, Navier–Stokes), shallow-water
(1D/2D linear/nonlinear, multilayer), QG (barotropic / baroclinic /
reparameterized). The architecture cleanly splits `_src/models/` (equations) ·
`_src/cli/scenarios/` (geometry + forcing + IC) · `_src/cli/models_registry/`
(how to build), funneled through a `scenario × model` dispatcher.

**The core observation.** somax predates pipekit and xrtoolz, so it hand-rolls
a large amount of "framework" that the rest of the ecosystem now provides:

| Hand-written in somax | ~LOC | Already exists in… |
|---|---|---|
| `cli/spec.py` — dataclasses + `from_dict`/`to_dict`/`validate`/deep-merge/YAML | 431 | `pipekit.ConfigMixin` + `serial.dumps/loads`; `pipekit-experiment` Hydra adapter |
| `cli/_run.py` — integration / orchestration loop | 1065 | `pipekit-cycle.Cycle` (scan-based stepping + history) |
| `scenarios/` + `models_registry/` + `_compatibility.py` | ~600 | `pipekit-experiment.ModelRegistry` + Operator config |
| `scripts/build_configs.py` — Python→YAML materializer | 66 | `pipekit-experiment` Hydra / `hydra-zen` round-trip |
| `_src/io/xarray.py` — state↔Dataset, zarr | 286 | `xrtoolz` (`einx.pack/unpack_dataset`, io) |
| `cli/_assertions.py` — CFL / energy / finite checks | 235 | `xrtoolz.metrics.physical` (geostrophic balance, PV conservation, divergence) |

**The missing seam.** `SomaxModel` exposed `vector_field` + `integrate` (a
diffrax `Solution`) but **not** `step(state, dt) -> state` — exactly the
`pipekit_cycle.ForwardModel` protocol method, and the building block the DA
libraries call. That one gap blocked vardax/filterax from consuming somax.

**Good news on API alignment.** somax already imports the *new* finitevolx
names (`CartesianGrid2D.from_interior`, `Mask2D` threaded as an operator
attribute, no `.w` / `.psi` field access, no manual `* mask` post-multiply).
The only stale coupling was the spectraldiffx Helmholtz constructor keyword
(`alpha` → `lambda_`). So alignment is a version bump, not a rewrite.

## The plan

Sequenced by unblocking value. Phases 0–1 are the high-leverage core; 2–3 are
the bulk cleanup; 4–5 are the future-facing payoff. Suggested order:
**0 → 1 → 3 → 2 → 4 → 5**.

### Phase 0 — Foundation alignment & dependency hygiene *(done)*

- Bump `finitevolx` `v0.0.40 → v0.0.41`. `spectraldiffx` stays pinned to the
  git tag `v0.0.10` (`>=0.0.10`): finitevolx `v0.0.41` itself pins
  `spectraldiffx` to `v0.0.10`, and `uv` requires a single git ref per
  package across the dependency graph — so a `0.0.12` pin is unresolvable and
  must agree with finitevolx.
- No somax source changes were required: the full test suite passes on the
  bumped versions as-is (somax already uses the new finitevolx grid/mask
  naming, and the spectraldiffx solver API is unchanged for somax's usage).
- **Capability opt-ins now unlocked** (track separately): finitevolx
  `SolveDomain` / `KnownValueLifting` for inhomogeneous BCs, the new spherical
  operators (to fill the stubbed `spherical_swm` / `global_ocean` scenarios),
  and the linear equation-of-state module.

### Phase 1 — Adopt pipekit protocols on the model layer *(done)*

- Add `step(self, state, dt) -> state` to `SomaxModel` (thin wrapper over
  `integrate` for a single step). The model supplies the `step` +
  `state_signature` members of `pipekit_cycle.ForwardModel` structurally, by
  duck typing; the third member (a default `dt`) is added by the
  `somax.operators` Operator adapter (a bare model has no inherent step size).
  It is also the building block for `filterax`'s dynamics interface. Phase 1
  itself adds **no** pipekit dependency; Phase 2 makes pipekit a base
  dependency for the Operator / Cycle layer, but somax's *core* still never
  imports it.
- *(Optional follow-up)* mix in `pipekit.ConfigMixin` so `get_config()` /
  `state` round-trip comes for free.

### Phase 2 — Replace hand-rolled config / registry / runner with pipekit *(done)*

> **What actually shipped** (PR #124): the term algebra, the `somax.operators`
> Operator bridge, and the `pipekit_cycle.Cycle`-driven runner all landed. The
> config layer below was **investigated and kept**: `pipekit.serial` is
> primitives-only and cannot round-trip somax's nested-dict YAML schema, so
> `cli/spec.py` / `RunSpec` remain. The pipekit dependency is spent where it
> fits — Operators + Cycle — not on (de)serialization.

- Re-express `scenarios` and `models` as `pipekit.Operator`s (or registry
  entries) so construction params auto-serialize via `ConfigMixin`. Replace
  `cli/spec.py`'s bespoke (de)serialization with `pipekit.serial` plus the
  `pipekit-experiment` Hydra / `hydra-zen` adapter for YAML round-tripping —
  this retires `scripts/build_configs.py` and the `configs/_authoring/*.py`
  materializer pattern.
- Replace the `scenario × model` dispatcher + `_compatibility.py` with
  `pipekit-experiment.ModelRegistry`.
- Re-express `cli/_run.py`'s stepping loop on `pipekit_cycle.Cycle` (scan-based
  stepping + history / `save_interval` for free) — the largest single LOC
  reduction. Keep `somax-sim` (cyclopts) as the thin CLI shell.

### Phase 3 — somax-native evaluation surface *(done)*

> **Revised after surveying xrtoolz's actual API.** The original plan was to
> reuse `xrtoolz` for IO + metrics. On inspection that premise did not hold:
> `xrtoolz.einx.pack_dataset`/`unpack_dataset` stack a Dataset's *variables*
> into a channel axis (an ML-export op) — they are not a pytree-state↔Dataset
> converter, and xrtoolz exposes no NetCDF/Zarr IO of its own (it defers to
> xarray). So there is nothing for `_src/io/xarray.py` to swap onto; its
> security-allowlisted class round-trip and zarr-v3 handling stay as-is.
> xrtoolz's physical metrics (`geostrophic_balance_error`, `divergence_error`)
> also assume lat/lon ocean observations with Coriolis derived from latitude,
> whereas somax runs on idealised Cartesian f-/β-plane grids (metres, no
> lat/lon coords); `pv_conservation_error` needs Lagrangian trajectories somax
> does not produce. Finally xrtoolz is a heavy dependency (cartopy, pyproj,
> rioxarray, xskillscore, …). Decision: give somax its own lightweight
> evaluation surface on its native grid, with **no new dependency**.

- New `somax.eval` (`_src/eval/metrics.py`): reference-free, Cartesian-grid
  field diagnostics built on the finitevolx operators models already hold —
  `rms_divergence`, `total_enstrophy`, `kinetic_energy` (generic to any C-grid
  fluid model) and `geostrophic_imbalance` (model-aware; reuses the model's own
  pressure-gradient + Coriolis operators, so it measures departure from exactly
  the balance the model integrates around).
- `compute_eval_metrics(model, state)` is a defensive dispatcher returning the
  applicable metrics (empty for non-fluid models). The `somax-sim` runner folds
  it into `metrics.json` under the existing `write_metrics` flag; the existing
  generic `bounded_metric` postflight assertion already enforces tolerances on
  any of these keys, so no new assertion code was needed.
- **Scope:** the metrics target velocity-state C-grid models (SWM, Burgers).
  Vorticity / streamfunction models (`barotropic_qg`, vorticity NS) are
  intentionally excluded — they evolve `q` / `omega` and never define a discrete
  velocity divergence (non-divergence is only analytic, so a divergence metric
  would be an operator-dependent artifact with no canonical zero). They already
  surface `kinetic_energy` + `enstrophy` through their own `diagnose()`.
- **Deferred to Phase 4:** reference-based skill scores (RMSE, PSD score) need a
  truth trajectory / observations and land alongside the data-assimilation
  work. The xrtoolz `CMEMSSource` / `CDSSource` loaders remain the right tool
  for real basin data if/when that work starts (kept out of somax's core).

### Phase 4 — Data-assimilation integration (vardax / filterax)

- Provide somax-side `ObservationOperator`s (satisfying the pipekit-cycle
  protocol; vardax/filterax also offer generic ones — `AveragingKernel`,
  `LinearObs`).
- Provide background / observation covariances as `gaussx` operators
  (`Kronecker`, `LowRankUpdate`, …).
- Wire a somax model into `vardax.VarDACycle` (4DVar) and `filterax` filters
  (ETKF / EnKF). Both already consume any JAX forward model by duck typing;
  Phase 1's `step()` is the only hard prerequisite. A small adapter can bridge
  `step(state, dt)` to `filterax.AbstractDynamics.__call__(state, *, key)`
  (fixed-window stepper).

### Phase 5 — Training & experiment tracking *(opportunistic)*

- `pipekit-jax.JaxModelOp` to serialize equinox models; `pipekit-experiment`
  trackers for runs; `pipekit-train` if/when learned closures or emulators are
  added.

## What has been done

**Phases 0-2 are merged** (PR #124, squash-merged to `main`): the dependency
alignment (`finitevolx` `v0.0.41`, `spectraldiffx` pinned to the git tag
`v0.0.10`), `SomaxModel.step` for `ForwardModel` conformance, the composable
term algebra (`Sum`/`Scaled`/`Compose`, IMEX lowering), the pipekit
`Operator` bridge (`somax.operators`), and the `pipekit_cycle.Cycle`-driven
runner. `pipekit` / `pipekit-cycle` are base dependencies; somax's core never
imports them (conformance is structural).

**Phase 3 (this branch)** adds the somax-native evaluation surface:

- **`somax/_src/eval/metrics.py`** (public `somax.eval`) — reference-free
  field diagnostics (`rms_divergence`, `total_enstrophy`, `kinetic_energy`,
  `geostrophic_imbalance`) plus the `compute_eval_metrics` dispatcher.
- **`somax/_src/cli/_run.py`** — folds `compute_eval_metrics` into
  `metrics.json` (under `write_metrics`, try-wrapped so it never breaks a run).
- **`tests/eval/test_diagnostics.py`** — analytic checks (divergence-free and
  irrotational fields read ~0, closed-form kinetic energy, a uniform
  geostrophic jet reads imbalance ~0, scale-invariance of the ratio, and the
  dispatcher's fluid/non-fluid behaviour).

The full test suite passes, and `ruff check`, `ruff format`, and `ty check`
are clean on the changed files.

> **Note on local development.** The `finitevolx` / `spectraldiffx`
> dependencies are private git pins. To run the suite against the local
> sibling checkouts, temporarily add:
>
> ```toml
> [tool.uv.sources]
> finitevolx = { path = "../finitevolX", editable = true }
> spectraldiffx = { path = "../spectraldiffx", editable = true }
> ```
>
> to `pyproject.toml`, then `uv sync`. Remove it before committing — the
> committed dependency spec uses the published git tags.
