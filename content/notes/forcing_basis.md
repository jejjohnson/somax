# Forcing Basis Functions

A design note for a reduced-order, differentiable forcing API in somax, assembled from external basis primitives (geonnax) rather than reimplemented here. It introduces a single `BasisForcing` that pairs a differentiable coefficient vector with a fixed spatial dictionary, a temporal gate, and a prior — and, crucially, the small **adapter** that lets such a forcing enter a somax model's right-hand side at all.

> **Status:** Proposed, with a landed vertical slice **and** the geonnax-facing builders wired in. This note was fact-checked against the somax tree (`core/forcing.py`, `core/terms.py`, `core/model.py`, `domain/domain.py`, `da/vardax_bridge.py`) and the pinned sibling repos. Several integration points the design needs **do not exist yet** and are called out explicitly below rather than assumed. The **dependency-free core** of the design shipped in `somax/_src/core/basis.py` (re-exported from `somax.core`): `SpatialBasis`, `TemporalBasis` / `ConstantInTime` / `FourierInTime`, `BasisForcing`, `TransformedForcing`, the `ForcingTerm` seam + `add_to`, and `control_filter` — covered by `tests/core/test_basis.py`, including the QG bit-for-bit parity (TODOs 1, 2, 3, 8 below). The **geonnax-facing layer** then shipped in `somax/_src/core/forcing_bank.py` (also re-exported from `somax.core`): `spatial_from_gabor` / `spatial_from_rbf` evaluate the public geonnax frames on `Domain.coords`, `GaussianWindowsInTime` wraps `gaussian_window_features`, `tile_in_time` builds the separable space-time frame, and the `ssh_geostrophic` / `sss_coastal` presets compose them (TODOs 4, 6 below, modulo the spectral presets — see next paragraph). Still open: the spectral-eigenbasis presets (HSGP Fourier, graph-Laplacian — blocked on public geonnax API, below), the Weaver–Courtier basis, and the weak-constraint vardax integration. The prior layer is a separate concern (companion note `content/notes/forcing_basis_flow_prior.md`, not yet written).

> **geonnax dependency — landed pin and public-vs-private.** `geonnax` is now a real, git-pinned somax dependency (`geonnax @ git+…@de70e81`, the 0.0.4 release commit). The forcing bank uses **only the public `geonnax.basis` surface**: `gabor_frame_grid`, `rbf_basis`, `wendland_c2/c4`, `gaussian_window_features`, `seasonal_features`. The **spectral eigenbases** the design also names — `fourier_basis` / `fourier_eigenvalues_1d` (box-Laplacian) and `graph_laplacian_eigpairs` — currently live in geonnax's *private* `geonnax._basis` namespace and are deliberately left out of `geonnax.basis.__all__`. Rather than import a private namespace, the HSGP-Fourier and graph-Laplacian presets are **deferred** until those primitives are promoted to a public API (a small geonnax change: add them as convenience re-exports in `geonnax.basis`, as the Gabor/RBF frames already are). Confirmed return contracts as of the pin: `gabor_frame_grid(x, bounds, *, n_scales, base_scale, oversample) -> (Phi (N,M), centers (M,d), scales (M,), wavenumbers (M,))` with `bounds` a build-time-concrete `(d, 2)` `[lo, hi]` box; `rbf_basis(x, centers (M,d), widths (M,), *, kernel) -> Phi (N,M)`; `gaussian_window_features(t (N,), centers (M,), widths (M,)) -> (N, M)`; `fourier_basis(x, num_basis_per_dim, L) -> (Phi, lam)`; `graph_laplacian_eigpairs(A, num_basis, *, normalized) -> (eigvals (M,), eigvecs (V,M))`.

> **What is real today vs. what this adds.** `core/forcing.py` defines `ForcingProtocol`, `ConstantForcing`, `NoForcing`, `SeasonalWindForcing` (learnable `tau0`), and `InterpolatedForcing` — but **no somax model currently consumes a `ForcingProtocol`.** The QG models bake a static `wind_forcing: Float[Array, "Ny Nx"]` times a learnable scalar `tau0` directly into the tendency (`qg/barotropic.py:159`, `qg/baroclinic.py:196`); the shallow-water models apply wind to the top layer directly. The model RHS is the **term algebra** (`core/terms.py`): a `Term` is `(t, state, args) -> tendency`, assembled by `build_diffrax_terms` into a `diffrax` solve. So this note is **not** "extend `forcing.py` without changing anything"; it is "add the missing seam (`ForcingTerm`) that lifts a `ForcingProtocol` field onto a state component as a tendency," and then build the reduced-order forcing on top of it. The existing `tau0 * wind_forcing` becomes the genuine, testable special case.

## 1. User story

As an ocean data-assimilation researcher running weak-constraint 4D-Var for SSH, I want to parameterise the model error in a reduced multiscale frame and solve only for its coefficients, so that the control is small and well-conditioned. **Caveat (see §9):** the existing `vardax_bridge` is a strong-constraint, autonomous forward-model adapter; the weak-constraint coefficient control is new work, not a free reuse.

As a modeller, I want to drive a shallow-water run with a seasonal-plus-tidal wind without hand-building the field, so that I write a few frequencies and a small amplitude vector instead of a full space-time array. This is the **static-coefficient** case and works through the existing `jax.grad` parameter path today (it is exactly the `tau0` pattern, generalised).

As someone assimilating sea surface salinity near river mouths, I want forcing localised where the physics is localised, so that I place radial basis functions on the coast rather than spending degrees of freedom on the open ocean.

As someone working with ocean colour, I want to model the forcing in log space because chlorophyll is lognormal, so that the field stays positive and the prior is applied where the variable is near-Gaussian.

In every case the goal is one object that produces a forcing **field** from a small differentiable **coefficient vector**, lifted into a model's RHS by a single adapter and differentiated by `jax.grad`.

## 2. Motivation

`forcing.py` already encodes the right backbone. `SeasonalWindForcing` is a single-mode temporal Fourier basis with a learnable amplitude, and `InterpolatedForcing` is the "interpolator is the temporal basis" idea expressed through `diffrax` paths. What is missing is (a) a first-class basis-expansion forcing that separates the reduced control you solve for from the basis that expands it into a field and carries the prior the variational cost needs, and (b) the adapter that connects *any* `ForcingProtocol` to the term-based RHS.

The key design choice is where the basis math lives, and the answer is: not here. geonnax is intended to be a pure-function basis zoo — Fourier and box-Laplacian eigenpairs, spherical harmonics, Slepian caps, graph-Laplacian eigenpairs, and (with the localized-bases work) a Wendland RBF basis, an overcomplete Gabor frame, and a Gaussian-window temporal feature. The prior over coefficients lives in the prior layer (pyrox's Hilbert-space-GP / sparse-spectrum, or gauss_flows' learned whitening). somax sits on top: it evaluates a geonnax basis on its `Domain`, attaches a prior, adds the temporal gate, lifts the result onto a state component, and hands it to `diffrax`. somax implements **one** basis of its own — the diffusion-operator (Weaver–Courtier) smoother, which is tied to the model grid and operators and therefore cannot be a pure geonnax function.

This separation matters for three reasons. The full space-time model error has $N_t N_s$ degrees of freedom, intractable and ill-posed to solve for directly; a basis reduces it to $m \ll N_t N_s$ coefficients. A choice of basis is a choice of covariance, $Q = \Phi \Lambda \Phi^\top$, so the basis is where the physics of the model error lives. And the per-variable structure of ocean fields means the right basis differs by variable, which is what the bank of presets provides.

> **Dependency reality.** `geonnax`, `pyrox`, and `gauss_flows` are **not** somax dependencies today — they appear nowhere in `pyproject.toml` or the source. The pinned siblings are `diffrax`, `finitevolx`, `vardax`, `gaussx`, `spectraldiffx`. Adopting this design means adding these as real, version-pinned dependencies and committing to the exact return contracts somax relies on (§8). Until they are pinned and their signatures confirmed, the geonnax-facing builders below are specifications, not working code. somax also does **not** currently use `einx` (it is a gaussx/finitevolx convention); the contractions here are written in plain `jax.numpy` to match somax.

## 3. Math background

The weak-constraint dynamics is an ODE whose right-hand side is the model operator plus a model-error forcing, both tendencies:

$$
\frac{\partial y}{\partial t}(\mathbf{x}, t) = \mathcal{F}[y](\mathbf{x}, t) + \varepsilon(\mathbf{x}, t)
$$

We expand the forcing in a space-time frame indexed by an atom label $a$ (a scale, a location, and a temporal centre):

$$
\varepsilon(\mathbf{x}, t) = \sum_{a=1}^{m} w_a \, \varphi_a(\mathbf{x}) \, \chi_a(t)
$$

where $\varphi_a$ is a spatial basis function (from geonnax) and $\chi_a$ a temporal window. The coefficients $w_a$ are the control. Assigning them a prior $w \sim \mathcal{N}(0, \Lambda)$ with diagonal $\Lambda = \mathrm{diag}(\sigma_a^2)$ induces a model-error covariance

$$
Q = \Phi \Lambda \Phi^\top
$$

so choosing the pair $(\Phi, \Lambda)$ is choosing $Q$. geonnax supplies $\Phi$; the variances $\Lambda$ come from the prior layer. The weak-constraint cost is an observation misfit plus the prior penalty:

$$
J[w, x_0] = \frac{1}{2} \sum_k (\mathbf{y}_k^{\mathrm{obs}} - \mathcal{H}_k y_k)^\top R_k^{-1} (\mathbf{y}_k^{\mathrm{obs}} - \mathcal{H}_k y_k) \; - \; \log p(w)
$$

(the misfit is *observation minus mapped state*). For the diagonal Gaussian prior $-\log p(w)$ is the quadratic $\frac{1}{2} \sum_a w_a^2 / \sigma_a^2$ up to a constant, and the flow-prior note generalises it to a learned density. Preconditioning uses the whitened control $u = \Lambda^{-1/2} w$, so the prior term becomes $\frac{1}{2} \lVert u \rVert^2$ and the Hessian is bounded below by the identity.

The spectral weighting of $\Lambda$ is where the basis metadata earns its place. For a spectral basis geonnax returns eigenvalues $\lambda_a$, and the prior layer sets $\sigma_a^2 = S(\sqrt{\lambda_a})$ for a kernel spectral density $S$ (the Hilbert-space-GP construction). For a frame geonnax returns the per-atom wavenumber $k_a$, and for SSH the steep mesoscale spectrum $E_{\mathrm{SSH}}(k) \propto k^{-\alpha}$ with $\alpha$ near four gives $\sigma_a^2 \propto k_a^{-\alpha}$, placing most variance at large scales.

## 4. Temporal structure

Two regimes follow from the choice of $\chi_a$. If each atom carries its own temporal window, the coefficients index space-time atoms directly and there is no per-step forcing. If the basis factorises into a shared spatial dictionary times a shared temporal basis, $\Phi = \Phi_t \otimes \Phi_s$, the coefficients arrange as $(m_t, m_s)$ and the construction is separable; per-step forcing is the degenerate case $\Phi_t = I$.

The temporal basis $\Phi_t$ maps $m_t$ temporal degrees of freedom to the model steps. Per-step forcing is the identity; an interpolator (zero-order-hold or linear — the existing `InterpolatedForcing`) is a piecewise temporal basis; and a smooth basis (Fourier via `seasonal_features`, or Gaussian windows via `gaussian_window_features`, both from geonnax) is the reduced case. The number of temporal degrees of freedom $m_t$ is set from the model-error correlation time $\tau_c$: a window spacing near $\tau_c$ matches the temporal smoothness of the error.

## 5. Numerics background

The continuous field is sampled on the model grid by calling a geonnax basis at the grid points, stacking atoms into a flat index $a = 1, \dots, m$:

$$
[\Phi_s]_{n,a} = \varphi_a(\mathbf{x}_n) \in \mathbb{R}^{N_s \times m}, \qquad [\mathbf{b}(t)]_a = \chi_a(t) \in \mathbb{R}^{m}
$$

and the forcing field at any time is one elementwise gating followed by one contraction:

$$
\varepsilon(\cdot, t) = \Phi_s (\mathbf{w} \odot \mathbf{b}(t)) \in \mathbb{R}^{N_s}
$$

Several somax-specific points govern the implementation.

**Grid points come from `Domain.coords`.** `Domain` exposes the flattened grid as the `coords` property (shape `(N_s, ndim)`); there is no `domain.points()` and no `domain.bounds` — bounds are the static `domain.xmin` / `domain.xmax` tuples, and the grid shape is `domain.Nx`. The dictionary $\Phi_s$ is precomputed once at construction by evaluating the chosen geonnax function on `domain.coords`, because `Domain` is static config and precomputation keeps `__call__` to two array ops.

**Synthesis is flat; the field is shaped.** $\Phi_s$ rows align to the *flattened* grid, so `synthesize` returns a flat `(N_s,)` vector. somax state fields are 2-D `(Ny, Nx)` (and states are multi-variable pytrees — `u, v, h` for shallow water, layered `q` for QG). `BasisForcing.__call__` therefore reshapes to `domain.Nx`, and the **placement** onto the right variable/layer is the adapter's job (§6), exactly as QG adds wind only to `dq[0]`.

**The forcing is a tendency evaluated at interior times.** It enters the `diffrax` vector field as a rate. Even the default fixed-step integrator (`model.integrate` defaults to `diffrax.ConstantStepSize`) evaluates the vector field at interior RK stage times $t + c_i\,\Delta t$, so the temporal gate must be a continuous function of $t$ — which the geonnax temporal features are, and which `diffrax` interpolation paths supply. (If a user switches to an adaptive `stepsize_controller`, the same continuity requirement holds at unpredictable interior times.)

**Adjoint.** The adjoint is the reverse-mode derivative of the `diffrax` solve. somax currently sets no explicit `adjoint=`, so diffrax's default `RecursiveCheckpointAdjoint` gives the discrete adjoint consistent with the cost; `BacksolveAdjoint` is available for O(1) memory if needed.

**Differentiability is scoped to `coeffs`.** The coefficients $w$ are the only learnable leaf; $\Phi_s$, the temporal centres/widths, and the prior variances are fixed. Because $\Phi_s$ is a large array it stays a normal array leaf (never `eqx.field(static=True)`, which would bloat the pytree definition); gradients are restricted to `coeffs` by partitioning at the optimiser with an explicit filter, `eqx.partition(forcing, control_filter)`, where `control_filter` is the mask selecting `coeffs` only.

## 6. The interface seam (`ForcingProtocol` → `Term`)

This is the part the original sketch omitted and everything else depends on. There are **two** call contracts in play and they do not match:

| Abstraction | Signature | Returns |
|---|---|---|
| `ForcingProtocol` (`core/forcing.py`) | `__call__(t, grid) -> field` | a field on the grid |
| `Term` (`core/terms.py`) | `__call__(t, state, args) -> tendency` | a pytree matching `state` |

The model RHS evaluates `Term`s, not `ForcingProtocol`s, and the term machinery never supplies a `grid` argument. So a forcing must be **lifted** into the term algebra by an adapter that (a) calls the forcing, (b) reshapes the flat synthesis to the field shape, and (c) places it onto the correct state component as a tendency:

```python
from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from somax._src.core.forcing import ForcingProtocol
from somax._src.core.terms import Term


class ForcingTerm(Term):
    """Lift a ForcingProtocol field into the term algebra as a tendency.

    Resolves the (t, grid) -> field  vs.  (t, state, args) -> tendency
    contract mismatch. `place` writes the field onto the target state
    component (e.g. dq[0], or the top-layer u/v), returning a tendency
    pytree that is zero everywhere else -- the generalisation of QG's
    `dq = dq.at[0].add(tau0 * wind_forcing)`.
    """

    forcing: ForcingProtocol
    place: Callable[[PyTree, Array], PyTree] = eqx.field(static=True)

    def __call__(self, t: float, state: PyTree, args: PyTree | None = None) -> PyTree:
        field = self.forcing(t, None)          # (Ny, Nx) after reshape
        zeros = jtu.tree_map(lambda leaf: leaf * 0.0, state)
        return self.place(zeros, field)        # tendency on one component


def add_to(component: str, layer: int | None = None) -> Callable[[PyTree, Array], PyTree]:
    """Build a `place` that adds `field` onto one named state component
    (optionally one layer). Mirrors the QG/SWM placement convention."""
    def _place(zeros: PyTree, field: Array) -> PyTree:
        leaf = getattr(zeros, component)
        leaf = leaf.at[layer].add(field) if layer is not None else leaf + field
        return eqx.tree_at(lambda s: getattr(s, component), zeros, leaf)
    return _place
```

With this seam, `ForcingTerm(forcing, place=add_to("q", layer=0))` drops a reduced-order forcing into any `TermModel` RHS and the existing `build_diffrax_terms` / `diffeqsolve` path integrates it. The `tau0 * wind_forcing` QG forcing is the special case `ForcingTerm(ConstantForcing(wind_forcing) scaled by tau0, place=add_to("q", 0))`, which is the right thing to assert in tests.

## 7. API proposal

A `SpatialBasis` holds the precomputed geonnax dictionary and the prior standard deviations, a `TemporalBasis` wraps a geonnax temporal feature, and `BasisForcing` composes them and preserves the `(t, grid)` call contract. Contractions are plain `jax.numpy` (somax does not use `einx`).

```python
from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from somax._src.core.forcing import ForcingProtocol


class SpatialBasis(eqx.Module):
    """A precomputed geonnax dictionary plus the per-mode prior std.

    `Phi` comes from evaluating a geonnax basis (fourier_basis, rbf_basis,
    gabor_frame_grid, ...) on `Domain.coords` at build time. `std` comes from
    the prior layer: a kernel spectral density of geonnax eigenvalues for a
    spectral basis, or a prescribed / wavenumber law for a frame.
    """

    Phi: Float[Array, "Ngrid m"]      # geonnax dictionary on the flattened grid
    std: Float[Array, "m"]            # Lambda^{1/2}, from the prior layer

    def synthesize(self, coeffs: Float[Array, "m"]) -> Float[Array, "Ngrid"]:
        return self.Phi @ coeffs

    def analyze(self, field: Float[Array, "Ngrid"]) -> Float[Array, "m"]:
        # Frame analysis (adjoint synthesis); exact only for orthonormal bases.
        return self.Phi.T @ field

    def prior_std(self) -> Float[Array, "m"]:
        return self.std


class TemporalBasis(eqx.Module):
    """Maps a scalar time to per-atom temporal weights b(t), wrapping a
    geonnax temporal feature (seasonal_features, gaussian_window_features)."""

    @abc.abstractmethod
    def weights(self, t: float) -> Float[Array, "m"]: ...


class BasisForcing(ForcingProtocol):
    """Reduced-order forcing: a fixed geonnax space-time frame driven by a
    differentiable coefficient vector (the DA control).

    `SeasonalWindForcing` is the special case of a one-mode Fourier temporal
    basis over a one-column spatial dictionary. Returns a field shaped to the
    model grid; `ForcingTerm` (Section 6) lifts it onto a state component.
    """

    coeffs: Float[Array, "m"]                 # learnable control, visible to jax.grad
    spatial: SpatialBasis                      # fixed geonnax dictionary + prior std
    temporal: TemporalBasis                    # fixed geonnax temporal gate
    grid_shape: tuple[int, ...] = eqx.field(static=True)  # domain.Nx, for reshape

    def __call__(self, t: float, grid: eqx.Module | None = None) -> Array:
        b = self.temporal.weights(t)               # (m,)
        active = self.coeffs * b                    # (m,)
        flat = self.spatial.synthesize(active)      # (Ngrid,)
        return flat.reshape(self.grid_shape)        # (Ny, Nx)

    def whiten(self, u: Float[Array, "m"]) -> "BasisForcing":
        # w = Lambda^{1/2} u for the diagonal prior; the flow-prior note
        # replaces this with a learned generative map.
        w = self.spatial.prior_std() * u
        return eqx.tree_at(lambda f: f.coeffs, self, w)

    def regularization(self) -> Float[Array, ""]:
        # 0.5 sum (w / sigma)^2 for the diagonal prior; the flow-prior note
        # replaces this with -prior.log_prob(w). Consumed by the DA cost (Section 9).
        std = self.spatial.prior_std()
        return 0.5 * jnp.sum((self.coeffs / std) ** 2)
```

A `SpatialBasis` is built by evaluating a geonnax function on `domain.coords`; the spectral or prescribed prior fills `std`:

```python
import jax.numpy as jnp
import numpy as np
from geonnax.basis import gabor_frame_grid          # public surface


def spatial_from_gabor(domain, *, n_scales, base_scale, slope, amp) -> SpatialBasis:
    xy = domain.coords                          # (Ngrid, ndim)
    # bounds is a build-time-concrete (ndim, 2) [lo, hi] box, from the static
    # xmin / xmax tuples — not (domain.xmin, domain.xmax) directly.
    bounds = np.stack([np.asarray(domain.xmin), np.asarray(domain.xmax)], axis=-1)
    Phi, centers, scales, wavenumbers = gabor_frame_grid(
        xy, bounds, n_scales=n_scales, base_scale=base_scale
    )
    std = jnp.sqrt(amp * wavenumbers ** (-slope))      # SSH-like spectral law
    return SpatialBasis(Phi=Phi, std=std)


# Deferred until the box-Laplacian eigenbasis is public (it currently lives in
# the private geonnax._basis). The shape, once promoted, is:
#
#   from geonnax.basis import fourier_basis        # (Phi, lam), private today
#   def spatial_from_fourier(domain, *, m, length, kernel_psd) -> SpatialBasis:
#       Phi, lam = fourier_basis(domain.coords, num_basis_per_dim=m, L=length)
#       std = jnp.sqrt(kernel_psd(jnp.sqrt(lam)))      # HSGP: S(sqrt(lambda))
#       return SpatialBasis(Phi=Phi, std=std)
```

Lognormal variables (ocean colour) wrap the synthesis in a transform:

```python
class TransformedForcing(ForcingProtocol):
    base: ForcingProtocol
    forward: callable = eqx.field(static=True)    # e.g. log10
    inverse: callable = eqx.field(static=True)    # e.g. lambda z: 10.0 ** z

    def __call__(self, t: float, grid: eqx.Module | None = None) -> Array:
        return self.inverse(self.base(t, grid))
```

Two decisions worth recording. The spatial dictionary is bound to a `Domain` at construction, because `Domain` is static config and precomputation keeps `__call__` cheap; the `grid` argument is retained only for `ForcingProtocol` conformance and may assert consistency with the build domain. And the basis math is delegated to geonnax: somax never reimplements an eigenfunction or a kernel, it evaluates one and weights it.

## 8. Bank of basis functions

Each preset is a thin factory: pick a geonnax spatial primitive, a geonnax temporal feature, and a prior weighting, then build a `BasisForcing`. The math of each basis lives in geonnax; the entries below name which primitive and which prior, not how the basis is computed.

| Preset | geonnax spatial primitive | temporal feature | prior weighting | transform |
|---|---|---|---|---|
| `ssh_geostrophic` | `gabor_frame_grid` (radial Gabor) | `gaussian_window_features` | wavenumber law, slope near 4 | none |
| `sst_frontal` | `gabor_frame_grid` + `fourier_basis` base | `gaussian_window_features` | HSGP small scale + smooth large | deseasonalise |
| `sss_coastal` | `rbf_basis` (Wendland) + `graph_laplacian_eigpairs` | `gaussian_window_features` | prescribed per centre + HSGP | none |
| `oc_chlorophyll` | `gabor_frame_grid` in log space | `gaussian_window_features` | wavenumber law | log10 / exp |
| `wind_seasonal_tidal` | low-mode `fourier_basis` or coarse `rbf_basis` | `seasonal_features` | small fixed | none |
| `global_largescale` | `real_spherical_harmonics` + `slepian_cap_basis` | `seasonal_features` | angular spectrum | none |

The spectral and Slepian bases use the **eigenvalue half** of the geonnax contract, so their `std` is a kernel spectral density of the returned eigenvalues. The Gabor frame and RBF basis use the **geometry half**, so their `std` is the wavenumber law or a prescribed per-centre value. These two return contracts — `(Phi, eigenvalues)` for spectral bases and `(Phi, centers, scales, wavenumbers)` for frames — are exactly what somax depends on and must be pinned (§ Dependency reality).

As landed in `forcing_bank.py` (`GaussianWindowsInTime` wraps `gaussian_window_features`; `tile_in_time` builds the separable space-time frame; the preset defaults to constant-in-time and switches to the time-distributed regime when `windows=(centers, widths)` is passed):

```python
def ssh_geostrophic(domain, *, n_scales=6, base_scale=20e3, slope=4.0,
                    amplitude=2e-6, oversample=1.0, windows=None) -> BasisForcing:
    spatial = spatial_from_gabor(domain, n_scales=n_scales, base_scale=base_scale,
                                 slope=slope, amplitude=amplitude, oversample=oversample)
    if windows is None:
        temporal = ConstantInTime(m=spatial.Phi.shape[1])
    else:
        spatial, temporal = tile_in_time(spatial, windows[0], windows[1])
    return BasisForcing(
        coeffs=jnp.zeros(spatial.Phi.shape[1]),
        spatial=spatial,
        temporal=temporal,
        grid_shape=tuple(domain.Nx),
    )
```

One basis stays in somax rather than geonnax: the diffusion-operator (Weaver–Courtier) smoother, whose columns are defined by integrating the heat equation with the model's own Laplacian on the model grid. It is an implicit basis tied to the somax operators, so it cannot be a pure geonnax function and lives in `somax/_src/core/basis.py` directly. "And more" is open by the same recipe: surface currents, wind stress, and sea-ice concentration are each a geonnax primitive plus a temporal feature plus a prior.

## 9. Data-assimilation integration — two distinct paths

The original sketch said this "reuses the `jax.grad` adjoint I already have through `vardax_bridge`." That is only half true, and the two halves require very different amounts of work. They should be kept separate.

**(a) Static-coefficient parameter estimation — works today.** If `coeffs` is a leaf of the *model* pytree (the `BasisForcing` lives inside a `ForcingTerm` inside the model's term tree), then `jax.grad` of any loss reaches `coeffs` through the existing differentiable RHS — this is exactly how `SeasonalWindForcing.tau0` and `Scaled.coeff` are already differentiable. This covers `wind_seasonal_tidal`, amplitude/scale tuning, and any *time-stationary* coefficient control. No vardax change is needed.

**(b) Weak-constraint model error — new work.** Solving for *time-distributed* model error $\varepsilon(\mathbf{x}, t)$ as a separate control is **not** what `da/vardax_bridge.py` does today. `SomaxForwardModel` is a flat-state `ForwardModel` (`step(state, dt)`) whose control is the state vector, and it is documented as **autonomous-only**: it passes a fixed `t0` on every substep, so a time-dependent forcing would be evaluated at the wrong absolute time for every substep after the first. Two things must change before (b) is real:

1. **Per-substep absolute time** must be threaded through the rollout, so `temporal.weights(t)` is sampled at the true substep time. This is a `vardax` / `pipekit_cycle.ForwardModel` contract change, not a somax-local fix.
2. **The control must include the coefficients** $w$ (or the whitened $u$), and the variational cost must add `BasisForcing.regularization()` (or, with the flow prior, $-\log p(w)$). This needs a hook in vardax's cost assembly; `whiten()` supplies the $u \mapsto w$ preconditioning map and `regularization()` the penalty term.

Until (1) and (2) land, weak-constraint `BasisForcing` should be exercised only through a somax-local `diffrax` adjoint (a hand-rolled 4D-Var loss over an explicit time grid), not through `vardax_bridge`. The note's headline user story is path (b); plan for it as a vardax-integration project, not a wire-up.

## 10. TODOs

1. ✅ `somax/_src/core/basis.py`: `SpatialBasis`, `TemporalBasis`, `BasisForcing`, `TransformedForcing` (the dependency-free core); the `spatial_from_*` builders that call geonnax landed in `somax/_src/core/forcing_bank.py`. → verified: `tests/core/test_basis.py` checks synthesis shapes, the `whiten` round-trip, the `regularization` value, and the `(Ngrid,) -> (Ny, Nx)` reshape.
2. **The `ForcingTerm` seam** (`core/terms.py` or `core/basis.py`) plus `add_to(component, layer)`. → verify: `ForcingTerm` wrapping a `ConstantForcing(wind_forcing)` scaled by `tau0` reproduces the QG `dq[0] += tau0 * wind_forcing` tendency bit for bit. **This is the highest-value first step** — it de-risks the whole design without geonnax.
3. Wrap the geonnax temporal features as `TemporalBasis` subclasses (`FourierInTime` over `seasonal_features`, `GaussianWindowsInTime` over `gaussian_window_features`, `SplineInTime` over `InterpolatedForcing`). → verify: a one-mode `FourierInTime` reproduces `SeasonalWindForcing`.
4. ✅ Bank presets in `forcing_bank.py`, each composing a geonnax primitive, a temporal feature, and a prior: `ssh_geostrophic` (Gabor frame + wavenumber-law prior), `sss_coastal` (placeable Wendland RBFs + prescribed prior), and `sst_frontal` (box-Laplacian HSGP + Matérn spectral prior), each with an optional Gaussian-window temporal gate via `tile_in_time`. Spectral / data-driven / wavelet builders also landed: `spatial_from_fourier` (Matérn HSGP), `spatial_from_graph_laplacian`, `spatial_from_eof` (DINEOF), `spatial_from_wavelet`, plus the `matern_spectral_density` helper — all consuming the geonnax bases made public in geonnax#25 (which also adds `eof_basis`, `divfree_basis`, `spherical_rbf_basis`, `wavelet_basis_1d/2d`). → verified in `tests/core/test_basis.py`. *Still deferred:* the vector `divfree_basis` (needs a vector placement seam — currently `add_to` writes a scalar field) and the on-sphere `spherical_rbf_basis` / `global_largescale` spherical-harmonic/Slepian presets (need a sphere-coordinate domain); both are geonnax-ready but await a somax interface extension.
5. The Weaver–Courtier diffusion basis, somax-local, using the model Laplacian. → verify: a single column matches a heat-kernel smoothing of a point source.
6. ✅ **Add and pin the basis dependency**: `geonnax` is git-pinned and the basis return contracts are confirmed against the pin (see the geonnax-dependency status block above). The spectral eigenbases were promoted to the public `geonnax.basis` API in geonnax#25 (the pin is bumped to that branch commit; re-pin to the merged SHA before a somax release). → verified: imports resolve and the documented return signatures match. *Still open:* the prior layer (`pyrox` / `gauss_flows`).
7. **Weak-constraint vardax integration** (path 9b): per-substep absolute time in the `ForwardModel` rollout, a coefficient control, and a cost hook consuming `regularization()` / `whiten()`. → verify: a small weak-constraint reconstruction recovers a known injected forcing. *(Vardax-side project, not a wire-up.)*
8. `control_filter` helper so optimisers update only `coeffs`. → verify: `jax.grad` of a loss has zero cotangent on `spatial.Phi`.
9. Propagating geostrophic atoms (advected centres) for Rossby-wave structure, by passing time-dependent centres to `gabor_frame`. → verify: a propagating single atom tracks the expected phase speed.
10. Docs: a tutorial page and notebook showing an SSH weak-constraint reconstruction, figures under `content/images/`; add `content/notes/forcing_basis.md` (this note) and the tutorial to `myst.yml`.

## References

- Beckers and Rixen (2003), EOF calculations and data filling from incomplete oceanographic datasets (DINEOF).
- Lindgren, Rue and Lindström (2011), the SPDE link between Gaussian fields and Markov random fields (Matérn).
- Weaver and Courtier (2001), correlation modelling via the diffusion equation.
- Solin and Särkkä (2020), Hilbert space methods for reduced-rank Gaussian process regression.
- Ubelmann, Le Traon and others, and Ardhuin and others, on the MIOST multiscale mapping.
- Le Guillou and others (2024), the QG weak-constraint reduced-order mapping (4DVarQG).
- Fablet and others (2021), 4DVarNet.
