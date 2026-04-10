# Spinup Duration for Baroclinic QG Double-Gyre Simulations

> **TL;DR.** For a 2–3 layer baroclinic QG double-gyre at 128²–256² resolution on a ~5000 km basin, the literature spans **2 years (demo-quality) to 40+ years (low-frequency variability studies)**. There is no single canonical number. somax ships **3 years as the default**, with a documented ladder for production (10 yr) and publication-grade (40 yr) work.

## Why this matters

The "spinup duration" is the length of integration that is **discarded** before analysis begins, to let transients from the initial conditions die out. Choosing it wrong means either:

- **Too short:** the WBC hasn't formed, the eastward jet hasn't extended, or the eddy field hasn't equilibrated. Downstream statistics are contaminated by adjustment transients.
- **Too long:** wasted compute, longer iteration cycles, harder to use the pipeline in CI.

The right choice depends on what you're trying to measure. somax provides a default that's defensible for general-purpose use and a ladder for users with stricter scientific requirements.

## Physical timescales

For a baroclinic QG double-gyre with basin width $L \sim 5000$ km, mid-latitude $f_0 \approx 10^{-4}~\text{s}^{-1}$, and first baroclinic deformation radius $R_d \sim 30$ km, the relevant adjustment timescales are:

| Process | Timescale | Order of magnitude |
|---|---|---|
| Inertial period $2\pi/f_0$ | hours | hours |
| Barotropic Rossby wave crossing $L/c_{0}$ | days–weeks | days |
| **First baroclinic Rossby wave crossing** $\tau_R = L/c_1$ | **months–years** | **~3 yr** for $c_1 \approx 2~\text{m/s}$ |
| Eddy-mean flow equilibration | years | ~5–10 yr |
| Decadal intrinsic variability modes | decades | 5–20 yr |

The first-baroclinic Rossby wave crossing time $\tau_R$ is the **physical floor** for any meaningful spinup. Anything less than $2$–$3 \tau_R$ leaves the basin in a state where the slowest large-scale waves haven't even propagated across it once.

## Phases of spinup

A typical baroclinic QG double-gyre, started from rest with a fixed wind forcing, passes through these phases:

1. **Barotropic adjustment** (months to ~1 year): Western boundary current forms, basin fills with Rossby waves. Eastward jet has not yet extended.
2. **First-baroclinic adjustment** (1–3 years): Eastward jet extends into the interior, first eddies are shed from the WBC, thermocline tilts. This is when the basic mean flow takes shape.
3. **Mean-flow / eddy equilibration** (~5–10 years): Eddy kinetic energy saturates, jet length and recirculation gyres stabilize. Statistical mean state is approximately stationary.
4. **Decadal variability envelope** (10–50 years): Required only if you want to *average over* the intrinsic 5–20 year modes Berloff, Hogg and co-authors identified.

## What the literature does

Concrete spinup durations from the canonical references and recent open-source implementations:

| Reference | Layers / grid / basin | Spinup discarded | Total run | Notes |
|---|---|---|---|---|
| Holland (1978) | 2L, ~20 km, 1000 km | informal, ~last few yr | multi-year | Original wind-driven 2-layer gyre |
| Berloff & McWilliams (1999) | 1.5–2L, eddy-permitting | ~10 yr | 40–100 yr | Low-frequency variability |
| Hogg, Dewar et al. (2003) | 3L, ~10 km (Q-GCM) | 10–20 yr | 50–100+ yr | Q-GCM coupled model |
| Hogg et al. (2005) | 3L, ~10 km | 10–20 yr | 50+ yr | Decadal mode quantization |
| Berloff, Hogg & Dewar (2007) | 3L, ~7.5 km, 3840 km | **40 yr** | 200 yr | "Turbulent oscillator" |
| Karabasov, Berloff & Goloviznin (2009) | 3L, 7.5–3.75 km | ~10 yr | multi-decadal | CABARET benchmark |
| Thiry et al. (2024 JAMES) | 3L, 256², 5120 km | **40 yr** | 60 yr | 20 yr analysis |
| **MQGeometry-1.0 (Thiry et al. 2024 GMD)** | **3L, 256², 5120 km** | **10 yr** | **50 yr** | Closest direct analog |
| **louity/qgsw-pytorch** | **3L, 256², 5120 km** | **2 yr** | **10 yr** | Other direct analog |

The two analog repos that match somax's target configuration most closely (`MQGeometry` and `qgsw-pytorch`, both 3-layer, 256², 5120 km basin) bracket the practical range: 2 years on the short end, 10 years on the long end. The Berloff/Hogg variability papers go to 40 years because they're after intrinsic decadal modes that no shorter run can resolve.

## Convergence diagnostics

The QG community typically monitors convergence via:

1. **Basin-integrated kinetic energy (KE) and eddy kinetic energy (EKE) per layer** as a time series. Upper-layer EKE is the most sensitive.
2. **Running-mean convergence**: the running mean of EKE over a sliding window (typically 1–2 years) should drift by less than a few percent over the previous window. Berloff & McWilliams (1999) and Hogg et al. (2005) use variants of this.
3. **Energy spectrum stationarity**: the $k^{-3}$ enstrophy range in the upper layer should be established and stable.
4. **Jet length and WBC separation latitude** stabilized within a standard deviation.
5. **EOFs of $\psi_1$** converged (only relevant for the publication-grade tier — needs the long 40+ yr runs).

There is no universal quantitative threshold. "5% running-mean drift per year" is a common informal criterion in QG papers but is rarely stated explicitly.

## somax's three-tier ladder

somax ships configs at three tiers:

| Tier | `spinup_duration_years` | Use case | Wallclock at 128² | Justification |
|---|---|---|---|---|
| **Default (CI / demo)** | **3** | Smoke test, "show me a turbulent jet", reference pipeline | minutes | Just clears the baroclinic Rossby adjustment time $\tau_R \approx 3$ yr; produces a post-WBC-formation state with the first generation of eddies. |
| **Production** | 10 | "Looks scientifically reasonable"; matches MQGeometry's `n_steps_save = 10 yr`; used by qgsw-pytorch's higher-end recommendation. | tens of minutes | Approaches statistical equilibrium of mean flow + EKE. |
| **Publication** | 40 | Required for decadal variability statistics (Berloff/Hogg conventions, Thiry et al. 2024). | hours | Captures intrinsic 5–20 yr modes. |

The default of **3 years** is chosen because:

- It clears $\tau_R$, the physical floor.
- It produces a turbulent state with a recognizable WBC and eastward jet.
- It runs in single-digit minutes at 128² on a modern CPU, making it CI-friendly.
- It exceeds qgsw-pytorch's 2-year default, which we consider too short to be defensible as a reference.
- It is more conservative than MQGeometry's 10 yr but produces qualitatively similar statistics for demonstration purposes.

For any work beyond demonstration, users should override to 10 or 40 years. The override mechanism is documented in the simulation pipeline guide.

## Honest uncertainty statement

There is no single canonical number. The 2–40 year spread in the literature reflects two genuinely different scientific goals:

- **(a) "Show me a turbulent jet"** — 2–5 years suffices.
- **(b) "Compute converged statistics of intrinsic decadal variability"** — 40+ years required.

For a general-purpose open-source reference config, **3 years is the shortest defensible minimum, 10 years is the community mode for routine work, and 40 years is publication-grade for variability studies**.

If you are using somax for research that depends on the absolute level of spinup, you should run a convergence study at your specific resolution and parameter set, not rely on the default.

## References

- Holland, W. R. (1978). The role of mesoscale eddies in the general circulation of the ocean. *Journal of Physical Oceanography* 8, 363–392.
- Berloff, P. S. & McWilliams, J. C. (1999). Large-scale low-frequency variability in wind-driven ocean gyres. *Journal of Physical Oceanography* 29, 1925–1949.
- Hogg, A. M., Dewar, W. K., Killworth, P. D. & Blundell, J. R. (2003). A quasi-geostrophic coupled model (Q-GCM). *Monthly Weather Review* 131, 2261.
- Hogg, A. M., Dewar, W. K., Killworth, P. D. & Blundell, J. R. (2005). Quantization of the low-frequency variability of the double-gyre circulation. *Journal of Physical Oceanography* 35, 2232.
- Berloff, P., Hogg, A. M. & Dewar, W. (2007). The turbulent oscillator: a mechanism of low-frequency variability of the wind-driven ocean gyres. *Journal of Physical Oceanography* 37, 2362–2386.
- Karabasov, S. A., Berloff, P. S. & Goloviznin, V. M. (2009). CABARET in the ocean gyres. *Ocean Modelling* 30, 155–168.
- Thiry, L., Li, L., Mémin, E. & Roullet, G. (2024). MQGeometry-1.0: a multi-layer quasi-geostrophic solver on non-rectangular geometries. *Geoscientific Model Development* 17, 1749.
- Thiry, L. et al. (2024). A unified formulation of quasi-geostrophic and shallow water equations via projection. *Journal of Advances in Modeling Earth Systems*. doi:10.1029/2024MS004510.
- louity/MQGeometry — [https://github.com/louity/MQGeometry](https://github.com/louity/MQGeometry)
- louity/qgsw-pytorch — [https://github.com/louity/qgsw-pytorch](https://github.com/louity/qgsw-pytorch)
