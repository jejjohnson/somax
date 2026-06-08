"""Geonnax-facing builders and the preset bank for reduced-order forcing.

This is the geonnax-facing layer that sits on top of the deliberately
dependency-free core in :mod:`somax._src.core.basis`. It evaluates a geonnax
spatial basis on a :class:`~somax._src.domain.domain.Domain` to build a
:class:`~somax._src.core.basis.SpatialBasis`, wraps a geonnax temporal feature
as a :class:`~somax._src.core.basis.TemporalBasis`, and assembles them into
ready-made :class:`~somax._src.core.basis.BasisForcing` presets (the "bank").

Only the **public** ``geonnax.basis`` surface is used: the overcomplete Gabor
frame (``gabor_frame_grid``), the placeable radial basis (``rbf_basis``), and
the Gaussian-in-time window (``gaussian_window_features``). geonnax returns the
synthesis matrix plus the *geometry* half of its basis contract (per-atom
wavenumbers for the frame); somax turns that geometry into a per-mode prior std
(``Lambda^{1/2}``) via a spectral law. The spectral *eigenbases*
(``fourier_basis``, ``graph_laplacian_eigpairs``) currently live in geonnax's
private ``geonnax._basis`` namespace, so the presets that need the eigenvalue
half of the contract (HSGP Fourier, graph-Laplacian) are deferred until those
primitives are promoted to the public API. See
``content/notes/forcing_basis.md`` for the full design.

The dictionary is evaluated once, at build time, on the static ``Domain``; the
resulting :class:`~somax._src.core.basis.BasisForcing` keeps ``__call__`` to one
elementwise gating plus one contraction.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from geonnax.basis import gabor_frame_grid, gaussian_window_features, rbf_basis
from jaxtyping import Array, Float

from somax._src.core.basis import (
    BasisForcing,
    ConstantInTime,
    SpatialBasis,
    TemporalBasis,
)
from somax._src.domain.domain import Domain


class GaussianWindowsInTime(TemporalBasis):
    """Localized temporal gate ``b_a(t) = exp(-(t - tau_a)^2 / (2 T_a^2))``.

    Wraps geonnax :func:`~geonnax.basis.gaussian_window_features`: each atom is
    a soft Gaussian window centred at ``centers[a]`` with width ``widths[a]``,
    the localized counterpart to
    :class:`~somax._src.core.basis.FourierInTime`. The centres and widths are
    fixed geometry (not part of the control); they are stored as array leaves
    but excluded from gradients by
    :func:`~somax._src.core.basis.control_filter`.

    Attributes:
        centers: Window centres ``tau_a`` of shape ``(m,)``.
        widths: Window widths ``T_a`` of shape ``(m,)``.
    """

    centers: Float[Array, " m"]
    widths: Float[Array, " m"]

    def weights(self, t: float) -> Float[Array, " m"]:
        """Return the Gaussian-window weights at scalar time ``t``."""
        # gaussian_window_features expects a batch of times (N,); evaluate the
        # single scalar time and drop the batch axis back to (m,).
        t_arr = jnp.atleast_1d(jnp.asarray(t))
        return gaussian_window_features(t_arr, self.centers, self.widths)[0]


def spatial_from_gabor(
    domain: Domain,
    *,
    n_scales: int,
    base_scale: float,
    slope: float = 4.0,
    amplitude: float = 1.0,
    oversample: float = 1.0,
) -> SpatialBasis:
    """Build a :class:`SpatialBasis` from a geonnax dyadic radial-Gabor frame.

    Evaluates :func:`~geonnax.basis.gabor_frame_grid` on ``domain.coords`` and
    fills the per-mode prior std from the frame's per-atom wavenumbers using the
    steep mesoscale spectral law ``sigma_a = sqrt(amplitude * k_a ** -slope)``,
    which places most variance at large scales (small wavenumber) — the
    weighting behind multiscale SSH mapping.

    Args:
        domain: The model domain; its ``coords`` (``(Ngrid, ndim)``) are the
            evaluation points and its static ``xmin`` / ``xmax`` the frame box.
        n_scales: Number of dyadic scales in the frame.
        base_scale: Finest envelope scale ``L_0`` (in domain units).
        slope: Spectral slope of the wavenumber prior law (``~4`` for SSH).
        amplitude: Overall prior variance scale.
        oversample: Centre density per scale (spacing ``L_s / oversample``).

    Returns:
        A :class:`SpatialBasis` whose ``Phi`` is the frame synthesis matrix and
        whose ``std`` follows the wavenumber law.
    """
    coords = domain.coords  # (Ngrid, ndim)
    # bounds is a build-time concrete value (read in Python to lay out the
    # centre grid), so build it from the static xmin/xmax with numpy.
    bounds = np.stack(
        [np.asarray(domain.xmin, dtype=float), np.asarray(domain.xmax, dtype=float)],
        axis=-1,
    )  # (ndim, 2)
    Phi, _centers, _scales, wavenumbers = gabor_frame_grid(
        coords,
        bounds,
        n_scales=n_scales,
        base_scale=base_scale,
        oversample=oversample,
    )
    std = jnp.sqrt(amplitude * wavenumbers ** (-slope))
    return SpatialBasis(Phi=Phi, std=std)


def spatial_from_rbf(
    domain: Domain,
    centers: Float[Array, "m ndim"],
    widths: Float[Array, " m"],
    *,
    kernel: str = "gaussian",
    std: Float[Array, " m"] | float = 1.0,
) -> SpatialBasis:
    """Build a :class:`SpatialBasis` from a geonnax placeable radial basis.

    Evaluates :func:`~geonnax.basis.rbf_basis` on ``domain.coords``, placing one
    column per ``(center, width)`` pair — put atoms where the physics is
    localized (a river mouth, a front) and leave the open ocean untouched. A
    radial basis has no eigendecomposition, so the prior std is *prescribed* per
    centre (the geometry half of the basis contract), defaulting to ones.

    Args:
        domain: The model domain; its ``coords`` are the evaluation points.
        centers: Atom centres of shape ``(m, ndim)``.
        widths: Per-atom width of shape ``(m,)`` (Gaussian length scale or
            Wendland support radius).
        kernel: ``"gaussian"`` (smooth global bump) or ``"wendland_c2"`` /
            ``"wendland_c4"`` (compact support).
        std: Prescribed per-centre prior std; a scalar is broadcast to ``(m,)``.

    Returns:
        A :class:`SpatialBasis` over the placed radial atoms.
    """
    coords = domain.coords
    Phi = rbf_basis(
        coords, jnp.asarray(centers), jnp.asarray(widths), kernel=kernel
    )  # (Ngrid, m)
    std_arr = jnp.broadcast_to(jnp.asarray(std, dtype=Phi.dtype), (Phi.shape[1],))
    return SpatialBasis(Phi=Phi, std=std_arr)


def tile_in_time(
    spatial: SpatialBasis,
    centers: Float[Array, " m_t"],
    widths: Float[Array, " m_t"],
) -> tuple[SpatialBasis, GaussianWindowsInTime]:
    """Lift a spatial dictionary into a separable space-time frame.

    Realises the separable construction ``Phi = Phi_t (x) Phi_s`` through the
    per-atom :class:`~somax._src.core.basis.BasisForcing` interface (whose
    temporal weights are 1:1 with the dictionary columns): each spatial atom is
    repeated once per temporal window, and each repeat is gated by that window.
    The resulting field is ``eps(x, t) = sum_{p,j} w_{p,j} phi_j(x) chi_p(t)``.

    The repeats are laid out in temporal-major blocks — column ``p * m_s + j``
    holds spatial atom ``j`` gated by window ``p`` — so the returned spatial
    ``std`` and the temporal ``centers`` / ``widths`` line up with the columns.

    Args:
        spatial: The space-only dictionary (``m_s`` atoms).
        centers: Temporal window centres of shape ``(m_t,)``.
        widths: Temporal window widths of shape ``(m_t,)``.

    Returns:
        ``(tiled_spatial, temporal)`` with ``tiled_spatial`` of
        ``m_t * m_s`` columns and a matching
        :class:`GaussianWindowsInTime` gate.
    """
    centers = jnp.asarray(centers)
    widths = jnp.asarray(widths)
    m_s = spatial.Phi.shape[1]
    Phi = jnp.tile(spatial.Phi, (1, centers.shape[0]))  # (Ngrid, m_t * m_s)
    std = jnp.tile(spatial.std, (centers.shape[0],))  # (m_t * m_s,)
    tiled = SpatialBasis(Phi=Phi, std=std)
    temporal = GaussianWindowsInTime(
        centers=jnp.repeat(centers, m_s), widths=jnp.repeat(widths, m_s)
    )
    return tiled, temporal


def ssh_geostrophic(
    domain: Domain,
    *,
    n_scales: int = 6,
    base_scale: float = 20e3,
    slope: float = 4.0,
    amplitude: float = 2e-6,
    oversample: float = 1.0,
    windows: tuple[Float[Array, " m_t"], Float[Array, " m_t"]] | None = None,
) -> BasisForcing:
    """SSH geostrophic preset: a radial-Gabor frame with a wavenumber-law prior.

    The overcomplete multiscale Gabor frame is the workhorse behind sea-surface
    -height mapping; the prior follows the steep mesoscale wavenumber law
    ``sigma^2 ~ k ** -slope`` (``slope`` near four). By default the forcing is
    constant in time (the static-coefficient case that the existing ``jax.grad``
    parameter path handles); pass ``windows`` to spread it over Gaussian time
    windows for the time-distributed (weak-constraint) case.

    Args:
        domain: The model domain.
        n_scales: Number of dyadic scales in the frame.
        base_scale: Finest envelope scale ``L_0`` (in domain units).
        slope: Spectral slope of the wavenumber prior law.
        amplitude: Overall prior variance scale.
        oversample: Centre density per scale.
        windows: Optional ``(centers, widths)`` for the temporal gate; ``None``
            keeps the forcing constant in time.

    Returns:
        A :class:`~somax._src.core.basis.BasisForcing` with zero initial
        coefficients, ready to drop into a model RHS via
        :class:`~somax._src.core.basis.ForcingTerm`.
    """
    spatial = spatial_from_gabor(
        domain,
        n_scales=n_scales,
        base_scale=base_scale,
        slope=slope,
        amplitude=amplitude,
        oversample=oversample,
    )
    if windows is None:
        temporal: TemporalBasis = ConstantInTime(m=spatial.Phi.shape[1])
    else:
        spatial, temporal = tile_in_time(spatial, windows[0], windows[1])
    return BasisForcing(
        coeffs=jnp.zeros(spatial.Phi.shape[1]),
        spatial=spatial,
        temporal=temporal,
        grid_shape=tuple(domain.Nx),
    )


def sss_coastal(
    domain: Domain,
    centers: Float[Array, "m ndim"],
    widths: Float[Array, " m"],
    *,
    kernel: str = "wendland_c2",
    std: Float[Array, " m"] | float = 1.0,
    windows: tuple[Float[Array, " m_t"], Float[Array, " m_t"]] | None = None,
) -> BasisForcing:
    """Coastal SSS preset: placeable radial atoms with a prescribed prior.

    Sea-surface-salinity forcing localised where the physics is — radial atoms
    placed at coastlines / river mouths rather than spread over the open ocean.
    Uses a compactly supported Wendland kernel by default so each atom is
    exactly zero past its width. As with :func:`ssh_geostrophic`, ``windows``
    switches from the constant-in-time to the time-distributed regime.

    Args:
        domain: The model domain.
        centers: Atom centres of shape ``(m, ndim)`` (e.g. river-mouth locations).
        widths: Per-atom width of shape ``(m,)``.
        kernel: Radial kernel name (compact-support Wendland by default).
        std: Prescribed per-centre prior std; a scalar is broadcast.
        windows: Optional ``(centers, widths)`` for the temporal gate.

    Returns:
        A :class:`~somax._src.core.basis.BasisForcing` over the placed atoms.
    """
    spatial = spatial_from_rbf(domain, centers, widths, kernel=kernel, std=std)
    if windows is None:
        temporal: TemporalBasis = ConstantInTime(m=spatial.Phi.shape[1])
    else:
        spatial, temporal = tile_in_time(spatial, windows[0], windows[1])
    return BasisForcing(
        coeffs=jnp.zeros(spatial.Phi.shape[1]),
        spatial=spatial,
        temporal=temporal,
        grid_shape=tuple(domain.Nx),
    )


__all__ = [
    "GaussianWindowsInTime",
    "spatial_from_gabor",
    "spatial_from_rbf",
    "ssh_geostrophic",
    "sss_coastal",
    "tile_in_time",
]
