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

import math

import jax.numpy as jnp
import numpy as np
from geonnax.basis import (
    eof_basis,
    fourier_basis,
    gabor_frame_grid,
    gaussian_window_features,
    graph_laplacian_eigpairs,
    rbf_basis,
    wavelet_basis_2d,
)
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


# --- spectral / data-driven / wavelet builders -------------------------------
#
# These consume the geonnax bases that became public in geonnax#25
# (fourier_basis, graph_laplacian_eigpairs, eof_basis, wavelet_basis_2d). They
# all return a scalar SpatialBasis that drops straight into BasisForcing; the
# vector divfree_basis and the on-sphere spherical_rbf_basis need a vector
# placement seam / sphere-coordinate domain respectively and are left as
# follow-ups (see content/notes/forcing_basis.md).


def matern_spectral_density(
    sqrt_lambda: Float[Array, " m"],
    *,
    variance: float = 1.0,
    length_scale: float = 1.0,
    nu: float = 1.5,
    ndim: int = 2,
) -> Float[Array, " m"]:
    r"""Matérn power spectral density ``S(omega)`` at ``omega = sqrt_lambda``.

    The Hilbert-space-GP prior variance of a Laplacian eigenmode with eigenvalue
    ``lambda`` is ``S(sqrt(lambda))`` (Solin & Särkkä 2020), so pairing this with
    :func:`spatial_from_fourier` builds a reduced-rank Matérn / SPDE field. With
    ``kappa = sqrt(2 nu) / length_scale`` the density is

    ``S(omega) = variance * c * (kappa^2 + omega^2) ** -(nu + ndim/2)``,

    ``c = 2^ndim pi^(ndim/2) Gamma(nu + ndim/2) (2 nu)^nu /
    (Gamma(nu) length_scale^(2 nu))``.

    geonnax deliberately keeps kernel spectral densities in the consuming
    library, so this closed-form (kernel-class-free) helper lives here.

    Args:
        sqrt_lambda: Square-root Laplacian eigenvalues ``omega`` of shape ``(m,)``.
        variance: Marginal variance ``sigma^2``.
        length_scale: Matérn length scale ``ell``.
        nu: Smoothness ``nu``.
        ndim: Spatial dimension ``d``.

    Returns:
        Per-mode variance of shape ``(m,)``.
    """
    kappa2 = 2.0 * nu / (length_scale**2)
    alpha = nu + ndim / 2.0
    const = (
        variance
        * (2.0**ndim)
        * (math.pi ** (ndim / 2.0))
        * math.gamma(alpha)
        * (2.0 * nu) ** nu
        / (math.gamma(nu) * length_scale ** (2.0 * nu))
    )
    return const * (kappa2 + sqrt_lambda**2) ** (-alpha)


def spatial_from_fourier(
    domain: Domain,
    *,
    num_basis_per_dim: int | tuple[int, ...],
    length_scale: float = 1.0,
    nu: float = 1.5,
    variance: float = 1.0,
) -> SpatialBasis:
    """Build a :class:`SpatialBasis` from the box-Laplacian (HSGP) eigenbasis.

    Evaluates geonnax :func:`~geonnax.basis.fourier_basis` on ``domain.coords``
    (shifted to the centred box ``[-L, L]^ndim``) and sets the per-mode prior
    std to the Matérn spectral density at the eigen-wavenumbers — the
    reduced-rank Matérn / SPDE construction. This is the principled smooth-field
    prior for variables like SST.

    Args:
        domain: The model domain.
        num_basis_per_dim: Per-axis number of 1D modes (``int`` broadcasts).
        length_scale: Matérn length scale.
        nu: Matérn smoothness.
        variance: Marginal variance.

    Returns:
        A :class:`SpatialBasis` with the Matérn HSGP prior std.
    """
    coords = domain.coords  # (Ngrid, ndim)
    xmin = jnp.asarray(domain.xmin)
    xmax = jnp.asarray(domain.xmax)
    centre = 0.5 * (xmin + xmax)
    half = 0.5 * (xmax - xmin)
    xy = coords - centre  # centred on the box [-half, half]
    L = tuple(float(h) for h in half)
    Phi, lam = fourier_basis(xy, num_basis_per_dim, L)
    var = matern_spectral_density(
        jnp.sqrt(lam),
        variance=variance,
        length_scale=length_scale,
        nu=nu,
        ndim=coords.shape[1],
    )
    return SpatialBasis(Phi=Phi, std=jnp.sqrt(var))


def spatial_from_graph_laplacian(
    adjacency: Float[Array, "V V"],
    n_modes: int,
    *,
    normalized: bool = True,
    regularization: float = 1e-3,
    smoothness: float = 2.0,
) -> SpatialBasis:
    """Build a :class:`SpatialBasis` from graph-Laplacian eigenvectors.

    Evaluates geonnax :func:`~geonnax.basis.graph_laplacian_eigpairs` and uses
    the low-frequency eigenvectors as the dictionary — the natural basis on an
    irregular / masked grid (e.g. an ocean basin with land removed), where the
    adjacency encodes the connectivity. The GMRF-style prior std decays with the
    eigenvalue, ``std = (lambda + regularization) ** (-smoothness / 2)``, so
    smooth (low-frequency) modes carry the most variance.

    Args:
        adjacency: Symmetric non-negative adjacency of shape ``(V, V)`` over the
            ``V`` (unmasked) grid nodes.
        n_modes: Number of low-frequency eigenpairs to keep.
        normalized: Use the symmetric normalized Laplacian if ``True``.
        regularization: Added to eigenvalues to bound the zero-mode variance.
        smoothness: Exponent of the eigenvalue decay in the prior.

    Returns:
        A :class:`SpatialBasis` whose ``Phi`` is ``(V, n_modes)``.
    """
    eigvals, eigvecs = graph_laplacian_eigpairs(
        jnp.asarray(adjacency), n_modes, normalized=normalized
    )
    std = (eigvals + regularization) ** (-0.5 * smoothness)
    return SpatialBasis(Phi=eigvecs, std=std)


def spatial_from_eof(
    data: Float[Array, "T N"],
    n_modes: int,
    *,
    center: bool = True,
) -> SpatialBasis:
    """Build a :class:`SpatialBasis` from empirical orthogonal functions (PCA).

    Evaluates geonnax :func:`~geonnax.basis.eof_basis` on a ``(T, Ngrid)`` data
    matrix (e.g. a stack of anomaly snapshots) and uses the leading EOFs as the
    dictionary — the data-driven reduced basis (DINEOF). The prior std is the
    per-mode sample standard deviation ``sigma_a / sqrt(T - 1)``.

    Args:
        data: Data matrix of shape ``(T, Ngrid)``.
        n_modes: Number of leading EOFs to keep.
        center: Subtract the sample mean before the SVD.

    Returns:
        A :class:`SpatialBasis` over the leading EOFs.
    """
    data = jnp.asarray(data)
    Phi, singular_values = eof_basis(data, n_modes, center=center)
    std = singular_values / jnp.sqrt(jnp.asarray(max(data.shape[0] - 1, 1), Phi.dtype))
    return SpatialBasis(Phi=Phi, std=std)


def spatial_from_wavelet(
    domain: Domain,
    *,
    wavelet: str = "haar",
    levels: int | None = None,
    std: Float[Array, " m"] | float = 1.0,
) -> SpatialBasis:
    """Build a :class:`SpatialBasis` from the orthonormal 2D wavelet basis.

    Evaluates geonnax :func:`~geonnax.basis.wavelet_basis_2d` for the domain's
    ``(Ny, Nx)`` grid (both must be powers of two) — a critically-sampled,
    non-redundant multiscale dictionary, the orthonormal counterpart to the
    Gabor frame. The orthonormal basis has no intrinsic spectrum, so the prior
    std is prescribed (defaulting to ones).

    Args:
        domain: A 2D model domain with power-of-two ``Nx``.
        wavelet: ``"haar"``, ``"db2"``, or ``"db4"``.
        levels: Decomposition levels (defaults to the full cascade).
        std: Prescribed per-mode prior std; a scalar is broadcast.

    Returns:
        A :class:`SpatialBasis` whose ``Phi`` is ``(Ngrid, Ngrid)`` orthonormal.

    Raises:
        ValueError: If the domain is not 2D.
    """
    if domain.ndim != 2:
        raise ValueError(f"spatial_from_wavelet needs a 2D domain; got {domain.ndim}D.")
    nx_shape = tuple(domain.Nx)
    ny, nx = int(nx_shape[0]), int(nx_shape[1])
    Phi = wavelet_basis_2d(ny, nx, wavelet=wavelet, levels=levels)  # (Ngrid, Ngrid)
    std_arr = jnp.broadcast_to(jnp.asarray(std, dtype=Phi.dtype), (Phi.shape[1],))
    return SpatialBasis(Phi=Phi, std=std_arr)


def sst_frontal(
    domain: Domain,
    *,
    num_basis_per_dim: int | tuple[int, ...] = 12,
    length_scale: float = 1.0,
    nu: float = 1.5,
    variance: float = 1.0,
    windows: tuple[Float[Array, " m_t"], Float[Array, " m_t"]] | None = None,
) -> BasisForcing:
    """SST preset: a smooth Matérn (HSGP) field over the box-Laplacian eigenbasis.

    A principled smooth-field prior for sea-surface temperature: the box
    eigenbasis weighted by the Matérn spectral density. Constant in time by
    default; pass ``windows`` for the time-distributed regime.

    Args:
        domain: The model domain.
        num_basis_per_dim: Per-axis number of 1D modes.
        length_scale: Matérn length scale.
        nu: Matérn smoothness.
        variance: Marginal variance.
        windows: Optional ``(centers, widths)`` Gaussian temporal gate.

    Returns:
        A :class:`~somax._src.core.basis.BasisForcing`.
    """
    spatial = spatial_from_fourier(
        domain,
        num_basis_per_dim=num_basis_per_dim,
        length_scale=length_scale,
        nu=nu,
        variance=variance,
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


__all__ = [
    "GaussianWindowsInTime",
    "matern_spectral_density",
    "spatial_from_eof",
    "spatial_from_fourier",
    "spatial_from_gabor",
    "spatial_from_graph_laplacian",
    "spatial_from_rbf",
    "spatial_from_wavelet",
    "ssh_geostrophic",
    "sss_coastal",
    "sst_frontal",
    "tile_in_time",
]
