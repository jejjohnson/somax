"""Characteristic scales for non-dimensionalising ocean models.

A :class:`Scales` object records the length, velocity, depth, time and
rotation scales that turn a dimensional model into a nondimensional one.
It is deliberately inert: it holds numbers and derives dimensionless
groups from them. Turning those numbers into an actual coordinate change
is :class:`somax._src.core.transforms.StateAffine`'s job, and turning
them into model parameters is each model's ``from_nondimensional``
factory.

Every field is ``static=True``, so a ``Scales`` is a compile-time
constant and dimensionless groups can be used in Python control flow
(resolution guards, ``if`` branches in a factory) without tracing.

There is no universal scale set. The characteristic time in particular
is *part of* the choice, not something derivable from ``L`` and ``U``:
quasi-geostrophic models are advective (``T = L/U``), shallow-water
models are inertial (``T = 1/f0``), spherical models are planetary
(``T = 1/Omega``), and a pure diffusion problem has no velocity scale at
all, so its time scale is diffusive (``T = L**2/kappa``). Pick the set
that matches the equation family with the corresponding classmethod,
and read it back off :attr:`Scales.kind`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import equinox as eqx


ScaleKind = Literal["advective", "inertial", "planetary", "diffusive"]

#: Standard gravity (m/s^2), the default for the ``g`` field.
GRAVITY = 9.81


class Scales(eqx.Module):
    """Characteristic scales of a run. All fields static.

    Attributes:
        L: Horizontal length scale [m].
        U: Velocity scale [m/s].
        H: Depth / layer-thickness scale [m].
        f0: Reference Coriolis parameter [1/s]. For the planetary set
            this is ``2 * Omega``, so that ``f(phi) = f0 * sin(phi)``
            and the Rossby number keeps its usual definition.
        T: Time scale [s]. Set by the constructor for the chosen family
            rather than derived, because the families disagree about it.
        g: Gravitational acceleration [m/s^2].
        kind: Which canonical set this is — ``"advective"``,
            ``"inertial"`` or ``"planetary"``. Recorded so that
            transforms, factories and the CLI can dispatch on it.
    """

    L: float = eqx.field(static=True)
    U: float = eqx.field(static=True)
    H: float = eqx.field(static=True)
    f0: float = eqx.field(static=True)
    T: float = eqx.field(static=True)
    g: float = eqx.field(static=True, default=GRAVITY)
    kind: ScaleKind = eqx.field(static=True, default="advective")

    # ------------------------------------------------------------------
    # Canonical scale sets, one per equation family
    # ------------------------------------------------------------------

    @classmethod
    def advective(
        cls,
        L: float,
        U: float,
        f0: float = 1.0,
        H: float = 1.0,
        g: float = GRAVITY,
    ) -> Scales:
        """Advective set: ``T = L / U``.

        The set for quasi-geostrophic models (barotropic, baroclinic,
        reparameterized) and for the pde1d / pde2d family, where the
        eddy turnover time is the time scale of interest.

        Args:
            L: Horizontal length scale (basin or eddy scale) [m].
            U: Velocity scale [m/s].
            f0: Reference Coriolis parameter [1/s]. Defaults to 1 for
                the non-rotating pde models, where it is unused.
            H: Depth scale [m].
            g: Gravitational acceleration [m/s^2].

        Returns:
            A ``Scales`` with ``kind="advective"``.

        Raises:
            ValueError: If ``L``, ``U``, ``H`` or ``g`` is not strictly
                positive.
        """
        _require_positive(L=L, U=U, H=H, g=g)
        return cls(L=L, U=U, H=H, f0=f0, T=L / U, g=g, kind="advective")

    @classmethod
    def inertial(
        cls,
        L: float,
        f0: float,
        H: float,
        rossby: float,
        g: float = GRAVITY,
    ) -> Scales:
        """Inertial set: ``T = 1 / f0`` and ``U = f0 * L * Ro``.

        The set for the shallow-water family, where the inertial period
        rather than the eddy turnover sets the time scale, and the
        velocity follows from the Rossby number instead of being given.

        Args:
            L: Horizontal length scale [m].
            f0: Reference Coriolis parameter [1/s].
            H: Depth / layer-thickness scale [m].
            rossby: Rossby number ``Ro = U / (f0 * L)``, which fixes
                the velocity scale.
            g: Gravitational acceleration [m/s^2].

        Returns:
            A ``Scales`` with ``kind="inertial"``.

        Raises:
            ValueError: If ``L``, ``H``, ``f0``, ``rossby`` or ``g`` is
                not strictly positive.
        """
        _require_positive(L=L, H=H, f0=f0, rossby=rossby, g=g)
        return cls(L=L, U=f0 * L * rossby, H=H, f0=f0, T=1.0 / f0, g=g, kind="inertial")

    @classmethod
    def planetary(
        cls,
        a: float,
        Omega: float,
        H: float,
        rossby: float,
        g: float = GRAVITY,
    ) -> Scales:
        """Planetary set: ``L = a``, ``T = 1/Omega``, ``U = 2 * Omega * a * Ro``.

        The set for spherical models, whose natural length scale is the
        planet radius. ``f0`` is stored as ``2 * Omega`` so that
        ``f(phi) = f0 * sin(phi)`` and :attr:`rossby` keeps its usual
        ``U / (f0 * L)`` definition — which is the conventional
        spherical Rossby number ``U / (2 * Omega * a)``.

        Args:
            a: Planet radius [m].
            Omega: Planet angular velocity [rad/s].
            H: Equivalent depth [m].
            rossby: Rossby number ``Ro = U / (2 * Omega * a)``.
            g: Gravitational acceleration [m/s^2].

        Returns:
            A ``Scales`` with ``kind="planetary"``.

        Raises:
            ValueError: If ``a``, ``Omega``, ``H``, ``rossby`` or ``g``
                is not strictly positive.
        """
        _require_positive(a=a, Omega=Omega, H=H, rossby=rossby, g=g)
        return cls(
            L=a,
            U=2.0 * Omega * a * rossby,
            H=H,
            f0=2.0 * Omega,
            T=1.0 / Omega,
            g=g,
            kind="planetary",
        )

    @classmethod
    def diffusive(
        cls,
        L: float,
        kappa: float,
        g: float = GRAVITY,
    ) -> Scales:
        """Diffusive set: ``T = L**2 / kappa``.

        The set for a pure diffusion problem, which has no velocity to
        build a time scale from. The implied velocity ``U = L/T =
        kappa/L`` is recorded so the derived properties stay
        well-defined, but it is a bookkeeping quantity rather than a
        physical flow speed.

        Under this scaling the nondimensional diffusivity is exactly 1,
        so a diffusion model has no free dimensionless number left.

        Args:
            L: Horizontal length scale [m].
            kappa: Diffusivity [m^2/s].
            g: Gravitational acceleration [m/s^2].

        Returns:
            A ``Scales`` with ``kind="diffusive"``.

        Raises:
            ValueError: If ``L``, ``kappa`` or ``g`` is not strictly
                positive.
        """
        _require_positive(L=L, kappa=kappa, g=g)
        return cls(
            L=L,
            U=kappa / L,
            H=1.0,
            f0=1.0,
            T=L**2 / kappa,
            g=g,
            kind="diffusive",
        )

    # ------------------------------------------------------------------
    # Derived dimensionless groups
    # ------------------------------------------------------------------

    @property
    def rossby(self) -> float:
        """Rossby number ``Ro = U / (f0 * L)``."""
        return self.U / (self.f0 * self.L)

    @property
    def burger(self) -> float:
        """Burger number ``Bu = g * H / (f0 * L)**2``."""
        return (self.g * self.H) / (self.f0 * self.L) ** 2

    @property
    def froude(self) -> float:
        """Froude number ``Fr = U / sqrt(g * H)``."""
        return self.U / math.sqrt(self.g * self.H)

    @property
    def lamb(self) -> float:
        """Lamb parameter ``eps = (f0 * L)**2 / (g * H)``, the inverse Burger.

        For the planetary set this is the usual
        ``4 * Omega**2 * a**2 / (g * H)``.
        """
        return 1.0 / self.burger

    @property
    def eta(self) -> float:
        """Geostrophic surface-height scale ``f0 * U * L / g`` [m].

        The thickness anomaly in geostrophic balance with a velocity
        ``U`` over a length ``L`` — the ``scale`` a transform gives to
        an ``h`` or ``eta`` field.
        """
        return self.f0 * self.U * self.L / self.g

    @property
    def vorticity(self) -> float:
        """Relative-vorticity scale ``U / L`` [1/s] — the scale for ``q``."""
        return self.U / self.L

    @property
    def streamfunction(self) -> float:
        """Streamfunction scale ``U * L`` [m^2/s] — the scale for ``psi``."""
        return self.U * self.L

    @property
    def omega(self) -> float:
        """Planet angular velocity ``f0 / 2`` [rad/s]."""
        return self.f0 / 2.0

    def dt_from_cfl(
        self,
        cfl: float,
        n_cells: int | Sequence[int],
        *,
        mode: str | None = None,
        diffusivity: float | None = None,
        extent: float | Sequence[float] | None = None,
    ) -> float:
        r"""Largest time step meeting a CFL target, in this set's time unit.

        Multi-dimensional by construction: pass one cell count per axis
        and the bound accounts for all of them. A single ``int`` is the
        1-D case.

        The advective bound is :math:`C \min_i \Delta x_i / \hat u`
        and the diffusive one
        :math:`C / (2 \hat\kappa \sum_i \Delta x_i^{-2})`, both in
        nondimensional units, where
        :math:`\hat u = U T / L` is the nondimensional velocity and
        :math:`\hat\kappa = \kappa T / L^2` the nondimensional
        diffusivity.

        Args:
            cfl: Target Courant number.
            n_cells: Interior cells across each axis. An ``int`` means
                one axis; a sequence gives one count per axis.
            mode: ``"advective"`` or ``"diffusive"``. Defaults to the
                diffusive form for a diffusive scale set and the
                advective form otherwise.
            diffusivity: Nondimensional diffusivity :math:`\kappa T/L^2`
                for the diffusive bound. For the models whose factories
                take a Reynolds or Péclet number this is ``1/Re`` or
                ``1/Pe`` — the value the model actually carries, which
                the scale set cannot know. Defaults to 1 for a
                diffusive scale set, where the scaling makes it exactly
                1 by construction; required otherwise.
            extent: Domain length along each axis in units of ``L``.
                Defaults to 1 per axis; pass ``(1.0, aspect)`` for a
                rectangular 2-D domain, whose ``y`` spacing is
                ``aspect/ny`` rather than ``1/ny``.

        Returns:
            The time step, expressed in units of :attr:`T` — the unit a
            nondimensional model built from these scales expects.

        Raises:
            ValueError: If ``cfl``, a cell count or an extent is not
                positive, if ``mode`` is unrecognised, if ``extent``
                and ``n_cells`` disagree in length, or if the diffusive
                bound is asked for without a diffusivity on a
                non-diffusive set.
        """
        _require_positive(cfl=cfl)
        counts = (n_cells,) if isinstance(n_cells, int) else tuple(n_cells)
        if not counts:
            raise ValueError("dt_from_cfl: n_cells must name at least one axis.")
        for axis, count in enumerate(counts):
            _require_positive(**{f"n_cells[{axis}]": float(count)})

        if extent is None:
            extents = (1.0,) * len(counts)
        else:
            extents = (
                (float(extent),)
                if isinstance(extent, (int, float))
                else tuple(float(e) for e in extent)
            )
        if len(extents) != len(counts):
            raise ValueError(
                f"dt_from_cfl: extent names {len(extents)} axes but n_cells "
                f"names {len(counts)}."
            )
        for axis, value in enumerate(extents):
            _require_positive(**{f"extent[{axis}]": value})

        spacings = tuple(e / c for e, c in zip(extents, counts, strict=True))

        if mode is None:
            mode = "diffusive" if self.kind == "diffusive" else "advective"

        if mode == "advective":
            # u_hat = U T / L: the speed the nondimensional model
            # actually carries. Dividing by it — the advective set is
            # the only one where it is 1.
            velocity = self.U * self.T / self.L
            return cfl * min(spacings) / velocity
        if mode == "diffusive":
            if diffusivity is None:
                if self.kind != "diffusive":
                    raise ValueError(
                        "dt_from_cfl: the diffusive bound needs the model's "
                        "nondimensional diffusivity (1/Re for the Burgers and "
                        "Navier-Stokes factories, 1/Pe where a Peclet number "
                        "is used). Only a diffusive scale set fixes it at 1."
                    )
                diffusivity = 1.0
            _require_positive(diffusivity=diffusivity)
            inverse_squares = sum(1.0 / spacing**2 for spacing in spacings)
            return cfl / (2.0 * diffusivity * inverse_squares)
        raise ValueError(
            f"dt_from_cfl: mode must be 'advective' or 'diffusive'; got {mode!r}."
        )

    def nondimensional(self) -> Scales:
        """The unit scale set of the same family.

        What a ``from_nondimensional`` factory builds its model in:
        ``L = 1`` and the family's own choice of what else is unity
        (``U = 1`` when advective, ``f0 = 1`` when inertial, ``Omega =
        1`` when planetary). Useful in tests, to state that a
        nondimensional run really is running at unit scales.

        Gravity is *not* carried across. Every other scale becomes 1,
        so keeping the SI ``9.81`` would change :attr:`burger`,
        :attr:`froude`, :attr:`lamb` and :attr:`eta`, handing a factory
        a different physical problem. It is instead set to whatever
        reproduces this set's Burger number at the new unit scales,
        ``g' = Bu (f0' L')**2 / H'`` — which is ``g H / U**2`` for the
        advective set, ``g H / (f0 L)**2`` for the inertial one, and
        four times that for the planetary one, whose ``f0'`` is ``2``.

        Returns:
            A ``Scales`` of the same ``kind`` with unit scales and the
            same dimensionless groups.
        """
        burger = self.burger
        if self.kind == "advective":
            # f0' = 1/Ro is what keeps the Rossby number, so the
            # gravity that keeps the Burger number scales with it.
            f0 = 1.0 / self.rossby
            return Scales.advective(L=1.0, U=1.0, f0=f0, H=1.0, g=burger * f0**2)
        if self.kind == "inertial":
            return Scales.inertial(L=1.0, f0=1.0, H=1.0, rossby=self.rossby, g=burger)
        if self.kind == "diffusive":
            return Scales.diffusive(L=1.0, kappa=1.0, g=burger)
        return Scales.planetary(
            a=1.0, Omega=1.0, H=1.0, rossby=self.rossby, g=4.0 * burger
        )


def _require_positive(**values: float) -> None:
    """Raise if any named value is not strictly positive or is not finite."""
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Scales: {name} must be a finite positive number; got {value!r}."
            )
