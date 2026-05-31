"""2D linear shallow water as a composable :class:`Term` algebra.

The second worked example of the term-kernel decomposition (after
:mod:`somax._src.models.pde2d.burgers_terms`). Where Burgers split into
two terms, the linear shallow water right-hand side splits cleanly into
*four* distinct physics kernels::

    gravity wave : dh/dt = -H0 * div(u, v)
                   du/dt = -g * dh/dx,   dv/dt = -g * dh/dy
    coriolis     : du/dt += f*v,         dv/dt -= f*u
    diffusion    : du/dt += nu*lap(u),   dv/dt += nu*lap(v)
    drag         : du/dt -= kappa*u,     dv/dt -= kappa*v

which compose into the RHS::

    rhs = gravity_wave + coriolis + nu * diffusion + (-kappa) * drag

The differentiable parameters enter as
:class:`~somax._src.core.terms.Scaled` coefficients (``nu`` on diffusion,
``-kappa`` on drag), so they stay JAX leaves; ``g``/``H0``/``f`` are
frozen constants baked into their kernels.

Two of the four terms are stiff — the gravity-wave operator (fast surface
waves) and the Laplacian diffusion — making this a richer IMEX showcase
than Burgers. ``LinearSWM2DTermModel.create(..., imex=True)`` tags the
diffusion term implicit so a splitting solver integrates it through its
implicit stage; the gravity-wave term is a further candidate.

The assembled :class:`LinearSWM2DTermModel` reproduces
:class:`~somax._src.models.swm.linear_2d.LinearShallowWater2D` exactly
(the tests assert identical tendencies and explicit trajectories).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import CartesianGrid2D, Coriolis2D, Difference2D, enforce_periodic
from jaxtyping import Array, PyTree

from somax._src.core.model import TermModel
from somax._src.core.terms import (
    IMPLICIT,
    Kind,
    Scaled,
    Sum,
    Term,
    _Kinded,
    implicit,
)
from somax._src.models.swm.linear_2d import LinearShallowWater2D, LinearSW2DState


class LinearSWMGravityWave(Term):
    r"""Gravity-wave / pressure kernel of the linear shallow water RHS.

    ``dh/dt = -H0 * div(u, v)``, ``du/dt = -g * dh/dx``,
    ``dv/dt = -g * dh/dy``. The fast surface-wave operator; the natural
    candidate for the implicit stage of a semi-implicit scheme.

    Args:
        diff: finitevolx C-grid difference operator.
        g: Gravitational acceleration (frozen).
        H0: Mean layer depth (frozen).
    """

    diff: Difference2D
    g: float = eqx.field(static=True)
    H0: float = eqx.field(static=True)

    def __call__(
        self, t: float, state: LinearSW2DState, args: PyTree | None = None
    ) -> LinearSW2DState:
        return LinearSW2DState(
            h=-self.H0 * self.diff.divergence(state.u, state.v),
            u=-self.g * self.diff.diff_x_T_to_U(state.h),
            v=-self.g * self.diff.diff_y_T_to_V(state.h),
        )


class LinearSWMCoriolis(Term):
    r"""Coriolis kernel: ``du/dt += f*v``, ``dv/dt -= f*u`` (no height term).

    Args:
        coriolis: finitevolx C-grid Coriolis operator.
        f_field: Precomputed Coriolis parameter field ``f(y)``.
    """

    coriolis: Coriolis2D
    f_field: Array

    def __call__(
        self, t: float, state: LinearSW2DState, args: PyTree | None = None
    ) -> LinearSW2DState:
        du_cor, dv_cor = self.coriolis(state.u, state.v, self.f_field)
        return LinearSW2DState(h=jnp.zeros_like(state.h), u=du_cor, v=dv_cor)


class LinearSWMDiffusion(Term):
    r"""Lateral diffusion kernel: ``laplacian(u)``, ``laplacian(v)``.

    Unscaled (the viscosity ``nu`` is applied via
    :class:`~somax._src.core.terms.Scaled` so it stays a differentiable
    leaf). Stiff — tag implicit for IMEX integration.

    Args:
        diff: finitevolx C-grid difference operator.
        kind: IMEX integration label (``"explicit"`` by default).
    """

    diff: Difference2D
    _kind: Kind = eqx.field(static=True, default="explicit")

    def __call__(
        self, t: float, state: LinearSW2DState, args: PyTree | None = None
    ) -> LinearSW2DState:
        return LinearSW2DState(
            h=jnp.zeros_like(state.h),
            u=self.diff.laplacian(state.u),
            v=self.diff.laplacian(state.v),
        )

    @property
    def kind(self) -> Kind:
        return self._kind


class LinearSWMDrag(Term):
    r"""Linear bottom-drag kernel returning ``(0, u, v)``.

    Unscaled; the drag coefficient ``kappa`` enters via a ``Scaled`` with
    a negative coefficient (``-kappa * drag``) so it stays differentiable.
    """

    def __call__(
        self, t: float, state: LinearSW2DState, args: PyTree | None = None
    ) -> LinearSW2DState:
        return LinearSW2DState(h=jnp.zeros_like(state.h), u=state.u, v=state.v)


def _assemble(
    *,
    diff: Difference2D,
    coriolis: Coriolis2D,
    f_field: Array,
    g: float,
    H0: float,
    nu: Array,
    kappa: Array,
    imex: bool,
) -> Term:
    """Build the ``gravity + coriolis + nu*diffusion - kappa*drag`` tree."""
    gravity = LinearSWMGravityWave(diff=diff, g=g, H0=H0)
    coriolis_term = LinearSWMCoriolis(coriolis=coriolis, f_field=f_field)
    diffusion = LinearSWMDiffusion(diff=diff, _kind=IMPLICIT if imex else "explicit")
    scaled_diffusion: Term = Scaled(diffusion, nu)
    if imex:
        scaled_diffusion = implicit(scaled_diffusion)
    drag = Scaled(LinearSWMDrag(), -kappa)
    return gravity + coriolis_term + scaled_diffusion + drag


class LinearSWM2DTermModel(TermModel):
    r"""2D linear shallow water assembled from composable term kernels.

    A faithful, term-based reconstruction of
    :class:`~somax._src.models.swm.linear_2d.LinearShallowWater2D`: same
    RHS, same boundary conditions. The viscosity / drag coefficients live
    in the term tree (as ``Scaled`` coefficients) and are read back via
    :attr:`nu` / :attr:`kappa`.

    Args:
        terms: The assembled RHS term tree (built by :meth:`create` /
            :meth:`from_model`).
        grid: The Arakawa C-grid (static; for diagnostics).
        bc_type: ``"periodic"`` or ``"wall"`` (static).
    """

    grid: CartesianGrid2D = eqx.field(static=True)
    bc_type: str = eqx.field(static=True, default="periodic")

    def apply_boundary_conditions(self, state: LinearSW2DState) -> LinearSW2DState:
        """Apply boundary conditions (periodic or free-slip wall).

        Mirrors :meth:`LinearShallowWater2D.apply_boundary_conditions`.
        """
        if self.bc_type == "periodic":
            return LinearSW2DState(
                h=enforce_periodic(state.h),
                u=enforce_periodic(state.u),
                v=enforce_periodic(state.v),
            )
        h, u, v = state.h, state.u, state.v
        # No normal flow: wall faces AND ghost faces.
        u = u.at[:, 0].set(0.0).at[:, -2].set(0.0).at[:, -1].set(0.0)
        v = v.at[0, :].set(0.0).at[-2, :].set(0.0).at[-1, :].set(0.0)
        # Free-slip: tangential velocity ghost = adjacent interior.
        u = u.at[0, :].set(u[1, :]).at[-1, :].set(u[-2, :])
        v = v.at[:, 0].set(v[:, 1]).at[:, -1].set(v[:, -2])
        # Height: zero-gradient at ghost cells.
        h = h.at[0, :].set(h[1, :]).at[-1, :].set(h[-2, :])
        h = h.at[:, 0].set(h[:, 1]).at[:, -1].set(h[:, -2])
        return LinearSW2DState(h=h, u=u, v=v)

    @property
    def nu(self) -> Array:
        """Lateral viscosity, read back from the diffusion kernel's coeff."""
        return _coeff_of(self.terms, LinearSWMDiffusion)

    @property
    def kappa(self) -> Array:
        """Bottom drag, read back as ``-`` the drag kernel's coeff."""
        return -_coeff_of(self.terms, LinearSWMDrag)

    @staticmethod
    def from_model(
        model: LinearShallowWater2D, *, imex: bool = False
    ) -> LinearSWM2DTermModel:
        """Build a term model matching an existing ``LinearShallowWater2D``.

        Reuses the source model's operators, Coriolis field, constants,
        and BC type so the two are numerically comparable.

        Args:
            model: A canonical LinearShallowWater2D instance.
            imex: Tag diffusion implicit (see module docstring).

        Returns:
            A ``LinearSWM2DTermModel`` mirroring ``model``.
        """
        rhs = _assemble(
            diff=model.diff,
            coriolis=model.coriolis,
            f_field=model.f_field,
            g=model.consts.gravity,
            H0=model.consts.H0,
            nu=jnp.asarray(model.params.lateral_viscosity),
            kappa=jnp.asarray(model.params.bottom_drag),
            imex=imex,
        )
        return LinearSWM2DTermModel(terms=rhs, grid=model.grid, bc_type=model.bc_type)

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 1e6,
        Ly: float = 1e6,
        g: float = 9.81,
        f0: float = 1e-4,
        beta: float = 0.0,
        H0: float = 100.0,
        lateral_viscosity: float = 0.0,
        bottom_drag: float = 0.0,
        bc: str = "periodic",
        *,
        imex: bool = False,
    ) -> LinearSWM2DTermModel:
        """Build a term-based linear SWM model.

        Args mirror :meth:`LinearShallowWater2D.create`, plus ``imex``
        (tag the diffusion term implicit). Delegates construction to the
        canonical model's factory, then wraps it via :meth:`from_model`,
        so the operators and Coriolis field match exactly.
        """
        base = LinearShallowWater2D.create(
            nx=nx,
            ny=ny,
            Lx=Lx,
            Ly=Ly,
            g=g,
            f0=f0,
            beta=beta,
            H0=H0,
            lateral_viscosity=lateral_viscosity,
            bottom_drag=bottom_drag,
            bc=bc,
        )
        return LinearSWM2DTermModel.from_model(base, imex=imex)


def _coeff_of(root: Term, kernel_cls: type) -> Array:
    """Return the ``Scaled`` coefficient wrapping a ``kernel_cls`` kernel."""

    def _find(term: Term) -> Array | None:
        if isinstance(term, Scaled) and isinstance(term.term, kernel_cls):
            return term.coeff
        if isinstance(term, Scaled | _Kinded):
            return _find(term.term)
        return None

    summands = root.terms if isinstance(root, Sum) else (root,)
    for summand in summands:
        coeff = _find(summand)
        if coeff is not None:
            return coeff
    raise ValueError(f"no {kernel_cls.__name__} kernel found in term tree")
