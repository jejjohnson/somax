"""2D Burgers equation expressed as a composable :class:`Term` algebra.

This is the worked example for the term-kernel decomposition (the
pipekit-refactor "Level 3" idea). The canonical :class:`Burgers2D`
hand-writes its right-hand side as one ``vector_field`` method::

    du/dt = -advection(u, v) + nu * laplacian(u)        (and likewise v)

Here the same physics is split into two reusable kernels —
:class:`Burgers2DAdvection` and :class:`Burgers2DDiffusion` — that
compose into the RHS::

    rhs = advection + nu * diffusion

Both kernels are :class:`~somax._src.core.terms.Term` instances, so the
assembled tree is differentiable end to end (the viscosity ``nu`` enters
as a :class:`~somax._src.core.terms.Scaled` coefficient, a JAX leaf) and
the assembled :class:`Burgers2DTermModel` still satisfies the somax model
contract (and hence ``pipekit_cycle.ForwardModel`` via ``step``).

The kernels are *integration-strategy neutral*: the same advection and
diffusion terms can be summed for a fully-explicit solve, or the
diffusion term can be tagged :func:`~somax._src.core.terms.implicit` so a
splitting (IMEX) solver routes the stiff Laplacian through its implicit
stage. ``Burgers2DTermModel.create(..., imex=True)`` does exactly that —
something the monolithic model cannot express without rewriting its
``build_terms``.

The accompanying tests assert this model is numerically identical to the
canonical :class:`Burgers2D` (same tendencies, same explicit trajectory),
which is the evidence that the decomposition is faithful.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    Advection2D as FVXAdvection2D,
    CartesianGrid2D,
    Difference2D,
    Interpolation2D,
    Mask2D,
    enforce_periodic,
)
from jaxtyping import Array, PyTree

from somax._src.core.model import TermModel
from somax._src.core.terms import IMPLICIT, Kind, Term, implicit
from somax._src.models.pde2d.burgers import Burgers2D, Burgers2DState


class Burgers2DAdvection(Term):
    r"""Advective tendency kernel ``-(u·∂x + v·∂y)`` for both velocities.

    Reproduces the advection half of :meth:`Burgers2D.vector_field`: the
    transport velocities are interpolated to faces and the C-grid
    advection operator is applied to each component. Marked explicit
    (the default) — advection is non-stiff.

    Args:
        advection: finitevolx C-grid advection operator.
        interp: finitevolx interpolation operator (T→U, T→V).
        method: Reconstruction method passed to the advection operator.
    """

    advection: FVXAdvection2D
    interp: Interpolation2D
    method: str = eqx.field(static=True, default="upwind1")

    def __call__(
        self, t: float, state: Burgers2DState, args: PyTree | None = None
    ) -> Burgers2DState:
        u_on_U = self.interp.T_to_U(state.u)
        v_on_V = self.interp.T_to_V(state.v)
        adv_u = self.advection(state.u, u_on_U, v_on_V, method=self.method)
        adv_v = self.advection(state.v, u_on_U, v_on_V, method=self.method)
        return Burgers2DState(u=adv_u, v=adv_v)


class Burgers2DDiffusion(Term):
    r"""Diffusive tendency kernel ``laplacian(·)`` for both velocities.

    Reproduces the (unscaled) diffusion half of
    :meth:`Burgers2D.vector_field`. The viscosity ``nu`` is *not* baked
    in here — it is applied by scaling this term (``nu * diffusion``) so
    the coefficient stays a differentiable JAX leaf and the kernel itself
    is reusable across models.

    Args:
        diff: finitevolx C-grid difference operator (provides
            ``laplacian``).
        kind: IMEX integration label. Defaults to ``"explicit"``;
            :meth:`Burgers2DTermModel.create` tags it implicit when
            ``imex=True``.
    """

    diff: Difference2D
    _kind: Kind = eqx.field(static=True, default="explicit")

    def __call__(
        self, t: float, state: Burgers2DState, args: PyTree | None = None
    ) -> Burgers2DState:
        return Burgers2DState(
            u=self.diff.laplacian(state.u),
            v=self.diff.laplacian(state.v),
        )

    @property
    def kind(self) -> Kind:
        return self._kind


class Burgers2DTermModel(TermModel):
    r"""2D Burgers model assembled from composable term kernels.

    A faithful, term-based reconstruction of :class:`Burgers2D`. The RHS
    is ``advection + nu * diffusion``; periodic boundary conditions are
    enforced before each term evaluation (matching the canonical model).

    The viscosity is **not** stored as a separate field — it lives inside
    the term tree as the :class:`~somax._src.core.terms.Scaled`
    coefficient on the diffusion kernel, which is the single source of
    truth for the dynamics (and the leaf ``jax.grad`` flows to). The
    :attr:`nu` property reads it back for inspection / diagnostics. This
    avoids duplicating a parameter outside the tree that actually drives
    integration — a leaf stored twice would let the two copies diverge
    and silently disconnect gradients.

    Args:
        terms: The assembled RHS term tree (built by :meth:`create`).
        grid: The Arakawa C-grid (static; used for diagnostics).
    """

    grid: CartesianGrid2D = eqx.field(static=True)

    def apply_boundary_conditions(self, state: Burgers2DState) -> Burgers2DState:
        """Enforce periodic boundary conditions on both velocity fields."""
        return Burgers2DState(
            u=enforce_periodic(state.u),
            v=enforce_periodic(state.v),
        )

    @property
    def nu(self) -> Array:
        """Kinematic viscosity, read back from the diffusion kernel's coeff.

        Walks the assembled term tree to the
        :class:`~somax._src.core.terms.Scaled` wrapping the
        :class:`Burgers2DDiffusion` kernel and returns its coefficient.
        On a gradient pytree this returns the gradient w.r.t. ``nu``.
        """
        from somax._src.core.terms import Scaled, _Kinded

        def _find(term: Term) -> Array | None:
            if isinstance(term, Scaled) and isinstance(term.term, Burgers2DDiffusion):
                return term.coeff
            if isinstance(term, Scaled | _Kinded):
                return _find(term.term)
            return None

        for summand in self.terms.terms:
            coeff = _find(summand)
            if coeff is not None:
                return coeff
        raise ValueError("no diffusion kernel found in term tree")

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 2.0,
        Ly: float = 2.0,
        nu: float = 0.01,
        method: str = "upwind1",
        mask: Mask2D | None = None,
        *,
        imex: bool = False,
    ) -> Burgers2DTermModel:
        """Build a term-based Burgers model.

        Args mirror :meth:`Burgers2D.create`, plus:

        Args:
            imex: When ``True``, tag the diffusion term implicit so that
                :func:`~somax._src.core.terms.build_diffrax_terms`
                produces a ``diffrax.MultiTerm`` and a splitting solver
                (e.g. ``diffrax.KenCarp3``) integrates the stiff Laplacian
                implicitly. When ``False`` (default) the whole RHS is one
                explicit ``ODETerm`` — numerically identical to
                :class:`Burgers2D`.

        Returns:
            A ``Burgers2DTermModel`` instance.
        """
        grid = CartesianGrid2D.from_interior(nx, ny, Lx, Ly)
        nu_arr = jnp.asarray(nu)

        advection = Burgers2DAdvection(
            advection=FVXAdvection2D(grid=grid, mask=mask),
            interp=Interpolation2D(grid=grid, mask=mask),
            method=method,
        )
        diffusion = Burgers2DDiffusion(
            diff=Difference2D(grid=grid, mask=mask),
            _kind=IMPLICIT if imex else "explicit",
        )

        scaled_diffusion = nu_arr * diffusion
        rhs: Term = advection + (
            implicit(scaled_diffusion) if imex else scaled_diffusion
        )

        return Burgers2DTermModel(terms=rhs, grid=grid)

    @staticmethod
    def from_model(model: Burgers2D, *, imex: bool = False) -> Burgers2DTermModel:
        """Build a term model matching an existing :class:`Burgers2D`.

        Reuses the source model's grid, mask, and advection method so the
        two are numerically comparable.

        Args:
            model: A canonical Burgers2D instance.
            imex: Tag diffusion implicit (see :meth:`create`).

        Returns:
            A ``Burgers2DTermModel`` mirroring ``model``.
        """
        nu_arr = jnp.asarray(model.params.nu)
        advection = Burgers2DAdvection(
            advection=model.advection,
            interp=model.interp,
            method=model.method,
        )
        diffusion = Burgers2DDiffusion(
            diff=model.diff,
            _kind=IMPLICIT if imex else "explicit",
        )
        scaled_diffusion = nu_arr * diffusion
        rhs: Term = advection + (
            implicit(scaled_diffusion) if imex else scaled_diffusion
        )
        return Burgers2DTermModel(terms=rhs, grid=model.grid)
