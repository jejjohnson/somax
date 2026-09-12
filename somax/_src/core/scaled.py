"""Integrate a model in transformed state coordinates.

:class:`ScaledModel` wraps any :class:`~somax._src.core.model.SomaxModel`
with a :class:`~somax._src.core.transforms.StateAffine` and a time
scale, so the solver sees transformed coordinates while the inner model
keeps its own. Because the chain rule for an affine map is exact, the
same wrapper serves both jobs:

* **Non-dimensionalisation** — the transform comes from a
  :class:`~somax._src.core.scales.Scales`, and the wrapped model
  integrates in nondimensional state and time.
* **Standardisation** — the transform comes from sample statistics, and
  a dimensional model is integrated on unit-variance state. This is the
  machine-learning baseline case: learned corrections, ensembles and
  optimisers all see O(1) numbers without the physics being touched.

Diagnostics stay in the inner model's units. ``ConservationDriftMonitor``
reports *relative* drift, which is scale-invariant, so a nondimensional
run gets the same conservation signal as a dimensional one and nothing
needs re-dimensionalising.
"""

from __future__ import annotations

import math
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax.tree_util as jtu
import paramax
from jaxtyping import PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.scales import Scales
from somax._src.core.transforms import StateAffine


class ScaledModel(SomaxModel):
    """Integrate ``inner`` in transformed coordinates.

    The wrapped model is a ``SomaxModel`` like any other: it integrates,
    steps, and reports diagnostics through the same interface. Every
    time passed to it — ``t0``, ``t1``, ``dt``, ``saveat`` — is in the
    *transformed* time unit, related to the inner model's by
    ``t_inner = time_scale * t_outer``.

    Attributes:
        inner: The model being wrapped. Its ``create()`` signature,
            ``vector_field`` and parameters are untouched.
        transform: The affine state map. ``forward`` takes an inner
            state to wrapped coordinates.
        time_scale: Inner time units per wrapped time unit, finite and
            strictly positive. For a nondimensionalising wrapper this
            is ``scales.T``; for a purely statistical one it stays
            at 1.
    """

    inner: SomaxModel
    transform: StateAffine
    time_scale: float = eqx.field(static=True, default=1.0)

    def __check_init__(self) -> None:
        """Reject a time scale that is not a usable change of variables.

        Zero is the dangerous one: every tendency is multiplied by it
        and the inner model is evaluated at ``t = 0`` forever, so the
        integration returns a frozen trajectory rather than reporting
        that the coordinate change is degenerate. NaN and infinity
        corrupt the solve outright. Negative is rejected too — the
        transform would run the model backwards in time, which nothing
        here is written for, and silently doing so would be worse than
        saying no.
        """
        if not math.isfinite(self.time_scale) or self.time_scale <= 0.0:
            raise ValueError(
                f"ScaledModel: time_scale must be a finite positive number; "
                f"got {self.time_scale!r}. It is inner time units per "
                f"wrapped time unit, so zero freezes the trajectory and a "
                f"negative value reverses it."
            )

    # ------------------------------------------------------------------
    # SomaxModel contract
    # ------------------------------------------------------------------

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> PyTree:
        r"""Tendency in transformed coordinates.

        With :math:`y = (x - \ell)/s` and :math:`t' = t / T`,

        .. math::

            \frac{dy}{dt'} = \frac{T}{s}\,
                \left.\frac{dx}{dt}\right|_{x = sy + \ell,\; t = T t'} .

        The Jacobian of an affine map is the constant diagonal
        :math:`1/s`, so this is exact — no approximation is introduced
        by integrating in the transformed variables.
        """
        inner_state = self.transform.inverse(state)
        tendency = self.inner.vector_field(t * self.time_scale, inner_state, args)
        return jtu.tree_map(
            lambda d, scale: self.time_scale * d / scale,
            tendency,
            self.transform.scale,
        )

    def build_terms(self) -> dfx.AbstractTerm:
        """Rescale the inner model's terms, preserving their structure.

        The inherited implementation would wrap the whole transformed
        right-hand side in a single ``ODETerm``, discarding an IMEX
        model's explicit/implicit split — so wrapping a
        ``TermModel.create(..., imex=True)`` would make it incompatible
        with :func:`somax.solvers.imex_solver`, and integrate its stiff
        diffusion explicitly under the default solver.

        Instead the inner term tree is built first and each ``ODETerm``
        in it is rescaled in place. Boundary conditions come from the
        inner terms, in the inner model's own coordinates, rather than
        being conjugated through the transform.
        """
        return _rescale_term(
            paramax.unwrap(self.inner).build_terms(),
            transform=self.transform,
            time_scale=self.time_scale,
        )

    def apply_boundary_conditions(self, state: PyTree) -> PyTree:
        """Apply the inner model's boundary conditions, conjugated.

        ``BC_y = forward . BC_x . inverse``. For the zeroing and masking
        boundary conditions somax models use, this agrees with applying
        the inner condition directly only when ``loc`` is zero on the
        affected cells — which :meth:`StateAffine.from_samples`
        guarantees for masked cells, and which holds by construction for
        the velocity and vorticity fields whose ``loc`` is zero.
        """
        inner_state = self.transform.inverse(state)
        return self.transform.forward(self.inner.apply_boundary_conditions(inner_state))

    def diagnose(self, state: PyTree) -> PyTree:
        """Diagnostics from the inner model, in the inner model's units.

        The state is mapped back before being handed over, so the
        numbers mean what they always did. They are *not*
        re-dimensionalised on top of that: a nondimensional inner model
        reports nondimensional diagnostics.
        """
        return self.inner.diagnose(self.transform.inverse(state))

    @property
    def state_signature(self) -> None:
        """Delegate to the inner model; the transform preserves structure."""
        return self.inner.state_signature

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_scales(
        cls,
        inner: SomaxModel,
        scales: Scales,
        state_cls: type,
        **per_field: Any,
    ) -> ScaledModel:
        """Wrap ``inner`` so it integrates in nondimensional coordinates.

        Args:
            inner: The dimensional model to wrap.
            scales: Characteristic scales. ``scales.T`` becomes the time
                scale, so times passed to the wrapper are in units of
                ``T``.
            state_cls: The ``State`` subclass ``inner`` integrates.
                Required because a model does not advertise its state
                type, and the transform needs the field names to pick
                per-field scales.
            **per_field: Forwarded to
                :meth:`StateAffine.from_scales` — how a multilayer model
                supplies its per-layer thickness scale.

        Returns:
            A ``ScaledModel`` whose state coordinates are nondimensional.
        """
        transform = StateAffine.from_scales(state_cls, scales, **per_field)
        return cls(inner=inner, transform=transform, time_scale=scales.T)

    @classmethod
    def standardised(
        cls,
        inner: SomaxModel,
        samples: PyTree,
        **from_samples_kw: Any,
    ) -> ScaledModel:
        """Wrap ``inner`` so it integrates on standardised state.

        The time scale stays at 1: standardising the state says nothing
        about time, and rescaling it as well would silently change what
        a ``dt`` means.

        Args:
            inner: The model to wrap.
            samples: Stacked states with a leading sample axis, from
                which the per-field or per-gridpoint mean and standard
                deviation are taken.
            **from_samples_kw: Forwarded to
                :meth:`StateAffine.from_samples` (``per_gridpoint``,
                ``eps``, ``mask``).

        Returns:
            A ``ScaledModel`` whose state coordinates have zero mean and
            unit variance over ``samples``.
        """
        transform = StateAffine.from_samples(samples, **from_samples_kw)
        return cls(inner=inner, transform=transform, time_scale=1.0)


def _rescale_term(
    term: dfx.AbstractTerm,
    *,
    transform: StateAffine,
    time_scale: float,
) -> dfx.AbstractTerm:
    """Map a diffrax term into transformed coordinates, structure intact.

    Recurses through ``MultiTerm`` so an IMEX split survives: each
    ``ODETerm`` keeps its place, and the solver still routes the
    explicit and implicit parts to the stages they were assembled for.

    Args:
        term: The inner model's diffrax term.
        transform: The affine state map.
        time_scale: Inner time units per wrapped time unit.

    Returns:
        A term of the same shape, evaluating in wrapped coordinates.

    Raises:
        TypeError: If the tree holds a term that is neither a
            ``MultiTerm`` nor an ``ODETerm`` — an SDE term, say, whose
            rescaling is not a chain rule on the drift alone.
    """
    if isinstance(term, dfx.MultiTerm):
        return dfx.MultiTerm(
            *(
                _rescale_term(sub, transform=transform, time_scale=time_scale)
                for sub in term.terms
            )
        )
    if isinstance(term, dfx.ODETerm):
        inner_vector_field = term.vector_field

        def rescaled(t: float, y: PyTree, args: PyTree | None = None) -> PyTree:
            tendency = inner_vector_field(t * time_scale, transform.inverse(y), args)
            return jtu.tree_map(
                lambda d, scale: time_scale * d / scale, tendency, transform.scale
            )

        return dfx.ODETerm(rescaled)
    raise TypeError(
        f"ScaledModel: cannot rescale a {type(term).__name__}. Only ODETerm "
        "and MultiTerm are supported; an affine change of variables is not a "
        "chain rule on the drift alone for a stochastic term."
    )
