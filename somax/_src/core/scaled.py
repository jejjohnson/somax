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

from typing import Any

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.scales import Scales
from somax._src.core.transforms import StateAffine


class ScaledModel(SomaxModel):
    """Integrate ``inner`` in transformed coordinates.

    The wrapped model is a ``SomaxModel`` like any other: it integrates,
    steps, and reports diagnostics through the same interface. Every
    time passed to it — ``t0``, ``t1``, ``dt``, ``save_at`` — is in the
    *transformed* time unit, related to the inner model's by
    ``t_inner = time_scale * t_outer``.

    Attributes:
        inner: The model being wrapped. Its ``create()`` signature,
            ``vector_field`` and parameters are untouched.
        transform: The affine state map. ``forward`` takes an inner
            state to wrapped coordinates.
        time_scale: Inner time units per wrapped time unit. For a
            nondimensionalising wrapper this is ``scales.T``; for a
            purely statistical one it stays at 1.
    """

    inner: SomaxModel
    transform: StateAffine
    time_scale: float = eqx.field(static=True, default=1.0)

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
