"""Shared helpers for the ``from_nondimensional`` model factories.

Every factory does the same three things: validate its dimensionless
inputs, turn them into the SI-shaped kwargs ``create()`` already
accepts, and hand back the unit :class:`~somax._src.core.scales.Scales`
the resulting model runs in. The pieces that are common to more than
one model live here so the mappings stay stated once.

Two Burger conventions appear in the layered models and it matters
which is meant:

* **Per interface** — ``Bu_k = g'_k H_k / (f0 L)**2``, one number per
  layer interface. This is what :func:`burger_to_g_prime` inverts, and
  what the factories accept, because it maps to the stratification
  one-to-one.
* **Per vertical mode** — ``(L_d,m / L)**2``, one number per baroclinic
  mode, where ``L_d,m`` comes from the vertical-mode eigenproblem. For
  two layers the internal mode has
  ``L_d**2 = g' H_1 H_2 / (f0**2 (H_1 + H_2))``, which is a specific
  combination of the interface values rather than any one of them.

Prescribing the modal radii directly would mean solving an inverse
eigenproblem, so the factories take the interface convention and expose
the resulting modal radii on the built model
(``model.modal.rossby_radii``) for the resolution guard to read.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def require_positive(context: str, **values: float) -> None:
    """Raise if any named value is not finite and strictly positive."""
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{context}: {name} must be a finite positive number; got {value!r}."
            )


def require_non_negative(context: str, **values: float) -> None:
    """Raise if any named value is negative or not finite."""
    for name, value in values.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{context}: {name} must be a finite non-negative number; "
                f"got {value!r}."
            )


def resolve_burger(
    context: str,
    burger: float | None,
    froude: float | None,
    rossby: float,
) -> float:
    """Settle the Burger number from ``burger`` or ``froude``.

    The two are not independent once the Rossby number is fixed:
    ``Bu = (Ro / Fr)**2``. Accepting both would let a caller state an
    inconsistent pair, so exactly one must be given.

    Args:
        context: Caller name, for error messages.
        burger: Burger number, or ``None``.
        froude: Froude number ``U / sqrt(gH)``, or ``None``.
        rossby: Rossby number, already validated.

    Returns:
        The Burger number.

    Raises:
        ValueError: If both or neither is supplied, or the value is not
            positive.
    """
    if (burger is None) == (froude is None):
        raise ValueError(
            f"{context}: give exactly one of burger or froude. They are "
            f"related by Bu = (Ro/Fr)**2 at fixed Ro, so supplying both "
            f"allows an inconsistent pair."
        )
    if froude is not None:
        require_positive(context, froude=froude)
        return (rossby / froude) ** 2
    require_positive(context, burger=burger)
    return float(burger)


def burger_to_g_prime(
    context: str,
    burger: Sequence[float],
    thickness: Sequence[float],
    f0: float = 1.0,
    length: float = 1.0,
) -> tuple[float, ...]:
    """Invert the per-interface Burger numbers into reduced gravities.

    ``Bu_k = g'_k H_k / (f0 L)**2`` gives
    ``g'_k = Bu_k * (f0 L)**2 / H_k``.

    Args:
        context: Caller name, for error messages.
        burger: Per-interface Burger numbers, top to bottom.
        thickness: Per-layer thicknesses in the same units the model
            will use, same length as ``burger``.
        f0: Reference Coriolis parameter of the target model.
        length: Horizontal length scale of the target model.

    Returns:
        Reduced gravities, one per interface.

    Raises:
        ValueError: If the sequences differ in length, are empty, or
            contain a non-positive entry.
    """
    if len(burger) != len(thickness):
        raise ValueError(
            f"{context}: burger and thickness_ratio must have the same "
            f"length; got {len(burger)} and {len(thickness)}."
        )
    if not burger:
        raise ValueError(f"{context}: burger must not be empty.")
    for index, (bu, h) in enumerate(zip(burger, thickness, strict=True)):
        require_positive(context, **{f"burger[{index}]": bu, f"thickness[{index}]": h})
    factor = (f0 * length) ** 2
    return tuple(
        float(bu) * factor / float(h) for bu, h in zip(burger, thickness, strict=True)
    )


def reject_derived_kwargs(
    context: str, derived: Sequence[str], **create_kw: object
) -> None:
    """Refuse forwarded arguments the factory derives for itself.

    ``**create_kw`` is a convenience for the knobs a dimensionless
    description says nothing about — ``bc``, ``method``, ``mask``,
    ``wind_profile``. Letting a stratification or a Coriolis parameter
    through it would silently win over the values derived from the
    dimensionless inputs, while the returned ``Scales`` still described
    the derived ones: the model and its scales would then be different
    systems.

    Args:
        context: Caller name, for error messages.
        derived: Argument names the factory sets itself.
        **create_kw: The forwarded arguments to check.

    Raises:
        ValueError: If any forwarded name is one the factory derives.
    """
    clash = sorted(set(create_kw) & set(derived))
    if clash:
        raise ValueError(
            f"{context}: {clash} is derived from the dimensionless inputs and "
            f"cannot be passed through to create(). Passing it would leave the "
            f"model and the returned Scales describing different systems; drop "
            f"it, or use create() directly for a dimensional build."
        )
