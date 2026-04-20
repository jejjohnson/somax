"""Small shared helpers for the Phase-3 model adapters."""

from __future__ import annotations

from typing import Any


def require_stratification(
    params: dict[str, Any], model_name: str
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Extract ``(H, g_prime)`` from a model ``params`` dict.

    Raises a :class:`ValueError` — not a bare ``KeyError`` — when a
    required ``model.stratification.<key>`` is missing, so the failure
    is legible when it bubbles up through the cyclopts CLI. Equal layer
    count between ``H`` and ``g_prime`` is enforced too, since the
    underlying model ``create()`` otherwise dies inside a stratification
    profile assertion that doesn't mention YAML keys.

    Args:
        params: The model-side params dict (``{"stratification": {...},
            "params": {...}}``).
        model_name: Registry name of the calling model, included in the
            error message so users can trace the config error back to
            the right ``model:`` block.

    Returns:
        A ``(H, g_prime)`` tuple-of-tuples ready to pass into the
        model's ``create()`` factory.
    """
    stratification = params.get("stratification") or {}
    missing = [key for key in ("H", "g_prime") if key not in stratification]
    if missing:
        raise ValueError(
            f"{model_name}: model.stratification missing required key(s) "
            f"{missing!r}. Both 'H' (layer thicknesses, m) and 'g_prime' "
            "(reduced gravities, m/s^2) are required for multilayer "
            "models; their lengths must match the number of layers."
        )
    H = tuple(float(v) for v in stratification["H"])
    g_prime = tuple(float(v) for v in stratification["g_prime"])
    if len(H) != len(g_prime):
        raise ValueError(
            f"{model_name}: len(model.stratification.H)={len(H)} but "
            f"len(model.stratification.g_prime)={len(g_prime)}; both must "
            "match the number of layers."
        )
    return H, g_prime
