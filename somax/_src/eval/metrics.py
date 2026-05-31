"""Reference-free evaluation metrics for somax model states.

These are *field-level diagnostics* computed on a model's own Cartesian
grid — distinct from ``model.diagnose()`` (per-model conserved invariants)
and from the per-field min/mean/max stats the runner already logs. They
give somax a quantitative evaluation surface for free-running simulations:
how divergent the flow is, how much enstrophy / kinetic energy it carries,
and how far it sits from geostrophic balance.

All metrics are **reference-free** — they need only the state itself, not a
ground-truth field. Reference-based skill scores (RMSE, PSD score, …) need
observations / a truth trajectory and belong with the data-assimilation
work in a later phase.

Design
------
- The generic metrics (:func:`rms_divergence`, :func:`total_enstrophy`,
  :func:`kinetic_energy`) take raw velocity arrays plus the finitevolx
  ``Difference2D`` / grid the model already holds, so they apply to any
  Arakawa C-grid fluid model (SWM, QG, Navier-Stokes).
- :func:`geostrophic_imbalance` is model-aware: it reuses the model's *own*
  pressure-gradient and Coriolis operators, so it measures departure from
  exactly the balance the model is built around.
- :func:`compute_eval_metrics` is a defensive dispatcher: it returns the
  applicable metrics for fluid models and an empty dict for everything else
  (Lorenz, diffusion, …), so the runner can call it unconditionally.

Reductions use the grid interior (dropping the one-cell ghost halo) to match
the convention in the models' ``diagnose`` methods and avoid contaminating
the metric with boundary / ghost-point artifacts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from jaxtyping import Array, Float


if TYPE_CHECKING:
    from somax._src.core.types import State


# One-cell ghost halo dropped from every reduction (matches ``diagnose``).
_INTERIOR = (slice(1, -1), slice(1, -1))


def _interior(arr: Float[Array, "Ny Nx"], *, interior: bool) -> Array:
    """Optionally trim the one-cell ghost halo before reducing."""
    return arr[_INTERIOR] if interior else arr


def rms_divergence(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    diff: Any,
    *,
    interior: bool = True,
) -> Float[Array, ""]:
    r"""Root-mean-square horizontal divergence ``sqrt(<(∇·u)²>)``.

    A diagnostic of how non-divergent the flow is. For incompressible /
    geostrophic flow it should stay near zero; a growing value flags
    spurious compressibility or numerical noise.

    Args:
        u: x-velocity field on its C-grid points, shape ``(Ny, Nx)``.
        v: y-velocity field on its C-grid points, shape ``(Ny, Nx)``.
        diff: A finitevolx ``Difference2D`` (the model's ``.diff``); its
            :meth:`divergence` lowers ``(u, v)`` to ``∇·u`` at tracer points.
        interior: Drop the one-cell ghost halo before reducing (default).

    Returns:
        Scalar RMS divergence (same units as ``∇·u``, i.e. 1/s for m/s
        velocities on a metre grid).
    """
    div = _interior(diff.divergence(u, v), interior=interior)
    return jnp.sqrt(jnp.mean(div**2))


def total_enstrophy(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    diff: Any,
    grid: Any,
    *,
    interior: bool = True,
) -> Float[Array, ""]:
    r"""Domain-integrated enstrophy ``0.5 ∫ ζ² dA`` with ``ζ = ∂v/∂x - ∂u/∂y``.

    Enstrophy is a robust health metric for 2D / quasi-2D turbulence: in the
    inviscid limit it is bounded, so runaway growth signals instability.

    Args:
        u: x-velocity field, shape ``(Ny, Nx)``.
        v: y-velocity field, shape ``(Ny, Nx)``.
        diff: finitevolx ``Difference2D``; its :meth:`curl` returns ζ.
        grid: The model's ``CartesianGrid2D`` (for the ``dx·dy`` cell area).
        interior: Drop the one-cell ghost halo before reducing (default).

    Returns:
        Scalar total enstrophy.
    """
    zeta = _interior(diff.curl(u, v), interior=interior)
    cell_area = grid.dx * grid.dy
    return 0.5 * jnp.sum(zeta**2) * cell_area


def kinetic_energy(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    grid: Any,
    *,
    interior: bool = True,
) -> Float[Array, ""]:
    r"""Domain-integrated kinetic energy ``0.5 ∫ (u² + v²) dA``.

    Mirrors the velocity term of the models' ``diagnose`` energy (summing the
    staggered components directly), so it tracks consistently alongside the
    model's own energy diagnostic.

    Args:
        u: x-velocity field, shape ``(Ny, Nx)``.
        v: y-velocity field, shape ``(Ny, Nx)``.
        grid: The model's ``CartesianGrid2D`` (for the ``dx·dy`` cell area).
        interior: Drop the one-cell ghost halo before reducing (default).

    Returns:
        Scalar kinetic energy (per unit depth).
    """
    ui = _interior(u, interior=interior)
    vi = _interior(v, interior=interior)
    cell_area = grid.dx * grid.dy
    return 0.5 * jnp.sum(ui**2 + vi**2) * cell_area


def geostrophic_imbalance(
    model: Any,
    state: State,
    *,
    interior: bool = True,
    eps: float = 1e-30,
) -> Float[Array, ""]:
    r"""Dimensionless ageostrophic fraction of a shallow-water-type state.

    Geostrophic balance is ``f x u = -g ∇η`` — equivalently, the
    pressure-gradient and Coriolis accelerations cancel. This metric forms
    that residual using the model's *own* operators::

        r_u = -g ∂η/∂x + f·v        (at U-points)
        r_v = -g ∂η/∂y - f·u        (at V-points)

    and returns ``rms(r) / rms(coriolis)`` — 0 for a perfectly geostrophic
    flow, O(1) when ageostrophic accelerations rival the Coriolis term.
    Reusing ``model.diff`` and ``model.coriolis`` keeps the residual exactly
    consistent with the balance the model integrates around (staggering,
    masks and the β-plane ``f`` field all included).

    Args:
        model: A shallow-water-type model exposing ``diff`` (with
            ``diff_x_T_to_U`` / ``diff_y_T_to_V``), ``coriolis``,
            ``f_field`` and ``consts.gravity``.
        state: A state with ``h`` (height perturbation η), ``u`` and ``v``.
        interior: Drop the one-cell ghost halo before reducing (default).
        eps: Floor added to the denominator to keep a motionless state
            (zero Coriolis term) finite.

    Returns:
        Scalar ageostrophic fraction in ``[0, ∞)``.
    """
    g = model.consts.gravity
    h, u, v = state.h, state.u, state.v

    # Leading-order momentum balance terms, on their native C-grid points.
    pg_u = -g * model.diff.diff_x_T_to_U(h)
    pg_v = -g * model.diff.diff_y_T_to_V(h)
    cor_u, cor_v = model.coriolis(u, v, model.f_field)

    res_u = _interior(pg_u + cor_u, interior=interior)
    res_v = _interior(pg_v + cor_v, interior=interior)
    cor_u_i = _interior(cor_u, interior=interior)
    cor_v_i = _interior(cor_v, interior=interior)

    residual = jnp.sqrt(jnp.mean(res_u**2) + jnp.mean(res_v**2))
    scale = jnp.sqrt(jnp.mean(cor_u_i**2) + jnp.mean(cor_v_i**2))
    return residual / (scale + eps)


def _is_fluid_state(state: State) -> bool:
    """A 2D C-grid velocity state we can compute field metrics for."""
    if not (hasattr(state, "u") and hasattr(state, "v")):
        return False
    return jnp.ndim(state.u) == 2 and jnp.ndim(state.v) == 2


def _supports_geostrophy(model: Any, state: State) -> bool:
    """Whether the model exposes the pieces ``geostrophic_imbalance`` needs."""
    consts = getattr(model, "consts", None)
    return (
        hasattr(state, "h")
        and hasattr(model, "coriolis")
        and hasattr(model, "f_field")
        and consts is not None
        and hasattr(consts, "gravity")
    )


def compute_eval_metrics(model: Any, state: State) -> dict[str, float]:
    """Compute every applicable reference-free metric for ``(model, state)``.

    A defensive dispatcher meant to be called unconditionally by the runner:
    it inspects the model / state and returns only the metrics that make
    sense. Non-fluid models (Lorenz, diffusion, …) and multilayer (3D) states
    yield an empty dict rather than an error.

    Args:
        model: A constructed somax model.
        state: The state to evaluate (typically the final integrated state).

    Returns:
        Flat ``{metric_name: float}`` dict — a subset of ``rms_divergence``,
        ``total_enstrophy``, ``kinetic_energy`` and ``geostrophic_imbalance``.
        Empty when no metric applies.
    """
    out: dict[str, float] = {}
    if not (hasattr(model, "diff") and hasattr(model, "grid")):
        return out
    if not _is_fluid_state(state):
        return out

    u, v, diff, grid = state.u, state.v, model.diff, model.grid
    out["rms_divergence"] = float(rms_divergence(u, v, diff))
    out["total_enstrophy"] = float(total_enstrophy(u, v, diff, grid))
    out["kinetic_energy"] = float(kinetic_energy(u, v, grid))
    if _supports_geostrophy(model, state):
        out["geostrophic_imbalance"] = float(geostrophic_imbalance(model, state))
    return out
