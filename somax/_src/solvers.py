"""Memory-safe IMEX solver construction for stiff somax models.

A term-based somax model can tag its stiff (diffusion / Laplacian) term
``implicit`` so a splitting integrator routes it through the implicit stage
(see :func:`somax.core.implicit` and ``*TermModel.create(..., imex=True)``).
Driving that with a bare ``diffrax.KenCarp3()`` has two footguns at scale:

1. **No implicit-solver tolerances.** With a fixed-step controller, diffrax
   raises unless the implicit solver's tolerances are set. An adaptive
   controller is the documented fix.
2. **Dense Jacobian OOM.** The default Newton root-finder uses
   ``lineax.AutoLinearSolver(well_posed=True)``, which materialises and
   factorises a dense ``N x N`` Jacobian of the implicit stage. For a 2-D
   field ``N = nx * ny``, so at ``256 x 256`` that Jacobian is ``65536 x
   65536`` (~17 GB in f32) — it runs out of memory, and even at ``64 x 64``
   the dense factorisation is ~100x slower than matrix-free (see #55).

:func:`imex_solver` returns a ``KenCarp3`` whose implicit stage uses a
**matrix-free** Krylov (GMRES) Newton solve — its memory is O(N), not
O(N^2), and its runtime is roughly flat in resolution. Pair it with
:func:`imex_stepsize_controller` (an adaptive PID controller carrying the
same tolerances)::

    from somax.solvers import imex_solver, imex_stepsize_controller

    sol = model.integrate(
        state0, t0, t1, dt,
        solver=imex_solver(),
        stepsize_controller=imex_stepsize_controller(),
    )

This is the recommended way to run any ``imex=True`` somax model at
≳128x128; the diffrax/lineax/optimistix pieces are all base dependencies.
"""

from __future__ import annotations

import diffrax as dfx
import lineax as lx
import optimistix as optx


def imex_solver(
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    gmres_restart: int = 20,
) -> dfx.AbstractSolver:
    """Build a ``KenCarp3`` IMEX solver with a matrix-free implicit stage.

    The implicit stage uses an ``optimistix.Newton`` root-finder backed by a
    ``lineax.GMRES`` linear solver, so the stiff (implicit) sub-problem is
    solved Jacobian-free — O(N) memory instead of the O(N^2) dense Jacobian
    that ``diffrax.KenCarp3()``'s default uses (the #55 OOM at 256x256).

    Args:
        rtol: Relative tolerance for the implicit Newton solve.
        atol: Absolute tolerance for the implicit Newton solve.
        gmres_restart: GMRES restart length (Krylov subspace size before
            restart). Larger converges in fewer outer iterations at higher
            per-iteration memory; 20 is a safe default.

    Returns:
        A ``diffrax`` IMEX solver suitable for ``model.integrate(...,
        solver=...)`` on a term model built with ``imex=True``. Use it with
        :func:`imex_stepsize_controller` (or any adaptive controller carrying
        matching tolerances).
    """
    root_finder = optx.Newton(
        rtol=rtol,
        atol=atol,
        linear_solver=lx.GMRES(rtol=rtol, atol=atol, restart=gmres_restart),
    )
    return dfx.KenCarp3(root_finder=root_finder)


def imex_stepsize_controller(
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> dfx.AbstractStepSizeController:
    """Build an adaptive PID controller for an IMEX solve.

    A fixed-step controller with an implicit solver requires the implicit
    tolerances to be specified and cannot reject a step when the Newton solve
    fails to converge; an adaptive controller is the documented fix (and the
    somax default ``ConstantStepSize`` does not satisfy the IMEX requirement).
    The tolerances here should match those passed to :func:`imex_solver`.

    Args:
        rtol: Relative tolerance for adaptive step-size control.
        atol: Absolute tolerance for adaptive step-size control.

    Returns:
        A ``diffrax.PIDController`` for ``model.integrate(...,
        stepsize_controller=...)``.
    """
    return dfx.PIDController(rtol=rtol, atol=atol)
