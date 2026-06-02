"""Public surface for somax's solver helpers.

:func:`imex_solver` / :func:`imex_stepsize_controller` build a memory-safe,
matrix-free IMEX integrator for stiff term-based models (those built with
``imex=True``). They avoid the dense-Jacobian OOM that a bare
``diffrax.KenCarp3()`` hits at large grids (#55) by solving the implicit
stage with a Krylov (GMRES) Newton root-finder — O(N) memory rather than
O(N^2).

Pure JAX / diffrax / lineax / optimistix (all base dependencies), so
importing this is cheap; it is kept out of ``somax``'s top-level
``__init__`` to mirror the module-per-surface layout (cf. :mod:`somax.eval`,
:mod:`somax.guards`).

Example::

    from somax.solvers import imex_solver, imex_stepsize_controller

    model = Burgers2DTermModel.create(nx=256, ny=256, nu=0.05, imex=True)
    sol = model.integrate(
        state0, t0=0.0, t1=1.0, dt=1e-3,
        solver=imex_solver(),
        stepsize_controller=imex_stepsize_controller(),
    )
"""

from somax._src.solvers import imex_solver, imex_stepsize_controller


__all__ = [
    "imex_solver",
    "imex_stepsize_controller",
]
