"""Models on a spherical Arakawa C-grid.

These use the full latitude-dependent Coriolis parameter
``f(phi) = 2 Omega sin(phi)`` rather than a beta-plane expansion, so
the planetary vorticity gradient varies correctly from equator to pole
instead of being frozen at a reference latitude.
"""

from __future__ import annotations

from somax._src.models.spherical.qg import (
    SphericalQG,
    SphericalQGDiagnostics,
    SphericalQGParams,
    SphericalQGPhysConsts,
    SphericalQGState,
)
from somax._src.models.spherical.swm import (
    SphericalSWM,
    SphericalSWMDiagnostics,
    SphericalSWMParams,
    SphericalSWMPhysConsts,
    SphericalSWMState,
)


__all__ = [
    "SphericalQG",
    "SphericalQGDiagnostics",
    "SphericalQGParams",
    "SphericalQGPhysConsts",
    "SphericalQGState",
    "SphericalSWM",
    "SphericalSWMDiagnostics",
    "SphericalSWMParams",
    "SphericalSWMPhysConsts",
    "SphericalSWMState",
]
