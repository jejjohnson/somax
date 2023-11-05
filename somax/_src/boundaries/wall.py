from somax._src.boundaries import padding
from somax._src.boundaries import zero_boundaries
import jax.numpy as jnp


def wall_boundaries_2D(u, grid: str = "h"):
    assert grid in ("h", "u", "v", "q")

    if grid == "h":
        # HEIGHT
        # wall boundaries, ∂xh(x=0=Lx),∂yh(y=0=Ly)
        u = wall_boundary_corners_2D(u)
    if grid == "u":
        # ZONAL VELOCITY
        # wall boundaries, u(x=0)=u(x=Lx)=0
        u = zero_boundaries(u[1:-1,:], ((1,1),(0,0)))
        u = wall_boundary_corners_2D(u)
    if grid == "v":
        # MERIDIONAL VELOCITY
        # wall boundaries, v(y=0)=v(y=Ly)=0
        u = zero_boundaries(u[:,1:-1], ((0,0),(1,1)))
        u = wall_boundary_corners_2D(u)
    if grid == "q":
        # POTENTIAL VORTICITY
        u = wall_boundary_corners_2D(u)
    return u


def wall_boundary_corners_2D(field):
    # fix corners to be average of neighbours
    field = field.at[0, 0].set(0.5 * (field[0, 1] + field[1, 0]))
    field = field.at[-1, 0].set(0.5 * (field[-2, 0] + field[-1, 1]))
    field = field.at[0, -1].set(0.5 * (field[1, -1] + field[0, -2]))
    field = field.at[-1, -1].set(0.5 * field[-1, -2] + field[-2, -1])
    return field