import jax.numpy as jnp


def fix_boundary_corners(field):
    # # fix corners to be average of neighbours
    field = field.at[0, 0].set(0.5 * (field[0, 1] + field[1, 0]))
    field = field.at[-1, 0].set(0.5 * (field[-2, 0] + field[-1, 1]))
    field = field.at[0, -1].set(0.5 * (field[1, -1] + field[0, -2]))
    field = field.at[-1, -1].set(0.5 * field[-1, -2] + field[-2, -1])
    return field


def wall_boundaries(u, grid: str = "h"):
    assert grid in ("h", "u", "v", "q")

    if grid == "h":
        u = fix_boundary_corners(u)
    if grid == "u":
        # ZONAL VELOCITY
        # wall boundaries
        # u(x=0=Lx,y=0=Ly)
        u = jnp.pad(
            u[1:-1, :], ((1, 1), (0, 0)), mode="constant", constant_values=0.0
        )
        u = fix_boundary_corners(u)
    if grid == "v":
        # wall boundaries, v(y=0=Ly)
        u = jnp.pad(
            u[:, 1:-1], ((0, 0), (1, 1)), mode="constant", constant_values=0.0
        )
        u = fix_boundary_corners(u)

    if grid == "q":
        u = fix_boundary_corners(u)
    return u


def doublegyre_boundaries(u, grid: str = "h"):
    assert grid in ("h", "u", "v", "q")

    if grid == "h":
        u = fix_boundary_corners(u)
    if grid == "u":
        # ZONAL VELOCITY
        # wall boundaries
        # u(x=0=Lx,y=0=Ly)
        u = jnp.pad(
            u[1:-1, :], ((1, 1), (0, 0)), mode="constant", constant_values=0.0
        )
        u = fix_boundary_corners(u)
    if grid == "v":
        # wall boundaries, v(y=0=Ly)
        u = jnp.pad(
            u[:, 1:-1], ((0, 0), (1, 1)), mode="constant", constant_values=0.0
        )
        u = fix_boundary_corners(u)

    if grid == "q":
        u = fix_boundary_corners(u)
    return u


def no_slip_padding(u, grid: str = "u"):

    if grid == "u":
        # pad with zeros
        u = jnp.pad(u, ((0, 0), (1, 1)), mode="constant", constant_values=0.0)
        # override with negative interior points
        u = u.at[:, 0].add(-u[:, 1])
        u = u.at[:, -1].add(-u[:, -2])

    if grid == "v":
        # pad with zeros
        u = jnp.pad(u, ((1, 1), (0, 0)), mode="constant", constant_values=0.0)
        # override with negative interior points
        u = u.at[0].add(-u[1])
        u = u.at[-1].add(-u[-2])

    return u


def flux_zero_padding(u):

    u = jnp.pad(
        u, pad_width=((1, 1), (1, 1)), mode="constant", constant_values=0.0
    )

    return u
