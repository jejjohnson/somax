from jaxtyping import Float, Array
from finitevolx import center_avg_2D
from somax._src.utils.constants import GRAVITY

def potential_vorticity(
    h: Float[Array, "Nx Ny"],
    vort_r: Float[Array, "Nx-1 Ny-1"],
    f: Float[Array, "Nx-1 Ny-1"] | float,
):
    """Calculates the potential vorticity according to the
    Shallow water equations.
    Eq:
        ζ = ∂v/∂x - ∂u/∂y
        q = (ζ + f) / h

    Args:
        h (Array): the height variable on the cell centers/nodes
            Size = [Nx Ny]
        vort_r (Array): the relative vorticity on the cell nodes/centers
            Size = [Nx-1 Ny-1]
        f (Array): the planetary vorticity on the cell nodes/centers
            Size = [Nx-1 Ny-1] | float

    Returns:
        pv (Array): the potential vorticity on the cell centers/nodes
            Size = [Nx-1 Ny-1]

    Example:
        >>> u, v, h = ...
        >>> vort_r = relative_vorticity(u=u[1:-1], v=v[:, 1:-1], dx=dx, dy=dy)
        >>> f = coriolis_fn(...)
        >>> q = potential_vorticity_height(h, vort_r, f)
    """
    # calculate h on pv
    h_on_q: Float[Array, "Nx-1 Ny-1"] = center_avg_2D(h)

    # potential vorticity, q = (ζ + f) / h
    q: Float[Array, "Nx-1 Ny-1"] = (vort_r + f) / h_on_q

    return q
