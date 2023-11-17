import fieldx._src.domain.utils as d_utils
from fieldx._src.domain.domain import Domain


def init_domain_from_bounds_and_step(xmin: float = 0.0, xmax: float = 1.0, dx: float = 0.1):
    """initialize 1d domain from bounds and step_size
    Eqs:
         Nx = 1 + ceil((x_max - x_min) / dx)
        Lx = (x1 - x0)

    Args:
        xmin (float): x min value
        xmax (float): maximum value
        Nx (int): the number of points for domain
    Returns:
        domain (Domain): the full domain
    """

    # calculate Nx
    Nx = d_utils.bounds_and_step_to_points(xmin=xmin, xmax=xmax, dx=dx)

    # calculate Lx
    Lx = d_utils.bounds_to_length(xmin=xmin, xmax=xmax)

    return Domain(xmin=xmin, xmax=xmax, dx=dx, Nx=Nx, Lx=Lx)