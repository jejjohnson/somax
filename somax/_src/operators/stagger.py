import functools as ft
import typing as tp

from somax._src.domain.base import Domain

DIRECTIONS = {
    "right": (1.0, 1.0),
    "left": (-1.0, -1.0),
    "inner": (1.0, -1.0),
    "outer": (-1.0, 1.0),
    None: (0.0, 0.0),
}

NUM_POINTS = {
    "right": 0,
    "left": 0,
    "inner": -2,
    "outer": 2,
    None: 0.0,
}

STAGGER_BOOLS = {"0": 1.0, "1": 0.5}

PADDING = {
    "both": (1, 1),
    "right": (0, 1),
    "left": (1, 0),
    None: (0, 0),
}


def domain_limits_transform(
    xmin: float,
    xmax: float,
    dx: float,
    Lx: float,
    Nx: int,
    direction: tp.Optional[str] = None,
    stagger: tp.Optional[bool] = None,
) -> tp.Tuple:
    """
    Transforms the domain limits and parameters based on the given inputs.

    Args:
        xmin (float): The minimum x-coordinate of the domain.
        xmax (float): The maximum x-coordinate of the domain.
        dx (float): The grid spacing.
        Lx (float): The length of the domain.
        Nx (int): The number of grid points.
        direction (str, optional): The direction of the transformation. Defaults to None.
        stagger (bool, optional): Whether to apply staggered transformation. Defaults to None.

    Returns:
        Tuple: A tuple containing the transformed xmin, xmax, Nx, and Lx values.
    """
    # convert staggers to bools
    if stagger is None:
        stagger = "0"
    stagger = str(int(stagger))

    # TODO: check size of dx
    left_dir, right_dir = DIRECTIONS[direction][0], DIRECTIONS[direction][1]
    stagger = STAGGER_BOOLS[stagger]
    left_dx = dx * stagger * left_dir
    right_dx = dx * stagger * right_dir
    xmin += left_dx
    xmax += right_dx
    Nx += NUM_POINTS[direction] * stagger
    Lx += dx * NUM_POINTS[direction] * stagger
    return xmin, xmax, int(Nx), Lx


def batch_domain_limits_transform(
    xmin: tp.Iterable[float],
    xmax: tp.Iterable[float],
    dx: tp.Iterable[float],
    Lx: tp.Iterable[float],
    Nx: tp.Iterable[float],
    direction: tp.Iterable[str] = None,
    stagger: tp.Iterable[bool] = None,
) -> tp.Tuple:
    if direction is None:
        direction = (None,) * len(xmin)

    if stagger is None:
        stagger = (False,) * len(xmin)

    msg = "Incorrect shapes"
    msg += f"\nxmin: {len(xmin)} | "
    msg += f"xmax: {len(xmax)} | "
    msg += f"dx: {len(dx)} | "
    msg += f"direction: {len(direction)} | "
    msg += f"stagger: {len(stagger)}"
    assert (
        len(xmin) == len(xmax) == len(dx) == len(direction) == len(stagger)
    ), msg

    limits = [
        domain_limits_transform(
            imin, imax, idx, iLx, iNx, idirection, istagger
        )
        for imin, imax, idx, iLx, iNx, idirection, istagger in zip(
            xmin, xmax, dx, Lx, Nx, direction, stagger
        )
    ]

    xmin, xmax, Nx, Lx = zip(*limits)
    return xmin, xmax, Nx, Lx


def stagger_domain(
    domain: Domain, direction: tp.Iterable[str], stagger: tp.Iterable[bool]
):
    msg = "Incorrect shapes"
    msg += f"\nxmin: {len(domain.xmin)} | "
    msg += f"xmax: {len(domain.xmax)} | "
    msg += f"dx: {len(domain.dx)} | "
    msg += f"direction: {len(direction)} | "
    msg += f"stagger: {len(stagger)}"
    assert (
        len(domain.xmin)
        == len(domain.xmax)
        == len(domain.dx)
        == len(direction)
        == len(stagger)
    ), msg

    # change domain limits
    xmin, xmax, Nx, Lx = batch_domain_limits_transform(
        domain.xmin,
        domain.xmax,
        domain.dx,
        domain.Lx,
        domain.Nx,
        direction,
        stagger,
    )

    domains = [
        Domain(xmin=ixmin, xmax=ixmax, dx=idx, Lx=iLx, Nx=iNx)
        for ixmin, ixmax, idx, iLx, iNx in zip(xmin, xmax, domain.dx, Lx, Nx)
    ]

    domain = ft.reduce(lambda a, b: a * b, domains)

    # print(domains[0], domains[1])
    # domain = sum(domains)
    # create new domain
    # domain = Domain(xmin=xmin, xmax=xmax, dx=domain.dx)

    return domain
