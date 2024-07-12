import math

import pytest

from somax._src.domain.utils import (
    bounds_and_points_to_step,
    bounds_and_step_to_points,
    bounds_to_length,
    length_and_points_to_step,
    length_and_step_to_points,
)


@pytest.mark.parametrize("xmin,xmax", [(0, 10), (10, 100), (0.11, 1.3534)])
def test_bounds_to_length(xmin, xmax):
    # Lx = abs(Xmax - Xmin)
    Lx_true = abs(xmax - xmin)

    Lx = bounds_to_length(xmin=xmin, xmax=xmax)

    assert Lx == Lx_true


@pytest.mark.parametrize(
    "xmin,xmax,Nx", [(0, 10, 10), (10, 100, 1000), (0.11, 1.3534, 6)]
)
def test_bounds_and_points_to_step(xmin, xmax, Nx):
    # Lx = abs(Xmax - Xmin)
    dx_true = (float(xmax) - float(xmin)) / (float(Nx) - 1.0)

    dx = bounds_and_points_to_step(xmin=xmin, xmax=xmax, Nx=Nx)

    assert dx == dx_true


@pytest.mark.parametrize("Lx,Nx", [(10.0, 10), (100.0, 52), (0.1, 100)])
def test_length_and_points_to_step(Lx, Nx):
    # Lx = abs(Xmax - Xmin)
    dx_true = float(Lx) / (float(Nx) - 1.0)

    dx = length_and_points_to_step(Lx=Lx, Nx=Nx)

    assert dx == dx_true


@pytest.mark.parametrize("Lx,dx", [(10.0, 0.1), (100.0, 10.0), (0.1, 100)])
def test_length_and_step_to_points(Lx, dx):
    # Nx = 1 + Lx / dx
    Nx_true = int(math.floor(1.0 + float(Lx) / float(dx)))

    Nx = length_and_step_to_points(Lx=Lx, dx=dx)

    assert Nx == Nx_true


@pytest.mark.parametrize(
    "xmin, xmax,dx", [(0.1, 10.0, 0.1), (0.11, 100.0, 10.0), (0.1, 53.65, 1.0)]
)
def test_bounds_and_step_to_points(xmin, xmax, dx):
    # Nx = 1 + ((xmax - xmin) / dx)
    Nx_true = 1 + int(math.floor(((float(xmax) - float(xmin)) / float(dx))))

    Nx = bounds_and_step_to_points(xmin=xmin, xmax=xmax, dx=dx)

    assert Nx == Nx_true
