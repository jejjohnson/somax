"""Tests for the Domain and TimeDomain classes."""

from __future__ import annotations

import jax.numpy as jnp

from somax._src.domain.domain import (
    Domain,
    init_domain_from_bounds_and_numpoints,
    init_domain_from_bounds_and_step,
)
from somax._src.domain.time import TimeDomain


def test_domain_from_bounds_and_numpoints_1d():
    domain = init_domain_from_bounds_and_numpoints(xmin=0.0, xmax=10.0, Nx=11)
    assert isinstance(domain, Domain)
    assert domain.Nx == (11,)
    assert domain.xmin == (0.0,)
    assert domain.xmax == (10.0,)


def test_domain_from_bounds_and_step():
    domain = init_domain_from_bounds_and_step(xmin=0.0, xmax=10.0, dx=1.0)
    assert isinstance(domain, Domain)
    assert domain.Nx == (11,)


def test_domain_direct_construction():
    domain = Domain(
        xmin=(0.0, 0.0),
        xmax=(10.0, 5.0),
        dx=(1.0, 1.0),
        Nx=(11, 6),
        Lx=(10.0, 5.0),
    )
    assert domain.ndim == 2
    assert domain.Nx == (11, 6)


def test_domain_coords():
    domain = init_domain_from_bounds_and_numpoints(xmin=0.0, xmax=1.0, Nx=5)
    coords = domain.coords_axis
    assert len(coords) == 1
    assert coords[0].shape == (5,)
    assert jnp.isclose(coords[0][0], 0.0)
    assert jnp.isclose(coords[0][-1], 1.0)


def test_domain_cell_volume_2d():
    dx, dy = 1.0, 0.5
    domain = Domain(
        xmin=(0.0, 0.0),
        xmax=(10.0, 5.0),
        dx=(dx, dy),
        Nx=(11, 11),
        Lx=(10.0, 5.0),
    )
    assert domain.cell_volume == dx * dy


def test_time_domain():
    td = TimeDomain(tmin=0.0, tmax=10.0, dt=0.1)
    assert td.tmin == 0.0
    assert td.tmax == 10.0
    assert td.dt == 0.1


def test_time_domain_from_numpoints():
    td = TimeDomain.from_numpoints(tmin=0.0, tmax=10.0, nt=101)
    assert td.tmin == 0.0
    assert td.tmax == 10.0
    assert jnp.isclose(td.dt, 0.1)


def test_time_domain_coords():
    td = TimeDomain(tmin=0.0, tmax=1.0, dt=0.25)
    coords = td.coords
    # jnp.arange does not include endpoint
    assert coords[0] == 0.0
    assert len(coords) == 4  # [0.0, 0.25, 0.5, 0.75]
