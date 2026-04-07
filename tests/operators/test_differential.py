import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import (
    x_avg_2D,
    y_avg_2D,
)
from finitevolx._src.constants import GRAVITY
from finitevolx._src.operators.operators import (
    absolute_vorticity,
    bernoulli_potential,
    difference,
    divergence,
    geostrophic_gradient,
    kinetic_energy,
    laplacian,
    relative_vorticity,
)

jax.config.update("jax_enable_x64", True)

rng = np.random.RandomState(123)

# sizes for all of the Arrays
Nx_center, Ny_center = 100, 50
Nx_face_u, Ny_face_u = Nx_center + 1, Ny_center
Nx_face_v, Ny_face_v = Nx_center, Ny_center + 1
Nx_node, Ny_node = Nx_center + 1, Ny_center + 1

# list of methods


@pytest.fixture()
def u_1d_ones():
    return jnp.ones(Nx_face_u)


@pytest.fixture()
def u_1d_randn():
    return rng.randn(Nx_face_u)


@pytest.fixture()
def v_1d_ones():
    return jnp.ones(Nx_face_v)


@pytest.fixture()
def v_1d_randn():
    return rng.randn(Nx_face_v)


@pytest.fixture()
def center_1d_ones():
    return jnp.ones(Nx_center)


@pytest.fixture()
def center_1d_randn():
    return rng.randn(Nx_center)


@pytest.fixture()
def node_1d_ones():
    return jnp.ones(Nx_center)


@pytest.fixture()
def node_1d_randn():
    return rng.randn(Nx_center)


@pytest.fixture()
def u_2d_ones():
    return jnp.ones((Nx_face_u, Ny_face_u))


@pytest.fixture()
def u_2d_randn():
    return rng.randn(Nx_face_u, Ny_face_u)


@pytest.fixture()
def v_2d_ones():
    return jnp.ones((Nx_face_v, Ny_face_v))


@pytest.fixture()
def v_2d_randn():
    return rng.randn(Nx_face_v, Ny_face_v)


@pytest.fixture()
def center_2d_ones():
    return jnp.ones((Nx_center, Ny_center))


@pytest.fixture()
def center_2d_randn():
    return rng.randn(Nx_center, Ny_center)


@pytest.fixture()
def node_2d_ones():
    return jnp.ones((Nx_node, Ny_node))


@pytest.fixture()
def node_2d_randn():
    return rng.randn(Nx_node, Ny_node)


def test_difference_1d_order1_ones(u_1d_ones):
    step_size = 0.1

    # classic difference
    du_dx_np = jnp.diff(u_1d_ones, n=1, axis=0) / step_size
    du_dx = difference(u_1d_ones, axis=0, step_size=step_size)

    np.testing.assert_array_equal(du_dx_np, du_dx)


def test_difference_1d_order1_random(u_1d_randn):
    step_size = 0.1

    # classic difference
    du_dx_np = jnp.diff(u_1d_randn, n=1, axis=0) / step_size
    du_dx = difference(u_1d_randn, axis=0, step_size=step_size)

    np.testing.assert_array_equal(du_dx_np, du_dx)


def test_difference_2d_order1_ones(u_2d_ones):
    step_size = 0.1

    # classic difference
    du_dx_np = jnp.diff(u_2d_ones, n=1, axis=0) / step_size
    du_dx = difference(u_2d_ones, axis=0, step_size=step_size)

    np.testing.assert_array_equal(du_dx_np, du_dx)

    # classic difference
    du_dy_np = jnp.diff(u_2d_ones, n=1, axis=1) / step_size
    du_dy = difference(u_2d_ones, axis=1, step_size=step_size)

    np.testing.assert_array_equal(du_dy_np, du_dy)


def test_difference_2d_order1_random(u_2d_randn):
    step_size = 0.1

    # classic difference
    du_dx_np = jnp.diff(u_2d_randn, n=1, axis=0) / step_size
    du_dx = difference(u_2d_randn, axis=0, step_size=step_size)

    np.testing.assert_array_equal(du_dx_np, du_dx)

    du_dy_np = jnp.diff(u_2d_randn, n=1, axis=1) / step_size
    du_dy = difference(u_2d_randn, axis=1, step_size=step_size)

    np.testing.assert_array_equal(du_dy_np, du_dy)


def test_difference_1d_order2_ones(u_1d_ones):
    step_size = 0.1
    derivative = 2

    # classic difference
    du_dx_np = jnp.diff(u_1d_ones, n=2, axis=0) / step_size**derivative
    du_dx = difference(u_1d_ones, axis=0, step_size=step_size, derivative=derivative)

    np.testing.assert_array_almost_equal(du_dx_np, du_dx)


def test_difference_1d_order2_random(u_1d_randn):
    step_size = 0.1
    derivative = 2

    # classic difference
    du_dx_np = jnp.diff(u_1d_randn, n=2, axis=0) / step_size**derivative
    du_dx = difference(u_1d_randn, axis=0, step_size=step_size, derivative=derivative)

    np.testing.assert_array_almost_equal(du_dx_np, du_dx)


def test_difference_2d_order2_ones(u_2d_ones):
    step_size = 0.1
    derivative = 2

    # classic difference
    du_dx_np = jnp.diff(u_2d_ones, n=derivative, axis=0) / step_size**derivative
    du_dx = difference(u_2d_ones, axis=0, step_size=step_size, derivative=derivative)

    np.testing.assert_array_almost_equal(du_dx_np, du_dx)

    # classic difference
    du_dy_np = jnp.diff(u_2d_ones, n=derivative, axis=1) / step_size**derivative
    du_dy = difference(u_2d_ones, axis=1, step_size=step_size, derivative=derivative)

    np.testing.assert_array_almost_equal(du_dy_np, du_dy)


def test_difference_2d_order2_random(u_2d_randn):
    step_size = 0.1
    derivative = 2
    # classic difference
    du_dx_np = jnp.diff(u_2d_randn, n=derivative, axis=0) / step_size**derivative
    du_dx = difference(u_2d_randn, axis=0, step_size=step_size, derivative=derivative)

    np.testing.assert_array_almost_equal(du_dx_np, du_dx)

    du_dy_np = jnp.diff(u_2d_randn, n=derivative, axis=1) / step_size**derivative
    du_dy = difference(u_2d_randn, axis=1, step_size=step_size, derivative=derivative)

    np.testing.assert_array_almost_equal(du_dy_np, du_dy)


def test_lap_ones(u_2d_ones):
    step_size = 0.1
    # classic difference
    d2u_dx2 = jnp.diff(u_2d_ones, n=2, axis=0) / step_size**2
    d2u_dy2 = jnp.diff(u_2d_ones, n=2, axis=1) / step_size**2
    lap_u_np = d2u_dx2[:, 1:-1] + d2u_dy2[1:-1, :]

    # wrapper function
    lap_u = laplacian(u_2d_ones, step_size=step_size)

    np.testing.assert_array_almost_equal(lap_u_np, lap_u)


def test_lap_random(u_2d_randn):
    step_size = 0.1

    # classic difference
    d2u_dx2 = jnp.diff(u_2d_randn, n=2, axis=0) / step_size**2
    d2u_dy2 = jnp.diff(u_2d_randn, n=2, axis=1) / step_size**2
    lap_u_np = d2u_dx2[:, 1:-1] + d2u_dy2[1:-1, :]

    # wrapper function
    lap_u = laplacian(u_2d_randn, step_size=step_size)

    np.testing.assert_array_almost_equal(lap_u_np, lap_u)


def test_divergence():
    u = rng.randn(50, 25)
    v = rng.randn(49, 26)

    dx, dy = 0.1, 0.2

    # gradient froms scratch
    du_dx_ = difference(u, axis=0, step_size=dx, derivative=1)
    dv_dy_ = difference(v, axis=1, step_size=dy, derivative=1)
    div_ = du_dx_ + dv_dy_
    # convenience function
    div = divergence(u, v, dx=dx, dy=dy)

    np.testing.assert_array_almost_equal(div_, div)


def test_relative_vorticity():
    u = rng.randn(50, 26)
    v = rng.randn(51, 25)

    dx, dy = 0.1, 0.2

    # gradient froms scratch
    du_dy_ = difference(u, axis=1, step_size=dy, derivative=1)
    dv_dx_ = difference(v, axis=0, step_size=dx, derivative=1)
    vort_r_ = dv_dx_ - du_dy_
    # convenience function
    vort_r = relative_vorticity(u, v, dx=dx, dy=dy)

    np.testing.assert_array_almost_equal(vort_r_, vort_r)


def test_absolute_vorticity():
    u = rng.randn(50, 26)
    v = rng.randn(51, 25)

    dx, dy = 0.1, 0.2

    # gradient froms scratch
    du_dy_ = difference(u, axis=1, step_size=dy, derivative=1)
    dv_dx_ = difference(v, axis=0, step_size=dx, derivative=1)
    vort_r_ = dv_dx_ + du_dy_
    # convenience function
    vort_r = absolute_vorticity(u, v, dx=dx, dy=dy)

    np.testing.assert_array_almost_equal(vort_r_, vort_r)


def test_geostrophic_gradient():
    psi = rng.randn(50, 25)
    dx, dy = 0.1, 0.2

    # gradient froms scratch
    u_ = -difference(psi, axis=1, step_size=dy, derivative=1)
    v_ = difference(psi, axis=0, step_size=dx, derivative=1)

    # convenience function
    u, v = geostrophic_gradient(psi, dx=dx, dy=dy)

    np.testing.assert_array_almost_equal(u_, u)
    np.testing.assert_array_almost_equal(v_, v)


def test_kinetic_energy_2d_ones(u_2d_ones, v_2d_ones, center_2d_ones):
    u = u_2d_ones
    v = v_2d_ones
    h = center_2d_ones

    # move u, v to h center
    u_c = x_avg_2D(u**2)
    v_c = y_avg_2D(v**2)
    ke_ = 0.5 * (u_c + v_c)

    ke = kinetic_energy(u=u, v=v)

    assert ke.shape == h.shape
    np.testing.assert_array_almost_equal(ke, ke_)


def test_bernoulli_potential_2d_ones(u_2d_ones, v_2d_ones, center_2d_ones):
    u = u_2d_ones
    v = v_2d_ones
    h = center_2d_ones

    # calculate work
    p_ = kinetic_energy(u=u, v=v)
    p_ += GRAVITY * h

    p = bernoulli_potential(h=h, u=u, v=v)

    assert p.shape == h.shape
    np.testing.assert_array_almost_equal(p, p_)
