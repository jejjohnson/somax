import jax
import jax.numpy as jnp
import numpy as np
import pytest

from somax._src.operators.interp import (
    avg_arithmetic,
    avg_geometric,
    avg_harmonic,
    avg_pool,
    avg_quadratic,
)

jax.config.update("jax_enable_x64", True)

rng = np.random.RandomState(123)


@pytest.fixture()
def u_1d_ones():
    return jnp.ones(100)


@pytest.fixture()
def u_2d_ones():
    return jnp.ones((100, 50))


@pytest.fixture()
def u_3d_ones():
    return jnp.ones((100, 50, 25))


@pytest.fixture()
def u_1d_randn():
    return rng.randn(100)


@pytest.fixture()
def u_2d_randn():
    return rng.randn(100, 50)


@pytest.fixture()
def u_3d_randn():
    return rng.randn(100, 50, 25)


def test_x_average_1D_arithmetic(u_1d_randn):
    u = u_1d_randn

    u_on_x = avg_arithmetic(u[1:], u[:-1])
    u_on_x_ = avg_pool(
        u, kernel_size=(2,), stride=(1,), padding="valid", mean_fn="arithmetic"
    )

    np.testing.assert_array_almost_equal(u_on_x, u_on_x_)


def test_x_average_1D_geometric(u_1d_randn):
    u = u_1d_randn

    u_on_x = avg_geometric(u[1:], u[:-1])
    u_on_x_ = avg_pool(
        u, kernel_size=(2,), stride=(1,), padding="valid", mean_fn="geometric"
    )

    np.testing.assert_array_almost_equal(u_on_x, u_on_x_)


def test_x_average_1D_harmonic(u_1d_randn):
    u = u_1d_randn
    u_on_x = avg_harmonic(u[:-1], u[1:])
    u_on_x_ = avg_pool(
        u, kernel_size=(2,), stride=(1,), padding="valid", mean_fn="harmonic"
    )

    np.testing.assert_array_almost_equal(u_on_x, u_on_x_)


def test_x_average_1D_quadratic(u_1d_randn):
    u = u_1d_randn
    u_on_x = avg_quadratic(u[:-1], u[1:])
    u_on_x_ = avg_pool(
        u, kernel_size=(2,), stride=(1,), padding="valid", mean_fn="quadratic"
    )

    np.testing.assert_array_almost_equal(u_on_x, u_on_x_)


def test_xy_average_2D_arithmetic(u_2d_randn):
    u = u_2d_randn

    u_on_x = avg_arithmetic(u[1:], u[:-1])
    u_on_x_ = avg_pool(
        u,
        kernel_size=(2, 1),
        stride=(1, 1),
        padding="valid",
        mean_fn="arithmetic",
    )

    np.testing.assert_array_almost_equal(u_on_x, u_on_x_)

    u_on_y = avg_arithmetic(u[:, 1:], u[:, :-1])
    u_on_y_ = avg_pool(
        u,
        kernel_size=(1, 2),
        stride=(1, 1),
        padding="valid",
        mean_fn="arithmetic",
    )

    np.testing.assert_array_almost_equal(u_on_y, u_on_y_)


def test_xy_average_2D_harmonic(u_2d_randn):
    u = u_2d_randn

    u_on_x = avg_harmonic(u[1:], u[:-1])
    u_on_x_ = avg_pool(
        u,
        kernel_size=(2, 1),
        stride=(1, 1),
        padding="valid",
        mean_fn="harmonic",
    )

    np.testing.assert_array_almost_equal(u_on_x, u_on_x_)

    u_on_y = avg_harmonic(u[:, 1:], u[:, :-1])
    u_on_y_ = avg_pool(
        u,
        kernel_size=(1, 2),
        stride=(1, 1),
        padding="valid",
        mean_fn="harmonic",
    )

    np.testing.assert_array_almost_equal(u_on_y, u_on_y_)


def test_xy_average_2D_geometric(u_2d_randn):
    u = u_2d_randn

    u_on_x = avg_geometric(u[1:], u[:-1])
    u_on_x_ = avg_pool(
        u,
        kernel_size=(2, 1),
        stride=(1, 1),
        padding="valid",
        mean_fn="geometric",
    )

    np.testing.assert_array_almost_equal(u_on_x, u_on_x_)

    u_on_y = avg_geometric(u[:, 1:], u[:, :-1])
    u_on_y_ = avg_pool(
        u,
        kernel_size=(1, 2),
        stride=(1, 1),
        padding="valid",
        mean_fn="geometric",
    )

    np.testing.assert_array_almost_equal(u_on_y, u_on_y_)


def test_xy_average_2D_quadratic(u_2d_randn):
    u = u_2d_randn

    u_on_x = avg_quadratic(u[1:], u[:-1])
    u_on_x_ = avg_pool(
        u,
        kernel_size=(2, 1),
        stride=(1, 1),
        padding="valid",
        mean_fn="quadratic",
    )

    np.testing.assert_array_almost_equal(u_on_x, u_on_x_)

    u_on_y = avg_quadratic(u[:, 1:], u[:, :-1])
    u_on_y_ = avg_pool(
        u,
        kernel_size=(1, 2),
        stride=(1, 1),
        padding="valid",
        mean_fn="quadratic",
    )

    np.testing.assert_array_almost_equal(u_on_y, u_on_y_)


def test_center_average_2D_arithmetic(u_2d_randn):
    u = u_2d_randn

    u_on_x = avg_arithmetic(u[1:], u[:-1])
    u_on_c = avg_arithmetic(u_on_x[:, 1:], u_on_x[:, :-1])
    u_on_c_ = avg_pool(
        u,
        kernel_size=(2, 2),
        stride=(1, 1),
        padding="valid",
        mean_fn="arithmetic",
    )

    np.testing.assert_array_almost_equal(u_on_c, u_on_c_)


def test_center_average_2D_geometric(u_2d_randn):
    u = u_2d_randn

    u_on_x = avg_geometric(u[1:], u[:-1])
    u_on_c = avg_geometric(u_on_x[:, 1:], u_on_x[:, :-1])
    u_on_c_ = avg_pool(
        u,
        kernel_size=(2, 2),
        stride=(1, 1),
        padding="valid",
        mean_fn="geometric",
    )

    np.testing.assert_array_almost_equal(u_on_c, u_on_c_)


# TODO: fix center average harmonic mean...
def test_center_average_2D_harmonic(u_2d_randn):
    u = u_2d_randn

    u_on_x = avg_harmonic(u[1:], u[:-1])
    u_on_c = avg_harmonic(u_on_x[:, 1:], u_on_x[:, :-1])
    u_on_c_ = avg_pool(
        u,
        kernel_size=(2, 2),
        stride=(1, 1),
        padding="valid",
        mean_fn="harmonic",
    )

    np.testing.assert_array_almost_equal(u_on_c, u_on_c_)


def test_center_average_2D_quadratic(u_2d_randn):
    u = u_2d_randn

    u_on_x = avg_quadratic(u[1:], u[:-1])
    u_on_c = avg_quadratic(u_on_x[:, 1:], u_on_x[:, :-1])
    u_on_c_ = avg_pool(
        u,
        kernel_size=(2, 2),
        stride=(1, 1),
        padding="valid",
        mean_fn="quadratic",
    )

    np.testing.assert_array_almost_equal(u_on_c, u_on_c_)
