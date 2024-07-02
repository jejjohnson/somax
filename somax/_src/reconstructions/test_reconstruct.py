import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.masks.utils import (
    init_mask_realish,
    init_mask_rect,
)
from finitevolx._src.reconstructions.reconstruct import (
    reconstruct,
    reconstruct_1pt,
    reconstruct_3pt,
    reconstruct_5pt,
)

jax.config.update("jax_enable_x64", True)

MASKS_RECT = init_mask_rect(6, "node")
MASKS_REAL = init_mask_realish(6, "center")

Q_ONES = jnp.ones_like(MASKS_RECT.center.values)
U_ONES = jnp.ones_like(MASKS_RECT.face_u.values)
V_ONES = jnp.ones_like(MASKS_RECT.face_v.values)

DIMS = [1, 0]
METHODS = ["linear", "weno", "wenoz"]
NUM_PTS = [1, 3, 5]
METHODS_DIMS = list(itertools.product(METHODS, DIMS))
METHODS_NUMPTS = list(itertools.product(METHODS, NUM_PTS))


def test_reconstruct_1pt_nomask():
    # take interior points
    u = U_ONES[1:-1]

    flux = reconstruct_1pt(q=Q_ONES, u=u, dim=0, u_mask=None)

    assert flux.shape == u.shape
    np.testing.assert_array_almost_equal(flux, np.ones_like(flux))


def test_reconstruct_1pt_mask():
    # take interior points
    u = U_ONES[1:-1]
    u_mask = MASKS_RECT.face_u[1:-1]

    flux = reconstruct_1pt(q=Q_ONES, u=u, dim=0, u_mask=u_mask)
    true_flux = u_mask.distbound1 * jnp.ones_like(flux)

    assert flux.shape == u.shape
    np.testing.assert_array_almost_equal(flux, true_flux)


@pytest.mark.parametrize("method", METHODS)
def test_reconstruct_3pt_mask_u(method):
    # take interior points
    u = jax.lax.slice_in_dim(U_ONES, axis=0, start_index=1, limit_index=-1)
    u_mask = MASKS_RECT.face_u[1:-1]

    # do flux
    flux = reconstruct_3pt(q=Q_ONES, u=u, u_mask=u_mask, dim=0, method=method)

    assert flux.shape == u.shape
    np.testing.assert_array_equal(flux, jnp.ones_like(flux))


@pytest.mark.parametrize("method", METHODS)
def test_reconstruct_3pt_mask_v(method):
    # take interior points
    v = jax.lax.slice_in_dim(V_ONES, axis=1, start_index=1, limit_index=-1)
    v_mask = MASKS_RECT.face_v[:, 1:-1]

    # do flux
    flux = reconstruct_3pt(q=Q_ONES, u=v, u_mask=v_mask, dim=1, method=method)

    assert flux.shape == v.shape
    np.testing.assert_array_equal(flux, jnp.ones_like(flux))


@pytest.mark.parametrize("method", ["linear", "weno", "wenoz"])
def test_reconstruct_3pt_nomask(method):
    # take interior points
    u = U_ONES[1:-1]

    # do flux
    flux = reconstruct_3pt(q=Q_ONES, u=u, u_mask=None, dim=0, method=method)

    assert flux.shape == u.shape
    np.testing.assert_array_equal(flux, jnp.ones_like(flux))


@pytest.mark.parametrize("method", ["linear", "weno", "wenoz"])
def test_reconstruct_5pt_mask(method):
    # take interior
    u = U_ONES[1:-1]
    u_mask = MASKS_RECT.face_u[1:-1]

    # do flux
    flux = reconstruct_5pt(q=Q_ONES, u=u, u_mask=u_mask, dim=0, method=method)

    assert flux.shape == u.shape
    np.testing.assert_array_equal(flux, jnp.ones_like(flux))


@pytest.mark.parametrize("method", ["linear", "weno", "wenoz"])
def test_reconstruct_5pt_nomask(method):
    # take interior
    u = U_ONES[1:-1]

    # do flux
    flux = reconstruct_5pt(q=Q_ONES, u=u, u_mask=None, dim=0, method=method)

    assert flux.shape == u.shape
    np.testing.assert_array_equal(flux, jnp.ones_like(flux))


@pytest.mark.parametrize("method,num_pts", METHODS_NUMPTS)
def test_reconstruct_nomask_u(method, num_pts):
    # take interior points
    u = U_ONES.copy()
    q = Q_ONES.copy()
    u = jax.lax.slice_in_dim(u, axis=0, start_index=1, limit_index=-1)

    # do flux
    flux = reconstruct(q=q, u=u, u_mask=None, dim=0, method=method, num_pts=num_pts)

    msg = "Incorrect shape..."
    msg += f"Shape: {flux.shape} | {u.shape}"
    assert flux.shape == u.shape, msg
    np.testing.assert_array_equal(flux, jnp.ones_like(flux))


@pytest.mark.parametrize("method,num_pts", METHODS_NUMPTS)
def test_reconstruct_nomask_v(method, num_pts):
    # take interior points
    v = V_ONES.copy()
    q = Q_ONES.copy()
    v = jax.lax.slice_in_dim(v, axis=1, start_index=1, limit_index=-1)

    # do flux
    flux = reconstruct(q=q, u=v, u_mask=None, dim=1, method=method, num_pts=num_pts)

    msg = "Incorrect shape..."
    msg += f"Shape: {flux.shape} | {v.shape}"
    assert flux.shape == v.shape, msg
    np.testing.assert_array_equal(flux, jnp.ones_like(flux))


@pytest.mark.parametrize("method,num_pts", METHODS_NUMPTS)
def test_reconstruct_mask_u(method, num_pts):
    # take interior points
    u = U_ONES.copy()
    q = Q_ONES.copy()
    u = jax.lax.slice_in_dim(u, axis=0, start_index=1, limit_index=-1)
    u_mask = MASKS_RECT.face_u[1:-1]

    # do flux
    flux = reconstruct(q=q, u=u, u_mask=u_mask, dim=0, method=method, num_pts=num_pts)

    true_flux = flux * u_mask.values
    msg = "Incorrect shape..."
    msg += f"Shape: {flux.shape} | {u.shape}"
    assert flux.shape == u.shape, msg
    np.testing.assert_array_equal(flux, true_flux)


@pytest.mark.parametrize("method,num_pts", METHODS_NUMPTS)
def test_reconstruct_mask_v(method, num_pts):
    # take interior points
    v = V_ONES.copy()
    q = Q_ONES.copy()
    v = jax.lax.slice_in_dim(v, axis=1, start_index=1, limit_index=-1)
    v_mask = MASKS_RECT.face_v[:, 1:-1]

    # do flux
    flux = reconstruct(q=q, u=v, u_mask=v_mask, dim=1, method=method, num_pts=num_pts)

    true_flux = flux * v_mask.values

    msg = "Incorrect shape..."
    msg += f"Shape: {true_flux.shape} | {v.shape}"
    assert true_flux.shape == v.shape, msg
    np.testing.assert_array_equal(flux, true_flux)
