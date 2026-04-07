from itertools import product

import jax
import numpy as np
import pytest

from finitevolx._src.reconstructions.upwind import (
    plusminus,
    upwind_1pt,
    upwind_3pt,
    upwind_5pt,
)

jax.config.update("jax_enable_x64", True)

RNG = np.random.RandomState(123)

U_RAND = RNG.randn(50, 25, 10)

METHODS = ["linear", "weno", "wenoz"]
DIMS = [0, 1, 2]
METHOD_DIMS = list(product(METHODS, DIMS))


def test_plusminus_way_pos():
    upos, uneg = plusminus(U_RAND, 1)

    assert np.greater_equal(upos, 0.0).all()
    assert np.less_equal(uneg, 0.0).all()


def test_plusminus_way_neg():
    upos, uneg = plusminus(U_RAND, -1)

    assert np.greater_equal(uneg, 0.0).all()
    assert np.less_equal(upos, 0.0).all()


def test_upwind_1pt():
    # X-AXIS
    qi_left, qi_right = upwind_1pt(U_RAND, dim=0)
    qi_left_, qi_right_ = U_RAND[:-1], U_RAND[1:]

    np.testing.assert_array_equal(qi_left_, qi_left)

    # Y-AXIS
    qi_left, qi_right = upwind_1pt(U_RAND, dim=1)
    qi_left_, qi_right_ = U_RAND[:, :-1], U_RAND[:, 1:]

    np.testing.assert_array_equal(qi_left_, qi_left)
    np.testing.assert_array_equal(qi_right_, qi_right)

    # Z-AXIS
    qi_left, qi_right = upwind_1pt(U_RAND, dim=2)
    qi_left_, qi_right_ = U_RAND[..., :-1], U_RAND[..., 1:]

    np.testing.assert_array_equal(qi_left_, qi_left)
    np.testing.assert_array_equal(qi_right_, qi_right)


@pytest.mark.parametrize("method, dim", METHOD_DIMS)
def test_upwind_3pt(method, dim):
    dim = 0
    dims = list(U_RAND.shape)

    dims[dim] -= 2

    qi_left, qi_right = upwind_3pt(U_RAND, dim=dim, method=method)

    assert qi_left.shape == tuple(dims)


@pytest.mark.parametrize("method, dim", METHOD_DIMS)
def test_upwind_5pt(method, dim):
    dim = 0
    dims = list(U_RAND.shape)

    dims[dim] -= 4

    qi_left, qi_right = upwind_5pt(U_RAND, dim=dim, method=method)

    assert qi_left.shape == tuple(dims)
