from typing import List, Tuple
import equinox as eqx
from jaxtyping import Array
import jax.numpy as jnp
import numpy as np
import einx


class Mode2LayerTransformer(eqx.Module):
    """
    Class representing a mode-to-layer transformer.

    This class provides methods to transform data from mode space to layer space and vice versa.

    Attributes:
        heights (Array): Array of heights.
        reduced_gravities (Array): Array of reduced gravities.
        Nz (int): Number of heights.
        A (Array): Transformation matrix from layer space to mode space.
        A_layer_2_mode (Array): Transformation matrix from mode space to layer space.
        ev_A (Array): Eigenvalues of the transformation matrix A.
        A_mode_2_layer (Array): Inverse transformation matrix from mode space to layer space.

    Methods:
        __init__(self, heights: List[float], reduced_gravities: List[float], correction: bool = False):
            Initializes the Mode2LayerTransformer object.
        transform(self, u: Array) -> Array:
            Transforms data from mode space to layer space.
        inverse_transform(self, u: Array) -> Array:
            Transforms data from layer space to mode space.
    """
    A: Array = eqx.static_field()
    A_layer_2_mode: Array = eqx.static_field()
    A_mode_2_layer: Array = eqx.static_field()
    ev_A: Array = eqx.static_field()
    
    def __init__(
        self,
        heights: List[float],
        reduced_gravities: List[float],
        correction: bool = False
        ):
        """
        Initializes the Mode2LayerTransformer object.

        Args:
            heights (List[float]): List of heights.
            reduced_gravities (List[float]): List of reduced gravities.
            correction (bool, optional): Flag indicating whether to apply correction. Defaults to False.
        """
        # initialize parameters
        A, A_layer_2_mode, ev_A, A_mode_2_layer = create_demposition_params(
            heights=heights, reduced_gravities=reduced_gravities, correction=correction
            )
        

        self.A = A
        self.A_layer_2_mode = A_layer_2_mode
        self.ev_A = ev_A
        self.A_mode_2_layer = A_mode_2_layer
        
    
    def transform(self, u: Array) -> Array:
        """
        Transforms data from mode space to layer space.

        Args:
            u (Array): Input data in mode space.

        Returns:
            Array: Transformed data in layer space.
        """
        # matrix multiplication
        u = einx.dot("... l, l ... -> l ...", self.A_layer_2_mode, u)
        
        return u
    
    def inverse_transform(self, u: Array) -> Array:
        """
        Transforms data from layer space to mode space.

        Args:
            u (Array): Input data in layer space.

        Returns:
            Array: Transformed data in mode space.
        """
        # matrix multiplication
        u = einx.dot("... l, l ... -> l ...", self.A_mode_2_layer, u)
        
        return u


def create_demposition_params(
    heights: List[float],
    reduced_gravities: List[float],
    correction: bool = False
) -> Tuple[Array, Array, Array, Array]:
    """
    Create decomposition parameters for multilayer quasi-geostrophic model.

    Args:
        heights (List[float]): List of heights for each layer.
        reduced_gravities (List[float]): List of reduced gravities for each layer.
        correction (bool, optional): Flag indicating whether to apply correction. Defaults to False.

    Returns:
        Tuple[Array, Array, Array, Array]: A tuple containing the following arrays:
            - A: Multilayer matrix A.
            - A_layer_2_mode: Layer-to-mode matrix.
            - ev_A: Eigenvalues of matrix A.
            - A_mode_2_layer: Mode-to-layer matrix.
    """
    num_layers = len(heights)

    msg = "Incorrect number of heights to reduced gravities."
    msg += f"\nHeights: {heights} | {num_layers}"
    msg += f"\nReduced Gravities: {reduced_gravities} | {len(reduced_gravities)}"
    assert num_layers - 1 == len(reduced_gravities), msg
    
    # calculate matrix M
    A = create_qg_multilayer_mat(heights, reduced_gravities, correction)
    A = jnp.asarray(A)

    # create layer to mode matrices
    ev_A, A_layer_2_mode, A_mode_2_layer = calculate_mode_matrices(A)
    ev_A = jnp.asarray(ev_A)
    A_layer_2_mode = jnp.asarray(A_layer_2_mode)
    A_mode_2_layer = jnp.asarray(A_mode_2_layer)
    return A, A_layer_2_mode, ev_A, A_mode_2_layer


def create_qg_multilayer_mat(
    heights: List[float],
    reduced_gravities: List[float],
    correction: bool = False,
) -> np.ndarray:
    """Computes the Matrix that is used to connect a stacked
    isopycnal Quasi-Geostrophic model.

    Args:
        heights (List[float]): The height for each layer. Size = [Nx]
        reduced_gravities (List[float]): The reduced gravities
            for each layer. Size = [Nx-1]
        correction (bool, optional): Flag indicating whether to apply
            correction terms. Defaults to False.

    Returns:
        np.ndarray: The Matrix connecting the layers. Size = [Nz, Nx]
    """
    num_heights = len(heights)

    # initialize matrix
    A = np.zeros((num_heights, num_heights))

    if num_heights == 1:
        A[0, 0] = 1.0 / (heights[0] * reduced_gravities[0])
    else:
        # top rows
        if correction:
            A[0, 0] = 1.0 / (heights[0] * 9.81) + 1.0 / (
                heights[0] * reduced_gravities[0]
            )
        else:
            A[0, 0] = 1.0 / (heights[0] * reduced_gravities[0])
        A[0, 1] = -1.0 / (heights[0] * reduced_gravities[0])

        # interior rows
        for i in range(1, num_heights - 1):
            A[i, i - 1] = -1.0 / (heights[i] * reduced_gravities[i - 1])
            A[i, i] = (
                1.0
                / heights[i]
                * (1 / reduced_gravities[i] + 1 / reduced_gravities[i - 1])
            )
            A[i, i + 1] = -1.0 / (
                heights[i] * reduced_gravities[num_heights - 2]
            )

        # bottom rows
        A[-1, -1] = 1.0 / (
            heights[num_heights - 1] * reduced_gravities[num_heights - 2]
        )
        A[-1, -2] = -1.0 / (
            heights[num_heights - 1] * reduced_gravities[num_heights - 2]
        )
    return A


def calculate_mode_matrices(A):
    """
    Calculate the mode matrices for a given matrix A.

    Parameters:
    A (numpy.ndarray): The input matrix.

    Returns:
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: A tuple containing the eigenvalues of A,
    the left-to-mode matrix, and the mode-to-left matrix.
    """
    ev_A, P = jnp.linalg.eig(A)
    ev_A = jnp.real(ev_A)
    Cl2m = jnp.linalg.inv(jnp.real(P))
    Cm2l = jnp.real(P)
    return ev_A, Cl2m, Cm2l