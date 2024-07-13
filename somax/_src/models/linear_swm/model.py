from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import (
    Array,
    ArrayLike,
    Float,
)

from somax._src.boundaries.base import zero_boundaries
from somax._src.constants import GRAVITY
from somax._src.domain.cartesian import CartesianDomain2D
from somax._src.masks.masks import MaskGrid
from somax._src.models.linear_swm.ops import (
    calculate_h_rhs,
    calculate_u_rhs,
    calculate_v_rhs,
)


class SWMState(eqx.Module):
    h: Array
    u: Array
    v: Array
    h_domain: CartesianDomain2D = eqx.field(static=True)
    u_domain: CartesianDomain2D = eqx.field(static=True)
    v_domain: CartesianDomain2D = eqx.field(static=True)
    masks: MaskGrid = eqx.field(static=True)


class LinearSWM(eqx.Module):
    """
    Linear shallow water model class.

    Args:
        gravity (float): Gravity constant. Default is GRAVITY.
        depth (float): Depth of the water. Default is 100.
        coriolis_param (float): Coriolis parameter. Default is 2e-4.
    """

    gravity: float = eqx.field(
        static=True,
    )
    depth: float = eqx.field(static=True)
    coriolis_param: float = eqx.field(static=True)

    def __init__(
        self,
        gravity=GRAVITY,
        depth=100,
        coriolis_param=2e-4,
    ):
        self.gravity = gravity
        self.depth = depth
        self.coriolis_param = coriolis_param

    @property
    def phase_speed(self):
        """
        Calculates the phase speed of the wave.

        Returns:
            float: The phase speed of the wave.
        """
        return jnp.sqrt(self.gravity * self.depth)

    @property
    def rossby_radius(self):
        """
        Calculates the Rossby radius of the model.

        Returns:
            float: The Rossby radius.
        """
        return jnp.sqrt(self.gravity * self.depth) / self.coriolis_param

    def apply_boundaries(self, u: ArrayLike, variable: str = "h"):
        """
        Apply boundary conditions to the input array.

        Parameters:
            u (ArrayLike): The input array.
            variable (str): The variable to which the boundary conditions should be applied.
                            Default is "h".

        Returns:
            ArrayLike: The input array with boundary conditions applied.
        """
        if variable == "u":
            u = zero_boundaries(u[1:-1], pad_width=((2, 2), (0, 0)))
        if variable == "v":
            u = zero_boundaries(u[:, 1:-1], pad_width=((0, 0), (2, 2)))
        return u

    def equation_of_motion(self, t: ArrayLike, state: SWMState, args: Any) -> SWMState:
        """
        Calculate the equation of motion for the linear shallow water model.

        Args:
            t (float): Time.
            state (State): Current state of the model.
            args (Any): Additional arguments.

        Returns:
            State: The time derivative of the state variables (h, u, v).
        """

        # extract dynamic states
        h, u, v = state.h, state.u, state.v

        # extract static states
        dx, dy = state.h_domain.resolution

        # calculate zonal velocity RHS ( ∂u/∂t = fv - g ∂h/∂x )
        u_rhs: Float[ArrayLike, "Dx-1 Dy"] = calculate_u_rhs(
            h, v, dx, self.coriolis_param, self.gravity
        )
        u_rhs: Float[ArrayLike, "Dx+1 Dy"] = self.apply_boundaries(u_rhs, "u")
        u_rhs *= state.masks.face_u.values
        

        # calculate meridonal velocity RHS: ( ∂v/∂t = - fu - g ∂h/∂y )
        v_rhs: Float[ArrayLike, "Dx Dy-1"] = calculate_v_rhs(
            h, u, dy, self.coriolis_param, self.gravity
        )
        v_rhs *= state.masks.face_v.values[..., 1:-1]
        v_rhs: Float[ArrayLike, "Dx Dy+1"] = self.apply_boundaries(v_rhs, "v")

        # calculate height RHS ( ∂h/∂t = - H (∂u/∂x + ∂v/∂y) )
        h_rhs: Float[ArrayLike, "Dx Dy"] = calculate_h_rhs(u, v, dx, dy, self.depth)
        h_rhs *= state.masks.center.values

        # update all state vectors
        state = eqx.tree_at(lambda x: x.h, state, h_rhs)
        state = eqx.tree_at(lambda x: x.u, state, u_rhs)
        state = eqx.tree_at(lambda x: x.v, state, v_rhs)

        return state
