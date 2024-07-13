from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import (
    Array,
    ArrayLike,
    Float,
)

from somax._src.boundaries.base import (
    zero_boundaries,
    zero_gradient_boundaries,
    no_slip_boundaries,
    no_flux_boundaries
)
from somax._src.constants import GRAVITY
from somax._src.domain.cartesian import CartesianDomain2D
from somax._src.masks.masks import MaskGrid
from somax._src.models.swm.ops import (
    calculate_h_linear_rhs,
    calculate_h_nonlinear_rhs,
    calculate_u_linear_rhs,
    calculate_u_nonlinear_rhs,
    calculate_v_linear_rhs,
    calculate_v_nonlinear_rhs,
    potential_vorticity
)
from somax._src.operators.average import x_avg_2D, y_avg_2D
from somax._src.reconstructions.flux import uv_center_flux
from somax._src.kinematics.derived import kinetic_energy


class SWMState(eqx.Module):
    h: Array
    u: Array
    v: Array
    h_domain: CartesianDomain2D = eqx.field(static=True)
    u_domain: CartesianDomain2D = eqx.field(static=True)
    v_domain: CartesianDomain2D = eqx.field(static=True)
    q_domain: CartesianDomain2D = eqx.field(static=True)
    masks: MaskGrid = eqx.field(static=True)


class SWM(eqx.Module):
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
    depth: ArrayLike = eqx.field(static=True)
    coriolis_f0: float = eqx.field(static=True)
    coriolis_beta: float = eqx.field(static=True)
    linear_mass: bool = eqx.field(static=True)
    linear_momentum: bool = eqx.field(static=True)
    mass_adv_scheme: str = eqx.field(static=True)
    mass_adv_stencil: int = eqx.field(static=True)
    momentum_adv_scheme: str = eqx.field(static=True)
    momentum_adv_stencil: int = eqx.field(static=True)

    def __init__(
        self,
        gravity=GRAVITY,
        depth=100,
        coriolis_f0=2e-4,
        coriolis_beta=2e-11,
        linear_mass: bool = False,
        mass_adv_scheme: str = "wenoz",
        mass_adv_stencil: int = 5,
        linear_momentum: bool = False,
        momentum_adv_scheme: str = "wenoz",
        momentum_adv_stencil: int = 5,
    ):
        self.gravity = gravity
        self.depth = depth
        self.coriolis_f0 = coriolis_f0
        self.coriolis_beta = coriolis_beta
        self.linear_mass = linear_mass
        self.mass_adv_scheme = mass_adv_scheme
        self.mass_adv_stencil = mass_adv_stencil
        self.linear_momentum = linear_momentum
        self.momentum_adv_scheme = momentum_adv_scheme
        self.momentum_adv_stencil = momentum_adv_stencil

    @property
    def phase_speed(self):
        """
        Calculates the phase speed of the wave.

        Returns:
            float: The phase speed of the wave.
        """
        return jnp.sqrt(self.gravity * self.depth)

    def coriolis_param(self, Y: Float[Array, "Dx Dy"]):
        return self.coriolis_f0 + Y * self.coriolis_beta

    def lateral_viscosity(self, dx: Array):
        return 1e-3 * self.coriolis_f0 * dx**2

    def rossby_radius(self, Y: Float[Array, "Dx Dy"]):
        """
        Calculates the Rossby radius of the model.

        Returns:
            float: The Rossby radius.
        """
        return jnp.sqrt(self.gravity * self.depth) / self.coriolis_param(Y=Y).mean()

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
        masks = state.masks

        # precalculate quantities
        if not self.linear_mass or not self.linear_momentum:
            # zero flux padding
            h_pad: Float[Array, "Nx+2 Ny+2"] = zero_gradient_boundaries(h, ((1, 1), (1, 1)))
            
            # calculate mass fluxes
            fluxes = uv_center_flux(
                h=h_pad, v=v, u=u,
                u_mask=masks.face_u, v_mask=masks.face_v,
                num_pts=self.mass_adv_stencil, method=self.mass_adv_scheme,
            )

            uh_flux: Float[Array, "Dx+1 Dy"] = fluxes[0]
            vh_flux: Float[Array, "Dx Dy+1"] = fluxes[1]

        if not self.linear_momentum:
            # planetary vorticity, f
            f_on_q: Float[Array, "Nx+1 Ny+1"] = self.coriolis_param(state.q_domain.grid[..., 0])
            # calculate potential vorticity, q = (ζ + f) / h
            u_pad: Float[Array, "Nx+1 Ny+2"] = no_slip_boundaries(u, ((0, 0), (1, 1)))
            v_pad: Float[Array, "Nx+2 Ny+1"] = no_slip_boundaries(v, ((1, 1), (0, 0)))
            q: Float[Array, "Dx-1 Dy-1"] = potential_vorticity(h=h_pad, u=u, v=v, f=f_on_q, dx=dx, dy=dy)
            q: Float[Array, "Dx+1 Dy+1"] = zero_gradient_boundaries(-q, ((1, 1), (1, 1)))
            q: Float[Array, "Dx+1 Dy+1"] = q.at[1:-1, 1:-1].set(-q[1:-1, 1:-1])
            # calculate kinetic energy, ke = 0.5 (u² + v²)
            u_on_h = x_avg_2D(u)
            v_on_h = y_avg_2D(v)
            ke: Float[Array, "Dx Dy"] = kinetic_energy(u=u_on_h, v=v_on_h)


        # Calculate height
        if self.linear_mass:
            #  RHS ( ∂h/∂t = - H (∂u/∂x + ∂v/∂y) )
            h_rhs: Float[ArrayLike, "Dx Dy"] = calculate_h_linear_rhs(
                u, v, dx, dy, self.depth
            )
            h_rhs *= state.masks.center.values
        else:
            h_rhs = calculate_h_nonlinear_rhs(uh_flux, vh_flux, dx, dy, self.depth)
            h_rhs *= state.masks.center.values

        # Calculate Zonal Velocity
        if self.linear_momentum:
            # Linear Momentum ( ∂u/∂t = fv - g ∂h/∂x )
            v_pad: Float[ArrayLike, "Dx+2 Dy+1"] = no_slip_boundaries(
                v, pad_width=((1, 1), (0, 0))
            )
            h_pad: Float[ArrayLike, "Dx+2 Dy"] = zero_gradient_boundaries(
                h, pad_width=((1, 1), (0, 0))
            )

            u_rhs: Float[ArrayLike, "Dx-1 Dy"] = calculate_u_linear_rhs(
                h_pad, v_pad, dx, self.coriolis_f0, self.gravity
            )
            # TODO: Change this!
            u_rhs *= state.masks.face_u.values

        else:
            u_rhs = calculate_u_nonlinear_rhs(
                h=h, q=q, dx=dx, vh_flux=vh_flux, ke=ke, u_mask=masks.face_u
            )
            u_rhs *= state.masks.face_u.values

        # Calculate Meridonal Velocity
        if self.linear_momentum:
            # Linear Momentum ( ∂v/∂t = - fu - g ∂h/∂y )
            u_pad: Float[ArrayLike, "Dx+1 Dy"] = no_slip_boundaries(
                u,
                pad_width=(
                    (0, 0),
                    (1, 1),
                ),
            )
            h_pad: Float[ArrayLike, "Dx Dy"] = zero_gradient_boundaries(
                h,
                pad_width=(
                    (0, 0),
                    (1, 1),
                ),
            )

            v_rhs: Float[ArrayLike, "Dx Dy+1"] = calculate_v_linear_rhs(
                h_pad, u_pad, dy, self.coriolis_f0, self.gravity
            )
            v_rhs *= state.masks.face_v.values
        else:
            
            v_rhs = calculate_v_nonlinear_rhs(
                h=h, q=q, dy=dy, uh_flux=uh_flux, ke=ke, v_mask=masks.face_v
            )
            v_rhs *= state.masks.face_v.values


        # update all state vectors
        state = eqx.tree_at(lambda x: x.h, state, h_rhs)
        state = eqx.tree_at(lambda x: x.u, state, u_rhs)
        state = eqx.tree_at(lambda x: x.v, state, v_rhs)

        return state
