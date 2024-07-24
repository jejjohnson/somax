import jax.numpy as jnp
from jaxtyping import Array, PyTree


from typing import Any, Optional

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
from somax._src.constants import OMEGA, GRAVITY, R_EARTH, RHO
from somax._src.domain.cartesian import CartesianDomain2D
from somax._src.masks.masks import MaskGrid
from somax._src.models.swm_double_gyre.ops import (
    calculate_h_linear_rhs,
    calculate_h_nonlinear_rhs,
    calculate_u_linear_rhs,
    calculate_u_nonlinear_rhs,
    calculate_v_linear_rhs,
    calculate_v_nonlinear_rhs,
    potential_vorticity,
)
from somax._src.kinematics.coriolis import coriolis_param, beta_param
from somax._src.operators.average import x_avg_2D, y_avg_2D
from somax._src.reconstructions.flux import uv_center_flux
from somax._src.kinematics.derived import kinetic_energy
from somax._src.reconstructions.base import reconstruct


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
    lat_0: ArrayLike = eqx.field(static=True)
    depth: ArrayLike = eqx.field(static=True)
    coriolis_f0: float = eqx.field(static=True)
    coriolis_beta: float = eqx.field(static=True)
    linear_mass: bool = eqx.field(static=True)
    linear_momentum: bool = eqx.field(static=True)
    mass_adv_scheme: str = eqx.field(static=True)
    mass_adv_stencil: int = eqx.field(static=True)
    momentum_adv_scheme: str = eqx.field(static=True)
    momentum_adv_stencil: int = eqx.field(static=True)
    diffusion: bool = eqx.field(static=True)
    forcing: Optional[PyTree] = None

    def __init__(
        self,
        gravity=GRAVITY,
        depth=100,
        lat_0: float | ArrayLike = 30.0,
        linear_mass: bool = False,
        mass_adv_scheme: str = "wenoz",
        mass_adv_stencil: int = 5,
        linear_momentum: bool = False,
        momentum_adv_scheme: str = "wenoz",
        momentum_adv_stencil: int = 5,
        diffusion: bool = False,
        forcing: Optional[PyTree] = None,
    ):
        self.gravity = gravity
        self.depth = depth
        self.lat_0 = lat_0
        self.coriolis_f0 = coriolis_param(lat_0, omega=OMEGA)
        self.coriolis_beta = beta_param(lat_0, omega=OMEGA, radius=R_EARTH)
        self.linear_mass = linear_mass
        self.mass_adv_scheme = mass_adv_scheme
        self.mass_adv_stencil = mass_adv_stencil
        self.linear_momentum = linear_momentum
        self.momentum_adv_scheme = momentum_adv_scheme
        self.momentum_adv_stencil = momentum_adv_stencil
        self.diffusion = diffusion
        self.forcing = forcing

    @property
    def phase_speed(self):
        """
        Calculates the phase speed of the wave.

        Returns:
            float: The phase speed of the wave.
        """
        return jnp.sqrt(self.gravity * self.depth)

    def coriolis_param(self, Y: Float[Array, "Dx Dy"], L_y: float):
        return self.coriolis_f0 + self.coriolis_beta * (Y - L_y / 2.0)

    def lateral_viscosity(self, dx: Array):
        return 1e-3 * self.coriolis_f0 * dx**2

    def rossby_radius(self, Y: Float[Array, "Dx Dy"]):
        """
        Calculates the Rossby radius of the model.

        Returns:
            float: The Rossby radius.
        """
        return jnp.sqrt(self.gravity * self.depth) / self.coriolis_param(Y=Y).mean()

    def apply_boundaries(self, u: Array, variable: str = "u_no_flow"):
        if variable == "u_no_flow":
            # apply no flow boundaries, u(x = 0) = u(x = Lx) = 0
            u = jnp.pad(u, ((1,1),(0,0)), mode="constant", constant_values=0.0)
        elif variable == "u_no_slip":
            # apply no slip boundaries, u(y = 0) = u(y = Ly) = 0.
            u = jnp.pad(u, ((0,0),(1,1)), mode="constant", constant_values=0.0)
        elif variable == "v_no_flow":
            # apply no flow boundaries, v(y = 0) = v(y = Ly) = 0,
            u = jnp.pad(u, ((0,0),(1,1)), mode="constant", constant_values=0.0)
        elif variable == "v_no_slip":
            # apply no flow boundaries, v(y = 0) = v(y = Ly) = 0,
            u = jnp.pad(u, ((1,1),(0,0)), mode="constant", constant_values=0.0)
        elif variable == "h_no_grad_ew":
            # apply no gradient boundaries, ∂xη(x = 0) = ∂xη(x = Lx) = 0
            u = jnp.pad(u, ((1,1),(0,0)), mode="edge")
        elif variable == "h_no_grad_ns":
            # apply no gradient boundaries, ∂yη(y = 0) = ∂yη(y = Ly) = 0
            u = jnp.pad(u, ((0,0),(1,1)), mode="edge")
        elif variable == "h_no_grad":
            # apply no gradient boundaries, ∂yη(y = 0) = ∂yη(y = Ly) = 0
            u = jnp.pad(u, ((1,1),(1,1)), mode="edge")
        else:
            raise ValueError(f"Unrecognized variable: {variable}")

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
        
        # apply boundary conditions
        h = self.apply_boundaries(h[1:-1, 1:-1], "h_no_grad")
        u = self.apply_boundaries(u[:, 1:-1], "u_no_slip")
        u = self.apply_boundaries(u[1:-1, :], "u_no_flow")
        v = self.apply_boundaries(v[1:-1, :], "v_no_slip")
        v = self.apply_boundaries(v[:, 1:-1], "v_no_flow")
        
        total_h = h + self.depth
        # precalculate quantities
        if not self.linear_mass or not self.linear_momentum:
            # zero flux padding
            # h_pad = jnp.pad(h, mode="edge", pad_width=((1,1),(1,1)))
            # print(f"H: {h.shape} | VH FLUX: {u.shape} | V Mask: {masks.face_u.shape}")
            uh_flux = reconstruct(q=total_h, u=u[1:-1], u_mask=masks.face_u[1:-1,:], dim=0, method=self.mass_adv_scheme, num_pts=self.mass_adv_stencil)
            vh_flux = reconstruct(q=total_h, u=v[:,1:-1], u_mask=masks.face_v[:,1:-1], dim=1, method=self.mass_adv_scheme, num_pts=self.mass_adv_stencil)
            
            # # zero pv boundaries
            uh_flux: Float[Array, "Dx+1 Dy+1"] = jnp.pad(uh_flux, ((1, 1), (0, 0)), mode="constant", constant_values=0.0)
            vh_flux: Float[Array, "Dx+1 Dy+1"] = jnp.pad(vh_flux, ((0, 0), (1, 1)), mode="constant", constant_values=0.0)
            
            # # mask values
            uh_flux *= masks.face_u.values
            vh_flux *= masks.face_v.values

        if not self.linear_momentum:
            # planetary vorticity, f = f₀ + β y
            f_on_q: Float[Array, "Nx-1 Ny-1"] = self.coriolis_param(state.q_domain.grid[1:-1,1:-1, 1], state.q_domain.size[1])

            # calculate potential vorticity, q = (ζ + f) / h
            q: Float[Array, "Dx-1 Dy-1"] = potential_vorticity(h=total_h, u=u, v=v, f=f_on_q, dx=dx, dy=dy)

            # # zero pv boundaries
            q: Float[Array, "Dx+1 Dy+1"] = zero_boundaries(q, ((1, 1), (1, 1)))
            
            # mask values
            q *= masks.node.values

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
            
        else:
            # apply RHS, ∂h/∂t + ∂/∂x((H+h)u) + ∂/∂y((H+h)v) = 0
            h_rhs = calculate_h_nonlinear_rhs(
                uh_flux,
                vh_flux,
                dx=dx, dy=dy,
                depth=1.0,
            )

        # Calculate Zonal Velocity
        if self.linear_momentum:
            u_rhs: Float[ArrayLike, "Dx-1 Dy"] = calculate_u_linear_rhs(
                h, v, dx, self.coriolis_f0, self.gravity
            )
            # TODO: Change this!
            u_rhs = self.apply_boundaries(u_rhs, "u_no_flow")
            
        else:
            u_rhs = calculate_u_nonlinear_rhs(
                h=total_h,
                q=q[1:-1, :],
                vh_flux=vh_flux,
                u_mask=masks.face_u[1:-1, :],
                ke=ke,
                dx=dx,
                num_pts=self.momentum_adv_stencil,
                method=self.momentum_adv_scheme,
            )
            u_rhs = self.apply_boundaries(u_rhs, "u_no_flow")
            

        # Calculate Meridonal Velocity
        if self.linear_momentum:
            # Linear Momentum ( ∂v/∂t = - fu - g ∂h/∂y )
            v_rhs: Float[ArrayLike, "Dx Dy+1"] = calculate_v_linear_rhs(
                h, u, dy, self.coriolis_f0, self.gravity
            )
            v_rhs = self.apply_boundaries(v_rhs, "v_no_flow")
            
        else:
            
            v_rhs = calculate_v_nonlinear_rhs(
                h=total_h,
                q=q[:, 1:-1],
                uh_flux=uh_flux,
                v_mask=masks.face_v[:, 1:-1],
                ke=ke,
                dy=dy,
                num_pts=self.momentum_adv_stencil,
                method=self.momentum_adv_scheme,
            )
            v_rhs = self.apply_boundaries(v_rhs, "v_no_flow")


        # apply masks
        h_rhs *= state.masks.center.values
        u_rhs *= state.masks.face_u.values
        v_rhs *= state.masks.face_v.values
        
        # apply forcing
        if self.forcing is not None:
            h_rhs = self.forcing(h_rhs, "h")
            u_rhs = self.forcing(u_rhs, "u")
            v_rhs = self.forcing(v_rhs, "v")
        
        
        # apply masks
        h_rhs *= state.masks.center.values
        u_rhs *= state.masks.face_u.values
        v_rhs *= state.masks.face_v.values
        
        # update all state vectors
        state = eqx.tree_at(lambda x: x.h, state, h_rhs)
        state = eqx.tree_at(lambda x: x.u, state, u_rhs)
        state = eqx.tree_at(lambda x: x.v, state, v_rhs)

        return state



