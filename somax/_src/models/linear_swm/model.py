import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array
from somax._src.constants import GRAVITY
from somax._src.models.linear_swm.ops import calculate_h_rhs, calculate_u_rhs, calculate_v_rhs
from somax._src.boundaries.base import no_slip_boundaries, zero_boundaries


class LinearSWM(eqx.Module):
    gravity: float = eqx.field(static=True,)
    depth: float = eqx.field(static=True)
    coriolis_param: float = eqx.field(static=True)

    def __init__(self, gravity=GRAVITY, depth=100, coriolis_param=2e-4):
        self.gravity = gravity
        self.depth = depth
        self.coriolis_param = coriolis_param

    @property
    def phase_speed(self):
        return jnp.sqrt(self.gravity * self.depth)
    
    @property
    def rossby_radius(self):
        return jnp.sqrt(self.gravity * self.depth) / self.coriolis_param
    
    def apply_boundaries(self, u: Array, variable: str="h"):
        if variable == "u":
            u = zero_boundaries(u[1:-1], pad_width=((2,2),(0,0)))
        if variable == "v":
            u = zero_boundaries(u[:, 1:-1], pad_width=((0,0),(2,2)))
        return u
    
    def equation_of_motion(self, t, state, args):

        (h, u, v) = state
        domain, masks = args
        dx, dy = domain.resolution
        

        u_rhs = calculate_u_rhs(h, v, dx, self.coriolis_param, self.gravity)
        u_rhs *= masks.face_u.values[1:-1, ...]
        u_rhs = self.apply_boundaries(u_rhs, "u")
        
        v_rhs = calculate_v_rhs(h, u, dy, self.coriolis_param, self.gravity)
        v_rhs *= masks.face_v.values[..., 1:-1]
        v_rhs = self.apply_boundaries(v_rhs, "v")
        
        h_rhs = calculate_h_rhs(u, v, dx, dy, self.depth)
        h_rhs *= masks.center.values

        return (h_rhs, u_rhs, v_rhs)
    


