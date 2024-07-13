import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array
from somax._src.constants import GRAVITY
from somax._src.models.linear_swm_field.ops import calculate_h_rhs, calculate_u_rhs, calculate_v_rhs
from somax._src.boundaries.base import no_slip_boundaries, zero_boundaries
from somax._src.field.base import Field, pad_x_field, pad_y_field, isel_x_field, isel_y_field


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
    
    def apply_boundaries(self, u: Field, variable: str="h") -> Field:
        if variable == "u":
            # select interior points
            u = isel_x_field(u, slice(2,-2))
            # pad with zeros (no-slip boundaries)
            u = pad_x_field(u, pad_width=(2,2), mode="constant", constant_values=0.0)
        elif variable == "v":
            # select interior points
            u = isel_y_field(u, slice(2,-2))
            # pad with zeros (no-slip boundaries)
            u = pad_y_field(u, pad_width=(2,2), mode="constant", constant_values=0.0)
        elif variable == "h":
            # do nothing
            pass
        else:
            msg = f"Unrecongized variable: {variable}"
            msg = f"\nShould be: 'u', 'v', or 'h'."
            raise ValueError(msg)
        return u
    
    def equation_of_motion(self, t, state, args):

        (h, u, v) = state
        masks = args
        
        u_rhs = calculate_u_rhs(h, v, u.domain, coriolis_param=self.coriolis_param, gravity=self.gravity)
        u_rhs *= masks.face_u.values
        u_rhs = self.apply_boundaries(u_rhs, "u")
        
        v_rhs = calculate_v_rhs(h, u, v.domain, self.coriolis_param, self.gravity)
        v_rhs *= masks.face_v.values
        v_rhs = self.apply_boundaries(v_rhs, "v")
        
        h_rhs = calculate_h_rhs(u, v, h.domain, self.depth)
        h_rhs *= masks.center.values
        h_rhs = self.apply_boundaries(h_rhs, "h")

        # replace values to ensure the same input/output structure
        h = h.replace_values(h_rhs.values)
        u = u.replace_values(u_rhs.values)
        v = v.replace_values(v_rhs.values)

        return (h, u, v)
    


