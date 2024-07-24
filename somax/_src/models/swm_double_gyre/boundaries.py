from jaxtyping import Array
import jax.numpy as jnp


def apply_boundaries(u: Array, variable: str="u_no_flow"):
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
    else:
        raise ValueError(f"Unrecognized variable: {variable}")

    return u