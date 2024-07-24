from jaxtyping import Float, Array
import jax.numpy as jnp
from somax._src.constants import RHO
import equinox as eqx


class WindForcing(eqx.Module):
    F_y: Float[Array, "Nx+1 Ny"]
    F_x: Float[Array, "Nx Ny+1"]
    
    def __init__(self, u_domain, v_domain, rho_0: float=RHO, H: float=100.0, F_0: float=0.12):
        
        y_coords = u_domain.grid[..., 1]
        L_y = u_domain.size[1]
        
        self.F_x = trade_wind(coords=y_coords, L=L_y, rho_0=rho_0, H=H, F_0=F_0)
        self.F_y = jnp.zeros(v_domain.shape)
        
    def __call__(self, u: Array, variable: str="u"):
        if variable == "u":
            u += self.F_x
        else:
            pass
        return u


def trade_wind(coords, L, rho_0: float=RHO, H: float=100.0, F_0: float=0.12):
    
    F = (
        jnp.cos(2 * jnp.pi * (coords / L - 0.5)) +
        2 * jnp.sin(2 * jnp.pi * (coords / L - 0.5))
    )
    
    param = F_0 / (rho_0 * H)
    return param * F
        
