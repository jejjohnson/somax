from typing import Tuple
import math
import equinox as eqx
from jaxtyping import Array, Float, ArrayLike
import jax
import jax.numpy as jnp
from somax._src.constants import CORIOLIS
from somax._src.masks.masks import MaskGrid
from somax._src.operators.differential import laplacian_2D
from somax._src.operators.average import center_avg_2D


def wind_curl(
    domain_size: float,
    y_coords: Array,
    domain_shape: Tuple[int, int],
    tau0: float = 0.08 / 1000.0,
):  
    Nx, Ny = domain_shape
    Lx, Ly = domain_size
    curl_tau = (
        - tau0
        * (2 * jnp.pi / Ly)
        * jnp.sin(2 * jnp.pi * y_coords / Ly)
    )
    
    return jnp.tile(curl_tau, (Nx, 1))


class WindForcing(eqx.Module):
    wind_forcing: Array = eqx.field(static=True)
    tau0: float = eqx.field(static=True)
    H_0: float = eqx.field(static=True)
    
    def __init__(self, y_coords: Array, domain_size: Tuple[int, int], domain_shape: Tuple[int, int], tau0: float=0.08 / 1000.0, H_0: float=400.0):
        self.tau0 = tau0
        self.H_0 = H_0
        self.wind_forcing = wind_curl(domain_size, y_coords, domain_shape, tau0) / H_0
    
    def __call__(self):
        return self.wind_forcing
    

class BottomDrag(eqx.Module):
    delta_ek: float = eqx.field(static=True)
    H_L: float = eqx.field(static=True)
    f0: float = eqx.field(static=True)
    
    def __init__(self, delta_ek: float=2.0, H_L: float=3_000, f0: float=CORIOLIS):
        self.delta_ek = delta_ek
        self.H_L = H_L
        self.f0 = f0
        
    @property
    def bottom_drag_coeff(self):
        return self.delta_ek / self.H_L * self.f0 / 2.0
    
    def __call__(self, psi: Array, dx: float, dy: float, masks: MaskGrid):
        
        batch_laplacian_2D = jax.vmap(laplacian_2D, in_axes=(0, None, None))
        
        forcing = batch_laplacian_2D(psi, dx, dy)
        
        # pad with zeros
        forcing = jnp.pad(forcing, ((0,0),(1,1),(1,1)), mode="constant", constant_values=0.0)
        
        # apply masks
        forcing *= masks.node.values
        
        # interpolate onto grid centers
        forcing = jax.vmap(center_avg_2D)(forcing)
        
        
        return self.bottom_drag_coeff * forcing[-1]
