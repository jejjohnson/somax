from typing import Any, List
import equinox as eqx
from somax._src.domain.cartesian import CartesianDomain2D
from jaxtyping import Array, ArrayLike, PyTree, Float
from somax._src.masks.masks import MaskGrid
from somax._src.models.mqg.decomp import Mode2LayerTransformer
from somax._src.operators.average import center_avg_2D
import jax
import jax.numpy as jnp
from somax._src.models.mqg.ops import compute_q_from_psi
from functools import partial


class MQGParams(eqx.Module):
    # coriolis [s^-1]
    f0: float = eqx.field(static=True)
    # coriolis gradient [m^-1 s^-1]
    beta: float = eqx.field(static=True)
    # wind stress magnitude m/s^2
    tau0: float = eqx.field(static=True)
    # [m]
    y0: float = eqx.field(static=True)
    # laplacian diffusion coef (m^2/s)
    a_2: float = eqx.field(static=True)
    # biharmonic diffusion coef (m^4/s)
    a_4: float = eqx.field(static=True)
    # boundary condition coef. (non-dim.)
    bcco: float = eqx.field(static=True)
    # eckman height [m]
    delta_ek: float = eqx.field(static=True)
    # stencil size for advection scheme
    num_pts: int = eqx.field(static=True)
    # method for advection scheme
    method: str = eqx.field(static=True)
    heights: Array = eqx.static_field()
    reduced_gravities: Array = eqx.static_field()
    
    def __init__(
        self,
        f0: float = 9.375e-5,
        beta: float = 1.754e-11,
        tau0: float = 2.0e-5,
        y0: float = 2_400_000.0,
        a_2: float = 0.0,
        a_4: float = 0.0,
        bcco: float = 0.2,
        delta_ek: float = 2.0,
        num_pts: int = 5,
        method: str = "wenoz",
        heights: List[float] = [400.0, 1_100.0, 2_600.0],
        reduced_gravities: List[float] = [0.025, 0.0125],
        ):
        self.f0 = f0
        self.beta = beta
        self.tau0 = tau0
        self.y0 = y0
        self.a_2 = a_2
        self.a_4 = a_4
        self.bcco = bcco
        self.delta_ek = delta_ek
        self.num_pts = num_pts
        self.method = method
        self.heights = jnp.asarray(heights)
        self.reduced_gravities = jnp.asarray(reduced_gravities)
        
    @property
    def Nz(self):
        return len(self.heights)
    
    @property
    def helmholtz_beta(self):
        return self.f0 ** 2
    
    
class MQGState(eqx.Module):
    psi: Array
    q: Array
    psi_domain: CartesianDomain2D = eqx.field(static=True)
    q_domain: CartesianDomain2D = eqx.field(static=True)
    masks: MaskGrid = eqx.field(static=True)
    
    @classmethod
    def init_from_psi(cls, psi: Array, psi_domain: CartesianDomain2D, masks: MaskGrid, params: MQGParams, transformer: Mode2LayerTransformer):
        
        # stagger the domain
        q_domain = psi_domain.stagger_x(direction="inner", stagger=True)
        q_domain = q_domain.stagger_y(direction="inner", stagger=True)
        
        dx, dy = psi_domain.resolution
        Y = psi_domain.grid[..., 1]
        
        
        q = compute_q_from_psi(psi=psi, masks=masks, dx=dx, dy=dy, Y=Y, y0=params.y0, beta=params.beta, transformer=transformer)
        
        return cls(psi=psi, q=q, psi_domain=psi_domain, q_domain=q_domain, masks=masks)

class MQGModel(eqx.Module):
    params: PyTree
    transformer: PyTree
    solver: PyTree
    
    def __init__(
        self,
        solver: PyTree,
        params: MQGParams,
        transformer: Mode2LayerTransformer,
    ):
        self.transformer = transformer
        self.solver = solver
        self.params = params
        
    def calculate_q_from_psi(self, psi: Float[Array, "Nx+1 Ny+1"], masks: MaskGrid, psi_domain: CartesianDomain2D) -> Float[Array, "Nx Ny"]:
        
        dx, dy = psi_domain.resolution
        Y = psi_domain.grid[..., 1]
        
        q = compute_q_from_psi(psi=psi, masks=masks, dx=dx, dy=dy, Y=Y, y0=self.params.y0, beta=self.params.beta, transformer=self.transformer)
        return q
    
    def calculate_psi_from_q(self, q: Float[Array, "Nx Ny"], masks: MaskGrid, q_domain: CartesianDomain2D) -> Float[Array, "Nx+1 Ny+1"]:
        
        Y = q_domain.grid[..., 1]
        
        beta_term = self.params.beta * (Y - self.params.y0)
        
        q_rhs = q - beta_term
        
        q_on_psi = jax.vmap(center_avg_2D)(q_rhs)
        
        psi = self.solver.solve(q_on_psi)
        
        return psi


    def equation_of_motion(self, t: ArrayLike, state: MQGState, args: Any | None = None) -> MQGState:
        return None

