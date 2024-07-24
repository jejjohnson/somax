from typing import Any, List
import equinox as eqx
from somax._src.domain.cartesian import CartesianDomain2D
from jaxtyping import Array, ArrayLike, PyTree
from somax._src.masks.masks import MaskGrid
from somax._src.models.mqg.decomp import Mode2LayerTransformer
import jax
import jax.numpy as jnp


class MQGState(eqx.Module):
    h: Array
    u: Array
    v: Array
    h_domain: CartesianDomain2D = eqx.field(static=True)
    u_domain: CartesianDomain2D = eqx.field(static=True)
    v_domain: CartesianDomain2D = eqx.field(static=True)
    q_domain: CartesianDomain2D = eqx.field(static=True)
    masks: MaskGrid = eqx.field(static=True)
    

class MQGModel(eqx.Module):
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
    transformer: PyTree
    heights: Array = eqx.static_field()
    reduced_gravities: Array = eqx.static_field()
    Nz: int = eqx.static_field()
    solver: PyTree
    
    def __init__(
        self,
        solver: PyTree,
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
        correction: bool = False
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
        with jax.default_device(jax.devices("cpu")[0]):
            transformer = Mode2LayerTransformer(heights, reduced_gravities, correction=False)
        self.transformer = transformer
        self.heights = jnp.asarray(heights)
        self.reduced_gravities = jnp.asarray(reduced_gravities)
        self.Nz = len(heights)
        self.solver = solver
        
    @property
    def lambd_sq(self):
        return self.transformer.ev_A * self.f0 ** 2


    def equation_of_motion(self, t: ArrayLike, state: MQGState, args: Any | None = None) -> MQGState:
        return None

