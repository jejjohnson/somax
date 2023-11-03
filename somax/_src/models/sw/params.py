import equinox as eqx
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)


class SWMParams(eqx.Module):
    gravity: float = eqx.field(static=True)
    depth: float = eqx.field(static=True)
    coriolis_f0: float = eqx.field(static=True)
    coriolis_beta: float = eqx.field(static=True)
    linear_mass: bool = eqx.field(static=True)
    linear_momentum: bool = eqx.field(static=True)
    mass_num_pts: int = eqx.field(static=True)
    mass_method: str = eqx.field(static=True)
    mom_num_pts: int = eqx.field(static=True)
    mom_method: str = eqx.field(static=True)

    def coriolis_param(self, Y: Float[Array, "Nx Ny"]):
        return self.coriolis_f0 + Y * self.coriolis_beta

    def lateral_viscosity(self, dx: Array):
        return 1e-3 * self.coriolis_f0 * dx**2

    @property
    def phase_speed(self):
        return jnp.sqrt(self.gravity * self.depth)

    def rossby_radius(self, Y: Float[Array, "Nx Ny"]):
        return jnp.sqrt(self.gravity * self.depth) / self.coriolis_param(Y=Y).mean()
