from typing import NamedTuple
import jax.numpy as jnp


class TimeDomain(NamedTuple):
    tmin: float
    tmax: float
    dt: float

    @classmethod
    def from_numpoints(cls, tmin, tmax, nt):
        dt = (tmax - tmin) / (float(nt) - 1)

        return cls(tmin=tmin, tmax=tmax, dt=dt)

    @property
    def coords(self):
        return jnp.arange(self.tmin, self.tmax, self.dt)