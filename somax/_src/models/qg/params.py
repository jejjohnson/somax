import equinox as eqx


class QGParams(eqx.Module):
    f0: float = 9.375e-5  # coriolis [s^-1]
    beta: float = 1.754e-11  # coriolis gradient [m^-1 s^-1]
    tau0: float = 2.0e-5  # wind stress magnitude m/s^2
    y0: float = 2_400_000.0  # [m]
    a_2: float = 0.0  # laplacian diffusion coef (m^2/s)
    a_4: float = 0.0  # biharmonic diffusion coef (m^4/s)
    bcco: float = 0.2  # boundary condition coef. (non-dim.)
    delta_ek: float = 2.0  # eckman height [m]
    num_pts: int = 5
    method: str = "wenoz"

    @property
    def zfbc(self):
        return self.bcco / (1.0 + 0.5 * self.bcco)
