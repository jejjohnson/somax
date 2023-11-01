import autoroot # noqa: F401
import diffrax as dfx
import einops
import equinox as eqx
from fieldx._src.domain.domain import Domain
import jax
from jax.config import config
import jax.numpy as jnp
from jaxtyping import (
    Array,
)
import numpy as np

config.update("jax_enable_x64", True)
import time

from fieldx._src.domain.time import TimeDomain
from finitevolx import (
    MaskGrid,
    center_avg_2D,
)
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd

from somax._src.models.qg.domain import LayerDomain
from somax._src.models.qg.elliptical import (
    DSTSolution,
    calculate_helmholtz_dst,
    compute_homogeneous_solution,
)
from somax._src.models.qg.operators import (
    advection_rhs,
    calculate_bottom_drag,
    calculate_potential_vorticity,
    calculate_wind_forcing,
    qg_rhs,
)
from somax._src.models.qg.params import QGParams
from somax._src.operators.dst import (
    compute_capacitance_matrices,

)


def plot_field(field):
    num_axis = len(field)
    fig, ax = plt.subplots(ncols=num_axis, figsize=(8, 2))

    for i in range(num_axis):
        pts = ax[i].imshow(field[i].T, origin="lower", cmap="coolwarm")
        plt.colorbar(pts)

    plt.tight_layout()
    plt.show()


# Low Resolution
# Nx, Ny = 128, 128
Nx, Ny = 256, 256
# High Resolution
# Nx, Ny = 769, 961

# Lx, Ly = 3840.0e3, 4800.0e3
Lx, Ly = 5_120.0e3, 5_120.0e3

dx, dy = Lx / Nx, Ly / Ny

xy_domain = Domain(
    xmin=(0.0, 0.0), xmax=(Lx, Ly), Lx=(Lx, Ly), Nx=(Nx, Ny), dx=(dx, dy)
)

params = QGParams(y0=0.5 * Ly)


# ====================================
# Boundaries
# ====================================
domain_type = "rectangular"

mask = jnp.ones((Nx, Ny))
mask = mask.at[0].set(0.0)
mask = mask.at[-1].set(0.0)
mask = mask.at[:, 0].set(0.0)
mask = mask.at[:, -1].set(0.0)

masks = MaskGrid.init_mask(mask, location="node")


# ====================================
# HEIGHTS
# ====================================
# heights
# heights = [350.0, 750.0, 2900.0]
heights = [400.0, 1_100.0, 2_600.0]

# reduced gravities
reduced_gravities = [0.025, 0.0125]

# initialize layer domain
layer_domain = LayerDomain(heights, reduced_gravities, correction=False)

# ====================================
# DST ELLIPTICAL SOLVER
# ====================================
H_mat = calculate_helmholtz_dst(xy_domain, layer_domain, params)

psi0 = jnp.ones(shape=(layer_domain.Nz,) + xy_domain.Nx)

# psi0 = np.load("/Users/eman/code_projects/data/qg_runs/psi_000y_360d.npy")[0]
lambda_sq = params.f0**2 * einops.rearrange(layer_domain.lambda_sq, "Nz -> Nz 1 1")

homsol = compute_homogeneous_solution(psi0, lambda_sq=lambda_sq, H_mat=H_mat)

# calculate homogeneous solution
homsol_i = jax.vmap(center_avg_2D)(homsol) * masks.center.values

homsol_mean = einops.reduce(homsol_i, "Nz Nx Ny -> Nz 1 1", reduction="mean")

# CALCULATE CAPCITANCE MATRIX
if domain_type == "octogonal":
    cap_matrices = compute_capacitance_matrices(
        H_mat, masks.nodes.irrbound_xids, masks.nodes.irrbound_yids
    )
else:
    cap_matrices = None


# DST SOLUTION
dst_sol = DSTSolution(
    homsol=homsol, homsol_mean=homsol_mean, H_mat=H_mat, capacitance_matrix=cap_matrices
)

# ====================================
# INITIAL VARIABLES
# ====================================

# PV
q = calculate_potential_vorticity(
    psi0,
    xy_domain,
    layer_domain,
    params=params,
    masks_psi=masks.node,
    masks_q=masks.center,
)

fn = jax.vmap(advection_rhs, in_axes=(0, 0, None, None, None, None, None, None))

div_flux = fn(
    q, psi0, xy_domain.dx[-2], xy_domain.dx[-1], 5, "wenoz", masks.face_u, masks.face_v
)

bottom_drag = calculate_bottom_drag(
    psi=psi0,
    domain=xy_domain,
    H_z=layer_domain.heights[-1],
    f0=params.f0,
    masks_psi=masks.node,
)

wind_forcing = calculate_wind_forcing(
    domain=xy_domain,
    H_0=layer_domain.heights[0],
    tau0=0.08 / 1_000.0,
)

dq, dpsi = qg_rhs(
    q=q,
    psi=psi0,
    domain=xy_domain,
    params=params,
    layer_domain=layer_domain,
    dst_sol=dst_sol,
    wind_forcing=wind_forcing,
    bottom_drag=bottom_drag,
    masks=masks,
)


# ====================================
# STATE VECTOR
# ====================================


class State(eqx.Module):
    q: Array
    psi: Array


# ====================================
# EQUATION OF MOTION
# ====================================


def vector_field(t: float, state: State, args) -> State:
    dq, dpsi = qg_rhs(
        q=state.q,
        psi=state.psi,
        domain=xy_domain,
        params=params,
        layer_domain=layer_domain,
        dst_sol=dst_sol,
        wind_forcing=wind_forcing,
        bottom_drag=bottom_drag,
        masks=masks,
    )

    state = eqx.tree_at(lambda x: x.q, state, dq)
    state = eqx.tree_at(lambda x: x.psi, state, dpsi)

    return state


logger.info("Initializing State...")

# psi0 = np.load("/Users/eman/code_projects/data/qg_runs/psi_000y_360d.npy")[0]
# psi0 = np.load("/Users/eman/code_projects/data/qg_runs/psi_0.986y_360.00d_octogonal.npy")


state_init = State(q=q, psi=psi0)

logger.info("Initializing Time Intervals...")
dt = 4_000

tmin = 0.0
num_days = 5 * 360
tmax = pd.to_timedelta(num_days, unit="days").total_seconds()
num_save = 20

t_domain = TimeDomain(tmin=tmin, tmax=tmax, dt=dt)
ts = jnp.linspace(tmin, tmax, num_save)
saveat = dfx.SaveAt(ts=ts)

solver = dfx.Bosh3()
stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-5)

logger.info("Starting Integration...")
# integration
t0 = time.time()
sol = dfx.diffeqsolve(
    terms=dfx.ODETerm(vector_field),
    solver=solver,
    t0=tmin,
    t1=tmax,
    dt0=dt,
    y0=state_init,
    saveat=saveat,
    args=None,
    stepsize_controller=stepsize_controller,
    max_steps=None,
)
logger.info("Done...!")
t1 = time.time() - t0

logger.info(f"Time taken: {t1 / 60:.2f} mins")

plot_field(sol.ys.psi[-1])
plot_field(sol.ys.q[-1])
n_years = num_days / 365

# logger.info(f"Saving...")
# output_dir = "/Users/eman/code_projects/data/qg_runs"
# fname = os.path.join(output_dir, f'psi_{n_years:.3f}y_{num_days:.2f}d_{domain_type}_.npy')
# np.save(fname, np.asarray(sol.ys.psi).astype('float32'))
#
# fname = os.path.join(output_dir, f'q_{n_years:.3f}y_{num_days:.2f}d_{domain_type}_.npy')
# np.save(fname, np.asarray(sol.ys.q).astype('float32'))

logger.info("Completed Script!")
