import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # second gpu


import jax

# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)

import autoroot  # noqa: F401
import typing as tp
import equinox as eqx
import einops

import math
from pathlib import Path
import diffrax as dfx
from jaxtyping import Array, Float
import jax
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import pandas as pd
from somax.domain import Domain
from somax.masks import MaskGrid
from somax.interp import x_avg_1D, x_avg_2D, y_avg_2D, center_avg_2D
from somax._src.models.qg.params import QGParams
from somax._src.models.qg.domain import LayerDomain
from somax._src.models.qg.elliptical import (
    DSTSolution,
    calculate_helmholtz_dst,
    compute_homogeneous_solution,
)
from somax._src.models.qg.forcing import (
    calculate_bottom_drag,
    calculate_wind_forcing,
)

from somax._src.models.qg.operators import (
    calculate_potential_vorticity,
    calculate_psi_from_pv,
    equation_of_motion,
)
import typer

import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


from loguru import logger


def plot_field(field, name=""):
    num_axis = len(field)
    fig, ax = plt.subplots(ncols=num_axis, figsize=(9, 3))
    vmin, vmax = np.min(field), np.max(field)
    vlim = np.min([np.abs(vmin), np.abs(vmax)])
    fig.suptitle(name)
    for i in range(num_axis):
        pts = ax[i].pcolormesh(
            field[i].T, cmap="coolwarm", vmin=-vlim, vmax=vlim, rasterized=True
        )
        plt.colorbar(pts, shrink=0.6)
        ax[i].set_aspect("equal")

    plt.tight_layout()
    plt.show()


def print_debug_quantity(quantity, name=""):
    size = quantity.shape
    min_ = jnp.min(quantity)
    max_ = jnp.max(quantity)
    mean_ = jnp.mean(quantity)
    median_ = jnp.mean(quantity)
    jax.debug.print(
        f"{name}: {size} | {min_:.6e} | {mean_:.6e} | {median_:.6e} | {max_:.6e}"
    )


def main(
    save_dir="./",
    diffusivity: float = 30.0,
    resolution: int = 512,
    method: str = "arakawa",
    num_pts: int = 3,
    dt: int = 2000,
    num_years: int = 50,
    stepsize_controller: bool = False,
):
    """
    high resoltion (>=512) - dt: 2000
    low resolution (<256) - dt: 4000

    stepsize (bool): options (constant, controller)
    """
    logger.info(f"Starting script...")
    output_dir = Path(save_dir).joinpath("qg_runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Save Directory: {output_dir}")

    # # Low Resolution (Non-Resolving)
    # Nx, Ny = 64, 64
    # Nx, Ny = 128, 128
    # Low Resolution
    # Nx, Ny = 256, 256
    # High Resolution
    logger.info(f"Resolution: {resolution}...")
    Nx, Ny = resolution, resolution
    # Nx, Ny = 769, 961

    # Lx, Ly = 3840.0e3, 4800.0e3
    # Lx, Ly = 4_000.0e3, 4_000.0e3
    Lx, Ly = 5_120.0e3, 5_120.0e3

    dx, dy = Lx / Nx, Ly / Ny

    logger.info(f"Create Domain...")
    xy_domain = Domain(
        xmin=(0.0, 0.0), xmax=(Lx, Ly), Lx=(Lx, Ly), Nx=(Nx, Ny), dx=(dx, dy)
    )

    # params = QGParams(
    #     y0=0.5 * Ly, a_2=1.,
    # )
    # params

    # explicit diffusivity
    if diffusivity > 0.0:
        params = QGParams(
            y0=0.5 * Ly,
            a_2=diffusivity,
            f0=9.4e-05,
            beta=1.7e-11,
            tau0=5e-5,
            # method='arakawa'
            method=method,
            num_pts=num_pts,
        )
    else:

        # NO diffusivity
        params = QGParams(
            y0=0.5 * Ly,
            a_2=0.0,
            f0=9.4e-05,
            beta=1.7e-11,
            tau0=5e-5,
            method=method,
            num_pts=num_pts,
        )

    # ====================================
    # Boundaries
    # ====================================
    logger.info(f"Create Boundaries...")
    domain_type = "rectangular"

    mask = jnp.ones((Nx, Ny))
    mask = mask.at[0].set(0.0)
    mask = mask.at[-1].set(0.0)
    mask = mask.at[:, 0].set(0.0)
    mask = mask.at[:, -1].set(0.0)
    masks = MaskGrid.init_mask(mask, location="node")

    # ====================================
    # Layer Domain
    # ====================================
    # heights = [350.0, 750.0, 2900.0]
    heights = [400.0, 1_100.0, 2_600.0]

    # reduced gravities
    reduced_gravities = [0.025, 0.0125]

    # initialize layer domain
    with jax.default_device(jax.devices("cpu")[0]):
        layer_domain = LayerDomain(
            heights, reduced_gravities, correction=False
        )

    # from jaxsw._src.operators.functional import elliptical as F_elliptical
    H_mat = calculate_helmholtz_dst(xy_domain, layer_domain, params)
    H_mat.shape

    psi0 = jnp.ones(shape=(layer_domain.Nz,) + xy_domain.Nx)

    # psi0 = np.load("/Users/eman/code_projects/data/qg_runs/psi_000y_360d.npy")[0]
    lambda_sq = params.f0**2 * einops.rearrange(
        layer_domain.lambda_sq, "Nz -> Nz 1 1"
    )

    homsol = compute_homogeneous_solution(
        psi0, lambda_sq=lambda_sq, H_mat=H_mat
    )
    print_debug_quantity(homsol, "HOMSOL")

    # calculate homogeneous solution
    homsol_i = jax.vmap(center_avg_2D)(homsol) * masks.center.values

    homsol_mean = einops.reduce(
        homsol_i, "Nz Nx Ny -> Nz 1 1", reduction="mean"
    )
    print_debug_quantity(homsol_mean, "HOMSOL MEAN")

    # CALCULATE CAPCITANCE MATRIX
    if domain_type == "octogonal":
        cap_matrices = compute_capacitance_matrices(
            H_mat, masks.node.irrbound_xids, masks.node.irrbound_yids
        )
    else:
        cap_matrices = None

    # DST SOLUTION
    dst_sol = DSTSolution(
        homsol=homsol,
        homsol_mean=homsol_mean,
        H_mat=H_mat,
        capacitance_matrix=cap_matrices,
    )

    wind_forcing = calculate_wind_forcing(
        domain=xy_domain,
        params=params,
        H_0=layer_domain.heights[0],
        tau0=0.08 / 1_000.0,
    )

    def forcing_fn(
        psi: Float[Array, "Nz Nx Ny"],
        dq: Float[Array, "Nz Nx-1 Ny-1"],
        domain: Domain,
        layer_domain: LayerDomain,
        params: QGParams,
        masks: MaskGrid,
    ) -> Float[Array, "Nz Nx Ny"]:

        # add wind forcing
        dq = dq.at[0].add(wind_forcing)

        # calculate bottom drag
        bottom_drag = calculate_bottom_drag(
            psi=psi,
            domain=domain,
            H_z=layer_domain.heights[-1],
            delta_ek=params.delta_ek,
            f0=params.f0,
            masks_psi=masks.node,
        )

        dq = dq.at[-1].add(bottom_drag)

        return dq

    logger.info(f"Create State...")

    class State(eqx.Module):
        q: Array
        psi: Array

    def vector_field(t: float, state: State, args) -> State:

        dq = equation_of_motion(
            q=state.q,
            psi=state.psi,
            params=params,
            domain=xy_domain,
            layer_domain=layer_domain,
            forcing_fn=forcing_fn,
            masks=masks,
        )

        dpsi = calculate_psi_from_pv(
            q=dq,
            params=params,
            domain=xy_domain,
            layer_domain=layer_domain,
            mask_node=masks.node,
            dst_sol=dst_sol,
            # include_beta=True
            remove_beta=False,
        )

        state = eqx.tree_at(lambda x: x.q, state, dq)
        state = eqx.tree_at(lambda x: x.psi, state, dpsi)

        return state

    def create_dataset(psi, q, year, time):

        import xarray as xr

        ds = xr.Dataset(
            {
                "q": (("z", "x_center", "y_center"), q),
                "psi": (("z", "x_node", "y_node"), psi),
            },
            coords={
                "x_node": xy_domain.coords_axis[0],
                "y_node": xy_domain.coords_axis[1],
                "year": year,
                "diffusivity": diffusivity,
                "method": method,
                "num_pts": num_pts,
                "resolution": resolution,
                "time": time,
                "dt": dt,
            },
        )

        return ds

    tmin = 0.0
    num_days = 360
    tmax = pd.to_timedelta(num_days, unit="days").total_seconds()
    num_save = 2
    num_steps = int(tmax / dt)

    logger.info(f"Number of Steps per year: {num_steps}...")

    from somax.domain import TimeDomain

    t_domain = TimeDomain(tmin=tmin, tmax=tmax, dt=dt)
    ts = jnp.linspace(tmin, tmax, num_save)

    def ssrk3_update_stage_1(x, x0, dt):
        # self.q += self.dt * dq_0
        return x + dt * x0

    def ssrk3_update_stage_2(x, x0, x1, dt):
        # self.q += (self.dt/4)*(dq_1 - 3*dq_0)
        return x + (dt / 4) * (x1 - 3 * x0)

    def ssrk3_update_stage_3(x, x0, x1, x2, dt):
        # self.q += (self.dt/12)*(8*dq_2 - dq_1 - dq_0)
        return x + (dt / 12) * (8 * x2 - x1 - x0)

    def timestepper_ssrk3(t, state, *args):
        """Time itegration with SSP-RK3 scheme."""

        # compute update (round 1)
        state_1 = vector_field(t, state, None)

        q = ssrk3_update_stage_1(state.q, state_1.q, dt)
        psi = ssrk3_update_stage_1(state.psi, state_1.psi, dt)

        state = eqx.tree_at(lambda x: x.q, state, q)
        state = eqx.tree_at(lambda x: x.psi, state, psi)

        # compute update (round 2)
        state_2 = vector_field(t, state, None)

        q = ssrk3_update_stage_2(state.q, state_1.q, state_2.q, dt)
        psi = ssrk3_update_stage_2(state.psi, state_1.psi, state_2.psi, dt)

        state = eqx.tree_at(lambda x: x.q, state, q)
        state = eqx.tree_at(lambda x: x.psi, state, psi)

        # compute update (round 3)
        state_3 = vector_field(t, state, None)

        q = ssrk3_update_stage_3(state.q, state_1.q, state_2.q, state_3.q, dt)
        psi = ssrk3_update_stage_3(
            state.psi, state_1.psi, state_2.psi, state_3.psi, dt
        )

        state = eqx.tree_at(lambda x: x.q, state, q)
        state = eqx.tree_at(lambda x: x.psi, state, psi)

        return state

    psi0 = jnp.zeros(shape=(layer_domain.Nz,) + xy_domain.Nx)
    # psi0 = np.load("/Users/eman/code_projects/data/qg_runs/psi_0.986y_360.00d_octogonal.npy")

    q0 = calculate_potential_vorticity(
        psi0,
        xy_domain,
        layer_domain,
        params=params,
        masks_psi=masks.node,
        masks_q=masks.center,
    )

    state = State(q=q0, psi=psi0)

    from tqdm.auto import trange, tqdm
    import time
    from functools import partial

    # jit to make it go brrrr
    fn = jax.jit(timestepper_ssrk3)

    pbar_year = trange(num_years, leave=True)
    logger.info(f"Starting Time Step...")
    logger.info(f"Starting Time Step...")

    t0_all = time.time()

    logger.info("Saving Initial Condition...")
    ds = create_dataset(psi0, q0, 0, 0)

    logger.info("Creating Dataset...")
    fname = output_dir.joinpath(
        f"soln_0y_r{resolution}_d{diffusivity}_{method}_pts{num_pts}.nc"
    )
    ds.to_netcdf(fname, engine="netcdf4")

    for iyear in pbar_year:
        pbar_year.set_description(f"Year - {iyear}")

        pbar_timestep = trange(num_steps, leave=False)
        t0 = time.time()
        for istep in pbar_timestep:
            state = fn(0, state, None)

        t1 = time.time() - t0
        logger.info(f"Year - {iyear+1} | Time Taken - {t1:.2f} secs")

        logger.info("Saving...")
        # print(state.psi.shape, state.q.shape)
        ds = create_dataset(state.psi, state.q, iyear + 1, t1)

        logger.info("Creating Dataset...")
        fname = output_dir.joinpath(
            f"soln_{iyear+1}y_r{resolution}_d{diffusivity}_{method}_pts{num_pts}.nc"
        )
        ds.to_netcdf(fname, engine="netcdf4")

    t1 = time.time() - t0_all
    logger.info("Completed Script!")
    logger.info(f"Total Time Taken - {t1:.2f} secs")


if __name__ == "__main__":
    typer.run(main)
