""" swm.py

2D shallow water model with:

- varying Coriolis force
- nonlinear terms
- no lateral friction
- periodic boundary conditions
- reconstructions, 5pt, improved weno
    * mass term: h --> u,v
    * momentum term: q --> uh,vh
"""
import autoroot
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # first gpu
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'FALSE'

import jax
# jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

from typing import Tuple
import typer
import numpy as np
import pandas as pd
import diffrax as dfx
import jax.numpy as jnp
from somax._src.domain.cartesian import CartesianDomain2D
from somax._src.models.swm.model import SWM, SWMState
from somax._src.models.swm.ics import init_h0_jet, init_partition
from somax._src.operators.average import x_avg_2D, y_avg_2D
from somax.masks import MaskGrid
from loguru import logger
from dataclasses import dataclass
from somax._src.constants import GRAVITY
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate
import hydra_zen
from hydra_zen import (
    MISSING,
    builds,
    make_custom_builds_fn,
    ZenStore,
    hydrated_dataclass,
    store,
)

sbuilds = make_custom_builds_fn(populate_full_signature=True)
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)


# CREATE MODELS
MODEL_WENOZ = sbuilds(
    SWM,
    gravity=GRAVITY,
    depth=100,
    coriolis_f0=2e-4,
    coriolis_beta=2e-11,
    linear_mass=False,
    linear_momentum=False,
    mass_adv_scheme="wenoz",
    mass_adv_stencil=5,
    momentum_adv_scheme="wenoz",
    momentum_adv_stencil=5,
)
MODEL_WENO = sbuilds(
    SWM,
    gravity= GRAVITY,
    depth=100,
    coriolis_f0=2e-4,
    coriolis_beta=2e-11,
    linear_mass=False,
    linear_momentum=False,
    mass_adv_scheme="weno",
    mass_adv_stencil=5,
    momentum_adv_scheme="weno",
    momentum_adv_stencil=5,
)
MODEL_LINEAR = sbuilds(
    SWM,
    gravity=GRAVITY,
    depth=100,
    coriolis_f0=2e-4,
    coriolis_beta=2e-11,
    linear_mass=False,
    linear_momentum=False,
    mass_adv_scheme="linear",
    mass_adv_stencil=5,
    momentum_adv_scheme="linear",
    momentum_adv_stencil=5,
)
store(MODEL_WENOZ, group="model", name="wenoz")
store(MODEL_WENO, group="model", name="weno")
store(MODEL_LINEAR, group="model", name="linear")

@dataclass
class Grid:
    shape: Tuple[int, int] = (200, 104)
    resolution: Tuple[float, float] = (5e3, 5e3)

    def init_domains(self):
        Nx, Ny = self.shape
        dx, dy = self.resolution

        x, y = (np.arange(Nx) * dx, np.arange(Ny) * dy)
        h_domain = CartesianDomain2D.init_from_coords((x, y))
        u_domain = h_domain.stagger_x(direction="outer", stagger=True)
        v_domain = h_domain.stagger_y(direction="outer", stagger=True)
        q_domain = h_domain.stagger_x(direction="outer", stagger=True).stagger_y(direction="outer", stagger=True)

        return h_domain, u_domain, v_domain, q_domain

GRID_STANDARD = sbuilds(Grid, shape=(200, 104), resolution=(5e3,5e3))

store(GRID_STANDARD, group="grid", name="standard")

@dataclass
class Period:
    num_time_steps: float = 365.0
    units: str = "days"
    num_save: int = 10

    @property
    def tmax(self):
        return pd.to_timedelta(self.num_time_steps, unit=self.units).total_seconds()

    @property
    def saveat(self):
        ts = jnp.linspace(0.0, self.tmax, self.num_save)
        return dfx.SaveAt(ts=ts)
    
PERIOD_STANDARD = sbuilds(Period, num_time_steps=10, units="days", num_save=10)

store(PERIOD_STANDARD, group="period", name="standard")


def run_simulation(model, grid, period):
    logger.info(f"Starting simulation...")

    logger.info(f"Initializing domains...")
    h_domain, u_domain, v_domain, q_domain = grid.init_domains()

    logger.info(f"Initializing masks...")
    mask = jnp.ones(h_domain.shape)
    mask = mask.at[0].set(0.0)
    mask = mask.at[-1].set(0.0)
    mask = mask.at[:, 0].set(0.0)
    mask = mask.at[:, -1].set(0.0)
    masks = MaskGrid.init_mask(mask, "center")

    logger.info(f"Initializing u domain")

    u0 = init_partition(u_domain)
    h0 = init_h0_jet(h_domain, model, u0=u0)
    v0 = jnp.zeros(v_domain.shape)

    logger.info(f"Initializing state...")
    state = SWMState(
        h=h0, u=u0, v=v0,
        h_domain=h_domain, 
        u_domain=u_domain, 
        v_domain=v_domain,
        q_domain=q_domain,
        masks=masks
    )

    # initializing time stepper
    # dt = 0.125 * min(*h_domain.resolution) / jnp.sqrt(model.gravity * jnp.mean(model.depth))
    dt = 4000

    # integration
    logger.info(f"Starting TimeStepping...")
    sol = dfx.diffeqsolve(
        terms=dfx.ODETerm(model.equation_of_motion),
        solver=dfx.Bosh3(),
        t0=0.0, t1=period.tmax, dt0=dt,
        y0=state,
        saveat=period.saveat,
        args=None,
        stepsize_controller=dfx.PIDController(rtol=1e-4, atol=1e-4),
        max_steps=None,
        progress_meter=dfx.TqdmProgressMeter()
    )

    logger.info(f"Finished Script...!")
    
    # put all components back
    h = sol.ys.h
    u = sol.ys.u
    u = jax.vmap(x_avg_2D)(u)
    v = sol.ys.v
    v = jax.vmap(y_avg_2D)(v)
    
    assert h.shape == u.shape == v.shape
    
    # def create_dataset(psi, q, year, time):

    #     import xarray as xr

    #     ds = xr.Dataset(
    #         {
    #             "u": (("z", "x", "y"), q),
    #             "psi": (("z", "x", "y"), psi),
    #         },
    #         coords={
    #             "x": h_domain.coords_axis[0],
    #             "y": h_domain.coords_axis[1],
    #             # "year": year,
    #             # "diffusivity": diffusivity,
    #             # "method": method,
    #             # "num_pts": num_pts,
    #             # "resolution": resolution,
    #             "time": time,
    #         },
    #     )

    #     return ds


store(
    run_simulation,
    hydra_defaults=[
        "_self_",
        # {"paths": "stream"},
        # {"data": "temperature"},
        # {"preprocess/period": "year"},
        # {"preprocess/coarsen": "halfres"},
        {"model": "wenoz"},
        {"grid": "standard"},
        {"period": "standard"}
        # {"training/params": "default"},
        # {"training/learning_rate": "cosine"},
        # {"training/loss": "mse"},
        # {"training/optimizer": "adam"},
    ],
)




if __name__ == "__main__":
    from hydra_zen import zen

    store.add_to_hydra_store()

    # Generate the CLI For train_fn
    z = zen(run_simulation)

    z.hydra_main(
        config_name="run_simulation",
        config_path=None,
        version_base="1.3",
    )