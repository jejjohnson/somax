import jax.numpy as jnp

from somax.masks import MaskGrid


def init_mask_rect(n: int = 10, init: str = "node"):
    mask = jnp.ones((n, n))
    mask = mask.at[0].set(0.0)
    mask = mask.at[-1].set(0.0)
    mask = mask.at[:, 0].set(0.0)
    mask = mask.at[:, -1].set(0.0)
    masks = MaskGrid.init_mask(mask, init)
    return masks


def init_mask_island(n: int = 10, init: str = "node"):
    mask = jnp.ones((n, n))
    mask = mask.at[0].set(0.0)
    mask = mask.at[-1].set(0.0)
    mask = mask.at[:, 0].set(0.0)
    mask = mask.at[:, -1].set(0.0)
    mask = mask.at[4:-4, 4:-4].set(0.0)
    masks = MaskGrid.init_mask(mask, init)
    return masks


def init_mask_channel(n: int = 10, init: str = "node"):
    mask = jnp.ones((n, 3 * n // 4))
    mask = mask.at[:, 0].set(0.0)
    mask = mask.at[:, -1].set(0.0)
    masks = MaskGrid.init_mask(mask, init)
    return masks


def init_mask_realish(n: int = 10, init: str = "center"):
    mask = jnp.ones((n, n))
    mask = mask.at[1, 0].set(0.0)
    mask = mask.at[n - 1, 2].set(0.0)
    mask = mask.at[0, n - 2].set(0.0)
    mask = mask.at[1, n - 2].set(0.0)
    mask = mask.at[0, n - 1].set(0.0)
    mask = mask.at[1, n - 1].set(0.0)
    mask = mask.at[2, n - 1].set(0.0)
    masks = MaskGrid.init_mask(mask, init)
    return masks
