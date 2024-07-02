import typing as tp

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from somax._src.interp.average import avg_pool


class NodeMask(eqx.Module):
    values: Array = eqx.field(static=True)
    not_values: Array = eqx.field(static=True)
    distbound1: Array = eqx.field(static=True)
    irrbound_xids: Array = eqx.field(static=True)
    irrbound_yids: Array = eqx.field(static=True)

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, item):
        values = self.values[item]
        not_values = self.not_values[item]
        distbound1 = self.distbound1[item]
        irrbound_xids = self.irrbound_xids[item]
        irrbound_yids = self.irrbound_yids[item]

        return NodeMask(
            values=values,
            not_values=not_values,
            distbound1=distbound1,
            irrbound_xids=irrbound_xids,
            irrbound_yids=irrbound_yids,
        )


class FaceMask(eqx.Module):
    values: Array = eqx.field(static=True)
    not_values: Array = eqx.field(static=True)
    distbound1: Array = eqx.field(static=True)
    distbound2: Array = eqx.field(static=True)
    distbound2plus: Array = eqx.field(static=True)
    distbound3plus: Array = eqx.field(static=True)

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, item):
        values = self.values[item]
        not_values = self.not_values[item]
        distbound1 = self.distbound1[item]
        distbound2 = self.distbound2[item]
        distbound2plus = self.distbound2plus[item]
        distbound3plus = self.distbound3plus[item]

        return FaceMask(
            values=values,
            not_values=not_values,
            distbound2=distbound2,
            distbound1=distbound1,
            distbound2plus=distbound2plus,
            distbound3plus=distbound3plus,
        )


class CenterMask(eqx.Module):
    values: Array = eqx.field(static=True)
    not_values: Array = eqx.field(static=True)
    values_interior: Array = eqx.field(static=True)
    distbound1: Array = eqx.field(static=True)

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, item):
        values = self.values[item]
        not_values = self.not_values[item]
        distbound1 = self.distbound1[item]
        values_interior = self.values_interior[item]

        return CenterMask(
            values=values,
            not_values=not_values,
            values_interior=values_interior,
            distbound1=distbound1,
        )


class MaskGrid(tp.NamedTuple):
    center: CenterMask
    face_u: FaceMask
    face_v: FaceMask
    node: NodeMask

    @classmethod
    def init_mask(cls, mask: Array, location: str = "node"):
        mtype = mask.dtype

        if location == "center":
            center, u, v, node = init_masks_from_center(mask)
        elif location == "node":
            center, u, v, node = init_masks_from_node(mask)
        else:
            raise ValueError(f"Unrecognized location: {location}")
        not_q = jnp.logical_not(center.astype(bool))
        not_u = jnp.logical_not(u.astype(bool))
        not_v = jnp.logical_not(v.astype(bool))
        not_psi = jnp.logical_not(node.astype(bool))

        # VARIABLE
        psi_irrbound_xids = jnp.logical_and(
            not_psi[1:-1, 1:-1],
            avg_pool(node, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)) > 1 / 18,
        )
        psi_irrbound_xids = jnp.where(psi_irrbound_xids)

        psi_distbound1 = jnp.logical_and(
            avg_pool(node.astype(mtype), (3, 3), stride=(1, 1), padding=(1, 1))
            < 17 / 18,
            node,
        )

        # TRACER
        q_distbound1 = jnp.logical_and(
            avg_pool(center, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            < 17 / 18,
            center,
        )
        q_interior = jnp.logical_and(jnp.logical_not(psi_distbound1), node)

        u_distbound1 = jnp.logical_and(
            avg_pool(u, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)) < 5 / 6,
            u,
        )
        v_distbound1 = jnp.logical_and(
            avg_pool(v, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)) < 5 / 6,
            v,
        )

        u_distbound2plus = jnp.logical_and(jnp.logical_not(u_distbound1), u)
        v_distbound2plus = jnp.logical_and(jnp.logical_not(v_distbound1), v)

        u_distbound2 = jnp.logical_and(
            avg_pool(u, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0)) < 9 / 10,
            u_distbound2plus,
        )
        v_distbound2 = jnp.logical_and(
            avg_pool(v, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)) < 9 / 10,
            v_distbound2plus,
        )

        u_distbound3plus = jnp.logical_and(
            jnp.logical_not(u_distbound2), u_distbound2plus
        )
        v_distbound3plus = jnp.logical_and(
            jnp.logical_not(v_distbound2), v_distbound2plus
        )

        # create variable mask
        node = NodeMask(
            values=node.astype(mtype),
            not_values=not_psi.astype(mtype),
            distbound1=psi_distbound1.astype(mtype),
            irrbound_xids=psi_irrbound_xids[0].astype(jnp.int32),
            irrbound_yids=psi_irrbound_xids[1].astype(jnp.int32),
        )

        # create tracer mask
        center = CenterMask(
            values=center.astype(mtype),
            not_values=not_q,
            distbound1=q_distbound1.astype(mtype),
            values_interior=q_interior.astype(mtype),
        )

        # create u velocity mask
        u = FaceMask(
            values=u.astype(mtype),
            not_values=not_u.astype(mtype),
            distbound1=u_distbound1.astype(mtype),
            distbound2=u_distbound2.astype(mtype),
            distbound2plus=u_distbound2plus.astype(mtype),
            distbound3plus=u_distbound3plus.astype(mtype),
        )

        # create v velocity mask
        v = FaceMask(
            values=v.astype(mtype),
            not_values=not_v,
            distbound1=v_distbound1.astype(mtype),
            distbound2plus=v_distbound2plus.astype(mtype),
            distbound2=v_distbound2.astype(mtype),
            distbound3plus=v_distbound3plus.astype(mtype),
        )

        return cls(node=node, face_u=u, face_v=v, center=center)


def init_masks_from_center(mask: Array):
    center = jnp.copy(mask)

    node = avg_pool(center, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)) > 7 / 8
    face_u = avg_pool(center, kernel_size=(2, 1), stride=(1, 1), padding=(1, 0)) > 3 / 4
    face_v = avg_pool(center, kernel_size=(1, 2), stride=(1, 1), padding=(0, 1)) > 3 / 4

    return center, face_u, face_v, node


def init_masks_from_node(mask: Array):
    node = jnp.copy(mask)
    center = avg_pool(node, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)) > 0.0
    face_u = avg_pool(center, kernel_size=(2, 1), stride=(1, 1), padding=(1, 0)) > 3 / 4
    face_v = avg_pool(center, kernel_size=(1, 2), stride=(1, 1), padding=(0, 1)) > 3 / 4

    return center, face_u, face_v, node
