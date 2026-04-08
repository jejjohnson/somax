"""Simulation checkpointing via orbax-checkpoint."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
from jaxtyping import PyTree


class SimulationCheckpointer(eqx.Module):
    """Manages periodic checkpointing during long simulations.

    Uses orbax-checkpoint for JAX-native pytree serialization.
    Compatible with DVC for versioning checkpoint files.

    Attributes:
        checkpoint_dir: Directory for checkpoint files.
        checkpoint_interval: Number of steps between saves.
    """

    checkpoint_dir: str = eqx.field(static=True)
    checkpoint_interval: int = eqx.field(static=True)

    def save(self, step: int, state: PyTree, model: eqx.Module) -> Path:
        """Save a checkpoint to disk.

        Args:
            step: Current simulation step.
            state: Model state pytree.
            model: Model instance (only ``model.params`` is saved).

        Returns:
            Path to the saved checkpoint directory.
        """
        import orbax.checkpoint as ocp

        path = Path(self.checkpoint_dir) / f"step_{step:08d}"
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(
            path,
            {"state": state, "params": model.params, "step": step},
        )
        return path

    def restore(
        self,
        step: int,
        target_state: PyTree,
        target_model: eqx.Module,
    ) -> tuple[PyTree, PyTree, int]:
        """Restore a checkpoint from disk.

        Args:
            step: Step number to restore.
            target_state: A state pytree with the expected structure
                (used as the restore target).
            target_model: A model instance whose ``params`` structure
                matches the saved checkpoint.

        Returns:
            Tuple of (restored_state, restored_params, restored_step).
        """
        import orbax.checkpoint as ocp

        path = Path(self.checkpoint_dir) / f"step_{step:08d}"
        checkpointer = ocp.StandardCheckpointer()
        ckpt = checkpointer.restore(
            path,
            target={
                "state": target_state,
                "params": target_model.params,
                "step": 0,
            },
        )
        return ckpt["state"], ckpt["params"], ckpt["step"]

    def should_save(self, step: int) -> bool:
        """Check whether a checkpoint should be saved at this step."""
        return step > 0 and step % self.checkpoint_interval == 0

    def latest_step(self) -> int | None:
        """Find the latest checkpoint step, or None if no checkpoints exist."""
        ckpt_dir = Path(self.checkpoint_dir)
        if not ckpt_dir.exists():
            return None
        steps = []
        for p in ckpt_dir.iterdir():
            if p.is_dir() and p.name.startswith("step_"):
                try:
                    steps.append(int(p.name.split("_")[1]))
                except ValueError:
                    continue
        return max(steps) if steps else None
