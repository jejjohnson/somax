"""Authored pipeline configs for somax-sim.

Each module in this directory defines one pipeline config as a plain
Python dict matching the :class:`somax._src.cli.spec.RunSpec` shape.
``scripts/build_configs.py`` imports these modules and materializes
them to ``configs/simulation/*.yaml`` for cyclopts and DVC to consume.

The plain-dict shape is the v0.1 decision (Q-J in the design doc); the
forward path migrates to ``hydra_zen.builds(...)`` per-file when type
safety becomes a real need.
"""
