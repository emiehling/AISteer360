"""Shared infrastructure for AISteer360 workbenches.

This package holds the workbench-agnostic layer (auth, database, agent dispatch, providers,
WebSocket relay, model catalog, secrets vault, common CSS). Individual workbenches under
`aisteer360.workbenches.<name>/` consume these primitives via the shared `create_workbench_app`
factory.
"""
