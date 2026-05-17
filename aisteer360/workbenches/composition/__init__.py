"""Composition Workbench.

Drag-and-drop builder for `SteeringPipeline` configurations with live inference preview. Where
the vector-calibration workbench's unit of work is a one-shot run, the composition workbench's
unit of work is a long-lived *session* — an agent process that holds a model in VRAM and serves
inference requests on demand.

The server stays inert; all compute (model loading, steering, generation) lives in the agent. See
`composition.agent.runner.SessionRunner`.
"""
