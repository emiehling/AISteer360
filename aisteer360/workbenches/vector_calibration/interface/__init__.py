"""FastAPI interface for the vector calibration workbench.

Exposes a small dashboard server that bridges the HTML UI to the
`VectorCalibrationWorkbench`.  The server holds models in GPU memory
across requests, runs the three-stage pipeline in a background task, streams
progress over WebSocket, and serves results as JSON.
"""
from .app import create_app

__all__ = ["create_app"]
