"""Runner API layer (FastAPI + WebSocket).

This package contains the *control-plane* of the runner:

- REST endpoints used by the dashboard to control runs.
- WebSocket broadcasting for real-time updates.

The API should stay thin; orchestration logic lives in `src.runner.core`.
"""
