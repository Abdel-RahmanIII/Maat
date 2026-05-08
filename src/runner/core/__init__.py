"""Runner core (orchestration + progress).

This package contains the control-plane logic that is independent from HTTP:

- `Orchestrator` which schedules workers and manages lifecycle state.
- Thread-safe progress trackers surfaced to the dashboard.
"""
