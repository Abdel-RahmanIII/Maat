"""Runner core (orchestration + progress).

This package contains the control-plane logic that is independent from HTTP:

- `Orchestrator` which schedules runners and manages lifecycle state.
- Disk-based progress tracking surfaced to the dashboard.
"""
