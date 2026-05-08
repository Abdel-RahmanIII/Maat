"""Runner workers (data-plane execution).

Workers run inside a `ThreadPoolExecutor` managed by the orchestrator.
Each worker processes exactly one unit of work:

- one puzzle (Experiment 1)
- one full game (Experiments 2 & 3)

Workers emit progress events via callbacks but do not know about WebSockets.
"""
