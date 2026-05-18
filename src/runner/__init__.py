"""Maat web-based experiment runner.

Provides:

- :class:`Orchestrator` — Single-experiment orchestration & thread management.
- :class:`RequestsManager` — Queue-based LLM request routing with rate limiting.
- :func:`create_app` — FastAPI application factory.
"""
