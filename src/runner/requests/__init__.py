"""Requests management layer for rate-limited LLM call routing.

Provides:

- :class:`RequestsManager` — Queue-based request router with RPM/RPD limiting.
- :class:`QueuedChatModel` — Drop-in LangChain model that routes through the manager.
"""
