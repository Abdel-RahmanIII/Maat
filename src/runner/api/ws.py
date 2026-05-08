"""WebSocket connection management for the runner dashboard.

The runner is multi-threaded (workers run in a `ThreadPoolExecutor`).
Those worker threads need to broadcast JSON events to all connected
WebSocket clients.

FastAPI WebSockets are asyncio-based; worker threads are not.

This module provides a small bridge:

- Maintain a thread-safe list of active `WebSocket` connections.
- Allow synchronous broadcast (`broadcast_sync`) from any thread by
  scheduling coroutine sends onto the app event loop.

The event payload is an untyped `dict[str, Any]` on purpose: the UI is
the contract, and the runner emits a flexible stream of event objects.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    """Manages active WebSocket connections and broadcasting."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = threading.Lock()
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        """Set the asyncio event loop used to schedule send operations."""

        self._event_loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        with self._lock:
            self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)

    def broadcast_sync(self, data: dict[str, Any]) -> None:
        """Broadcast JSON to all clients from a non-async context.

        If the event loop is not ready (e.g. during startup/shutdown),
        this becomes a no-op.
        """

        loop = self._event_loop
        if loop is None:
            return

        with self._lock:
            dead: list[WebSocket] = []
            for ws in self._connections:
                try:
                    asyncio.run_coroutine_threadsafe(ws.send_json(data), loop)
                except Exception:
                    dead.append(ws)

            for ws in dead:
                try:
                    self._connections.remove(ws)
                except ValueError:
                    pass
