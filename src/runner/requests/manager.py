"""Central requests manager for routing LLM calls across multiple API keys."""

import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Tracks rate limits and usage for a single API key."""

    api_key: str
    rpm_limit: int = 15
    rpd_limit: int = 1500

    _requests_this_minute: int = field(default=0, init=False)
    _requests_today: int = field(default=0, init=False)
    _minute_start: float = field(default_factory=time.time, init=False)
    _day_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc).date(), init=False)
    _exhausted_until: float = field(default=0.0, init=False)
    _exhausted_for_day: bool = field(default=False, init=False)

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def can_accept(self) -> bool:
        """Check if this API key can accept a new request right now."""
        with self._lock:
            if self._exhausted_for_day:
                return False

            now = time.time()
            if now < self._exhausted_until:
                return False

            # Check day rollover
            today = datetime.now(timezone.utc).date()
            if today > self._day_start:
                self._day_start = today
                self._requests_today = 0
                self._exhausted_for_day = False

            # Check minute rollover
            if now - self._minute_start >= 60.0:
                self._minute_start = now
                self._requests_this_minute = 0

            if self._requests_today >= self.rpd_limit:
                self._exhausted_for_day = True
                return False

            if self._requests_this_minute >= self.rpm_limit:
                self._exhausted_until = self._minute_start + 60.0
                return False

            return True

    def record_usage(self) -> None:
        """Record that a request was made."""
        with self._lock:
            self._requests_this_minute += 1
            self._requests_today += 1

    def mark_exhausted_minute(self) -> None:
        """Mark this key as exhausted for the current minute."""
        with self._lock:
            self._exhausted_until = time.time() + 60.0

    def mark_exhausted_day(self) -> None:
        """Mark this key as exhausted for the current day."""
        with self._lock:
            self._exhausted_for_day = True


@dataclass
class QueueItem:
    """A single LLM request item in the queue."""

    messages: list[BaseMessage]
    model_kwargs: dict[str, Any]
    invoke_kwargs: dict[str, Any]
    future: Future
    retries: int = 0
    max_retries: int = 3


class RequestsManager:
    """Manages a queue of LLM requests and routes them across multiple API keys.

    Parameters
    ----------
    api_keys:
        List of Google API keys to distribute requests across.
    rpm_limit:
        Requests per minute limit shared by all API keys.
    rpd_limit:
        Requests per day limit shared by all API keys.
    on_global_rpd_limit:
        Optional callback invoked when ALL API keys have exhausted their
        daily quota. Typically wired to the Orchestrator's pause method.
    """

    def __init__(
        self,
        api_keys: list[str],
        *,
        rpm_limit: int = 15,
        rpd_limit: int = 1500,
        on_global_rpd_limit: Optional[Callable[[], None]] = None,
    ):
        if not api_keys:
            raise ValueError("At least one API key must be provided")

        self._api_configs = [
            APIConfig(api_key=k, rpm_limit=rpm_limit, rpd_limit=rpd_limit)
            for k in api_keys
        ]
        self._queue: queue.Queue[QueueItem] = queue.Queue()
        self._on_global_rpd_limit = on_global_rpd_limit

        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)

        # Cached model instances per API key (avoids re-instantiation overhead)
        self._model_cache: dict[str, dict[str, ChatGoogleGenerativeAI]] = {}
        self._cache_lock = threading.Lock()

    def start(self) -> None:
        """Start the background worker thread."""
        self._stop_event.clear()
        if not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

    def stop(self) -> None:
        """Stop the background worker thread gracefully."""
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    def submit(self, messages: list[BaseMessage], model_kwargs: dict[str, Any], invoke_kwargs: dict[str, Any] = None) -> Future:
        """Submit a request to the queue. Returns a Future."""
        future = Future()
        self._queue.put(QueueItem(
            messages=messages, 
            model_kwargs=model_kwargs, 
            invoke_kwargs=invoke_kwargs or {}, 
            future=future
        ))
        return future

    def _get_or_create_model(
        self, api_key: str, model_name: str, model_kwargs: dict[str, Any]
    ) -> ChatGoogleGenerativeAI:
        """Return a cached model for this API key + model name, or create one.

        Models are cached per (api_key, model_name) to avoid constructor
        overhead and to allow HTTP connection reuse across requests.
        """
        cache_key = f"{api_key}:{model_name}"
        with self._cache_lock:
            if cache_key in self._model_cache:
                return self._model_cache[cache_key]

        # Build outside lock to avoid blocking other threads
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            **model_kwargs,
        )

        with self._cache_lock:
            # Double-check; another thread may have created it
            if cache_key not in self._model_cache:
                self._model_cache[cache_key] = llm
            return self._model_cache[cache_key]

    def _worker_loop(self) -> None:
        """Continuously pulls from the queue and routes requests to available APIs."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            selected_config = None
            all_rpd_exhausted = True

            while selected_config is None and not self._stop_event.is_set():
                all_rpd_exhausted = True
                for config in self._api_configs:
                    if not config._exhausted_for_day:
                        all_rpd_exhausted = False
                    if config.can_accept():
                        selected_config = config
                        break

                if all_rpd_exhausted:
                    logger.error("All API keys have exhausted their Daily Quota (RPD).")
                    if self._on_global_rpd_limit:
                        try:
                            self._on_global_rpd_limit()
                        except Exception as e:
                            logger.error(f"Error calling on_global_rpd_limit: {e}")
                    
                    if not item.future.done():
                        item.future.set_exception(Exception("All API keys reached RPD limit."))
                    break

                if selected_config is None:
                    # All keys hit RPM but not RPD. Sleep a bit and check again.
                    time.sleep(1.0)

            if all_rpd_exhausted or self._stop_event.is_set():
                if not item.future.done() and self._stop_event.is_set():
                    item.future.set_exception(Exception("RequestsManager stopped"))
                continue

            # We have a config, record usage and invoke
            selected_config.record_usage()

            # Prepare kwargs for model construction (strip model_name/model keys)
            kwargs_copy = item.model_kwargs.copy()
            model_name = kwargs_copy.pop("model_name", "gemma-4-31b-it")
            if "model" in kwargs_copy:
                model_name = kwargs_copy.pop("model")

            try:
                llm = self._get_or_create_model(
                    api_key=selected_config.api_key,
                    model_name=model_name,
                    model_kwargs=kwargs_copy,
                )

                response = llm.invoke(item.messages, **item.invoke_kwargs)
                item.future.set_result(response)

            except Exception as e:
                err_name = type(e).__name__
                err_msg = str(e).lower()

                if "resourceexhausted" in err_name.lower() or "429" in err_msg:
                    if "quota" in err_msg and "day" in err_msg:
                        logger.warning("API key hit daily quota.")
                        selected_config.mark_exhausted_day()
                    else:
                        logger.warning("API key hit RPM limit.")
                        selected_config.mark_exhausted_minute()
                    
                    # Re-queue to be handled by another API key
                    self._queue.put(item)

                elif any(err in err_name.lower() for err in ["internalservererror", "serviceunavailable"]) or "500" in err_msg or "503" in err_msg:
                    logger.warning(f"Server error during LLM call: {e}")
                    item.retries += 1
                    if item.retries <= item.max_retries:
                        # Re-queue the item without sleeping
                        self._queue.put(item)
                    else:
                        logger.error(f"Max retries reached for request: {e}")
                        item.future.set_exception(e)

                elif "invalidargument" in err_name.lower() or "400" in err_msg:
                    logger.error(f"Bad request during LLM call: {e}")
                    item.future.set_exception(e)

                else:
                    logger.error(f"Unexpected error in LLM call: {e}")
                    item.future.set_exception(e)


_global_manager: Optional[RequestsManager] = None

def set_global_manager(manager: Optional[RequestsManager]) -> None:
    """Set the global RequestsManager instance."""
    global _global_manager
    _global_manager = manager

def get_global_manager() -> Optional[RequestsManager]:
    """Get the global RequestsManager instance."""
    return _global_manager
