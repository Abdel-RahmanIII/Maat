"""Central requests manager for routing LLM calls across multiple API keys.

The manager owns the per-key models, queueing, quota tracking, dispatch
selection, and retry/error handling. The code is intentionally split into
small helpers so the request flow can be read top-to-bottom.

Pause / Resume
--------------
When all API keys exhaust their daily quota (RPD), the manager fires
``on_global_rpd_limit`` so the orchestrator can pause the entire run.
All pending queue items are drained and their futures set with a
``RequestsPausedError`` so runners know to stop cleanly.

Resume is manual: the orchestrator calls ``resume()`` after the user
triggers it from the dashboard.

Error Handling
--------------
Transient server errors (503 "model overloaded", 500 internal) are
retried with **exponential backoff** up to ``max_retries`` attempts.
Rate-limit errors (429 / ``ResourceExhausted``) rotate keys and requeue.
Permanent failures (400 bad request, unknown) immediately fail the future.
"""

from __future__ import annotations

import enum
import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Optional, Sequence

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import ModelConfig

logger = logging.getLogger(__name__)


_REQUEST_SEND_DELAY_SECONDS = 5


# ── Custom exceptions ────────────────────────────────────────────────────


class RequestsPausedError(Exception):
    """Raised on pending futures when the manager is paused."""


class RequestTerminatedError(Exception):
    """Raised when a request is permanently killed (max retries exceeded).

    Unlike ``RequestsPausedError`` (system-wide pause), this indicates a
    single request could not be completed after all retry attempts.
    """


# ── Error classification ────────────────────────────────────────────────


class ErrorCategory(enum.Enum):
    """Structured classification of LLM API errors."""

    RATE_LIMIT_RPM = "rate_limit_rpm"
    RATE_LIMIT_RPD = "rate_limit_rpd"
    SERVER_503 = "server_503"
    SERVER_500 = "server_500"
    BAD_REQUEST = "bad_request"
    UNKNOWN = "unknown"


def _classify_error(error: Exception) -> ErrorCategory:
    """Map an exception to a structured ``ErrorCategory``.

    Inspects both the exception type name and the string representation
    to handle the variety of error formats from Google's API.
    """

    err_name = type(error).__name__.lower()
    err_msg = str(error).lower()

    # ── Timeouts ──
    # Treat timeouts as permanent failures (same behavior as 400 bad request).
    # This is intentional: callers should adjust prompts / load / routing rather
    # than retrying indefinitely.
    if (
        "deadlineexceeded" in err_name
        or "timeout" in err_name
        or "timed out" in err_msg
        or "timeout" in err_msg
        or ("deadline" in err_msg and "exceed" in err_msg)
    ):
        return ErrorCategory.BAD_REQUEST

    # ── Rate limits ──
    if "resourceexhausted" in err_name or "429" in err_msg:
        if "quota" in err_msg and "day" in err_msg:
            return ErrorCategory.RATE_LIMIT_RPD
        return ErrorCategory.RATE_LIMIT_RPM

    # ── Server errors ──
    if "503" in err_msg or "serviceunavailable" in err_name or "overloaded" in err_msg:
        return ErrorCategory.SERVER_503

    if "500" in err_msg or "internalservererror" in err_name:
        return ErrorCategory.SERVER_500

    # ── Client errors ──
    if "invalidargument" in err_name or "400" in err_msg:
        return ErrorCategory.BAD_REQUEST

    return ErrorCategory.UNKNOWN


# ── Per-key usage tracker ────────────────────────────────────────────────


@dataclass
class APIConfig:
    """Tracks rate limits and usage for a single API key."""

    api_key: str
    rpm_limit: int = 15
    rpd_limit: int = 1500

    _requests_this_minute: int = field(default=0, init=False)
    _requests_today: int = field(default=0, init=False)
    _minute_start: float = field(default_factory=time.time, init=False)
    _day_start: date = field(default_factory=lambda: datetime.now(timezone.utc).date(), init=False)
    _exhausted_until: float = field(default=0.0, init=False)
    _exhausted_for_day: bool = field(default=False, init=False)

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def can_accept(self) -> bool:
        """Check whether the key can accept a request right now."""

        with self._lock:
            if self._exhausted_for_day:
                return False

            now = time.time()
            if now < self._exhausted_until:
                return False

            # Reset daily counters after UTC day rollover.
            today = datetime.now(timezone.utc).date()
            if today > self._day_start:
                self._day_start = today
                self._requests_today = 0
                self._exhausted_for_day = False

            # Reset minute counters after 60 seconds.
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
        """Record a successful request against this key."""

        with self._lock:
            self._requests_this_minute += 1
            self._requests_today += 1

    def usage_snapshot(self) -> tuple[int, int]:
        """Return the current RPM and RPD counters."""

        with self._lock:
            return self._requests_this_minute, self._requests_today

    def mark_exhausted_minute(self) -> None:
        """Mark this key as exhausted for the current minute."""

        with self._lock:
            self._exhausted_until = time.time() + 60.0

    def mark_exhausted_day(self) -> None:
        """Mark this key as exhausted for the current day."""

        with self._lock:
            self._exhausted_for_day = True


# ── Queue item ───────────────────────────────────────────────────────────


@dataclass
class QueueItem:
    """A single queued LLM request."""

    messages: list[BaseMessage]
    tools: Sequence[Any] | None
    invoke_kwargs: dict[str, Any]
    future: Future
    retries: int = 0
    max_retries: int = 5


# ── RequestsManager ─────────────────────────────────────────────────────


class RequestsManager:
    """Queue-based request router with per-key models and quota tracking.

    All LLM requests flow through a single queue.  A background worker
    thread pulls items and dispatches them to the least-loaded API key.

    Pause semantics
    ~~~~~~~~~~~~~~~
    When the manager is paused (RPD exhaustion or external signal):

    1. All items currently in the queue have their futures set with
       ``RequestsPausedError``.
    2. New ``submit()`` calls immediately return a future that is
       already resolved with ``RequestsPausedError``.
    3. The worker loop blocks on ``_pause_event.wait()``.

    Resume clears the pause and lets the worker loop continue.  The
    orchestrator is responsible for re-submitting work items.

    Error handling
    ~~~~~~~~~~~~~~
    - **503 / 500**: Exponential backoff (``backoff_base ** retries``,
      capped at ``backoff_max``), up to ``max_retries`` attempts.
    - **429 / RPM**: Key rotated, request requeued instantly.
    - **429 / RPD**: Key marked exhausted for the day; if all keys
      exhausted, ``on_global_rpd_limit`` fires.
    - **400 / Unknown**: Future fails immediately.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        rpm_limit: int = 15,
        rpd_limit: int = 1500,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
        request_send_delay: float = _REQUEST_SEND_DELAY_SECONDS,
        request_timeout: float = 300.0,
        on_global_rpd_limit: Optional[Callable[[], None]] = None,
    ) -> None:
        self.model_config = model_config
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.request_send_delay = request_send_delay
        self.request_timeout = request_timeout  # 5 minutes default
        api_keys = model_config.api_keys

        if not api_keys:
            raise ValueError("At least one API key must be provided")

        # Build one model and one usage tracker per API key.
        self._api_configs = [
            APIConfig(api_key=key, rpm_limit=rpm_limit, rpd_limit=rpd_limit)
            for key in api_keys
        ]
        self._models = [self._build_model(api_key=key) for key in api_keys]
        self._queue: queue.Queue[QueueItem] = queue.Queue()
        self._on_global_rpd_limit = on_global_rpd_limit

        logger.info(
            "build RequestsManager with %d API keys | model=%s | request_timeout=%.0fs",
            len(self._api_configs),
            self.model_config.model_name,
            self.request_timeout,
        )

        # Lifecycle
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Initially NOT paused
        self._paused = False
        self._paused_lock = threading.Lock()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)

    def _build_model(self, api_key: str) -> ChatGoogleGenerativeAI:
        """Construct a model instance for one API key."""

        return ChatGoogleGenerativeAI(
            model=self.model_config.model_name,
            temperature=self.model_config.temperature,
            max_output_tokens=self.model_config.max_output_tokens,
            google_api_key=api_key,
            request_timeout=self.request_timeout,
            max_retries=0,
            thinking_level="minimal",
        )

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background worker thread."""

        self._stop_event.clear()
        self._paused = False
        self._pause_event.set()
        if not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

    def stop(self) -> None:
        """Stop the background worker thread gracefully."""

        self._stop_event.set()
        self._pause_event.set()  # Unblock if paused
        self._drain_queue()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

    def pause(self) -> None:
        """Pause the manager: drain the queue and block the worker loop."""

        with self._paused_lock:
            if self._paused:
                return
            self._paused = True

        self._pause_event.clear()
        self._drain_queue()
        logger.info("[RequestsManager] ⏸  Paused — queue drained.")

    def resume(self) -> None:
        """Resume the manager after a pause."""

        with self._paused_lock:
            if not self._paused:
                return
            self._paused = False

        self._pause_event.set()
        logger.info("[RequestsManager] ▶  Resumed.")

    @property
    def is_paused(self) -> bool:
        """Whether the manager is currently paused."""

        with self._paused_lock:
            return self._paused

    # ── Submit ───────────────────────────────────────────────────────

    def submit(
        self,
        messages: Sequence[BaseMessage],
        *,
        tools: Sequence[Any] | None = None,
        invoke_kwargs: dict[str, Any] | None = None,
    ) -> Future:
        """Enqueue a request and return a Future for the eventual response.

        If the manager is paused, returns a future that is immediately
        resolved with ``RequestsPausedError``.
        """

        future: Future = Future()

        with self._paused_lock:
            if self._paused:
                future.set_exception(RequestsPausedError("RequestsManager is paused"))
                return future

        self._queue.put(
            QueueItem(
                messages=list(messages),
                tools=tools,
                invoke_kwargs=invoke_kwargs or {},
                future=future,
                max_retries=self.max_retries,
            )
        )
        return future

    # ── Status ───────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Return current rate-limit / usage status for the dashboard.

        Replaces the old ``RateLimiter.get_status()`` contract.
        """

        keys_status: list[dict[str, Any]] = []
        total_rpm = 0
        total_rpd = 0

        for i, cfg in enumerate(self._api_configs):
            rpm, rpd = cfg.usage_snapshot()
            total_rpm += rpm
            total_rpd += rpd
            keys_status.append({
                "key_index": i,
                "rpm_current": rpm,
                "rpm_limit": cfg.rpm_limit,
                "rpd_current": rpd,
                "rpd_limit": cfg.rpd_limit,
                "exhausted_for_day": cfg._exhausted_for_day,
            })

        return {
            "paused": self.is_paused,
            "queue_size": self._queue.qsize(),
            "num_keys": len(self._api_configs),
            "total_rpm": total_rpm,
            "total_rpd": total_rpd,
            "keys": keys_status,
        }

    # ── Key selection ────────────────────────────────────────────────

    def _select_config_index(self) -> int | None:
        """Pick the least-loaded key that can accept a request.

        Returns:
            The selected index, ``None`` if all keys are temporarily rate
            limited, or ``-1`` if every key is exhausted for the day.
        """

        candidates: list[tuple[int, int, int]] = []
        all_rpd_exhausted = True

        for index, config in enumerate(self._api_configs):
            if not config._exhausted_for_day:
                all_rpd_exhausted = False
            if config.can_accept():
                rpm_count, rpd_count = config.usage_snapshot()
                candidates.append((index, rpm_count, rpd_count))

        if candidates:
            candidates.sort(key=lambda item: (item[1], item[2], item[0]))
            return candidates[0][0]

        if all_rpd_exhausted:
            return -1

        return None

    def _wait_for_available_key(self) -> int | None:
        """Wait until a key is available or the manager is stopped."""

        while not self._stop_event.is_set():
            # Check if paused
            if not self._pause_event.is_set():
                return None

            selected_index = self._select_config_index()
            if selected_index is not None:
                return selected_index
            time.sleep(1.0)

        return None

    # ── Logging ──────────────────────────────────────────────────────

    def _log_request(self, selected_index: int, messages: Sequence[BaseMessage]) -> None:
        """Emit a compact log line for the queued request."""

        num_messages = len(messages)
        total_chars = sum(len(str(getattr(msg, "content", ""))) for msg in messages)
        logger.info(
            "[RequestsManager] ── REQUEST ── %s | key=%d | msgs=%d | chars=%d",
            self.model_config.model_name,
            selected_index,
            num_messages,
            total_chars,
        )

    def _log_success(self, response: Any, elapsed_ms: float) -> None:
        """Log a successful response in a single place."""

        usage = getattr(response, "usage_metadata", {}) or {}
        in_tokens = int(usage.get("input_tokens", 0) or 0)
        out_tokens = int(usage.get("output_tokens", 0) or 0)
        logger.info(
            "[RequestsManager] ── SUCCESS ── %.0fms | in=%d out=%d tokens",
            elapsed_ms,
            in_tokens,
            out_tokens,
        )

    # ── Error handling ───────────────────────────────────────────────

    def _set_future_exception(self, item: QueueItem, error: Exception) -> None:
        """Set an exception on the future only if it is still pending."""

        if not item.future.done():
            item.future.set_exception(error)

    def _compute_backoff(self, retries: int) -> float:
        """Compute the backoff delay for the current retry attempt."""

        return min(self.backoff_base ** retries, self.backoff_max)

    def _handle_rate_limit(
        self,
        item: QueueItem,
        selected_config: APIConfig,
        category: ErrorCategory,
    ) -> None:
        """Mark the exhausted key and requeue the item for another key."""

        if category == ErrorCategory.RATE_LIMIT_RPD:
            logger.warning(
                "[RequestsManager] ── 429 RPD ── API key hit daily quota → marked exhausted"
            )
            selected_config.mark_exhausted_day()
        else:
            logger.warning(
                "[RequestsManager] ── 429 RPM ── API key hit minute limit → rotating key"
            )
            selected_config.mark_exhausted_minute()

        # Requeue for another key
        self._queue.put(item)

    def _handle_server_error(
        self,
        item: QueueItem,
        category: ErrorCategory,
        error: Exception,
    ) -> None:
        """Retry server errors with exponential backoff.

        The backoff sleep happens on the *caller's* thread (a per-request
        dispatch thread), so it never blocks the main dispatcher loop or
        other concurrent requests.
        """

        item.retries += 1
        code = "503" if category == ErrorCategory.SERVER_503 else "500"

        if item.retries <= item.max_retries:
            delay = self._compute_backoff(item.retries)
            logger.warning(
                "[RequestsManager] ── %s RETRY %d/%d ── backing off %.1fs | %s",
                code,
                item.retries,
                item.max_retries,
                delay,
                _short_error(error),
            )
            # Sleep on THIS thread only — other requests are unaffected.
            self._backoff_sleep(delay)
            if not self._stop_event.is_set():
                # Re-submit so the dispatcher picks it up with a fresh key selection.
                self._queue.put(item)
            else:
                self._set_future_exception(item, error)
        else:
            terminated_error = RequestTerminatedError(
                f"Request failed after {item.max_retries} retries — last error: {error}"
            )
            logger.error(
                "[RequestsManager] ── %s TERMINATED ── failed after %d retries | %s",
                code,
                item.max_retries,
                _short_error(error),
            )
            self._set_future_exception(item, terminated_error)

    def _handle_bad_request(self, item: QueueItem, error: Exception) -> None:
        """Fail immediately — the request itself is invalid."""

        logger.error(
            "[RequestsManager] ── 400 BAD REQUEST ── %s",
            _short_error(error),
        )
        self._set_future_exception(item, error)

    def _handle_unknown_error(self, item: QueueItem, error: Exception) -> None:
        """Fail immediately — unrecognized error type."""

        logger.error(
            "[RequestsManager] ── UNKNOWN ERROR ── %s: %s",
            type(error).__name__,
            _short_error(error),
        )
        self._set_future_exception(item, error)

    def _handle_worker_exception(
        self,
        item: QueueItem,
        selected_config: APIConfig,
        error: Exception,
    ) -> None:
        """Classify the error and dispatch to the appropriate handler."""

        category = _classify_error(error)

        if category in (ErrorCategory.RATE_LIMIT_RPM, ErrorCategory.RATE_LIMIT_RPD):
            self._handle_rate_limit(item, selected_config, category)
        elif category in (ErrorCategory.SERVER_503, ErrorCategory.SERVER_500):
            self._handle_server_error(item, category, error)
        elif category == ErrorCategory.BAD_REQUEST:
            self._handle_bad_request(item, error)
        else:
            self._handle_unknown_error(item, error)

    def _backoff_sleep(self, delay: float) -> None:
        """Sleep for *delay* seconds, checking the stop event every second."""

        end = time.time() + delay
        while not self._stop_event.is_set():
            remaining = end - time.time()
            if remaining <= 0:
                break
            time.sleep(min(1.0, remaining))

    def _manager_send_sleep(self) -> None:
        """Pause briefly on the manager thread after each request handoff."""

        if self.request_send_delay > 0:
            self._backoff_sleep(self.request_send_delay)

    # ── Queue management ─────────────────────────────────────────────

    def _drain_queue(self) -> None:
        """Drain all pending items from the queue, failing their futures."""

        drained = 0
        while True:
            try:
                item = self._queue.get_nowait()
                self._set_future_exception(item, RequestsPausedError("RequestsManager paused — request cancelled"))
                drained += 1
            except queue.Empty:
                break

        if drained:
            logger.info("[RequestsManager] Drained %d pending requests from queue.", drained)

    # ── Dispatch thread ──────────────────────────────────────────────

    def _dispatch_item(self, item: QueueItem, selected_index: int) -> None:
        """Execute one request on a dedicated thread.

        Running each request on its own thread means that backoff sleeps
        and retries are fully isolated — one slow/failing request cannot
        block other concurrent requests from being dispatched.
        """

        selected_config = self._api_configs[selected_index]
        selected_config.record_usage()

        self._log_request(selected_index, item.messages)

        try:
            api_call_start = time.perf_counter()
            llm = self._models[selected_index]
            runner = llm.bind_tools(item.tools) if item.tools else llm
            response = runner.invoke(item.messages, **item.invoke_kwargs)
            api_call_elapsed_ms = (time.perf_counter() - api_call_start) * 1000

            self._log_success(response, api_call_elapsed_ms)
            item.future.set_result(response)

        except Exception as error:
            self._handle_worker_exception(item, selected_config, error)

    # ── Worker loop ──────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """Lightweight dispatcher: dequeue → select key → hand off to thread.

        The loop itself never sleeps for backoff — request retries happen on
        each dispatch thread, while the fixed send pacing happens here on the
        manager thread.  This keeps the dispatcher free to serve all
        concurrent workers without stalling.
        """

        while not self._stop_event.is_set():
            # Block if paused
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # If paused while waiting for an item, drain and loop
            if not self._pause_event.is_set():
                self._set_future_exception(item, RequestsPausedError("RequestsManager paused"))
                continue

            selected_index = self._wait_for_available_key()

            if selected_index is None:
                # Stopped or paused
                if self._stop_event.is_set():
                    self._set_future_exception(item, RequestTerminatedError("RequestsManager stopped"))
                else:
                    # Paused mid-wait
                    self._set_future_exception(item, RequestsPausedError("RequestsManager paused"))
                continue

            if selected_index == -1:
                # All keys exhausted for the day — trigger pause
                logger.error(
                    "[RequestsManager] ── RPD EXHAUSTED ── all API keys have hit their daily quota"
                )
                # Requeue the item before pausing so it gets drained properly
                self._queue.put(item)
                self.pause()
                if self._on_global_rpd_limit:
                    try:
                        self._on_global_rpd_limit()
                    except Exception as error:
                        logger.error("Error calling on_global_rpd_limit: %s", error)
                continue

            # Hand off to a dedicated thread — dispatcher is immediately free.
            t = threading.Thread(
                target=self._dispatch_item,
                args=(item, selected_index),
                daemon=True,
            )
            t.start()
            self._manager_send_sleep()


# ── Helpers ──────────────────────────────────────────────────────────────


def _short_error(error: Exception, max_len: int = 120) -> str:
    """Return a truncated, single-line error message for logs."""

    msg = str(error).replace("\n", " ").strip()
    if len(msg) > max_len:
        return msg[:max_len] + "…"
    return msg


# ── Global singleton ─────────────────────────────────────────────────────


_global_manager: Optional[RequestsManager] = None


def set_global_manager(manager: Optional[RequestsManager]) -> None:
    """Set the global RequestsManager instance."""

    global _global_manager
    _global_manager = manager


def get_global_manager() -> Optional[RequestsManager]:
    """Get the global RequestsManager instance."""

    return _global_manager
