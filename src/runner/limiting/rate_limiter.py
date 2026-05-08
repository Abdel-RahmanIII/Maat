"""Thread-safe global API rate limiter.

Enforces RPM (requests per minute) and RPD (requests per day) limits across all
worker threads. TPM (tokens per minute) is tracked but never blocks.

Blocking behaviour
------------------
- RPM full → caller blocks until the oldest request exits the 60s window.
- RPD reached → caller blocks until midnight local time.

No fail-fast and no exceptions: callers simply block.

This module is intentionally independent from FastAPI: it can be used in
headless runs or tests that still want deterministic throttling.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RPD_STATE_FILE = "results/.rpd_state.json"


class RateLimiter:
    """Global thread-safe rate limiter with RPM / RPD / TPM tracking."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        # Limits (set via configure)
        self._rpm: int = 15
        self._rpd: int = 1500
        self._tpm: int | None = None

        # Sliding-window timestamps for RPM
        self._rpm_window: deque[float] = deque()

        # Daily counter
        self._rpd_date: date = date.today()
        self._rpd_count: int = 0

        # Token tracking per-minute window
        self._tpm_window: deque[tuple[float, int]] = deque()
        self._total_tokens_today: int = 0

        # Statistics
        self._total_requests: int = 0
        self._total_blocks_rpm: int = 0
        self._total_blocks_rpd: int = 0

        # Persistence path (relative to project root)
        self._rpd_state_path: Path | None = None

        # Optional callbacks for real-time dashboard updates
        self._on_status_change: list[Any] = []

    # ── Configuration ────────────────────────────────────────────────

    def configure(
        self,
        *,
        rpm: int = 15,
        rpd: int = 1500,
        tpm: int | None = None,
        project_root: Path | str | None = None,
    ) -> None:
        """Set rate limits. Call once before starting workers."""

        with self._lock:
            self._rpm = rpm
            self._rpd = rpd
            self._tpm = tpm

            if project_root:
                self._rpd_state_path = Path(project_root) / _RPD_STATE_FILE
                self._load_rpd_state()

        logger.info("Rate limiter configured: RPM=%d, RPD=%d, TPM=%s", rpm, rpd, tpm or "unlimited")

    def on_status_change(self, callback: Any) -> None:
        """Register a callback for status changes."""

        self._on_status_change.append(callback)

    # ── Core API ─────────────────────────────────────────────────────

    def acquire(self) -> float:
        """Block until an API request slot is available.

        Returns the wall-clock time (seconds) spent waiting.
        """

        start = time.monotonic()

        with self._condition:
            # RPD
            self._rotate_day()
            while self._rpd_count >= self._rpd:
                self._total_blocks_rpd += 1
                wait_secs = self._seconds_until_midnight()
                logger.warning(
                    "RPD limit reached (%d/%d). Sleeping %.0fs until midnight.",
                    self._rpd_count,
                    self._rpd,
                    wait_secs,
                )
                self._notify_status()
                self._condition.wait(timeout=wait_secs + 1)
                self._rotate_day()

            # RPM
            now = time.time()
            self._prune_rpm_window(now)
            while len(self._rpm_window) >= self._rpm:
                self._total_blocks_rpm += 1
                oldest = self._rpm_window[0]
                wait_secs = max(0.05, oldest + 60.0 - now)
                logger.debug(
                    "RPM limit reached (%d/%d). Waiting %.1fs.",
                    len(self._rpm_window),
                    self._rpm,
                    wait_secs,
                )
                self._notify_status()
                self._condition.wait(timeout=wait_secs + 0.1)
                now = time.time()
                self._prune_rpm_window(now)
                self._rotate_day()

                while self._rpd_count >= self._rpd:
                    wait_rpd = self._seconds_until_midnight()
                    self._condition.wait(timeout=wait_rpd + 1)
                    self._rotate_day()

            # Record
            self._rpm_window.append(time.time())
            self._rpd_count += 1
            self._total_requests += 1
            self._save_rpd_state()
            self._notify_status()

        return time.monotonic() - start

    def record_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Record token usage after an API call (non-blocking)."""

        total = prompt_tokens + completion_tokens
        now = time.time()

        with self._lock:
            self._tpm_window.append((now, total))
            self._total_tokens_today += total
            cutoff = now - 60.0
            while self._tpm_window and self._tpm_window[0][0] < cutoff:
                self._tpm_window.popleft()

        # Wake any waiters
        with self._condition:
            self._condition.notify_all()

    def get_status(self) -> dict[str, Any]:
        """Return current rate-limiter state for the dashboard."""

        with self._lock:
            now = time.time()
            self._prune_rpm_window(now)
            self._rotate_day()

            cutoff = now - 60.0
            while self._tpm_window and self._tpm_window[0][0] < cutoff:
                self._tpm_window.popleft()
            current_tpm = sum(t for _, t in self._tpm_window)

            if len(self._rpm_window) >= self._rpm:
                next_slot = max(0.0, self._rpm_window[0] + 60.0 - now)
            else:
                next_slot = 0.0

            return {
                "rpm_current": len(self._rpm_window),
                "rpm_limit": self._rpm,
                "rpd_current": self._rpd_count,
                "rpd_limit": self._rpd,
                "tpm_current": current_tpm,
                "tpm_limit": self._tpm,
                "tokens_today": self._total_tokens_today,
                "next_rpm_slot_seconds": round(next_slot, 1),
                "rpd_resets_at": str(datetime.combine(date.today() + timedelta(days=1), dt_time.min)),
                "total_requests": self._total_requests,
                "total_blocks_rpm": self._total_blocks_rpm,
                "total_blocks_rpd": self._total_blocks_rpd,
            }

    # ── Internal helpers ─────────────────────────────────────────────

    def _prune_rpm_window(self, now: float) -> None:
        cutoff = now - 60.0
        while self._rpm_window and self._rpm_window[0] < cutoff:
            self._rpm_window.popleft()

    def _rotate_day(self) -> None:
        today = date.today()
        if today != self._rpd_date:
            logger.info("Day rolled over (%s → %s). Resetting RPD counter from %d.", self._rpd_date, today, self._rpd_count)
            self._rpd_date = today
            self._rpd_count = 0
            self._total_tokens_today = 0
            self._save_rpd_state()

    @staticmethod
    def _seconds_until_midnight() -> float:
        now = datetime.now()
        midnight = datetime.combine(now.date() + timedelta(days=1), dt_time.min)
        return max(0.0, (midnight - now).total_seconds())

    def _load_rpd_state(self) -> None:
        if not self._rpd_state_path or not self._rpd_state_path.exists():
            return
        try:
            data = json.loads(self._rpd_state_path.read_text(encoding="utf-8"))
            saved_date = data.get("date", "")
            if saved_date == str(date.today()):
                self._rpd_count = data.get("count", 0)
                self._rpd_date = date.today()
                logger.info("Loaded RPD state: %d requests today.", self._rpd_count)
            else:
                logger.info("RPD state is stale (%s). Starting fresh.", saved_date)
        except Exception:
            logger.warning("Could not read RPD state file.", exc_info=True)

    def _save_rpd_state(self) -> None:
        if not self._rpd_state_path:
            return
        try:
            self._rpd_state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"date": str(self._rpd_date), "count": self._rpd_count}
            self._rpd_state_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            logger.warning("Could not save RPD state.", exc_info=True)

    def _notify_status(self) -> None:
        for cb in self._on_status_change:
            try:
                cb(self.get_status())
            except Exception:
                pass


_instance: RateLimiter | None = None
_instance_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    """Return the global rate limiter singleton."""

    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = RateLimiter()
    return _instance
