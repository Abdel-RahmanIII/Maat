from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta

import pytest

from src.runner.limiting.rate_limiter import RateLimiter, get_rate_limiter


def test_rate_limiter_singleton():
    rl1 = get_rate_limiter()
    rl2 = get_rate_limiter()
    assert rl1 is rl2


def test_rate_limiter_configure(tmp_path):
    rl = RateLimiter()
    rl.configure(rpm=10, rpd=100, tpm=1000, project_root=tmp_path)
    status = rl.get_status()
    assert status["rpm_limit"] == 10
    assert status["rpd_limit"] == 100
    assert status["tpm_limit"] == 1000
    assert rl._rpd_state_path == tmp_path / "results/.rpd_state.json"


def test_rate_limiter_tpm_tracking():
    rl = RateLimiter()
    rl.configure(rpm=100, rpd=1000, tpm=5000)
    rl.record_tokens(prompt_tokens=10, completion_tokens=20)
    status = rl.get_status()
    assert status["tpm_current"] == 30
    assert status["tokens_today"] == 30


def test_rate_limiter_acquire_no_block(monkeypatch):
    rl = RateLimiter()
    rl.configure(rpm=100, rpd=1000)
    
    start = time.monotonic()
    wait_time = rl.acquire()
    
    assert wait_time < 0.1
    assert rl._total_requests == 1
    assert rl._rpd_count == 1
    assert len(rl._rpm_window) == 1


def test_rate_limiter_rpm_blocking(monkeypatch):
    rl = RateLimiter()
    rl.configure(rpm=2, rpd=100)
    
    # We will mock time to simulate sliding window without actually waiting 60s
    current_time = 1000.0
    
    def fake_time():
        return current_time
    
    def fake_monotonic():
        return current_time
        
    def fake_wait(timeout=None):
        nonlocal current_time
        if timeout:
            current_time += timeout
    
    monkeypatch.setattr(time, "time", fake_time)
    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    monkeypatch.setattr(rl._condition, "wait", fake_wait)
    
    # Acquire 2 slots (fills RPM)
    rl.acquire()
    current_time += 1.0
    rl.acquire()
    
    # Third acquire should block until the first slot expires (at t=1060)
    # The oldest is 1000.0, so wait is 1000.0 + 60.0 - 1001.0 = 59.0
    wait_time = rl.acquire()
    
    assert wait_time >= 59.0
    assert rl._total_blocks_rpm == 1
    assert rl._total_requests == 3


def test_rate_limiter_rpd_blocking(monkeypatch):
    rl = RateLimiter()
    rl.configure(rpm=100, rpd=1)
    
    current_time = 1000.0
    
    def fake_time():
        return current_time
    
    def fake_monotonic():
        return current_time
        
    def fake_wait(timeout=None):
        nonlocal current_time
        if timeout:
            current_time += timeout
            
    # Mock date and midnight calculation
    def fake_seconds_until_midnight():
        return 3600.0 # 1 hour
        
    monkeypatch.setattr(time, "time", fake_time)
    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    monkeypatch.setattr(rl._condition, "wait", fake_wait)
    monkeypatch.setattr(rl, "_seconds_until_midnight", fake_seconds_until_midnight)
    
    # Override _rotate_day so it doesn't accidentally reset rpd_count based on real date
    original_rotate_day = rl._rotate_day
    
    def fake_rotate_day():
        # Simulate rotation only if we advanced enough time, but for this test we manually control it
        pass
        
    monkeypatch.setattr(rl, "_rotate_day", fake_rotate_day)
    
    # 1st acquire passes
    rl.acquire()
    assert rl._rpd_count == 1
    
    # We now mock rotate_day to reset after the wait
    calls = 0
    def fake_rotate_day_with_reset():
        nonlocal calls
        calls += 1
        if calls > 1:
            rl._rpd_count = 0
            rl._rpd_date = date.today() + timedelta(days=1)
            
    # 2nd acquire should block for RPD
    wait_time = 0
    def do_acquire():
        nonlocal wait_time
        # Before wait, replace rotate_day so when it loops it resets
        monkeypatch.setattr(rl, "_rotate_day", fake_rotate_day_with_reset)
        wait_time = rl.acquire()
        
    do_acquire()
    assert rl._total_blocks_rpd == 1
    assert wait_time >= 3600.0
    assert rl._rpd_count == 1 # Was reset to 0, then incremented by acquire
    assert rl._total_requests == 2


def test_rate_limiter_persistence(tmp_path):
    rl1 = RateLimiter()
    rl1.configure(rpm=10, rpd=100, project_root=tmp_path)
    
    # Make a request
    rl1.acquire()
    assert rl1._rpd_count == 1
    
    # Ensure state file is created
    state_file = tmp_path / "results/.rpd_state.json"
    assert state_file.exists()
    
    data = json.loads(state_file.read_text())
    assert data["count"] == 1
    assert data["date"] == str(date.today())
    
    # Create new instance and load
    rl2 = RateLimiter()
    rl2.configure(rpm=10, rpd=100, project_root=tmp_path)
    
    assert rl2._rpd_count == 1
    
    # Test stale date
    stale_date = str(date.today() - timedelta(days=1))
    state_file.write_text(json.dumps({"count": 50, "date": stale_date}))
    
    rl3 = RateLimiter()
    rl3.configure(rpm=10, rpd=100, project_root=tmp_path)
    
    # Should be 0 because it's a new day
    assert rl3._rpd_count == 0


def test_rate_limiter_concurrency():
    rl = RateLimiter()
    rl.configure(rpm=1000, rpd=10000)
    
    def worker():
        for _ in range(50):
            rl.acquire()
            
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    assert rl._total_requests == 500
    assert rl._rpd_count == 500
    assert len(rl._rpm_window) == 500
    
def test_rate_limiter_status_callbacks():
    rl = RateLimiter()
    rl.configure(rpm=10, rpd=100)
    
    events = []
    def callback(status):
        events.append(status)
        
    rl.on_status_change(callback)
    
    rl.acquire()
    assert len(events) == 1
    assert events[0]["rpd_current"] == 1
    
    rl.record_tokens(prompt_tokens=10, completion_tokens=10)
    # Callback is not directly called on record_tokens unless wait completes
    # We can trigger it by a blocked acquire, but that's tested.
