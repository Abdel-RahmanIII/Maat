from __future__ import annotations

import threading
import time

from src.runner.core.progress import ConditionProgress, ExperimentProgress

def test_condition_progress_initialization():
    cp = ConditionProgress("A", 10)
    assert cp.condition == "A"
    assert cp.total == 10
    assert cp.completed == 0
    assert cp.failed == 0
    assert cp.in_progress == 0
    assert cp.valid_count == 0

def test_condition_progress_record_start():
    cp = ConditionProgress("A", 10)
    cp.record_start()
    assert cp.in_progress == 1

def test_condition_progress_record_complete():
    cp = ConditionProgress("A", 10)
    cp.record_start()
    cp.record_complete(is_valid=True)
    assert cp.completed == 1
    assert cp.in_progress == 0
    assert cp.valid_count == 1

    cp.record_start()
    cp.record_complete(is_valid=False)
    assert cp.completed == 2
    assert cp.in_progress == 0
    assert cp.valid_count == 1

def test_condition_progress_record_failure():
    cp = ConditionProgress("A", 10)
    cp.record_start()
    cp.record_failure()
    assert cp.failed == 1
    assert cp.in_progress == 0
    assert cp.completed == 0

def test_condition_progress_to_dict():
    cp = ConditionProgress("A", 10)
    cp.record_start()
    cp.record_complete(is_valid=True)
    cp.record_start()
    cp.record_failure()
    
    d = cp.to_dict()
    assert d == {
        "condition": "A",
        "total": 10,
        "completed": 1,
        "failed": 1,
        "in_progress": 0,
        "valid_count": 1,
    }

def test_experiment_progress_initialization():
    ep = ExperimentProgress(1, ["A", "B"])
    assert ep.experiment == 1
    assert ep.status == "pending"
    assert ep.started_at is None
    assert ep.output_dir is None

def test_experiment_progress_init_condition():
    ep = ExperimentProgress(1, ["A", "B"])
    ep.init_condition("A", 10)
    assert "A" in ep.conditions_progress
    assert ep.conditions_progress["A"].total == 10

def test_experiment_progress_to_dict():
    ep = ExperimentProgress(1, ["A"])
    ep.init_condition("A", 10)
    ep.status = "running"
    ep.started_at = "2023-01-01T00:00:00"
    
    d = ep.to_dict()
    assert d["experiment"] == 1
    assert d["status"] == "running"
    assert d["started_at"] == "2023-01-01T00:00:00"
    assert "A" in d["conditions"]
    assert d["conditions"]["A"]["total"] == 10

def test_condition_progress_concurrency():
    cp = ConditionProgress("A", 1000)
    
    def worker():
        for _ in range(100):
            cp.record_start()
            time.sleep(0.001)
            cp.record_complete(is_valid=True)
            
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    assert cp.completed == 1000
    assert cp.valid_count == 1000
    assert cp.in_progress == 0
    assert cp.failed == 0

def test_condition_progress_concurrency_failure():
    cp = ConditionProgress("B", 1000)
    
    def worker():
        for i in range(100):
            cp.record_start()
            time.sleep(0.001)
            if i % 2 == 0:
                cp.record_complete(is_valid=False)
            else:
                cp.record_failure()
            
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    assert cp.completed == 500
    assert cp.failed == 500
    assert cp.valid_count == 0
    assert cp.in_progress == 0
