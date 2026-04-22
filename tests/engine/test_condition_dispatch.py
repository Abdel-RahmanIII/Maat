import pytest

from src.state import TurnState
from src.engine.condition_dispatch import dispatch_turn, _import_runner

def test_import_runner_valid():
    runner = _import_runner("A")
    assert callable(runner)
    assert runner.__name__ == "run_condition_a"

def test_import_runner_invalid():
    with pytest.raises(ValueError, match="Unknown condition"):
        _import_runner("INVALID")
