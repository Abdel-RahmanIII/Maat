from pathlib import Path

import pytest

from src.engine.config_loader import load_experiment_config

def test_load_experiment_config(tmp_path: Path):
    config_content = """
experiment: 1
conditions: ["A", "B"]
puzzle_data: "src/data/some_puzzles.jsonl"
output_dir: "results/exp1"
generation_strategy: "generator_only"
delay_seconds: 1.5
max_api_retries: 2
backoff_base: 1.0
backoff_max: 10.0
model:
  model_name: "test-model"
  temperature: 0.1
  max_output_tokens: 500
"""
    yaml_path = tmp_path / "exp1.yaml"
    yaml_path.write_text(config_content)
    
    cfg = load_experiment_config(yaml_path)
    
    assert cfg["experiment"] == 1
    assert cfg["conditions"] == ["A", "B"]
    assert cfg["generation_strategy"] == "generator_only"
    assert cfg["delay_seconds"] == 1.5
    assert cfg["max_api_retries"] == 2
    assert cfg["backoff_base"] == 1.0
    assert cfg["backoff_max"] == 10.0
    
    assert cfg["model_config"].model_name == "test-model"
    assert cfg["model_config"].temperature == 0.1
    assert cfg["model_config"].max_output_tokens == 500
    
    # Path resolution
    assert isinstance(cfg["puzzle_data"], Path)
    assert "some_puzzles.jsonl" in str(cfg["puzzle_data"])

def test_load_experiment_config_invalid(tmp_path: Path):
    config_content = """
experiment: 4
conditions: ["A"]
"""
    yaml_path = tmp_path / "exp4.yaml"
    yaml_path.write_text(config_content)
    
    with pytest.raises(ValueError, match="experiment must be 1, 2, or 3"):
        load_experiment_config(yaml_path)
