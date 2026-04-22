"""Maat metrics package.

Provides structured data models, a real-time per-turn collector, and
aggregate metric computation for all experiments.
"""

from src.metrics.collector import MetricsCollector, infer_game_phase
from src.metrics.definitions import (
    ConditionMetrics,
    CriticAccuracy,
    DescriptiveStats,
    ErrorTypeRSR,
    FIRDeltaEntry,
    FSTEntry,
    GameRecord,
    LegalityDegradationBin,
    PhaseStratifiedFIR,
    QuartileErrorDist,
    ToolCallDistribution,
    TurnRecord,
)

__all__ = [
    # Collector
    "MetricsCollector",
    "infer_game_phase",
    # Data models
    "TurnRecord",
    "GameRecord",
    "ConditionMetrics",
    "CriticAccuracy",
    "DescriptiveStats",
    "ErrorTypeRSR",
    "FIRDeltaEntry",
    "FSTEntry",
    "LegalityDegradationBin",
    "PhaseStratifiedFIR",
    "QuartileErrorDist",
    "ToolCallDistribution",
]
