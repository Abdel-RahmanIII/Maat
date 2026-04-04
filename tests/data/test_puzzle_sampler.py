from __future__ import annotations

from pathlib import Path

from src.data.puzzle_sampler import PuzzleRecord, classify_phase, load_puzzles, stratified_sample


def test_classify_phase_opening() -> None:
    assert classify_phase("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") == "opening"


def test_classify_phase_middlegame() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 20"
    assert classify_phase(fen) == "middlegame"


def test_classify_phase_endgame() -> None:
    fen = "8/8/8/8/8/8/4k3/4K3 w - - 0 40"
    assert classify_phase(fen) == "endgame"


def test_load_puzzles_skips_invalid_fen(tmp_path: Path) -> None:
    csv_path = tmp_path / "puzzles.csv"
    csv_path.write_text(
        "PuzzleId,FEN,Rating\n"
        "p1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,1200\n"
        "p2,not_a_fen,1500\n",
        encoding="utf-8",
    )

    records = load_puzzles(csv_path)

    assert len(records) == 1
    assert records[0].puzzle_id == "p1"


def test_stratified_sample_per_phase_counts() -> None:
    records: list[PuzzleRecord] = []
    for phase in ("opening", "middlegame", "endgame"):
        for index in range(8):
            records.append(
                PuzzleRecord(
                    puzzle_id=f"{phase}-{index}",
                    fen="8/8/8/8/8/8/4k3/4K3 w - - 0 40",
                    rating=1000 + index * 100,
                    phase=phase,
                    fullmove_number=40,
                )
            )

    sampled = stratified_sample(records, per_phase=4, seed=7, rating_buckets=4)

    opening_count = sum(1 for record in sampled if record.phase == "opening")
    middlegame_count = sum(1 for record in sampled if record.phase == "middlegame")
    endgame_count = sum(1 for record in sampled if record.phase == "endgame")

    assert opening_count == 4
    assert middlegame_count == 4
    assert endgame_count == 4
