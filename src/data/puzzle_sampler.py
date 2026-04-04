from __future__ import annotations

import csv
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import chess

LICHESS_PUZZLE_CSV_URL = "https://database.lichess.org/lichess_db_puzzle.csv"
PHASES = ("opening", "middlegame", "endgame")


@dataclass(frozen=True)
class PuzzleRecord:
    puzzle_id: str
    fen: str
    rating: int
    phase: str
    fullmove_number: int


def _non_pawn_material_points(board: chess.Board) -> int:
    piece_values = {
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    total = 0
    for piece_type, value in piece_values.items():
        total += len(board.pieces(piece_type, chess.WHITE)) * value
        total += len(board.pieces(piece_type, chess.BLACK)) * value

    return total


def classify_phase(fen: str) -> str:
    """Classify opening/middlegame/endgame using the plan heuristics."""

    board = chess.Board(fen)
    move_number = board.fullmove_number
    non_pawn_material = _non_pawn_material_points(board)

    if move_number <= 15:
        return "opening"

    if move_number > 35 or non_pawn_material <= 13:
        return "endgame"

    return "middlegame"


def download_puzzle_csv(destination: str | Path, url: str = LICHESS_PUZZLE_CSV_URL) -> Path:
    """Download a puzzle CSV file to the provided destination path."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response:
        destination_path.write_bytes(response.read())

    return destination_path


def _safe_int(raw: str | None, default: int = 0) -> int:
    if raw is None:
        return default

    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _row_value(row: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def load_puzzles(csv_path: str | Path) -> list[PuzzleRecord]:
    """Load Lichess puzzle rows and enrich with phase metadata."""

    puzzles: list[PuzzleRecord] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            puzzle_id = _row_value(row, "PuzzleId", "puzzle_id", "id")
            fen = _row_value(row, "FEN", "fen")
            rating_raw = _row_value(row, "Rating", "rating")

            if not puzzle_id or not fen:
                continue

            try:
                board = chess.Board(fen)
            except ValueError:
                continue

            record = PuzzleRecord(
                puzzle_id=puzzle_id,
                fen=fen,
                rating=_safe_int(rating_raw, default=0),
                phase=classify_phase(fen),
                fullmove_number=board.fullmove_number,
            )
            puzzles.append(record)

    return puzzles


def _bucketize_by_rating(records: list[PuzzleRecord], bucket_count: int) -> list[list[PuzzleRecord]]:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive.")

    if not records:
        return [[] for _ in range(bucket_count)]

    sorted_records = sorted(records, key=lambda record: record.rating)
    buckets: list[list[PuzzleRecord]] = [[] for _ in range(bucket_count)]

    total = len(sorted_records)
    for index, record in enumerate(sorted_records):
        bucket_index = min((index * bucket_count) // total, bucket_count - 1)
        buckets[bucket_index].append(record)

    return buckets


def _sample_phase_records(
    records: list[PuzzleRecord],
    target_count: int,
    rating_buckets: int,
    rng: random.Random,
) -> list[PuzzleRecord]:
    if target_count <= 0 or not records:
        return []

    buckets = _bucketize_by_rating(records, rating_buckets)

    base = target_count // rating_buckets
    remainder = target_count % rating_buckets
    requested = [base + (1 if i < remainder else 0) for i in range(rating_buckets)]

    sampled: list[PuzzleRecord] = []
    leftovers: list[PuzzleRecord] = []

    for bucket_index, bucket in enumerate(buckets):
        take_count = min(requested[bucket_index], len(bucket))
        chosen = rng.sample(bucket, take_count) if take_count > 0 else []
        sampled.extend(chosen)

        remaining = [record for record in bucket if record not in chosen]
        leftovers.extend(remaining)

    desired_total = min(target_count, len(records))
    still_needed = desired_total - len(sampled)
    if still_needed > 0 and leftovers:
        sampled.extend(rng.sample(leftovers, min(still_needed, len(leftovers))))

    return sampled


def stratified_sample(
    records: list[PuzzleRecord],
    per_phase: int = 100,
    seed: int = 42,
    rating_buckets: int = 4,
) -> list[PuzzleRecord]:
    """Sample puzzles by phase with rating stratification inside each phase."""

    rng = random.Random(seed)
    phase_records = {phase: [] for phase in PHASES}

    for record in records:
        if record.phase in phase_records:
            phase_records[record.phase].append(record)

    sampled_all: list[PuzzleRecord] = []
    for phase in PHASES:
        sampled_all.extend(
            _sample_phase_records(
                records=phase_records[phase],
                target_count=per_phase,
                rating_buckets=rating_buckets,
                rng=rng,
            )
        )

    rng.shuffle(sampled_all)
    return sampled_all


def sample_from_csv(
    csv_path: str | Path,
    per_phase: int = 100,
    seed: int = 42,
    rating_buckets: int = 4,
) -> list[PuzzleRecord]:
    """Load puzzles from CSV then produce a phase-stratified sample."""

    records = load_puzzles(csv_path)
    return stratified_sample(
        records=records,
        per_phase=per_phase,
        seed=seed,
        rating_buckets=rating_buckets,
    )
