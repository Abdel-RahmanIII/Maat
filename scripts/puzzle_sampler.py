from __future__ import annotations

import csv
import json
import random
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import chess

LICHESS_PUZZLE_CSV_URL = "https://database.lichess.org/lichess_db_puzzle.csv"
PHASES = ("opening", "middlegame", "endgame")
DIFFICULTIES = ("easy", "medium", "hard")
ENDGAME_THEME_HINTS = (
	"rookendgame",
	"queenendgame",
	"pawnendgame",
	"bishopendgame",
	"knightendgame",
	"queenvsrook",
)


@dataclass(frozen=True)
class PuzzleRecord:
	puzzle_id: str
	fen: str
	rating: int
	phase: str
	fullmove_number: int
	moves: str = ""
	rating_deviation: int = 0
	popularity: int = 0
	nb_plays: int = 0
	themes: tuple[str, ...] = ()
	piece_count: int = 0
	major_pieces: int = 0
	difficulty: str = "medium"
	phase_source: str = "heuristic"
	heuristic_phase: str = "middlegame"


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


def parse_fen_features(fen: str) -> tuple[int | None, int | None, int | None]:
	try:
		board = chess.Board(fen)
	except ValueError:
		return None, None, None

	fullmove = board.fullmove_number
	piece_count = len(board.piece_map())
	pawns = len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK))
	major_pieces = piece_count - pawns
	return fullmove, piece_count, major_pieces


def _classify_phase_from_features(fullmove: int, piece_count: int) -> str:
	if fullmove <= 12:
		return "opening"
	if piece_count <= 10 or fullmove >= 40:
		return "endgame"
	return "middlegame"


def classify_phase(fen: str) -> str:
	"""Classify opening/middlegame/endgame using move and piece-count heuristics."""

	board = chess.Board(fen)
	return _classify_phase_from_features(board.fullmove_number, len(board.piece_map()))


def assign_difficulty(rating: int) -> str:
	if rating < 1300:
		return "easy"
	if rating < 1700:
		return "medium"
	return "hard"


def _phase_from_themes(themes: tuple[str, ...]) -> str | None:
	theme_set = set(themes)

	if "opening" in theme_set:
		return "opening"
	if "endgame" in theme_set or any(tag in theme_set for tag in ENDGAME_THEME_HINTS):
		return "endgame"
	if "middlegame" in theme_set:
		return "middlegame"

	return None


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


def _parse_themes(raw: str | None) -> tuple[str, ...]:
	if not raw:
		return ()

	values = [value.strip().lower() for value in raw.split(",")]
	return tuple(value for value in values if value)


def load_puzzles(csv_path: str | Path) -> list[PuzzleRecord]:
	"""Load Lichess puzzle rows and enrich with metadata for sampling."""

	puzzles: list[PuzzleRecord] = []
	with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			puzzle_id = _row_value(row, "PuzzleId", "puzzle_id", "id")
			fen = _row_value(row, "FEN", "fen")
			moves = _row_value(row, "Moves", "moves") or ""

			if not puzzle_id or not fen:
				continue

			fullmove, piece_count, major_pieces = parse_fen_features(fen)
			if fullmove is None or piece_count is None or major_pieces is None:
				continue

			themes = _parse_themes(_row_value(row, "Themes", "themes"))
			theme_phase = _phase_from_themes(themes)
			heuristic_phase = _classify_phase_from_features(fullmove, piece_count)
			phase = theme_phase or heuristic_phase
			phase_source = "theme" if theme_phase else "heuristic"

			rating = _safe_int(_row_value(row, "Rating", "rating"), default=0)
			record = PuzzleRecord(
				puzzle_id=puzzle_id,
				fen=fen,
				rating=rating,
				phase=phase,
				fullmove_number=fullmove,
				moves=moves,
				rating_deviation=_safe_int(
					_row_value(row, "RatingDeviation", "rating_deviation"),
					default=0,
				),
				popularity=_safe_int(_row_value(row, "Popularity", "popularity"), default=0),
				nb_plays=_safe_int(_row_value(row, "NbPlays", "nb_plays", "nbplays"), default=0),
				themes=themes,
				piece_count=piece_count,
				major_pieces=major_pieces,
				difficulty=assign_difficulty(rating),
				phase_source=phase_source,
				heuristic_phase=heuristic_phase,
			)
			puzzles.append(record)

	return puzzles


def apply_quality_filters(
	records: list[PuzzleRecord],
	max_rating_deviation: int = 75,
	min_popularity: int = 50,
	min_nb_plays: int = 100,
) -> list[PuzzleRecord]:
	"""Keep puzzles with stable rating estimates and sufficient play/popularity."""

	return [
		record
		for record in records
		if record.rating_deviation < max_rating_deviation
		and record.popularity > min_popularity
		and record.nb_plays >= min_nb_plays
	]


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


def _extract_row_values(row: PuzzleRecord | dict[str, Any]) -> tuple[str | None, str | None]:
	if isinstance(row, PuzzleRecord):
		return row.fen, row.moves

	fen = row.get("FEN") or row.get("fen")
	moves = row.get("Moves") or row.get("moves")
	return fen, moves


def sanity_check(row: PuzzleRecord | dict[str, Any]) -> bool:
	"""Return True if the row has a valid non-terminal board and legal first move."""

	try:
		fen, moves = _extract_row_values(row)
		if not fen or not moves:
			return False

		board = chess.Board(fen)
		if board.is_game_over():
			return False

		first_move = moves.split()[0]
		move = chess.Move.from_uci(first_move)
		return move in board.legal_moves
	except (IndexError, ValueError):
		return False


def _sample_records_from_pool(
	records: list[PuzzleRecord],
	target_count: int,
	rng: random.Random,
	enforce_sanity: bool,
) -> list[PuzzleRecord]:
	if target_count <= 0 or not records:
		return []

	shuffled = records[:]
	rng.shuffle(shuffled)

	sampled: list[PuzzleRecord] = []
	for record in shuffled:
		if enforce_sanity and not sanity_check(record):
			continue

		sampled.append(record)
		if len(sampled) >= target_count:
			break

	return sampled


def stratified_sample_phase_difficulty(
	records: list[PuzzleRecord],
	target_per_cell: int = 34,
	final_target: int = 300,
	seed: int = 42,
	enforce_sanity: bool = True,
) -> list[PuzzleRecord]:
	"""Sample by phase+difficulty with theme-first and heuristic top-up.

	Each phase+difficulty cell samples from themed records first, then tops up
	from heuristic records in the same cell until ``target_per_cell`` is reached
	or candidates are exhausted.
	"""

	rng = random.Random(seed)
	themed_pools = {(phase, difficulty): [] for phase in PHASES for difficulty in DIFFICULTIES}
	heuristic_pools = {(phase, difficulty): [] for phase in PHASES for difficulty in DIFFICULTIES}

	for record in records:
		difficulty = record.difficulty if record.difficulty in DIFFICULTIES else assign_difficulty(record.rating)

		if record.phase_source == "theme" and record.phase in PHASES:
			themed_pools[(record.phase, difficulty)].append(record)
			continue

		heuristic_phase = record.heuristic_phase or record.phase
		if heuristic_phase in PHASES:
			heuristic_pools[(heuristic_phase, difficulty)].append(
				replace(
					record,
					phase=heuristic_phase,
					phase_source="heuristic",
					heuristic_phase=heuristic_phase,
					difficulty=difficulty,
				)
			)

	sampled: list[PuzzleRecord] = []
	for phase in PHASES:
		for difficulty in DIFFICULTIES:
			key = (phase, difficulty)
			themed_candidates = themed_pools[key]
			themed_sampled = _sample_records_from_pool(
				records=themed_candidates,
				target_count=target_per_cell,
				rng=rng,
				enforce_sanity=enforce_sanity,
			)

			remaining = target_per_cell - len(themed_sampled)
			heuristic_sampled: list[PuzzleRecord] = []
			if remaining > 0:
				heuristic_sampled = _sample_records_from_pool(
					records=heuristic_pools[key],
					target_count=remaining,
					rng=rng,
					enforce_sanity=enforce_sanity,
				)

			sampled.extend(themed_sampled)
			sampled.extend(heuristic_sampled)

	rng.shuffle(sampled)
	if final_target > 0:
		return sampled[:final_target]
	return sampled


def build_prompt_input(record: PuzzleRecord) -> dict[str, Any]:
	board = chess.Board(record.fen)
	solution_uci = record.moves.split()[0] if record.moves.strip() else ""
	return {
		"puzzle_id": record.puzzle_id,
		"fen": record.fen,
		"phase": record.phase,
		"difficulty": record.difficulty,
		"rating": record.rating,
		"board_ascii": str(board),
		"solution_uci": solution_uci,
		"legal_moves": [move.uci() for move in board.legal_moves],
	}


def build_experiment_inputs(records: list[PuzzleRecord]) -> list[dict[str, Any]]:
	return [build_prompt_input(record) for record in records]


def _record_to_csv_row(record: PuzzleRecord) -> dict[str, Any]:
	row = asdict(record)
	row["themes"] = ",".join(record.themes)
	return row


def write_sampled_csv(records: list[PuzzleRecord], destination: str | Path) -> Path:
	destination_path = Path(destination)
	destination_path.parent.mkdir(parents=True, exist_ok=True)

	fieldnames = list(_record_to_csv_row(records[0]).keys()) if records else list(PuzzleRecord.__annotations__.keys())
	with destination_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for record in records:
			writer.writerow(_record_to_csv_row(record))

	return destination_path


def write_phase_difficulty_collections(
	records: list[PuzzleRecord],
	output_directory: str | Path,
	file_prefix: str = "collection",
	include_empty: bool = False,
) -> dict[tuple[str, str], Path]:
	"""Write one CSV file per (phase, difficulty) collection.

	By default, only non-empty collections are written.
	"""

	output_dir = Path(output_directory)
	output_dir.mkdir(parents=True, exist_ok=True)

	collections: dict[tuple[str, str], list[PuzzleRecord]] = {
		(phase, difficulty): []
		for phase in PHASES
		for difficulty in DIFFICULTIES
	}

	for record in records:
		phase = record.phase if record.phase in PHASES else record.heuristic_phase
		if phase not in PHASES:
			continue

		difficulty = record.difficulty if record.difficulty in DIFFICULTIES else assign_difficulty(record.rating)
		collections[(phase, difficulty)].append(record)

	written: dict[tuple[str, str], Path] = {}
	for phase in PHASES:
		for difficulty in DIFFICULTIES:
			key = (phase, difficulty)
			cell_records = collections[key]
			if not cell_records and not include_empty:
				continue

			filename = f"{file_prefix}_{phase}_{difficulty}.csv"
			destination = output_dir / filename
			write_sampled_csv(cell_records, destination)
			written[key] = destination

	return written


def write_experiment_inputs_jsonl(inputs: list[dict[str, Any]], destination: str | Path) -> Path:
	destination_path = Path(destination)
	destination_path.parent.mkdir(parents=True, exist_ok=True)

	with destination_path.open("w", encoding="utf-8") as handle:
		for item in inputs:
			handle.write(json.dumps(item, ensure_ascii=True) + "\n")

	return destination_path


def _resolve_logger(enable_logs: bool, logger: Callable[[str], None] | None) -> Callable[[str], None] | None:
	if logger is not None:
		return logger
	if not enable_logs:
		return None

	def _default_logger(message: str) -> None:
		print(message)

	return _default_logger


def _emit_log(log_fn: Callable[[str], None] | None, message: str) -> None:
	if log_fn is not None:
		log_fn(message)


def _pause_for_phase(
	*,
	pause_between_phases: bool,
	phase_number: int,
	total_phases: int,
	input_func: Callable[[str], str] | None,
	log_fn: Callable[[str], None] | None,
) -> None:
	if not pause_between_phases or phase_number >= total_phases:
		return

	reader = input_func
	if reader is None:
		if not (sys.stdin.isatty() and sys.stdout.isatty()):
			_emit_log(log_fn, f"[phase {phase_number}/{total_phases}] Non-interactive terminal detected; continuing.")
			return
		reader = input

	try:
		reader(f"[phase {phase_number}/{total_phases}] Press Enter to continue...")
	except EOFError:
		_emit_log(log_fn, f"[phase {phase_number}/{total_phases}] Input unavailable; continuing.")


def prepare_experiment_dataset(
	csv_path: str | Path,
	*,
	target_per_cell: int = 34,
	final_target: int = 300,
	seed: int = 42,
	max_rating_deviation: int = 75,
	min_popularity: int = 50,
	min_nb_plays: int = 100,
	enforce_sanity: bool = True,
	sampled_csv_output: str | Path | None = None,
	inputs_jsonl_output: str | Path | None = None,
	collections_output_dir: str | Path | None = None,
	collections_file_prefix: str = "collection",
	enable_logs: bool = True,
	pause_between_phases: bool = True,
	logger: Callable[[str], None] | None = None,
	input_func: Callable[[str], str] | None = None,
) -> tuple[list[PuzzleRecord], list[dict[str, Any]]]:
	phase_total = 5
	log_fn = _resolve_logger(enable_logs=enable_logs, logger=logger)
	pipeline_started = time.perf_counter()
	_emit_log(log_fn, f"[pipeline] Starting dataset preparation from {Path(csv_path)}")

	phase_started = time.perf_counter()
	records = load_puzzles(csv_path)
	_emit_log(
		log_fn,
		f"[phase 1/{phase_total}] Loaded {len(records):,} rows in {time.perf_counter() - phase_started:.2f}s",
	)
	_pause_for_phase(
		pause_between_phases=pause_between_phases,
		phase_number=1,
		total_phases=phase_total,
		input_func=input_func,
		log_fn=log_fn,
	)

	phase_started = time.perf_counter()
	filtered = apply_quality_filters(
		records,
		max_rating_deviation=max_rating_deviation,
		min_popularity=min_popularity,
		min_nb_plays=min_nb_plays,
	)
	_emit_log(
		log_fn,
		f"[phase 2/{phase_total}] Quality filter kept {len(filtered):,}/{len(records):,} rows in {time.perf_counter() - phase_started:.2f}s",
	)
	_pause_for_phase(
		pause_between_phases=pause_between_phases,
		phase_number=2,
		total_phases=phase_total,
		input_func=input_func,
		log_fn=log_fn,
	)

	phase_started = time.perf_counter()
	sampled = stratified_sample_phase_difficulty(
		filtered,
		target_per_cell=target_per_cell,
		final_target=final_target,
		seed=seed,
		enforce_sanity=enforce_sanity,
	)
	_emit_log(
		log_fn,
		f"[phase 3/{phase_total}] Sampled {len(sampled):,} rows in {time.perf_counter() - phase_started:.2f}s",
	)
	_pause_for_phase(
		pause_between_phases=pause_between_phases,
		phase_number=3,
		total_phases=phase_total,
		input_func=input_func,
		log_fn=log_fn,
	)

	phase_started = time.perf_counter()
	inputs = build_experiment_inputs(sampled)
	_emit_log(
		log_fn,
		f"[phase 4/{phase_total}] Built {len(inputs):,} experiment dicts in {time.perf_counter() - phase_started:.2f}s",
	)
	_pause_for_phase(
		pause_between_phases=pause_between_phases,
		phase_number=4,
		total_phases=phase_total,
		input_func=input_func,
		log_fn=log_fn,
	)

	phase_started = time.perf_counter()

	if sampled_csv_output is not None:
		write_sampled_csv(sampled, sampled_csv_output)
		_emit_log(log_fn, f"[phase 5/{phase_total}] Wrote sampled CSV to {Path(sampled_csv_output)}")
	if inputs_jsonl_output is not None:
		write_experiment_inputs_jsonl(inputs, inputs_jsonl_output)
		_emit_log(log_fn, f"[phase 5/{phase_total}] Wrote inputs JSONL to {Path(inputs_jsonl_output)}")
	if collections_output_dir is not None:
		write_phase_difficulty_collections(
			sampled,
			collections_output_dir,
			file_prefix=collections_file_prefix,
		)
		_emit_log(log_fn, f"[phase 5/{phase_total}] Wrote collections to {Path(collections_output_dir)}")

	_emit_log(
		log_fn,
		f"[phase 5/{phase_total}] Output phase finished in {time.perf_counter() - phase_started:.2f}s",
	)
	_emit_log(
		log_fn,
		f"[pipeline] Finished in {time.perf_counter() - pipeline_started:.2f}s",
	)

	return sampled, inputs


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
