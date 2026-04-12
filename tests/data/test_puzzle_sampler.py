from __future__ import annotations

from pathlib import Path

from scripts.puzzle_sampler import (
    PuzzleRecord,
    apply_quality_filters,
    assign_difficulty,
    build_prompt_input,
    classify_phase,
    load_puzzles,
    prepare_experiment_dataset,
    sanity_check,
    stratified_sample,
    stratified_sample_phase_difficulty,
    write_phase_difficulty_collections,
)


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


def test_apply_quality_filters_thresholds() -> None:
    records = [
        PuzzleRecord("ok", "8/8/8/8/8/8/4k3/4K3 w - - 0 40", 1300, "endgame", 40, rating_deviation=74, popularity=51, nb_plays=100),
        PuzzleRecord("rd", "8/8/8/8/8/8/4k3/4K3 w - - 0 40", 1300, "endgame", 40, rating_deviation=75, popularity=51, nb_plays=100),
        PuzzleRecord("pop", "8/8/8/8/8/8/4k3/4K3 w - - 0 40", 1300, "endgame", 40, rating_deviation=74, popularity=50, nb_plays=100),
        PuzzleRecord("plays", "8/8/8/8/8/8/4k3/4K3 w - - 0 40", 1300, "endgame", 40, rating_deviation=74, popularity=51, nb_plays=99),
    ]

    filtered = apply_quality_filters(records)

    assert [record.puzzle_id for record in filtered] == ["ok"]


def test_assign_difficulty() -> None:
    assert assign_difficulty(1299) == "easy"
    assert assign_difficulty(1300) == "medium"
    assert assign_difficulty(1699) == "medium"
    assert assign_difficulty(1700) == "hard"


def test_cell_level_fallback_tops_up_with_heuristic_when_themed_is_short() -> None:
    themed = PuzzleRecord(
        puzzle_id="themed-open-easy",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        rating=1000,
        phase="opening",
        fullmove_number=1,
        moves="e2e4",
        popularity=70,
        nb_plays=500,
        rating_deviation=40,
        difficulty="easy",
        phase_source="theme",
        heuristic_phase="opening",
    )
    heuristic = PuzzleRecord(
        puzzle_id="heur-open-easy",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        rating=1000,
        phase="opening",
        fullmove_number=1,
        moves="e2e4",
        popularity=70,
        nb_plays=500,
        rating_deviation=40,
        difficulty="easy",
        phase_source="heuristic",
        heuristic_phase="opening",
    )

    sampled = stratified_sample_phase_difficulty(
        [themed, heuristic],
        target_per_cell=2,
        final_target=10,
        seed=11,
    )

    opening_easy = [
        record
        for record in sampled
        if record.phase == "opening" and record.difficulty == "easy"
    ]
    assert len(opening_easy) == 2
    opening_easy_ids = {record.puzzle_id for record in opening_easy}
    assert opening_easy_ids == {"themed-open-easy", "heur-open-easy"}


def test_cell_level_fallback_uses_heuristic_only_if_themed_missing() -> None:
    heuristic = PuzzleRecord(
        puzzle_id="heur-end-hard",
        fen="8/8/8/8/8/8/4k3/4K3 w - - 0 40",
        rating=2200,
        phase="endgame",
        fullmove_number=40,
        moves="e1e2",
        popularity=70,
        nb_plays=500,
        rating_deviation=40,
        difficulty="hard",
        phase_source="heuristic",
        heuristic_phase="endgame",
    )

    sampled = stratified_sample_phase_difficulty(
        [heuristic],
        target_per_cell=2,
        final_target=10,
        seed=11,
        enforce_sanity=False,
    )

    endgame_hard = [
        record
        for record in sampled
        if record.phase == "endgame" and record.difficulty == "hard"
    ]
    assert len(endgame_hard) == 1
    assert endgame_hard[0].puzzle_id == "heur-end-hard"


def test_sanity_check_rejects_illegal_first_move() -> None:
    bad = {
        "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Moves": "e2e5",
    }

    assert sanity_check(bad) is False


def test_build_prompt_input_contains_expected_fields() -> None:
    record = PuzzleRecord(
        puzzle_id="p1",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        rating=1200,
        phase="opening",
        fullmove_number=1,
        moves="e2e4",
        difficulty="medium",
    )

    payload = build_prompt_input(record)

    assert payload["puzzle_id"] == "p1"
    assert payload["solution_uci"] == "e2e4"
    assert payload["phase"] == "opening"
    assert "e2e4" in payload["legal_moves"]


def test_prepare_experiment_dataset_writes_outputs(tmp_path: Path) -> None:
    csv_path = tmp_path / "puzzles.csv"
    csv_path.write_text(
        "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes\n"
        "p1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,e2e4,1100,40,70,500,opening\n"
        "p2,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 20,d2d4,1500,40,70,500,middlegame\n"
        "p3,8/8/8/8/8/8/4k3/4K3 w - - 0 40,e1e2,2100,40,70,500,endgame\n",
        encoding="utf-8",
    )
    sampled_csv = tmp_path / "sampled.csv"
    inputs_jsonl = tmp_path / "inputs.jsonl"

    sampled, inputs = prepare_experiment_dataset(
        csv_path,
        target_per_cell=1,
        final_target=3,
        sampled_csv_output=sampled_csv,
        inputs_jsonl_output=inputs_jsonl,
    )

    assert sampled_csv.exists()
    assert inputs_jsonl.exists()
    assert len(sampled) == len(inputs)


def test_write_phase_difficulty_collections_creates_separate_files(tmp_path: Path) -> None:
    records = [
        PuzzleRecord(
            puzzle_id="open-easy",
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            rating=1100,
            phase="opening",
            fullmove_number=1,
            moves="e2e4",
            difficulty="easy",
            phase_source="theme",
        ),
        PuzzleRecord(
            puzzle_id="mid-medium",
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 20",
            rating=1500,
            phase="middlegame",
            fullmove_number=20,
            moves="d2d4",
            difficulty="medium",
            phase_source="theme",
        ),
        PuzzleRecord(
            puzzle_id="end-hard",
            fen="4k3/8/8/8/8/8/8/4K2R w - - 0 40",
            rating=2100,
            phase="endgame",
            fullmove_number=40,
            moves="h1h8",
            difficulty="hard",
            phase_source="theme",
        ),
    ]

    output_dir = tmp_path / "collections"
    written = write_phase_difficulty_collections(records, output_dir, file_prefix="cell")

    assert len(written) == 3
    assert (output_dir / "cell_opening_easy.csv").exists()
    assert (output_dir / "cell_middlegame_medium.csv").exists()
    assert (output_dir / "cell_endgame_hard.csv").exists()


def test_prepare_experiment_dataset_writes_collection_files(tmp_path: Path) -> None:
    csv_path = tmp_path / "puzzles.csv"
    csv_path.write_text(
        "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes\n"
        "p1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,e2e4,1100,40,70,500,opening\n"
        "p2,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 20,d2d4,1500,40,70,500,middlegame\n"
        "p3,4k3/8/8/8/8/8/8/4K2R w - - 0 40,h1h8,2100,40,70,500,endgame\n",
        encoding="utf-8",
    )
    collections_dir = tmp_path / "collections"

    prepare_experiment_dataset(
        csv_path,
        target_per_cell=1,
        final_target=3,
        collections_output_dir=collections_dir,
        collections_file_prefix="set",
        enforce_sanity=False,
    )

    assert (collections_dir / "set_opening_easy.csv").exists()
    assert (collections_dir / "set_middlegame_medium.csv").exists()
    assert (collections_dir / "set_endgame_hard.csv").exists()


def test_prepare_experiment_dataset_logs_and_pauses(tmp_path: Path) -> None:
    csv_path = tmp_path / "puzzles.csv"
    csv_path.write_text(
        "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes\n"
        "p1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,e2e4,1100,40,70,500,opening\n",
        encoding="utf-8",
    )

    captured_logs: list[str] = []
    prompts: list[str] = []

    def fake_logger(message: str) -> None:
        captured_logs.append(message)

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    prepare_experiment_dataset(
        csv_path,
        target_per_cell=1,
        final_target=1,
        enforce_sanity=False,
        enable_logs=True,
        pause_between_phases=True,
        logger=fake_logger,
        input_func=fake_input,
    )

    assert len(prompts) == 4
    assert any("[phase 1/5]" in message for message in captured_logs)
    assert any("[phase 5/5]" in message for message in captured_logs)
