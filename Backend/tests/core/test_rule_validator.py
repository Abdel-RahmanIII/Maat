from core.rule_validator import RuleValidator
from core.state_manager import StateManager


def test_validator_accepts_legal_uci() -> None:
    manager = StateManager()
    validator = RuleValidator(manager)

    result = validator.validate_move("e2e4")

    assert result.is_valid is True
    assert result.normalized_move_uci == "e2e4"
    assert result.validation_stage is None


def test_validator_accepts_san_and_normalizes_to_uci() -> None:
    manager = StateManager()
    validator = RuleValidator(manager)

    result = validator.validate_move("e4")

    assert result.is_valid is True
    assert result.normalized_move_uci == "e2e4"
    assert result.validation_stage is None


def test_validator_rejects_illegal_move_at_legality_stage() -> None:
    """e2e5 is valid UCI syntax but illegal on the board — stage must be 'legality'."""
    manager = StateManager()
    validator = RuleValidator(manager)

    result = validator.validate_move("e2e5")

    assert result.is_valid is False
    assert result.error_code == "illegal_move"
    assert result.validation_stage == "legality"


def test_validator_rejects_nonsense_at_syntax_stage() -> None:
    """'hello' cannot be parsed as UCI or SAN — stage must be 'syntax'."""
    manager = StateManager()
    validator = RuleValidator(manager)

    result = validator.validate_move("hello")

    assert result.is_valid is False
    assert result.error_code == "unsupported_format"
    assert result.validation_stage == "syntax"


def test_validator_rejects_empty_at_syntax_stage() -> None:
    manager = StateManager()
    validator = RuleValidator(manager)

    result = validator.validate_move("   ")

    assert result.is_valid is False
    assert result.error_code == "syntax_error"
    assert result.validation_stage == "syntax"


def test_validator_rejects_move_after_terminal() -> None:
    """Fool's mate — any further move must be rejected."""
    manager = StateManager()
    validator = RuleValidator(manager)
    for move in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        manager.apply_validated_move_uci(move) if validator.validate_move(move).is_valid else None

    result = validator.validate_move("e2e4")

    assert result.is_valid is False
    assert result.error_code == "game_already_terminal"
    assert result.validation_stage == "syntax"