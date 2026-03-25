from core.schemas import ValidationErrorCode
from core.state_manager import StateManager
from core.validator import RuleValidator


def test_validator_accepts_legal_uci() -> None:
    manager = StateManager()
    validator = RuleValidator(manager)

    result = validator.validate_move("e2e4")

    assert result.is_valid is True
    assert result.normalized_move_uci == "e2e4"


def test_validator_accepts_san_and_normalizes_to_uci() -> None:
    manager = StateManager()
    validator = RuleValidator(manager)

    result = validator.validate_move("e4")

    assert result.is_valid is True
    assert result.normalized_move_uci == "e2e4"


def test_validator_rejects_illegal_move() -> None:
    manager = StateManager()
    validator = RuleValidator(manager)

    result = validator.validate_move("e2e5")

    assert result.is_valid is False
    assert result.error_code == ValidationErrorCode.ILLEGAL_MOVE


def test_validator_rejects_unsupported_format() -> None:
    manager = StateManager()
    validator = RuleValidator(manager)

    result = validator.validate_move("hello")

    assert result.is_valid is False
    assert result.error_code == ValidationErrorCode.UNSUPPORTED_FORMAT
