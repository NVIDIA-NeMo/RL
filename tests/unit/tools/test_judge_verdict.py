import importlib.util
from pathlib import Path

import pytest

G_PATH = (
    Path(__file__).resolve().parents[3] / "tools/super_rl/gym_overlays/judge_verdict.py"
)
spec = importlib.util.spec_from_file_location("judge_verdict", G_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "text,expected",
    [("[[A=B]]", True), ("[[A!=B]]", False), ("[[A!=B]] then [[A=B]]", False)],
)
def test_first_valid_verdict_is_authoritative(text: str, expected: bool) -> None:
    assert (
        module.first_verdict(text, equal_label="[[A=B]]", not_equal_label="[[A!=B]]")
        is expected
    )


@pytest.mark.parametrize("text", ["", "I cannot decide", "[[broken]]"])
def test_missing_verdict_is_not_a_negative(text: str) -> None:
    with pytest.raises(ValueError, match="no valid verdict"):
        module.first_verdict(text, equal_label="[[A=B]]", not_equal_label="[[A!=B]]")
