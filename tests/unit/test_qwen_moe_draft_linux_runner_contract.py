from __future__ import annotations

from pathlib import Path


def test_focused_qwen_moe_runner_uses_the_locked_python_module() -> None:
    runner = Path(__file__).resolve().parents[2] / "scripts" / "run_focused_qwen_moe_draft_tests.sh"

    contents = runner.read_text()

    assert "uv run --frozen --no-sync python -m pytest" in contents
    assert "sys.executable" in contents
    assert "sys.prefix" in contents
    assert "pytest.__file__" in contents
    assert "/opt/nemo_rl_venv/bin/pytest" not in contents
    assert "tests/unit/models/megatron/test_dflash_asymmetric_tp_export.py" in contents


if __name__ == "__main__":
    test_focused_qwen_moe_runner_uses_the_locked_python_module()
