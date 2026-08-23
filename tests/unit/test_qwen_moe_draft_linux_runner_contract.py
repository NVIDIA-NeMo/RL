# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path


def test_focused_qwen_moe_runner_uses_the_locked_python_module() -> None:
    runner = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_focused_qwen_moe_draft_tests.sh"
    )

    contents = runner.read_text()

    assert "uv run --frozen --no-sync python -m pytest" in contents
    assert "sys.executable" in contents
    assert "sys.prefix" in contents
    assert "pytest.__file__" in contents
    assert "/opt/nemo_rl_venv/bin/pytest" not in contents
    assert "tests/unit/models/megatron/test_dflash_asymmetric_tp_export.py" in contents
    assert contents.count("--mcore-only") == 1
    assert "readonly MCORE_TESTS=(" in contents
    assert "readonly CONTRACT_TEST=" in contents
    assert '"${MCORE_TESTS[@]}"' in contents
    assert '"${CONTRACT_TEST}"' in contents


if __name__ == "__main__":
    test_focused_qwen_moe_runner_uses_the_locked_python_module()
