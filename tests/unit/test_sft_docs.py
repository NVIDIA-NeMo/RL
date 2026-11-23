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

"""Regression tests for https://github.com/NVIDIA-NeMo/RL/issues/1548.

The SFT guide's ``ResponseDataset`` template must list every top-level
``data`` key that ``setup_data`` (``examples/run_sft.py``) and the SFT
dataloader setup (``nemo_rl/algorithms/sft.py``) read directly. Those keys
have no defaults anywhere (``DataConfig`` is a ``TypedDict``), so a config
copied from an incomplete template loads the datasets fine and then fails
with ``KeyError: 'add_bos'``.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SFT_DOC = REPO_ROOT / "docs/guides/sft.md"

# Top-level `data` keys read directly (data_config[...]) by setup_data and
# the SFT dataloader setup. Keep in sync with examples/run_sft.py and
# nemo_rl/algorithms/sft.py.
REQUIRED_TOP_LEVEL_KEYS = (
    "max_input_seq_length",
    "add_bos",
    "add_eos",
    "add_generation_prompt",
    "shuffle",
    "num_workers",
)


def _extract_response_dataset_template() -> dict:
    """Return the ResponseDataset `data:` block from docs/guides/sft.md.

    The block is identified by its `default.dataset_name`, not by position:
    the guide contains other `data:` blocks (e.g. the Megatron packed
    example), and a block containing a bare `...` line does not parse as a
    mapping at all, so position-based matching would silently test the
    wrong template.
    """
    blocks = re.findall(r"```yaml\n(.*?)```", SFT_DOC.read_text(), re.DOTALL)
    for block in blocks:
        # A bare `...` line ends the YAML document, so a template that still
        # uses it as a placeholder does not parse. Users copy the template
        # with the placeholder dropped, so parse it that way too.
        candidates = (
            block,
            "\n".join(line for line in block.splitlines() if line.strip() != "..."),
        )
        for candidate in candidates:
            try:
                parsed = yaml.safe_load(candidate)
            except yaml.YAMLError:
                continue
            if (
                isinstance(parsed, dict)
                and isinstance(parsed.get("data"), dict)
                and isinstance(parsed["data"].get("default"), dict)
                and parsed["data"]["default"].get("dataset_name") == "ResponseDataset"
            ):
                return parsed["data"]
    raise AssertionError(
        "ResponseDataset data template not found in docs/guides/sft.md"
    )


@pytest.fixture(scope="module")
def data_config() -> dict:
    return _extract_response_dataset_template()


def test_template_has_required_top_level_keys(data_config: dict) -> None:
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in data_config]
    assert not missing, (
        "SFT guide ResponseDataset template is missing required top-level "
        f"data keys: {missing} (see "
        "https://github.com/NVIDIA-NeMo/RL/issues/1548)"
    )


def test_template_top_level_key_types(data_config: dict) -> None:
    assert isinstance(data_config["add_bos"], bool)
    assert isinstance(data_config["add_eos"], bool)
    assert isinstance(data_config["add_generation_prompt"], bool)
    assert isinstance(data_config["shuffle"], bool)
    assert isinstance(data_config["num_workers"], int)


def test_template_has_dataset_level_fields(data_config: dict) -> None:
    default = data_config["default"]
    assert default.get("dataset_name") == "ResponseDataset"
    assert "input_key" in default
    assert "output_key" in default
    assert "processor" in default
    assert "data_path" in data_config["train"]
    assert "data_path" in data_config["validation"]
