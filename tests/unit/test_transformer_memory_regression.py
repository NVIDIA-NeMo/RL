# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import pytest
from packaging import version


def test_transformers_version_memory_regression():
    """
    Test that transformers version is within the safe range [4.54, 4.56).

    This test exists because of a memory regression in transformers>=4.54,<4.56
    where KV cache is incorrectly treated as trainable, causing significant memory
    pressure with higher TP settings and long sequence lengths.

    If this test fails, it means:
    - Either transformers has been upgraded to >=4.56 (good!)
    - Or downgraded to <4.54 (unexpected)

    In either case, you should:
    1. Remove this test file (tests/unit/test_transformer_memory_regression.py)
    2. Reinstate the nightly test: tests/test_suites/llm/dpo-mistral-nemo-instruct-2407-1n8g-fsdp2tp8-actckpt-long.sh
    3. Update the GitHub issue: https://github.com/NVIDIA-NeMo/RL/issues/1343

    Related upstream issue: https://github.com/huggingface/transformers/issues/39795
    """
    import transformers

    transformers_version = version.parse(transformers.__version__)

    # Expected range: >= 4.54 and < 4.56
    min_version = version.parse("4.54.0")
    max_version = version.parse("4.56.0")

    is_in_expected_range = min_version <= transformers_version < max_version

    if not is_in_expected_range:
        error_message = (
            f"\n{'=' * 80}\n"
            f"Transformers version {transformers.__version__} is OUTSIDE the expected range [4.54, 4.56).\n"
            f"\n"
            f"This is GOOD NEWS if you've upgraded to >=4.56 (memory regression is fixed)!\n"
            f"\n"
            f"ACTION REQUIRED:\n"
            f"1. Remove this test file: tests/unit/test_transformer_memory_regression.py\n"
            f"2. Reinstate the nightly test that was disabled:\n"
            f"   tests/test_suites/llm/dpo-mistral-nemo-instruct-2407-1n8g-fsdp2tp8-actckpt-long.sh\n"
            f"3. Update and close GitHub issue: https://github.com/NVIDIA-NeMo/RL/issues/1343\n"
            f"\n"
            f"Background: transformers [4.54, 4.56) had a memory regression where KV cache\n"
            f"was incorrectly treated as trainable, causing OOMs with high TP and long sequences.\n"
            f"See: https://github.com/huggingface/transformers/issues/39795\n"
            f"{'=' * 80}\n"
        )
        pytest.fail(error_message)

    # If we're in the expected range, the test passes silently
    assert is_in_expected_range, "Transformers version should be in range [4.54, 4.56)"
