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
"""Compatibility with the upstream remote-tokenizer fix in Transformers 5.15."""

from unittest.mock import Mock

import pytest
import transformers
from packaging.version import Version
from transformers import AutoTokenizer, PreTrainedConfig
from transformers.models.auto import tokenization_auto


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.15.0"),
    reason="The mcore environment uses the upstream Transformers 5.15 fix",
)
def test_blocklisted_tokenizer_honors_explicit_remote_code(monkeypatch):
    # Import the policy package to exercise its version-gated compatibility patch.
    import nemo_rl.models.policy  # noqa: F401

    class RemoteConfig(PreTrainedConfig):
        model_type = "deepseek_v3"

    tokenizer = object()
    remote_class = Mock()
    remote_class.from_pretrained.return_value = tokenizer
    monkeypatch.setattr(
        tokenization_auto,
        "get_tokenizer_config",
        lambda *args, **kwargs: {
            "tokenizer_class": "TestRemoteTokenizer",
            "auto_map": {"AutoTokenizer": ["remote.TestRemoteTokenizer", None]},
        },
    )
    loader = Mock(return_value=remote_class)
    monkeypatch.setattr(tokenization_auto, "get_class_from_dynamic_module", loader)
    blocklist = tokenization_auto.MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS
    assert "deepseek_v3" in blocklist
    result = AutoTokenizer.from_pretrained(
        "offline-test", config=RemoteConfig(), trust_remote_code=True
    )
    assert result is tokenizer
    loader.assert_called_once()
    assert "deepseek_v3" in blocklist
