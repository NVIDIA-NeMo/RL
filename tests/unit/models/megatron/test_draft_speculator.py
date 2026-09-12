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

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.models.policy.draft_config import DFlashDraftConfig, Eagle3DraftConfig

pytestmark = pytest.mark.mcore


def test_disabled_resolver_import_does_not_require_megatron_runtime() -> None:
    """Reading disabled draft config must not import Megatron Bridge."""
    program = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "megatron.bridge" or name.startswith("megatron.bridge."):
        raise ModuleNotFoundError(f"blocked test import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import

from nemo_rl.models.megatron.draft.training import resolve_draft_speculator
from nemo_rl.models.policy.draft_config import Eagle3DraftConfig

assert resolve_draft_speculator(Eagle3DraftConfig(enabled=False)) is None
"""

    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_disabled_draft_config_resolves_to_none() -> None:
    """Disabled and absent draft settings do not create a speculator."""
    from nemo_rl.models.megatron.draft.training import resolve_draft_speculator

    assert resolve_draft_speculator(Eagle3DraftConfig(enabled=False)) is None
    assert resolve_draft_speculator(None) is None


@patch("nemo_rl.models.megatron.draft.training.build_draft_model")
def test_eagle3_speculator_delegates_existing_model_build(
    mock_build: MagicMock,
) -> None:
    """The enabled EAGLE-3 speculator uses the existing draft model builder."""
    from nemo_rl.models.megatron.draft.training import resolve_draft_speculator

    expected = MagicMock()
    mock_build.return_value = expected
    config = Eagle3DraftConfig(enabled=True, model_name="draft")
    speculator = resolve_draft_speculator(config)
    assert speculator is not None
    model_provider = SimpleNamespace()
    pg_collection = MagicMock()
    policy_model_chunk = MagicMock()

    result = speculator.build_model(
        model_provider=model_provider,
        pg_collection=pg_collection,
        policy_model_chunk=policy_model_chunk,
    )

    assert result is expected
    mock_build.assert_called_once_with(
        model_provider=model_provider,
        draft_config=config,
        pg_collection=pg_collection,
        policy_model_chunk=policy_model_chunk,
    )


def test_dflash_provider_builds_body_from_target_dimensions() -> None:
    from nemo_rl.models.megatron.draft.dflash import DFlashBody
    from nemo_rl.models.megatron.draft.training import resolve_draft_speculator

    config = DFlashDraftConfig(
        enabled=True,
        gamma=3,
        anchors_per_sample=2,
        mask_token_id=99,
        target_hidden_state_layer_ids=[1, 3],
        num_layers=2,
    )
    provider = SimpleNamespace(
        hidden_size=32,
        ffn_hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=8,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        num_layers=4,
        use_cpu_initialization=True,
        fp16=False,
        bf16=False,
        params_dtype=__import__("torch").float32,
        init_method_std=0.02,
        layernorm_epsilon=1e-6,
        rotary_base=1_000_000.0,
    )
    speculator = resolve_draft_speculator(config)
    assert speculator is not None

    body = speculator.build_model(
        model_provider=provider,
        pg_collection=SimpleNamespace(tp=None),
        policy_model_chunk=MagicMock(),
    )

    assert isinstance(body, DFlashBody)
    assert body.config.hidden_size == 32
    assert body.config.num_target_taps == 2
    assert body.config.num_hidden_layers == 2
