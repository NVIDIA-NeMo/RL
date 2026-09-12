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

from nemo_rl.models.policy.draft_config import Eagle3DraftConfig

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
    assert resolve_draft_speculator({"enabled": False}) is None


def test_raw_mapping_draft_config_resolves_to_speculator() -> None:
    """A hand-built PolicyConfig carries a plain dict under "draft".

    Only ``Policy.__init__`` normalizes that entry into a model, so setup paths
    reached without it would otherwise hit ``AttributeError: 'dict' object has
    no attribute 'enabled'``.
    """
    from nemo_rl.models.megatron.draft.training import resolve_draft_speculator

    speculator = resolve_draft_speculator({"enabled": True, "model_name": "draft"})
    assert speculator is not None
    assert isinstance(speculator.config, Eagle3DraftConfig)
    assert speculator.config.model_name == "draft"


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
