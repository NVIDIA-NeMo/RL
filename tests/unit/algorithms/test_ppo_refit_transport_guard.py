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
"""PPO must reject ``refit_transport`` on every backend, not only on vLLM.

The rejection lived inside ``if generation_config["backend"] == "vllm":``, but
PPO also supports SGLang (``ppo.py`` asserts ``backend in ("vllm", "sglang")``),
so the same YAML that raises under GRPO ran to completion under PPO with the
transport silently dropped.

CPU-only: the guard is at the top of ``setup``, above any cluster or model
construction.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.ppo import setup

# Anything that is not the default collective path.
_TRANSPORTS = ["nixl", "nccl_reshard"]


def _master_config(backend: str, transport):
    generation = {"backend": backend}
    if transport is not None:
        generation["refit_transport"] = transport
    return SimpleNamespace(
        policy={"generation": generation},
        value={},
        env={},
        loss_fn=SimpleNamespace(),
        ppo=SimpleNamespace(),
        data={},
        logger={},
        cluster={},
    )


def _run(backend: str, transport):
    setup(
        master_config=_master_config(backend, transport),
        tokenizer=MagicMock(),
        dataset=MagicMock(),
        val_dataset=None,
    )


@pytest.mark.parametrize("transport", _TRANSPORTS)
def test_sglang_with_a_transport_is_rejected(transport):
    """This is the case that used to run silently."""
    with pytest.raises(ValueError, match="refit_transport"):
        _run("sglang", transport)


@pytest.mark.parametrize("transport", _TRANSPORTS)
def test_vllm_with_a_transport_is_still_rejected(transport):
    """The pre-existing rejection must survive the restructuring."""
    with pytest.raises(ValueError, match="refit_transport"):
        _run("vllm", transport)


def test_the_error_names_the_backend_that_was_configured():
    """A message that only says 'not supported by PPO' sends the reader to the
    vLLM branch, which is not where their config went wrong."""
    with pytest.raises(ValueError, match="sglang"):
        _run("sglang", "nixl")


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_no_transport_gets_past_the_guard(backend):
    """The supported pairing. Setup fails later on the mocks, which is what
    keeps this from passing vacuously if the guard were made unconditional."""
    with pytest.raises(Exception) as excinfo:
        _run(backend, None)
    assert "refit_transport" not in str(excinfo.value)
