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

"""Un-sharing the vLLM drafter lm_head before draft refit.

vLLM 0.26's spec-decode proposer aliases the drafter's ``lm_head`` module to
the target's when their weights are identical at engine init
(``SpecDecodeBaseProposer._maybe_share_lm_head``). With a co-training draft,
loading the trained (diverged) draft head through that alias would overwrite
the serving target's head. ``_unshare_draft_lm_head`` must detect the alias
and give the drafter private storage — and must not touch anything otherwise.
"""

import types

import pytest
import torch

pytestmark = pytest.mark.vllm

pytest.importorskip("vllm", reason="vllm_backend requires a vLLM install")

from nemo_rl.models.generation.vllm.vllm_backend import (  # noqa: E402
    VllmInternalWorkerExtension,
)


def _make_backend(target_lm_head):
    backend = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(
            model=types.SimpleNamespace(lm_head=target_lm_head)
        )
    )
    backend._unshare_draft_lm_head = types.MethodType(
        VllmInternalWorkerExtension._unshare_draft_lm_head, backend
    )
    return backend


def _lm_head(weight):
    head = torch.nn.Module()
    head.weight = torch.nn.Parameter(weight, requires_grad=False)
    return head


def test_aliased_lm_head_is_unshared_and_target_protected():
    target_head = _lm_head(torch.randn(16, 8))
    # vLLM attaches loader metadata to parameters via set_weight_attrs; the
    # private copy must keep it, or the next load_weights on the drafter fails.
    target_head.weight.weight_loader = lambda param, loaded: None
    draft_model = types.SimpleNamespace(lm_head=target_head)  # vLLM-style alias
    backend = _make_backend(target_head)

    backend._unshare_draft_lm_head(draft_model)

    assert draft_model.lm_head is not target_head
    assert (
        draft_model.lm_head.weight.untyped_storage().data_ptr()
        != target_head.weight.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(draft_model.lm_head.weight, target_head.weight)
    assert draft_model.lm_head.weight.weight_loader is target_head.weight.weight_loader

    # A subsequent draft-head update must leave the target untouched.
    before = target_head.weight.detach().clone()
    with torch.no_grad():
        draft_model.lm_head.weight += 1.0
    torch.testing.assert_close(target_head.weight.detach(), before)


def test_distinct_lm_head_is_left_alone():
    target_head = _lm_head(torch.randn(16, 8))
    own_head = _lm_head(torch.randn(16, 8))
    draft_model = types.SimpleNamespace(lm_head=own_head)
    backend = _make_backend(target_head)

    backend._unshare_draft_lm_head(draft_model)

    assert draft_model.lm_head is own_head  # no copy when not aliased


def test_missing_lm_head_is_a_noop():
    backend = _make_backend(target_lm_head=None)
    draft_model = types.SimpleNamespace()
    backend._unshare_draft_lm_head(draft_model)  # must not raise


def _make_refit_backend(model_runner):
    backend = types.SimpleNamespace(model_runner=model_runner)
    backend._get_drafter_model = types.MethodType(
        VllmInternalWorkerExtension._get_drafter_model, backend
    )
    backend._load_draft_weights = types.MethodType(
        VllmInternalWorkerExtension._load_draft_weights, backend
    )
    return backend


def test_speculator_attribute_is_a_drafter_fallback():
    # The V2 model runner names the owner `speculator` instead of `drafter`.
    model = types.SimpleNamespace()
    runner = types.SimpleNamespace(
        drafter=None, speculator=types.SimpleNamespace(model=model)
    )
    assert _make_refit_backend(runner)._get_drafter_model() is model


def test_missing_drafter_raises_on_last_pp_rank(monkeypatch):
    import nemo_rl.models.generation.vllm.vllm_backend as vllm_backend_module

    monkeypatch.setattr(
        vllm_backend_module,
        "get_pp_group",
        lambda: types.SimpleNamespace(is_last_rank=True),
    )
    backend = _make_refit_backend(types.SimpleNamespace(drafter=None))
    with pytest.raises(RuntimeError, match="no vLLM drafter"):
        backend._load_draft_weights([("lm_head.weight", torch.zeros(1))])


def test_missing_drafter_is_expected_on_earlier_pp_ranks(monkeypatch):
    import nemo_rl.models.generation.vllm.vllm_backend as vllm_backend_module

    monkeypatch.setattr(
        vllm_backend_module,
        "get_pp_group",
        lambda: types.SimpleNamespace(is_last_rank=False),
    )
    backend = _make_refit_backend(types.SimpleNamespace(drafter=None))
    backend._load_draft_weights([("lm_head.weight", torch.zeros(1))])  # must not raise
