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
"""DraftRuntimeLossWrapper: loss composition and policy-gradient isolation."""

import torch

from nemo_rl.algorithms.loss import DraftRuntimeLossWrapper


class _FakeDraftRuntime:
    def __init__(self, draft_loss: torch.Tensor, loss_weight: float):
        self._draft_loss = draft_loss
        self.loss_weight = loss_weight
        self.seen_prepared_data = None

    def compute_loss(self, prepared_data):
        self.seen_prepared_data = prepared_data
        return self._draft_loss, {"draft_loss": self._draft_loss.detach().clone()}


def _fake_prepare_fn(next_token_logits, data, loss_fn):
    return {"logprobs": next_token_logits}, data


def _make_policy_loss_fn(policy_param: torch.Tensor):
    def loss_fn(*, data, global_valid_seqs, global_valid_toks, logprobs):
        loss = (logprobs.sum() * policy_param).sum()
        return loss, {"policy_loss": loss.detach().clone()}

    return loss_fn


def test_combined_loss_composition_and_metrics_merge():
    policy_param = torch.tensor(2.0, requires_grad=True)
    draft_param = torch.tensor(3.0, requires_grad=True)
    draft_loss = draft_param * 5.0
    runtime = _FakeDraftRuntime(draft_loss, loss_weight=0.5)

    wrapper = DraftRuntimeLossWrapper(
        loss_fn=_make_policy_loss_fn(policy_param),
        prepare_fn=_fake_prepare_fn,
        draft_runtime=runtime,
    )
    logits = torch.ones(2, 3)
    combined, metrics = wrapper(logits, {"x": 1}, None, None)

    expected_policy = logits.sum() * 2.0
    assert torch.isclose(combined, expected_policy + 0.5 * draft_loss)
    assert "policy_loss" in metrics and "draft_loss" in metrics
    assert runtime.seen_prepared_data == {"x": 1}


def test_zero_loss_weight_isolates_policy_gradients():
    logits = torch.ones(2, 3)

    # Reference: policy-only gradient.
    ref_param = torch.tensor(2.0, requires_grad=True)
    ref_loss, _ = _make_policy_loss_fn(ref_param)(
        data=None, global_valid_seqs=None, global_valid_toks=None, logprobs=logits
    )
    ref_loss.backward()

    policy_param = torch.tensor(2.0, requires_grad=True)
    draft_param = torch.tensor(3.0, requires_grad=True)
    runtime = _FakeDraftRuntime(draft_param * 5.0, loss_weight=0.0)
    wrapper = DraftRuntimeLossWrapper(
        loss_fn=_make_policy_loss_fn(policy_param),
        prepare_fn=_fake_prepare_fn,
        draft_runtime=runtime,
    )
    combined, _ = wrapper(logits, {}, None, None)
    combined.backward()

    assert torch.equal(policy_param.grad, ref_param.grad)


def test_draft_gradients_flow_only_through_draft_term():
    policy_param = torch.tensor(2.0, requires_grad=True)
    draft_param = torch.tensor(3.0, requires_grad=True)
    runtime = _FakeDraftRuntime(draft_param * 5.0, loss_weight=0.25)
    wrapper = DraftRuntimeLossWrapper(
        loss_fn=_make_policy_loss_fn(policy_param),
        prepare_fn=_fake_prepare_fn,
        draft_runtime=runtime,
    )
    combined, _ = wrapper(torch.ones(2, 3), {}, None, None)
    combined.backward()

    assert torch.isclose(draft_param.grad, torch.tensor(0.25 * 5.0))
    # The policy gradient must be independent of the draft term entirely.
    assert torch.isclose(policy_param.grad, torch.tensor(6.0))


def test_draft_loss_scale_multiplies_draft_term_only():
    """Under context parallelism the trainer backprops
    (dp*cp/cp_gradient_fanout) * combined_loss; the draft term is replicated
    across CP ranks and needs the full dp*cp multiplier, so the wrapper
    rescales it by cp_gradient_fanout (no-op at cp=1)."""
    policy_param = torch.tensor(2.0, requires_grad=True)
    draft_param = torch.tensor(3.0, requires_grad=True)
    runtime = _FakeDraftRuntime(draft_param * 5.0, loss_weight=0.5)
    wrapper = DraftRuntimeLossWrapper(
        loss_fn=_make_policy_loss_fn(policy_param),
        prepare_fn=_fake_prepare_fn,
        draft_runtime=runtime,
        draft_loss_scale=4.0,
    )
    logits = torch.ones(2, 3)
    combined, _ = wrapper(logits, {}, None, None)
    expected_policy = logits.sum() * 2.0
    assert torch.isclose(combined, expected_policy + 0.5 * 4.0 * (3.0 * 5.0))
