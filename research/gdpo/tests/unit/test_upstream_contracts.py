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
"""Pin the GDPO worker to the NeMo RL interfaces it overrides and calls.

The other unit tests stub these seams out, so a signature change upstream
would pass them and only fail inside a GPU training step. These tests check
the GDPO code against the *real* upstream signatures on CPU instead.
"""

import inspect
import types

import torch
from gdpo import train_gdpo
from gdpo.worker import DTensorGDPOPolicyWorker

from nemo_rl.models.automodel.train import (
    LogprobsPostProcessor,
    LossPostProcessor,
    automodel_forward_backward,
)

_WORKER_CLS = DTensorGDPOPolicyWorker.__ray_metadata__.modified_class


def test_forward_backward_reads_only_what_the_base_worker_passes(monkeypatch):
    # The base worker's default _forward_backward is automodel_forward_backward
    # called with **kwargs, so its parameters are exactly what train() passes.
    passed = {
        name: object()
        for name in inspect.signature(automodel_forward_backward).parameters
    }
    captured = {}
    monkeypatch.setattr(
        "gdpo.worker.gdpo_forward_backward", lambda **kwargs: captured.update(kwargs)
    )
    worker = types.SimpleNamespace(_gdpo_train_elbo_scorer=lambda sequence_dim: None)

    _WORKER_CLS._forward_backward(worker, **passed)

    assert captured["data_iterator"] is passed["data_iterator"]
    assert captured["post_processing_fn"] is passed["post_processing_fn"]


def test_train_scorer_builds_the_logprobs_post_processor_upstream_accepts():
    built = []

    def make_logprobs_post_processor(**kwargs):
        # The real constructor raises TypeError on renamed or removed kwargs.
        built.append(LogprobsPostProcessor(**kwargs, shift_targets=False))
        return built[-1]

    worker = types.SimpleNamespace(
        cfg={},
        enable_seq_packing=False,
        sampling_params=None,
        _make_logprobs_post_processor=make_logprobs_post_processor,
    )

    _WORKER_CLS._gdpo_train_elbo_scorer(worker, sequence_dim=1)

    assert len(built) == 1


def test_training_loop_calls_the_loss_with_its_current_signature():
    loss_signature = inspect.signature(LossPostProcessor.__call__)
    calls = []

    def loss_fn(*args, **kwargs):
        # Raises TypeError for a missing required or unexpected argument.
        loss_signature.bind(None, *args, **kwargs)
        calls.append(kwargs)
        return kwargs["logits"].sum(), {"loss": 1.0}

    weight = torch.ones(2, 3, requires_grad=True)
    microbatch = types.SimpleNamespace(processed_inputs=object(), data_dict={})

    results = train_gdpo.gdpo_forward_backward(
        data_iterator=iter([microbatch]),
        post_processing_fn=loss_fn,
        elbo_scorer=lambda mb: weight * 2.0,
        forward_only=False,
        global_valid_seqs=torch.tensor(2),
        global_valid_toks=torch.tensor(6),
        sequence_dim=1,
        dp_size=1,
        cp_size=1,
        num_global_batches=1,
        num_valid_microbatches=1,
        on_microbatch_start=None,
    )

    assert len(calls) == 1 and len(results) == 1
    assert torch.equal(weight.grad, torch.full((2, 3), 2.0))
