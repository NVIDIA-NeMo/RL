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

"""Tests that the worker scores each quadrature point against its own mask.

These run on CPU with no distributed backend: the model forward is stubbed so
the test can observe exactly what each quadrature point was handed.
"""

import contextlib
import math
import types

import pytest
import torch
from gdpo import SdmcElboEstimator, SdmcLikelihoodConfig
from gdpo.worker import AutomodelGDPOPolicyWorker

from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch
from nemo_rl.models.automodel.train import LogprobsPostProcessor

# The worker is a Ray actor class; the plain class carrying the methods sits
# behind __ray_metadata__ and is what we bind the method under test from.
_WORKER_CLS = AutomodelGDPOPolicyWorker.__ray_metadata__.modified_class

MASK_ID = 99
PAD_ID = 0


class FakeTokenizer:
    pad_token_id = PAD_ID


class FakeProcessedInputs:
    def __init__(self, input_ids):
        self.input_ids = input_ids
        self.target_ids = None


class FakeMicrobatch:
    def __init__(self, input_ids, token_mask):
        self.processed_inputs = FakeProcessedInputs(input_ids)
        self.data_dict = {"token_mask": token_mask}


def make_worker(quadrature="gauss-3", mc_samples=1):
    """A stub carrying only what _gdpo_elbo_logprobs actually touches."""
    worker = types.SimpleNamespace(
        model=object(),
        device_mesh=None,
        cp_size=1,
        tokenizer=FakeTokenizer(),
        allow_flash_attn_args=False,
        sampling_params=None,
        elbo_estimator=SdmcElboEstimator(
            SdmcLikelihoodConfig(quadrature=quadrature, mc_samples=mc_samples),
            MASK_ID,
        ),
        _autocast_context=lambda: contextlib.nullcontext(),
    )
    worker._gdpo_elbo_logprobs = types.MethodType(
        _WORKER_CLS._gdpo_elbo_logprobs, worker
    )
    return worker


@pytest.fixture
def batch():
    # Row layout: two prompt tokens then four scorable completion tokens.
    input_ids = torch.tensor([[5, 6, 11, 12, 13, 14], [7, 8, 21, 22, 23, 24]])
    token_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]], dtype=torch.bool
    )
    return FakeMicrobatch(input_ids, token_mask)


@pytest.fixture
def spy(monkeypatch):
    """Record the input_ids each forward is prepared with."""
    seen = []

    def fake_prepare(model, processed_inputs, **kwargs):
        seen.append(processed_inputs.input_ids.clone())
        return types.SimpleNamespace(
            model_context_factory=lambda: contextlib.nullcontext()
        )

    def fake_forward(**kwargs):
        mb = kwargs["processed_mb"]
        return (
            torch.zeros_like(mb.processed_inputs.input_ids, dtype=torch.float32),
            {},
            mb,
        )

    monkeypatch.setattr("gdpo.worker.prepare_model_forward", fake_prepare)
    monkeypatch.setattr("gdpo.worker.forward_with_post_processing_fn", fake_forward)
    return seen


class TestQuadratureIsolation:
    def test_one_forward_is_prepared_per_quadrature_point(self, batch, spy):
        make_worker("gauss-3")._gdpo_elbo_logprobs(
            processed_mb=batch, post_processing_fn=None, sequence_dim=1
        )
        assert len(spy) == 3

    def test_each_point_is_prepared_against_its_own_masked_view(self, batch, spy):
        """Each point must see its own mask.

        Hoisting prepare_model_forward out of score_fn would snapshot the clean
        sequence for every point.
        """
        make_worker("gauss-3")._gdpo_elbo_logprobs(
            processed_mb=batch, post_processing_fn=None, sequence_dim=1
        )
        clean = torch.tensor([[5, 6, 11, 12, 13, 14], [7, 8, 21, 22, 23, 24]])
        for prepared_ids in spy:
            assert not torch.equal(prepared_ids, clean), (
                "a quadrature point was prepared against the unmasked sequence"
            )
            assert (prepared_ids == MASK_ID).any()

    def test_prompt_positions_are_never_masked(self, batch, spy):
        make_worker("gauss-3")._gdpo_elbo_logprobs(
            processed_mb=batch, post_processing_fn=None, sequence_dim=1
        )
        for prepared_ids in spy:
            assert (prepared_ids[:, :2] != MASK_ID).all()

    def test_more_points_prepare_more_forwards(self, batch, spy):
        make_worker("gauss-5")._gdpo_elbo_logprobs(
            processed_mb=batch, post_processing_fn=None, sequence_dim=1
        )
        assert len(spy) == 5

    def test_mc_samples_multiply_the_forward_count(self, batch, spy):
        make_worker("gauss-2", mc_samples=3)._gdpo_elbo_logprobs(
            processed_mb=batch, post_processing_fn=None, sequence_dim=1
        )
        assert len(spy) == 6


class TestInputRestoration:
    def test_the_microbatch_is_left_clean_afterwards(self, batch, spy):
        before = batch.processed_inputs.input_ids.clone()
        make_worker()._gdpo_elbo_logprobs(
            processed_mb=batch, post_processing_fn=None, sequence_dim=1
        )
        assert torch.equal(batch.processed_inputs.input_ids, before)
        assert batch.processed_inputs.target_ids is None

    def test_inputs_are_restored_even_when_the_forward_raises(self, batch, monkeypatch):
        before = batch.processed_inputs.input_ids.clone()

        monkeypatch.setattr(
            "gdpo.worker.prepare_model_forward",
            lambda *a, **k: types.SimpleNamespace(
                model_context_factory=lambda: contextlib.nullcontext()
            ),
        )

        def boom(**kwargs):
            raise RuntimeError("forward exploded")

        monkeypatch.setattr("gdpo.worker.forward_with_post_processing_fn", boom)
        with pytest.raises(RuntimeError, match="forward exploded"):
            make_worker()._gdpo_elbo_logprobs(
                processed_mb=batch, post_processing_fn=None, sequence_dim=1
            )
        assert torch.equal(batch.processed_inputs.input_ids, before)
        assert batch.processed_inputs.target_ids is None


class TestOutput:
    def test_the_result_keeps_the_batch_shape(self, batch, spy):
        out = make_worker()._gdpo_elbo_logprobs(
            processed_mb=batch, post_processing_fn=None, sequence_dim=1
        )
        assert out.shape == batch.processed_inputs.input_ids.shape

    def test_real_automodel_scoring_preserves_clean_targets_and_gradients(self, batch):
        class UniformDiffusionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(100))

            def forward(self, input_ids, attention_mask=None, use_cache=False):
                return types.SimpleNamespace(
                    logits=self.logits.expand(*input_ids.shape, -1)
                )

        input_ids = batch.processed_inputs.input_ids
        microbatch = ProcessedMicrobatch(
            processed_inputs=ProcessedInputs(
                input_ids=input_ids,
                seq_len=input_ids.shape[1],
                position_ids=torch.arange(input_ids.shape[1]).expand_as(input_ids),
            ),
            data_dict={
                **batch.data_dict,
                "input_lengths": torch.full((input_ids.shape[0],), input_ids.shape[1]),
            },
            original_batch_size=input_ids.shape[0],
            original_seq_len=input_ids.shape[1],
        )
        worker = make_worker("gauss-1")
        worker.model = UniformDiffusionModel()

        result = worker._gdpo_elbo_logprobs(
            processed_mb=microbatch,
            post_processing_fn=LogprobsPostProcessor(cfg={}, shift_targets=False),
            sequence_dim=1,
        )

        # At t=1/2 exactly two of four completion tokens are scored, each
        # weighted by 1/t. A uniform 100-token model gives -4*log(100).
        torch.testing.assert_close(
            result.sum(dim=1), torch.full((input_ids.shape[0],), -4 * math.log(100))
        )
        result.sum().backward()
        assert torch.isfinite(worker.model.logits.grad).all()
        assert worker.model.logits.grad[MASK_ID] < 0
        assert worker.model.logits.grad[input_ids[:, 2:]].sum() > 0
        assert microbatch.processed_inputs.input_ids is input_ids
