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
import types

import pytest
import torch
from gdpo import SdmcElboEstimator, SdmcLikelihoodConfig
from gdpo.worker import DTensorGDPOPolicyWorker

# The worker is a Ray actor class; the plain class carrying the methods sits
# behind __ray_metadata__ and is what we bind the method under test from.
_WORKER_CLS = DTensorGDPOPolicyWorker.__ray_metadata__.modified_class

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
