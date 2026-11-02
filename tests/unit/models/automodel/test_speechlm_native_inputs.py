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

"""Native public Speech input expansion through sampled-token projection."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("nemo.collections.speechlm2")
pytest.importorskip("nemo_automodel.components.utils.model_utils")

pytestmark = pytest.mark.automodel

from lightning import LightningModule  # noqa: E402
from nemo_automodel.components.models.common import BackendConfig  # noqa: E402
from nemo_automodel.components.models.nemotron_v3.model import (  # noqa: E402
    NemotronHForCausalLM,
)
from omegaconf import DictConfig  # noqa: E402
from transformers import PretrainedConfig  # noqa: E402

from nemo_rl.algorithms.logits_sampling_utils import (  # noqa: E402
    TrainingSamplingParams,
)
from nemo_rl.models.automodel.speechlm import SpeechLMPolicy  # noqa: E402


class Perception(torch.nn.Module):
    """Small learned audio encoder fixture with unequal audio-token lengths."""

    def __init__(self, hidden):
        super().__init__()
        self.proj = torch.nn.Linear(1, hidden)
        self.encoder = torch.nn.Identity()

    def forward(self, input_signal, input_signal_length):
        return self.proj(input_signal[:, :3, None]), torch.div(
            input_signal_length, 2, rounding_mode="floor"
        )


class NativeFixture(SpeechLMPolicy):
    @property
    def sampling_rate(self):
        return 16000

    @property
    def text_pad_id(self):
        return 0

    @property
    def audio_locator_tag_id(self):
        return 99

    def __init__(self):
        LightningModule.__init__(self)
        self.native_model = {
            "projection_chunk_size": 2,
            "max_expanded_sequence_length": 32,
            "deterministic_perception": True,
        }
        self.cfg = DictConfig({"packed_sequences": False})
        self._use_tp = False
        self._device_mesh = None
        config = PretrainedConfig(
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            hidden_size=8,
            attention_bias=False,
            attention_dropout=0.0,
            intermediate_size=16,
            mlp_bias=False,
            mlp_hidden_act="relu2",
            layer_norm_epsilon=1e-5,
            num_hidden_layers=2,
            vocab_size=12,
            torch_dtype="float32",
            initializer_range=0.02,
            rescale_prenorm_residual=True,
            residual_in_fp32=False,
            layers_block_type=["attention", "mlp"],
            n_routed_experts=2,
            num_experts_per_tok=1,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            moe_intermediate_size=8,
            norm_topk_prob=False,
            moe_shared_expert_intermediate_size=8,
            num_nextn_predict_layers=1,
            mtp_hybrid_override_pattern="*E",
        )
        backend = BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            dispatcher="torch",
            enable_hf_state_dict_adapter=False,
        )
        self.llm = NemotronHForCausalLM(
            config,
            backend=backend,
            num_nextn_predict_layers=2,
            mtp_use_repeated_layer=True,
        ).float()  # Older native constructor defaults linear storage to BF16.
        self.llm.mtp.requires_grad_(False)
        self.llm.lm_head.requires_grad_(False)
        self.perception = Perception(config.hidden_size)


def test_native_audio_expansion_projection_and_audio_gradient():
    torch.manual_seed(42)
    model = NativeFixture()
    model.train()
    batch = {
        "input_ids": torch.tensor([[1, 99, 2, 3, 4, 0], [1, 99, 2, 5, 0, 0]]),
        "audio_samples": torch.randn(2, 6),
        "audio_lengths": torch.tensor([[6], [4]]),
    }
    mask = torch.tensor([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 0]])
    expanded = model.prepare_inputs(
        {
            "input_ids": batch["input_ids"],
            "audios": batch["audio_samples"],
            "audio_lens": batch["audio_lengths"].flatten(),
            "loss_mask": mask.bool(),
        },
        include_mtp_inputs=False,
    )
    assert expanded["input_embeds"].shape == (2, 7, 8)
    assert expanded["target_ids"].shape == (2, 7)
    dense = model.llm(
        inputs_embeds=expanded["input_embeds"],
        attention_mask=expanded["attention_mask"],
        compute_mtp=False,
    ).logits
    values = -F.cross_entropy(
        (dense / 0.8).flatten(0, 1), expanded["target_ids"].flatten(), reduction="none"
    ).reshape_as(expanded["target_ids"])
    expected = torch.zeros((2, 5)).masked_scatter(
        mask[:, 1:].bool(), values[expanded["target_ids"] != -100]
    )
    actual = model.next_token_logprobs(
        batch, mask, TrainingSamplingParams(temperature=0.8)
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert actual.shape == (2, 5)
    assert torch.count_nonzero(actual[~mask[:, 1:].bool()]) == 0
    assert torch.count_nonzero(model.perception.proj.weight.grad) > 0
    assert model.llm.lm_head.weight.grad is None
    assert not model.perception.training
    assert model.llm.mtp_config.num_layers == 2
    assert model.llm.mtp_config.use_repeated_layer is True
    assert all(parameter.grad is None for parameter in model.llm.mtp.parameters())


def test_native_projection_rejects_filtered_sampling_and_expanded_overflow():
    model = NativeFixture()
    batch = {
        "input_ids": torch.tensor([[1, 99, 2, 3]]),
        "audio_samples": torch.randn(1, 6),
        "audio_lengths": torch.tensor([6]),
    }
    mask = torch.tensor([[0, 0, 0, 1]])
    with pytest.raises(ValueError, match="top_p"):
        model.next_token_logprobs(batch, mask, TrainingSamplingParams(top_p=0.9))
    model.native_model["max_expanded_sequence_length"] = 3
    with pytest.raises(ValueError, match="context contract"):
        model.next_token_logprobs(batch, mask)


def test_standard_microbatch_dispatch_runs_gspo_loss_and_audio_gradient():
    from nemo_rl.algorithms.loss import ClippedPGLossFn
    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch
    from nemo_rl.models.automodel.train import (
        LogprobsPostProcessor,
        LossPostProcessor,
        forward_with_post_processing_fn,
        prepare_model_forward,
    )

    torch.manual_seed(42)
    model = NativeFixture().train()
    # NativeFixture bypasses production construction; use the metadata that the
    # public wrapper supplies to the standard worker dispatcher.
    model.config = SimpleNamespace(model_type="nemo_speechlm")
    ids = torch.tensor([[1, 99, 2, 3, 4, 0], [1, 99, 2, 5, 0, 0]])
    mask = torch.tensor([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 0]])
    processed = ProcessedInputs(
        input_ids=ids,
        seq_len=6,
        vlm_kwargs={
            "audio_samples": torch.randn(2, 6),
            "audio_lengths": torch.tensor([[6], [4]]),
        },
    )
    data = BatchedDataDict(
        {"input_ids": ids, "token_mask": mask, "sample_mask": torch.ones(2)}
    )
    microbatch = ProcessedMicrobatch(data, processed, 2, 6)
    sampling = TrainingSamplingParams(temperature=0.8)
    prepared = prepare_model_forward(
        model,
        processed,
        device_mesh=None,
        cp_size=1,
        padding_token_id=0,
        is_reward_model=False,
        allow_flash_attn_args=False,
    )
    old, metrics, returned = forward_with_post_processing_fn(
        model=model,
        prepared=prepared,
        post_processing_fn=LogprobsPostProcessor(
            cfg={},
            sampling_params=sampling,
        ),
        processed_mb=microbatch,
        global_valid_seqs=None,
        global_valid_toks=None,
        sampling_params=sampling,
    )
    assert returned is microbatch
    torch.testing.assert_close(metrics["logprobs"], old)
    assert old.shape == ids.shape
    assert torch.count_nonzero(old[:, 0]) == 0
    assert torch.count_nonzero(old[~mask.bool()]) == 0
    with torch.no_grad():
        expanded = model.prepare_inputs(
            {
                "input_ids": ids,
                "audios": processed.vlm_kwargs["audio_samples"],
                "audio_lens": processed.vlm_kwargs["audio_lengths"].flatten(),
                "loss_mask": mask.bool(),
            },
            include_mtp_inputs=False,
        )
        dense = model.llm(
            inputs_embeds=expanded["input_embeds"],
            attention_mask=expanded["attention_mask"],
            compute_mtp=False,
        ).logits
        values = -F.cross_entropy(
            (dense / sampling.temperature).flatten(0, 1),
            expanded["target_ids"].flatten(),
            reduction="none",
        ).reshape_as(expanded["target_ids"])
        expected = torch.zeros_like(ids, dtype=values.dtype)
        expected[:, 1:] = torch.zeros_like(old[:, 1:]).masked_scatter(
            mask[:, 1:].bool(), values[expanded["target_ids"] != -100]
        )
    torch.testing.assert_close(old, expected)
    data["prev_logprobs"] = old.detach()
    data["generation_logprobs"] = data["prev_logprobs"].clone()
    data["advantages"] = torch.tensor([[1.0], [-0.5]]).expand_as(ids)
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            sequence_level_importance_ratios=True,
            token_level_loss=False,
            reference_policy_kl_penalty=0.0,
            ratio_clip_min=0.2,
            ratio_clip_max=0.2,
        )
    )
    loss, _, returned = forward_with_post_processing_fn(
        model=model,
        prepared=prepared,
        post_processing_fn=LossPostProcessor(
            loss_fn=loss_fn,
            cfg={},
            cp_mesh=None,
            cp_size=1,
            dp_size=1,
            sampling_params=sampling,
        ),
        processed_mb=microbatch,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=mask.sum(),
        sampling_params=sampling,
    )
    # Ratios are one, so the GSPO sequence mean is -(1 - .5) / 2.
    torch.testing.assert_close(loss, torch.tensor(-0.25))
    assert returned is microbatch
    loss.backward()
    assert torch.count_nonzero(model.perception.proj.weight.grad) > 0
    assert model.llm.lm_head.weight.grad is None
    assert all(parameter.grad is None for parameter in model.llm.mtp.parameters())
