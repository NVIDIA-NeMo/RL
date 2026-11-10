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

"""Native SpeechLM model through NeMo RL's unmodified Automodel dispatch.

Requires the policy environment (NeMo Speech and Automodel main).
"""

import json

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("nemo.collections.speechlm2")
pytest.importorskip("nemo_automodel.components.models.nemotron_v3.model")

from lightning import LightningModule  # noqa: E402
from nemo_automodel.components.models.common import BackendConfig  # noqa: E402
from nemo_automodel.components.models.nemotron_v3.model import (  # noqa: E402
    NemotronHForCausalLM,
)
from omegaconf import DictConfig  # noqa: E402
from speechlm_gspo.data import Nemotron3_5TranscribeProcessor  # noqa: E402
from speechlm_gspo.model import SpeechLMPolicy, register_speechlm  # noqa: E402
from tokenizers import Tokenizer, models  # noqa: E402
from transformers import (  # noqa: E402
    AutoProcessor,
    PretrainedConfig,
    PreTrainedTokenizerFast,
)

from nemo_rl.algorithms.logits_sampling_utils import (  # noqa: E402
    TrainingSamplingParams,
)

PAD, AUDIO = 0, 12
IDS = torch.tensor([[1, AUDIO, 2, 3, 4, PAD], [1, AUDIO, 2, 5, PAD, PAD]])
RESPONSE_MASK = torch.tensor([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 0]])
AUDIO_KWARGS = {
    "audio_samples": torch.randn(2, 6, generator=torch.Generator().manual_seed(0)),
    "audio_lengths": torch.tensor([[6], [4]]),
}


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
        return PAD

    @property
    def audio_locator_tag_id(self):
        return AUDIO

    def __init__(self):
        LightningModule.__init__(self)
        self.native_model = {
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
        ).float()
        self.llm.mtp.requires_grad_(False)
        self.llm.lm_head.requires_grad_(False)
        self.perception = Perception(config.hidden_size)


def _dense_response_logprobs(model, temperature):
    """Reference: project every expanded position, keep the response targets."""
    expanded = model.prepare_inputs(
        {
            "input_ids": IDS,
            "audios": AUDIO_KWARGS["audio_samples"],
            "audio_lens": AUDIO_KWARGS["audio_lengths"].flatten(),
            "loss_mask": RESPONSE_MASK.bool(),
        },
        include_mtp_inputs=False,
    )
    logits = model.llm(
        inputs_embeds=expanded["input_embeds"],
        attention_mask=expanded["attention_mask"],
        compute_mtp=False,
    ).logits
    values = -F.cross_entropy(
        (logits / temperature).flatten(0, 1),
        expanded["target_ids"].flatten(),
        reduction="none",
    ).reshape_as(expanded["target_ids"])
    expected = torch.zeros(IDS.shape)
    expected[:, 1:] = expected[:, 1:].masked_scatter(
        RESPONSE_MASK[:, 1:].bool(), values[expanded["target_ids"] != -100]
    )
    return expected


def test_auto_processor_resolves_checkpoint_processor_class(tmp_path):
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
    ).save_pretrained(tmp_path)
    # Class name recorded in the released checkpoint's processor_config.json.
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"processor_class": "Nemotron3_5TranscribeProcessor"})
    )
    register_speechlm()
    processor = AutoProcessor.from_pretrained(
        tmp_path, trust_remote_code=True, use_fast=True
    )
    assert type(processor) is Nemotron3_5TranscribeProcessor
    assert processor.tokenizer.unk_token == "<unk>"


def test_forward_returns_text_aligned_logits():
    torch.manual_seed(42)
    model = NativeFixture().train()
    logits = model(IDS, **AUDIO_KWARGS)
    assert logits.shape == (*IDS.shape, 13)
    assert torch.isneginf(logits[..., AUDIO]).all()
    assert model.llm.lm_head.weight.shape[0] == 12
    logprobs = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, IDS[:, 1:, None])
    actual = torch.zeros(IDS.shape)
    actual[:, 1:] = logprobs.squeeze(-1)
    response = RESPONSE_MASK.bool()
    torch.testing.assert_close(
        actual[response], _dense_response_logprobs(model, 1.0)[response]
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    reference_gradients = torch.autograd.grad(
        _dense_response_logprobs(model, 1.0)[response].sum(), parameters
    )
    actual[response].sum().backward()
    for parameter, reference in zip(parameters, reference_gradients):
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference)
    assert torch.count_nonzero(model.perception.proj.weight.grad) > 0
    assert model.llm.lm_head.weight.grad is None
    assert all(parameter.grad is None for parameter in model.llm.mtp.parameters())
    assert not model.perception.training


def test_forward_keeps_native_width_for_in_vocab_placeholder():
    class InVocabFixture(NativeFixture):
        @property
        def audio_locator_tag_id(self):
            return 11

    model = InVocabFixture()
    ids = IDS.masked_fill(IDS == AUDIO, 11)
    logits = model(ids, **AUDIO_KWARGS)
    assert logits.shape == (*IDS.shape, 12)
    assert torch.isfinite(logits).all()


def test_forward_rejects_expanded_overflow_and_kv_cache():
    model = NativeFixture()
    with pytest.raises(ValueError, match="KV cache"):
        model(IDS, **AUDIO_KWARGS, use_cache=True)
    model.native_model["max_expanded_sequence_length"] = 3
    with pytest.raises(ValueError, match="max_expanded_sequence_length"):
        model(IDS, **AUDIO_KWARGS)


@pytest.fixture
def singleton_tp_mesh(tmp_path):
    from torch.distributed.device_mesh import init_device_mesh

    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_standard_microbatch_dispatch_runs_gspo_loss_and_audio_gradient(
    chunk_size, singleton_tp_mesh
):
    from nemo_rl.algorithms.loss import ClippedPGLossFn
    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig
    from nemo_rl.algorithms.utils import compute_seq_logprob_errors
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
    processed = ProcessedInputs(
        input_ids=IDS,
        attention_mask=torch.ones_like(IDS, dtype=torch.bool),
        seq_len=IDS.shape[1],
        vlm_kwargs=dict(AUDIO_KWARGS),
    )
    data = BatchedDataDict(
        {
            "input_ids": IDS,
            "input_lengths": (IDS != PAD).sum(1),
            "token_mask": RESPONSE_MASK,
            "sample_mask": torch.ones(2),
        }
    )
    microbatch = ProcessedMicrobatch(data, processed, 2, IDS.shape[1])
    sampling = TrainingSamplingParams(temperature=0.8)
    prepared = prepare_model_forward(
        model,
        processed,
        device_mesh=None,
        cp_size=1,
        padding_token_id=PAD,
        is_reward_model=False,
        allow_flash_attn_args=False,
    )
    old, metrics, returned = forward_with_post_processing_fn(
        model=model,
        prepared=prepared,
        post_processing_fn=LogprobsPostProcessor(
            cfg={"logprob_chunk_size": chunk_size}, sampling_params=sampling
        ),
        processed_mb=microbatch,
        global_valid_seqs=None,
        global_valid_toks=None,
        sampling_params=sampling,
    )
    assert returned is microbatch
    assert old.shape == IDS.shape
    assert torch.count_nonzero(old[:, 0]) == 0
    assert torch.isneginf(old[IDS == AUDIO]).all()
    response = RESPONSE_MASK.bool()
    with torch.no_grad():
        expected = _dense_response_logprobs(model, sampling.temperature)
    torch.testing.assert_close(old[response], expected[response])

    data["prev_logprobs"] = old.detach()
    data["generation_logprobs"] = torch.where(response, old.detach(), 0.0)
    errors, valid = compute_seq_logprob_errors(
        policy_logprobs=old[:, 1:],
        generation_logprobs=data["generation_logprobs"][:, 1:],
        token_mask=RESPONSE_MASK[:, 1:],
        sample_mask=data["sample_mask"],
    )
    assert valid.all()
    torch.testing.assert_close(errors, torch.ones(2))
    data["advantages"] = torch.tensor([[1.0], [-0.5]]).expand_as(IDS)
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            sequence_level_importance_ratios=True,
            token_level_loss=False,
            reference_policy_kl_penalty=0.0,
            ratio_clip_min=0.2,
            ratio_clip_max=0.2,
            metrics_level="full",
        )
    )
    loss, metrics, returned = forward_with_post_processing_fn(
        model=model,
        prepared=prepared,
        post_processing_fn=LossPostProcessor(
            loss_fn=loss_fn,
            cfg={"logprob_chunk_size": chunk_size},
            cp_mesh=None,
            cp_size=1,
            dp_size=1,
            sampling_params=sampling,
            tp_mesh=singleton_tp_mesh,
        ),
        processed_mb=microbatch,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=RESPONSE_MASK.sum(),
        sampling_params=sampling,
    )
    # Ratios are one, so the GSPO sequence mean is -(1 - .5) / 2.
    torch.testing.assert_close(loss, torch.tensor(-0.25))
    assert returned is microbatch
    assert all(
        torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values()
    )
    reference_logprobs = _dense_response_logprobs(model, sampling.temperature)
    masked_logratios = (reference_logprobs - expected) * RESPONSE_MASK
    sequence_logratios = masked_logratios.sum(1) / RESPONSE_MASK.sum(1)
    reference_loss = -(sequence_logratios.exp() * torch.tensor([1.0, -0.5])).mean()
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    reference_gradients = torch.autograd.grad(reference_loss, parameters)
    loss.backward()
    for parameter, reference in zip(parameters, reference_gradients):
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference)
    assert torch.count_nonzero(model.perception.proj.weight.grad) > 0
    assert model.llm.lm_head.weight.grad is None
    assert all(parameter.grad is None for parameter in model.llm.mtp.parameters())
