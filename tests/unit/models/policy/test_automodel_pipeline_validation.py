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

"""Driver-side validation of Automodel pipeline-parallel configs."""

from types import SimpleNamespace
from typing import Any

import pydantic
import pytest

from nemo_rl.models.policy import AutomodelPipelineConfig
from nemo_rl.models.policy.utils import (
    automodel_pipeline_parallel_size,
    validate_automodel_pipeline_config,
)


def _config(**automodel_overrides: Any) -> dict[str, Any]:
    automodel_cfg = {
        "enabled": True,
        "tensor_parallel_size": 1,
        "context_parallel_size": 1,
        "pipeline_parallel_size": 2,
        "pipeline": {"pp_schedule": "1f1b", "microbatches_per_step": 4},
        "defer_fsdp_grad_sync": False,
        "sequence_parallel": False,
        "cpu_offload": False,
        "lora_cfg": {"enabled": False},
    }
    automodel_cfg.update(automodel_overrides)
    return {
        "model_name": "tiny-model",
        "train_micro_batch_size": 2,
        "logprob_batch_size": 2,
        "automodel_cfg": automodel_cfg,
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "generation": {"backend": "vllm", "colocated": {"enabled": True}},
    }


def _validate(config: dict[str, Any], *, is_vlm: bool = False, model_config=None):
    return validate_automodel_pipeline_config(
        config, is_vlm=is_vlm, model_config=model_config
    )


def test_pipeline_parallel_size_defaults_to_one_when_absent():
    assert automodel_pipeline_parallel_size({"enabled": True}) == 1
    assert automodel_pipeline_parallel_size({"pipeline_parallel_size": 4}) == 4


def test_pp_size_one_needs_no_pipeline_block():
    config = _config(pipeline_parallel_size=1)
    del config["automodel_cfg"]["pipeline"]
    assert _validate(config) is None


def test_valid_config_returns_parsed_pipeline_block():
    pipeline_cfg = _validate(
        _config(), model_config=SimpleNamespace(tie_word_embeddings=False)
    )
    assert pipeline_cfg == AutomodelPipelineConfig(
        pp_schedule="1f1b", microbatches_per_step=4, layers_per_stage=None
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"pipeline_parallel_size": 0}, "must be >= 1"),
        ({"pipeline": None}, "valid dictionary"),
        (
            {"pipeline": {"pp_schedule": "1f1b", "microbatches_per_step": 1}},
            "at least one microbatch per pipeline stage",
        ),
        (
            {
                "pipeline": {
                    "pp_schedule": "1f1b",
                    "microbatches_per_step": 4,
                    "layers_per_stage": 2,
                }
            },
            "applies only to pp_schedule=interleaved_1f1b",
        ),
        (
            {
                "pipeline_parallel_size": 3,
                "pipeline": {
                    "pp_schedule": "interleaved_1f1b",
                    "microbatches_per_step": 10,
                },
            },
            "multiple of max",
        ),
    ],
)
def test_rejects_invalid_pipeline_settings(overrides, message):
    with pytest.raises((ValueError, pydantic.ValidationError), match=message):
        _validate(_config(**overrides))


def test_pipeline_block_is_required():
    config = _config()
    del config["automodel_cfg"]["pipeline"]
    with pytest.raises(ValueError, match="requires a policy.automodel_cfg.pipeline"):
        _validate(config)


def test_pipeline_block_rejects_unknown_keys():
    with pytest.raises(pydantic.ValidationError):
        _validate(
            _config(
                pipeline={
                    "pp_schedule": "1f1b",
                    "microbatches_per_step": 4,
                    "pp_microbatch_size": 1,
                }
            )
        )


def test_interleaved_accepts_rounded_microbatches():
    pipeline_cfg = _validate(
        _config(
            pipeline_parallel_size=4,
            pipeline={"pp_schedule": "interleaved_1f1b", "microbatches_per_step": 12},
        )
    )
    assert pipeline_cfg.pp_schedule == "interleaved_1f1b"


@pytest.mark.parametrize(
    "overrides", [{"expert_parallel_size": 8}, {"context_parallel_size": 2}]
)
def test_accepts_expert_and_context_parallelism(overrides):
    assert _validate(_config(**overrides)) is not None


def test_defer_fsdp_grad_sync_must_be_explicit():
    config = _config()
    del config["automodel_cfg"]["defer_fsdp_grad_sync"]
    with pytest.raises(ValueError, match="defer_fsdp_grad_sync"):
        _validate(config)


def test_logprob_batch_size_must_match_train_micro_batch_size():
    config = _config()
    config["logprob_batch_size"] = 4
    with pytest.raises(ValueError, match="logprob_batch_size"):
        _validate(config)


@pytest.mark.parametrize(
    ("mutate", "name"),
    [
        (
            lambda c: c["sequence_packing"].update(enabled=True),
            "sequence_packing.enabled",
        ),
        (
            lambda c: c["dynamic_batching"].update(enabled=True),
            "dynamic_batching.enabled",
        ),
        (
            lambda c: c["automodel_cfg"].update(sequence_parallel=True),
            "automodel_cfg.sequence_parallel",
        ),
        (lambda c: c["automodel_cfg"].update(cpu_offload=True), "cpu_offload"),
        (
            lambda c: c["automodel_cfg"]["lora_cfg"].update(enabled=True),
            "lora_cfg.enabled",
        ),
        (lambda c: c.update(router_replay={"enabled": True}), "router_replay"),
        (
            lambda c: c.update(reward_model_cfg={"enabled": True}),
            "reward_model_cfg",
        ),
        (lambda c: c.update(quant_cfg="int4"), "quant_cfg"),
        (
            lambda c: c["generation"]["colocated"].update(enabled=False),
            "non-colocated generation",
        ),
        (
            lambda c: c["generation"].update(refit_transport="nixl"),
            "refit_transport",
        ),
    ],
)
def test_rejects_unsupported_combinations(mutate, name):
    config = _config()
    mutate(config)
    with pytest.raises(ValueError, match="does not support") as excinfo:
        _validate(config)
    assert name in str(excinfo.value)


def test_rejects_vlm():
    with pytest.raises(ValueError, match="multimodal"):
        _validate(_config(), is_vlm=True)


def test_rejects_tied_embeddings():
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        _validate(_config(), model_config=SimpleNamespace(tie_word_embeddings=True))
    # VLM-style configs keep the flag on the text config.
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        _validate(
            _config(),
            model_config=SimpleNamespace(
                text_config=SimpleNamespace(tie_word_embeddings=True)
            ),
        )
