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

from types import SimpleNamespace

import pytest

from nemo_rl.modelopt import utils as modelopt_utils
from nemo_rl.weight_sync.nccl_reshard_utils import (
    check_nccl_reshard_refit_support,
)


def _valid_nvfp4_config(*, mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        policy={
            "quant_cfg": None,
            "precision": "bfloat16",
            "generation": {
                "backend": "vllm",
                "colocated": {"enabled": False},
                "real_quant": True,
                "quant_cfg": f"NVFP4_{mode.upper()}",
                "real_quant_calibration_path": (
                    "/artifacts/calibration.safetensors" if mode == "w4a4" else None
                ),
                "vllm_cfg": {
                    "precision": "bfloat16",
                    "tensor_parallel_size": 1,
                    "expert_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                },
            },
            "megatron_cfg": {"enabled": True},
            "dtensor_cfg": {"enabled": False},
        }
    )


@pytest.mark.parametrize("mode", ["w4a16", "w4a4"])
def test_validator_accepts_plain_bf16_to_supported_nvfp4(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    monkeypatch.setattr(
        modelopt_utils,
        "resolve_nvfp4_real_quant_mode",
        lambda _quant_cfg: mode,
    )

    check_nccl_reshard_refit_support(_valid_nvfp4_config(mode=mode))


def test_validator_rejects_unsupported_policy_precision_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        modelopt_utils,
        "resolve_nvfp4_real_quant_mode",
        lambda _quant_cfg: "w4a16",
    )
    config = _valid_nvfp4_config(mode="w4a16")
    config.policy["precision"] = "bf16"

    with pytest.raises(ValueError, match="policy.precision must be 'bfloat16'"):
        check_nccl_reshard_refit_support(config)


def test_validator_rejects_non_nvfp4_real_quant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_non_nvfp4(_quant_cfg: str) -> str:
        raise ValueError("enabled weights are not NVFP4")

    monkeypatch.setattr(
        modelopt_utils,
        "resolve_nvfp4_real_quant_mode",
        reject_non_nvfp4,
    )

    with pytest.raises(ValueError, match="must resolve to a supported NVFP4 mode"):
        check_nccl_reshard_refit_support(_valid_nvfp4_config(mode="w4a16"))


@pytest.mark.parametrize(
    ("precision", "is_mx"),
    [("fp8", False), ("fp8", True), ("bfloat16", True)],
)
def test_validator_rejects_mixed_generation_quantization_modes(
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
    is_mx: bool,
) -> None:
    monkeypatch.setattr(
        modelopt_utils,
        "resolve_nvfp4_real_quant_mode",
        lambda _quant_cfg: "w4a16",
    )
    config = _valid_nvfp4_config(mode="w4a16")
    config.policy["generation"]["vllm_cfg"].update(
        {"precision": precision, "is_mx": is_mx}
    )

    with pytest.raises(ValueError, match="real NVFP4.*vllm_cfg"):
        check_nccl_reshard_refit_support(config)


def test_validator_rejects_tensor_parallel_nvfp4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        modelopt_utils,
        "resolve_nvfp4_real_quant_mode",
        lambda _quant_cfg: "w4a16",
    )
    config = _valid_nvfp4_config(mode="w4a16")
    config.policy["generation"]["vllm_cfg"]["tensor_parallel_size"] = 2

    with pytest.raises(ValueError, match="tensor_parallel_size must be 1"):
        check_nccl_reshard_refit_support(config)


def test_validator_rejects_expert_parallel_nvfp4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        modelopt_utils,
        "resolve_nvfp4_real_quant_mode",
        lambda _quant_cfg: "w4a16",
    )
    config = _valid_nvfp4_config(mode="w4a16")
    config.policy["generation"]["vllm_cfg"]["expert_parallel_size"] = 2

    with pytest.raises(ValueError, match="expert_parallel_size must be 1"):
        check_nccl_reshard_refit_support(config)


@pytest.mark.parametrize("calibration_path", [None, "", "   "])
def test_validator_rejects_w4a4_without_calibration(
    monkeypatch: pytest.MonkeyPatch,
    calibration_path: str | None,
) -> None:
    monkeypatch.setattr(
        modelopt_utils,
        "resolve_nvfp4_real_quant_mode",
        lambda _quant_cfg: "w4a4",
    )
    config = _valid_nvfp4_config(mode="w4a4")
    config.policy["generation"]["real_quant_calibration_path"] = calibration_path

    with pytest.raises(ValueError, match="non-empty.*real_quant_calibration_path"):
        check_nccl_reshard_refit_support(config)
