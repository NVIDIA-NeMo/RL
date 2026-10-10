# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = pytest.mark.mcore


def inherited_fp8_provider() -> SimpleNamespace:
    return SimpleNamespace(
        fp8="hybrid",
        fp8_recipe="mxfp8",
        fp8_param=True,
        moe_router_padding_for_quantization=True,
        moe_router_padding_for_fp8=True,
        quant_recipe=object(),
    )


@pytest.mark.parametrize(
    "fp8_cfg",
    [
        None,
        {"enabled": False},
        {"enabled": True, "fp8": "hybrid", "fp8_recipe": "mxfp8", "fp8_param": True},
    ],
)
def test_bf16_only_clears_explicitly_disabled_fp8(
    fp8_cfg: dict[str, Any] | None,
) -> None:
    from nemo_rl.models.megatron.setup import _apply_precision_config

    provider = inherited_fp8_provider()
    recipe = provider.quant_recipe
    options = {"pipeline_dtype": "bfloat16"}
    if fp8_cfg is not None:
        options["fp8_cfg"] = fp8_cfg
    _apply_precision_config(provider, {"megatron_cfg": options}, torch.bfloat16)
    disabled = fp8_cfg is not None and fp8_cfg["enabled"] is False
    assert provider.fp8 == (None if disabled else "hybrid")
    assert provider.fp8_param is (not disabled)
    assert provider.moe_router_padding_for_quantization is (not disabled)
    assert provider.moe_router_padding_for_fp8 is (not disabled)
    assert provider.quant_recipe is (None if disabled else recipe)


def test_bf16_fp8_disable_preserves_explicit_fp4() -> None:
    from nemo_rl.models.megatron.setup import _apply_precision_config

    provider = inherited_fp8_provider()
    recipe = provider.quant_recipe
    _apply_precision_config(
        provider,
        {
            "megatron_cfg": {
                "pipeline_dtype": "bfloat16",
                "fp8_cfg": {"enabled": False},
                "fp4_cfg": {"enabled": True, "fp4": "e2m1", "fp4_recipe": "nvfp4"},
            }
        },
        torch.bfloat16,
    )
    assert provider.fp4 == "e2m1"
    assert provider.fp4_recipe == "nvfp4"
    assert provider.quant_recipe is recipe
    assert provider.moe_router_padding_for_quantization is True


def test_bf16_fp8_disable_preserves_explicit_te_recipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from nemo_rl.models.megatron import setup

    provider = inherited_fp8_provider()
    path = tmp_path / "precision.yaml"
    path.write_text("{}")
    recipe = SimpleNamespace(matchers=[], configs={})
    monkeypatch.setattr(setup, "load_quantization_recipe", lambda _: recipe)
    setup._apply_precision_config(
        provider,
        {
            "megatron_cfg": {
                "pipeline_dtype": "bfloat16",
                "fp8_cfg": {"enabled": False},
                "te_precision_config_file": str(path),
            }
        },
        torch.bfloat16,
    )
    assert provider.quant_recipe is recipe
    assert provider.moe_router_padding_for_quantization is True


@pytest.mark.parametrize(
    "optimizer_name,distributed,layerwise,layout,expected",
    [
        ("muon", True, False, True, True),
        ("muon", False, False, True, False),
        ("adam", True, False, True, False),
        ("muon", False, True, False, True),
        ("muon", False, False, False, None),
        ("muon", True, True, False, None),
    ],
)
def test_muon_selection_and_compact_validation(
    monkeypatch: pytest.MonkeyPatch,
    optimizer_name: str,
    distributed: bool,
    layerwise: bool,
    layout: bool,
    expected: bool | None,
) -> None:
    from nemo_rl.models.megatron import setup

    for name in [
        "ConfigContainer",
        "OptimizerConfig",
        "TrainingConfig",
        "LoggerConfig",
        "DistributedInitConfig",
        "DistributedDataParallelConfig",
        "SchedulerConfig",
        "TokenizerConfig",
    ]:
        monkeypatch.setattr(setup, name, SimpleNamespace)
    config = {
        "train_global_batch_size": 2,
        "tokenizer": {"name": "test-tokenizer"},
        "megatron_cfg": {
            "optimizer": {
                "optimizer": optimizer_name,
                "use_distributed_optimizer": distributed,
                "use_layer_wise_distributed_optimizer": layerwise,
            },
            "use_layer_wise_param_layout": layout,
            "distributed_data_parallel_config": {
                "overlap_param_gather": False,
                "grad_reduce_in_fp32": True,
                "overlap_grad_reduce": False,
                "data_parallel_sharding_strategy": "no_shard",
            },
            "train_iters": 2,
            "scheduler": {},
        },
    }

    def build() -> SimpleNamespace:
        return setup._create_megatron_config(
            SimpleNamespace(), SimpleNamespace(), config, "test-model", torch.bfloat16
        )

    if expected is None:
        with pytest.raises(
            ValueError, match="use_layer_wise_param_layout=False requires"
        ):
            build()
    else:
        result = build()
        assert result.optimizer.use_layer_wise_distributed_optimizer is expected
        assert result.ddp.use_distributed_optimizer is distributed


@pytest.mark.parametrize("layout", [True, False])
def test_layerwise_flags_reach_model_construction(
    monkeypatch: pytest.MonkeyPatch, layout: bool
) -> None:
    from nemo_rl.models.megatron import setup

    def get_model(
        *, use_layer_wise_distributed_optimizer=False, use_layer_wise_param_layout=True
    ):
        return use_layer_wise_distributed_optimizer, use_layer_wise_param_layout

    monkeypatch.setattr(setup, "get_model", get_model)
    kwargs = setup._get_layer_wise_model_kwargs(
        {"megatron_cfg": {"use_layer_wise_param_layout": layout}},
        SimpleNamespace(use_layer_wise_distributed_optimizer=True),
    )
    assert get_model(**kwargs) == (True, layout)


def test_non_layerwise_preserves_older_bridge_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.models.megatron import setup

    monkeypatch.setattr(setup, "get_model", lambda: None)
    assert (
        setup._get_layer_wise_model_kwargs(
            {"megatron_cfg": {}},
            SimpleNamespace(use_layer_wise_distributed_optimizer=False),
        )
        == {}
    )


@pytest.mark.parametrize("supports_layerwise,compact", [(False, False), (True, True)])
def test_layerwise_rejects_incompatible_bridge(
    monkeypatch: pytest.MonkeyPatch, supports_layerwise: bool, compact: bool
) -> None:
    from nemo_rl.models.megatron import setup

    def get_model(*, use_layer_wise_distributed_optimizer: bool = False) -> None:
        pass

    monkeypatch.setattr(
        setup, "get_model", get_model if supports_layerwise else lambda: None
    )
    with pytest.raises(ValueError, match="upgrade Megatron-Bridge"):
        setup._get_layer_wise_model_kwargs(
            {"megatron_cfg": {"use_layer_wise_param_layout": not compact}},
            SimpleNamespace(use_layer_wise_distributed_optimizer=True),
        )
