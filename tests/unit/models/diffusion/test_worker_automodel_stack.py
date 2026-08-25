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
"""Tests for the diffusion worker's NeMo Automodel model-lifecycle stack.

These tests never load a real checkpoint: the Automodel entry points are
monkeypatched (pipeline loading) or exercised on tiny pure-torch modules
(LoRA). They skip cleanly when ``nemo_automodel`` is not installed, mirroring
how the diffusers-dependent tests handle minimal installs.
"""

import os

import pytest
import torch

pytest.importorskip("nemo_automodel")

from nemo_rl.models.diffusion.workers.flow_grpo_worker import (
    load_diffusion_pipeline,
)


class _FakePipe:
    def __init__(self):
        self.to_calls: list = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self


def test_load_diffusion_pipeline_uses_automodel_loader(monkeypatch):
    from nemo_automodel._diffusers import auto_diffusion_pipeline as adp

    seen: dict = {}

    def fake_from_pretrained(model_name, **kwargs):
        seen["model_name"] = model_name
        seen.update(kwargs)
        return _FakePipe(), {}

    monkeypatch.setattr(
        adp.NeMoAutoDiffusionPipeline,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    pipe, managers = load_diffusion_pipeline(
        "Qwen/Qwen-Image", dtype=torch.bfloat16, device=torch.device("cpu")
    )

    assert isinstance(pipe, _FakePipe)
    assert managers == {}
    assert seen["model_name"] == "Qwen/Qwen-Image"
    assert seen["torch_dtype"] is torch.bfloat16
    assert seen["device"] == torch.device("cpu")
    assert seen["load_for_training"] is True
    # Device-only move happens after loading: the pin's _move_module_to_device
    # would cast fp32 rotary buffers to bf16 and skew prompt embeddings.
    assert seen["move_to_device"] is False
    assert pipe.to_calls == [((torch.device("cpu"),), {})]
    # Stage one keeps the worker's manual DP all-reduce: no parallel managers.
    assert seen["parallel_scheme"] is None
    assert seen["peft_cfg"] is None


def test_load_diffusion_pipeline_returns_two_tuple(monkeypatch):
    from nemo_automodel._diffusers import auto_diffusion_pipeline as adp

    fake_managers = {"transformer": object()}
    monkeypatch.setattr(
        adp.NeMoAutoDiffusionPipeline,
        "from_pretrained",
        staticmethod(lambda model_name, **kwargs: (_FakePipe(), fake_managers)),
    )

    pipe, managers = load_diffusion_pipeline(
        "Qwen/Qwen-Image", dtype=torch.float32, device=torch.device("cpu")
    )
    assert isinstance(pipe, _FakePipe)
    assert managers is fake_managers


# ---------------------------------------------------------------------------
# LoRA: schema -> PeftConfig mapping and fail-loud injection check
# ---------------------------------------------------------------------------
class _TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.to_q = torch.nn.Linear(4, 4)
        self.attn.to_k = torch.nn.Linear(4, 4)


class _TinyTransformer(torch.nn.Module):
    def __init__(self, num_blocks: int = 2) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            _TinyBlock() for _ in range(num_blocks)
        )


def _lora_cfg_dict(**overrides):
    cfg = {
        "enabled": True,
        "rank": 2,
        "alpha": 4,
        "target_modules": ["*.attn.to_q", "*.attn.to_k"],
        "dropout": 0.0,
        "exclude_modules": None,
    }
    cfg.update(overrides)
    return cfg


def test_build_peft_config_maps_schema_fields():
    from nemo_rl.models.diffusion.workers.flow_grpo_worker import build_peft_config

    peft_cfg = build_peft_config(_lora_cfg_dict(rank=64, alpha=128))
    assert peft_cfg.dim == 64
    assert peft_cfg.alpha == 128
    assert peft_cfg.target_modules == ["*.attn.to_q", "*.attn.to_k"]
    assert peft_cfg.exclude_modules == []
    assert peft_cfg.dropout == 0.0
    # The loader pins LoRA weights to the base dtype (bf16); leave it unset here.
    assert peft_cfg.lora_dtype is None


def test_full_path_targets_inject_and_pass_fail_loud_check():
    from nemo_automodel.components._peft.lora import (
        LinearLoRA,
        apply_lora_to_linear_modules,
    )

    from nemo_rl.models.diffusion.workers.flow_grpo_worker import (
        assert_lora_targets_hit,
        build_peft_config,
    )

    model = _TinyTransformer(num_blocks=3)
    hits = apply_lora_to_linear_modules(model, build_peft_config(_lora_cfg_dict()))
    assert hits == 6  # 3 blocks x {to_q, to_k}
    assert sum(isinstance(m, LinearLoRA) for m in model.modules()) == 6
    assert_lora_targets_hit(model, ["*.attn.to_q", "*.attn.to_k"])


def test_bare_suffix_targets_match_nothing_and_fail_loud():
    from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules

    from nemo_rl.models.diffusion.workers.flow_grpo_worker import (
        assert_lora_targets_hit,
        build_peft_config,
    )

    model = _TinyTransformer()
    # peft-style bare suffixes: ModuleMatcher anchors the whole FQN, so this
    # silently injects nothing — exactly the failure the check must catch.
    hits = apply_lora_to_linear_modules(
        model, build_peft_config(_lora_cfg_dict(target_modules=["to_q", "to_k"]))
    )
    assert hits == 0
    with pytest.raises(RuntimeError, match="full-path"):
        assert_lora_targets_hit(model, ["to_q", "to_k"])


def test_partially_missed_targets_fail_loud_and_name_offenders():
    from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules

    from nemo_rl.models.diffusion.workers.flow_grpo_worker import (
        assert_lora_targets_hit,
        build_peft_config,
    )

    model = _TinyTransformer()
    targets = ["*.attn.to_q", "*.attn.to_v"]  # to_v does not exist
    apply_lora_to_linear_modules(
        model, build_peft_config(_lora_cfg_dict(target_modules=targets))
    )
    with pytest.raises(RuntimeError, match=r"to_v"):
        assert_lora_targets_hit(model, targets)


# ---------------------------------------------------------------------------
# KL reference: lora_scale_zero context manager
# ---------------------------------------------------------------------------
def _lora_model_with_nonzero_delta() -> torch.nn.Module:
    from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules

    from nemo_rl.models.diffusion.workers.flow_grpo_worker import build_peft_config

    torch.manual_seed(0)
    model = _TinyTransformer()
    apply_lora_to_linear_modules(model, build_peft_config(_lora_cfg_dict()))
    # lora_B is zero-initialized; make the LoRA delta observable.
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param)
    return model


def test_lora_scale_zero_matches_base_forward_bitwise():
    from nemo_rl.models.diffusion.workers.flow_grpo_worker import lora_scale_zero

    model = _lora_model_with_nonzero_delta()
    x = torch.randn(2, 4)
    block = model.transformer_blocks[0]
    base_out = torch.nn.functional.linear(
        x, block.attn.to_q.weight, block.attn.to_q.bias
    )

    with_lora = block.attn.to_q(x)
    assert not torch.equal(with_lora, base_out)  # delta actually active

    saved_scale = block.attn.to_q.scale
    with lora_scale_zero(model):
        assert block.attn.to_q.scale == 0.0
        assert torch.equal(block.attn.to_q(x), base_out)  # bitwise base-only
    assert block.attn.to_q.scale == saved_scale
    assert torch.equal(block.attn.to_q(x), with_lora)


def test_lora_scale_zero_restores_scales_on_exception():
    from nemo_automodel.components._peft.lora import LinearLoRA

    from nemo_rl.models.diffusion.workers.flow_grpo_worker import lora_scale_zero

    model = _lora_model_with_nonzero_delta()
    scales = [m.scale for m in model.modules() if isinstance(m, LinearLoRA)]
    with pytest.raises(ValueError):
        with lora_scale_zero(model):
            raise ValueError("boom")
    assert [m.scale for m in model.modules() if isinstance(m, LinearLoRA)] == scales


def test_build_no_adapter_forward_zeroes_scales_and_matches_call_convention():
    from nemo_automodel.components._peft.lora import LinearLoRA

    from nemo_rl.models.diffusion.workers.flow_grpo_worker import (
        build_no_adapter_forward,
    )

    model = _lora_model_with_nonzero_delta()
    calls = []

    def forward_fn(transformer, **kwargs):
        assert transformer is model
        # must run with every LoRA branch disabled
        assert all(
            m.scale == 0.0 for m in transformer.modules() if isinstance(m, LinearLoRA)
        )
        calls.append(kwargs)
        return kwargs["hidden_states"] * 0

    fwd = build_no_adapter_forward(model, forward_fn)
    x = torch.ones(2, 3)
    # pipeline._denoise_step's calling convention: no positional transformer argument
    out = fwd(hidden_states=x, timestep=torch.tensor([1.0]))
    assert torch.equal(out, torch.zeros(2, 3))
    assert calls and "timestep" in calls[0]
    # scales restored after the closure returns
    assert all(m.scale != 0.0 for m in model.modules() if isinstance(m, LinearLoRA))


# ---------------------------------------------------------------------------
# Checkpointing: Automodel Checkpointer round-trip on a tiny LoRA model
# ---------------------------------------------------------------------------
def _cleanup_dcp_planner_cache() -> None:
    """Clear DCP SavePlanner class-level plan caches (shared across tests)."""
    from torch.distributed.checkpoint.planner import SavePlanner

    for attr in (
        "_cached_save_plan",
        "_cached_all_plans",
        "_cached_global_plan",
        "_cached_metadata",
        "_cached_final_save_plan",
    ):
        cache = getattr(SavePlanner, attr, None)
        if cache is not None:
            cache.clear()


@pytest.fixture
def single_process_group():
    """Single-process gloo group: the worker always has one initialized."""
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    _cleanup_dcp_planner_cache()
    yield
    _cleanup_dcp_planner_cache()


def _lora_model_and_optimizer(seed: int):
    from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules

    from nemo_rl.models.diffusion.workers.flow_grpo_worker import build_peft_config

    torch.manual_seed(seed)
    model = _TinyTransformer()
    peft_cfg = build_peft_config(_lora_cfg_dict())
    apply_lora_to_linear_modules(model, peft_cfg)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-2
    )
    return model, optimizer, peft_cfg


def test_checkpointer_lora_round_trip_and_layout(tmp_path, single_process_group):
    from nemo_rl.models.diffusion.workers.flow_grpo_worker import build_checkpointer

    model, optimizer, peft_cfg = _lora_model_and_optimizer(seed=0)
    # One real step so LoRA weights and Adam moments are all nonzero.
    x = torch.randn(4, 4)
    loss = sum(
        b.attn.to_q(x).square().mean() + b.attn.to_k(x).square().mean()
        for b in model.transformer_blocks
    )
    loss.backward()
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad += torch.randn_like(p.grad)
    optimizer.step()

    path = str(tmp_path / "step_1")
    ckpt = build_checkpointer(is_peft=True, dp_rank=0)
    ckpt.save_model(model, path, peft_config=peft_cfg)
    ckpt.save_optimizer(optimizer, model, path)

    # Layout contract for FlowGRPOPolicyWorker.save_checkpoint / load_checkpoint.
    assert os.path.isfile(os.path.join(path, "model", "adapter_model.safetensors"))
    assert os.path.isfile(os.path.join(path, "model", "adapter_config.json"))
    assert os.path.isfile(os.path.join(path, "optim", ".metadata"))

    # Fresh model with different base weights: only the adapter is restored.
    model2, optimizer2, _ = _lora_model_and_optimizer(seed=1)
    ckpt2 = build_checkpointer(is_peft=True, dp_rank=0)
    ckpt2.load_model(model2, os.path.join(path, "model"))
    ckpt2.load_optimizer(optimizer2, model2, path)

    lora1 = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    lora2 = {k: v for k, v in model2.state_dict().items() if "lora_" in k}
    assert lora1 and lora1.keys() == lora2.keys()
    for k in lora1:
        assert torch.equal(lora1[k], lora2[k]), k

    state1 = optimizer.state_dict()["state"]
    state2 = optimizer2.state_dict()["state"]
    assert len(state1) == len(state2) > 0
    for pid in state1:
        assert torch.allclose(state1[pid]["exp_avg"], state2[pid]["exp_avg"])
        assert torch.allclose(state1[pid]["exp_avg_sq"], state2[pid]["exp_avg_sq"])


# ---------------------------------------------------------------------------
# Misc: inlined calculate_shift and worker venv registration
# ---------------------------------------------------------------------------
def test_inlined_calculate_shift_matches_diffusers():
    qwenimage = pytest.importorskip("diffusers.pipelines.qwenimage.pipeline_qwenimage")

    from nemo_rl.models.diffusion.workers.flow_grpo_worker import calculate_shift

    for seq_len in (256, 1024, 4096, 64 * 64):
        assert calculate_shift(seq_len) == qwenimage.calculate_shift(seq_len)
    assert calculate_shift(1024, 128, 2048, 0.4, 1.2) == qwenimage.calculate_shift(
        1024, 128, 2048, 0.4, 1.2
    )


def test_flow_grpo_worker_registered_with_automodel_diffusion_env():
    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
    )

    exe = ACTOR_ENVIRONMENT_REGISTRY[
        "nemo_rl.models.diffusion.workers.flow_grpo_worker.FlowGRPOPolicyWorker"
    ]
    assert "--extra automodel" in exe
    assert "--extra diffusion" in exe


def test_lora_schema_defaults_are_full_path_patterns():
    from nemo_rl.models.diffusion.interfaces import DiffusionLoraCfg

    cfg = DiffusionLoraCfg()
    assert cfg.target_modules, "defaults must not be empty"
    assert all(t.startswith("*.") for t in cfg.target_modules)


def test_exemplar_yaml_targets_are_full_path_patterns():
    yaml = pytest.importorskip("yaml")

    with open("examples/configs/flow_grpo_qwen_image_ocr.yaml") as f:
        cfg = yaml.safe_load(f)
    targets = cfg["policy"]["lora_cfg"]["target_modules"]
    assert len(targets) == 12
    assert all(t.startswith("*.") for t in targets)
