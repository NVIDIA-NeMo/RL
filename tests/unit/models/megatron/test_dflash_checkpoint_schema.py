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

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from megatron.core.model_parallel_config import ModelParallelConfig

from nemo_rl.models.megatron.draft.dflash import DFlashBody, DFlashBodyConfig

pytestmark = pytest.mark.mcore


_PUBLIC_ARTIFACT_REPO = "z-lab/Qwen3-8B-DFlash-b16"
_PUBLIC_ARTIFACT_REVISION = (
    "9b41424b7109f9c5413454f481b09a82b85333f4"  # pragma: allowlist secret
)
_PUBLIC_CONFIG_SHA256 = "9834d608c9ca53d5548b415471ae9e8ebc9aab6cedfc2a7af95b6bd097373102"  # pragma: allowlist secret
_PUBLIC_SAFETENSORS_HEADER_BYTES = 6_232
_PUBLIC_SAFETENSORS_HEADER_SHA256 = "6724cbb4ec77638c24d878ce60aa4fbf0505f9ad3bc2b00110176767baf50856"  # pragma: allowlist secret


_LAYER_SUFFIXES = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _tiny_config() -> DFlashBodyConfig:
    return DFlashBodyConfig(
        hidden_size=8,
        intermediate_size=12,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=2,
        num_target_taps=2,
        rope_theta=10_000.0,
    )


def test_qwen3_8b_defaults_are_pinned_and_frozen() -> None:
    config = DFlashBodyConfig()

    assert config.hidden_size == 4096
    assert config.intermediate_size == 12288
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.num_hidden_layers == 5
    assert config.num_target_taps == 5
    assert config.rope_theta == 1_000_000.0
    assert config.rms_norm_eps == 1e-6
    with pytest.raises(FrozenInstanceError):
        config.hidden_size = 8  # type: ignore[misc]


def test_hf_checkpoint_loader_forwards_an_explicit_revision(tmp_path: Path) -> None:
    from nemo_rl.models.megatron.draft.utils import _load_checkpoint_state

    repo_id = "deepseek-ai/dspark_qwen3_8b_block7"
    revision = "03326e5043815da1f81b109078b2889737c26017"  # pragma: allowlist secret
    with (
        patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as fetch,
        patch(
            "nemo_rl.models.megatron.draft.utils._load_checkpoint_from_directory",
            return_value={},
        ),
    ):
        state = _load_checkpoint_state(repo_id, revision=revision)

    assert state == {}
    assert fetch.call_args.kwargs["repo_id"] == repo_id
    assert fetch.call_args.kwargs["revision"] == revision


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rope_theta", float("nan")),
        ("rope_theta", float("inf")),
        ("rms_norm_eps", float("nan")),
        ("rms_norm_eps", float("inf")),
        ("initializer_range", float("nan")),
        ("initializer_range", float("inf")),
    ],
)
def test_config_rejects_nonfinite_positive_float_fields(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        DFlashBodyConfig(**{field: value})


def test_exact_public_body_state_dict_schema_and_shapes() -> None:
    artifact_provenance = (
        f"{_PUBLIC_ARTIFACT_REPO}@{_PUBLIC_ARTIFACT_REVISION}; "
        f"config.json sha256={_PUBLIC_CONFIG_SHA256}; "
        f"model.safetensors header[{_PUBLIC_SAFETENSORS_HEADER_BYTES}] "
        f"sha256={_PUBLIC_SAFETENSORS_HEADER_SHA256}"
    )
    with torch.device("meta"):
        body = DFlashBody(DFlashBodyConfig())
    state = body.state_dict()
    expected_keys = {"fc.weight", "hidden_norm.weight", "norm.weight"}
    expected_keys.update(
        f"layers.{layer_index}.{suffix}"
        for layer_index in range(5)
        for suffix in _LAYER_SUFFIXES
    )

    assert set(state) == expected_keys, artifact_provenance
    assert {tensor.dtype for tensor in state.values()} == {torch.bfloat16}, (
        artifact_provenance
    )
    assert tuple(state["fc.weight"].shape) == (4096, 20480)
    assert tuple(state["hidden_norm.weight"].shape) == (4096,)
    assert tuple(state["norm.weight"].shape) == (4096,)
    for layer_index in range(5):
        prefix = f"layers.{layer_index}."
        assert tuple(state[prefix + "input_layernorm.weight"].shape) == (4096,)
        assert tuple(state[prefix + "self_attn.q_proj.weight"].shape) == (
            4096,
            4096,
        )
        assert tuple(state[prefix + "self_attn.k_proj.weight"].shape) == (
            1024,
            4096,
        )
        assert tuple(state[prefix + "self_attn.v_proj.weight"].shape) == (
            1024,
            4096,
        )
        assert tuple(state[prefix + "self_attn.o_proj.weight"].shape) == (
            4096,
            4096,
        )
        assert tuple(state[prefix + "self_attn.q_norm.weight"].shape) == (128,)
        assert tuple(state[prefix + "self_attn.k_norm.weight"].shape) == (128,)
        assert tuple(state[prefix + "post_attention_layernorm.weight"].shape) == (4096,)
        assert tuple(state[prefix + "mlp.gate_proj.weight"].shape) == (
            12288,
            4096,
        )
        assert tuple(state[prefix + "mlp.up_proj.weight"].shape) == (
            12288,
            4096,
        )
        assert tuple(state[prefix + "mlp.down_proj.weight"].shape) == (
            4096,
            12288,
        )


def test_body_preserves_explicit_mcore_precision_config() -> None:
    parallel_config = ModelParallelConfig(
        tensor_model_parallel_size=1,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
    )

    body = DFlashBody(_tiny_config(), parallel_config=parallel_config)

    assert body.parallel_config is parallel_config
    assert {parameter.dtype for parameter in body.parameters()} == {torch.float32}


def test_body_rejects_mcore_config_with_wrong_tensor_parallel_size() -> None:
    parallel_config = ModelParallelConfig(
        tensor_model_parallel_size=2,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="tensor_model_parallel_size"):
        DFlashBody(_tiny_config(), parallel_config=parallel_config)


def test_body_owns_no_target_embedding_head_or_mask_tensor() -> None:
    body = DFlashBody(_tiny_config())
    forbidden_fragments = ("embed_tokens", "lm_head", "mask_embedding", "mask_token")

    assert all(
        fragment not in name
        for name in body.state_dict()
        for fragment in forbidden_fragments
    )
    assert all(
        fragment not in name
        for name, _ in body.named_modules()
        for fragment in forbidden_fragments
    )


def test_strict_state_dict_load_accepts_exact_schema_and_rejects_drift() -> None:
    torch.manual_seed(41)
    source = DFlashBody(_tiny_config())
    exact_state = source.state_dict()
    restored = DFlashBody(_tiny_config())

    incompatible = restored.load_state_dict(exact_state, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    for name, parameter in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], parameter)

    missing_state = dict(exact_state)
    del missing_state["norm.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(missing_state, strict=True)
    unexpected_state = dict(exact_state)
    unexpected_state["mask_embedding.weight"] = torch.empty(1, 8)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        restored.load_state_dict(unexpected_state, strict=True)


def test_sharded_state_dict_preserves_public_names_and_prefix(tmp_path: Path) -> None:
    mapping = pytest.importorskip("megatron.core.dist_checkpointing.mapping")
    transformer_utils = pytest.importorskip("megatron.core.transformer.utils")
    created_process_group = False
    if not torch.distributed.is_initialized():
        init_file = tmp_path / "dflash_schema_dist_init"
        torch.distributed.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=0,
            world_size=1,
        )
        created_process_group = True

    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        singleton_groups = [
            torch.distributed.new_group([group_rank])
            for group_rank in range(world_size)
        ]
        body = DFlashBody(
            _tiny_config(),
            tp_group=torch.distributed.group.WORLD,
        )
        sharded = body.sharded_state_dict(
            prefix="draft.",
            sharded_offsets=(),
            metadata={"dp_cp_group": singleton_groups[rank]},
        )
        nested = transformer_utils.sharded_state_dict_default(
            body,
            "parent.draft.",
            (),
            {"dp_cp_group": singleton_groups[rank]},
            tp_group=torch.distributed.group.WORLD,
        )

        assert set(sharded) == {f"draft.{name}" for name in body.state_dict()}
        assert set(nested) == {f"parent.draft.{name}" for name in body.state_dict()}
        for name, sharded_tensor in sharded.items():
            assert isinstance(sharded_tensor, mapping.ShardedTensor)
            assert sharded_tensor.key == name
        assert tuple(sharded["draft.fc.weight"].global_shape) == (8, 16)
        assert tuple(
            sharded["draft.layers.0.self_attn.q_proj.weight"].global_shape
        ) == (8, 8)
        assert tuple(
            sharded["draft.layers.0.self_attn.o_proj.weight"].global_shape
        ) == (8, 8)
    finally:
        if created_process_group:
            torch.distributed.destroy_process_group()


def test_megatron_sharded_checkpoint_round_trip(tmp_path: Path) -> None:
    dist_checkpointing = pytest.importorskip("megatron.core.dist_checkpointing")
    mapping = pytest.importorskip("megatron.core.dist_checkpointing.mapping")
    created_process_group = False
    if not torch.distributed.is_initialized():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            torch.distributed.init_process_group("gloo")
        else:
            init_file = tmp_path / "dflash_dist_init"
            torch.distributed.init_process_group(
                "gloo",
                init_method=f"file://{init_file}",
                rank=0,
                world_size=1,
            )
        created_process_group = True

    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        singleton_groups = [
            torch.distributed.new_group([group_rank])
            for group_rank in range(world_size)
        ]
        metadata = {"dp_cp_group": singleton_groups[rank]}
        torch.manual_seed(43)
        source = DFlashBody(
            _tiny_config(),
            tp_group=torch.distributed.group.WORLD,
        )
        sharded = source.sharded_state_dict(prefix="draft.", metadata=metadata)
        assert all(
            isinstance(value, mapping.ShardedTensor) for value in sharded.values()
        )

        checkpoint_paths = [
            str(tmp_path / "dflash_dcp") if torch.distributed.get_rank() == 0 else ""
        ]
        torch.distributed.broadcast_object_list(checkpoint_paths, src=0)
        checkpoint_dir = checkpoint_paths[0]
        if torch.distributed.get_rank() == 0:
            Path(checkpoint_dir).mkdir(parents=True)
        torch.distributed.barrier()
        dist_checkpointing.save({"model": sharded}, checkpoint_dir)
        restored = DFlashBody(
            _tiny_config(),
            tp_group=torch.distributed.group.WORLD,
        )
        template = restored.sharded_state_dict(
            prefix="draft.",
            metadata=metadata,
        )
        loaded: dict[str, Any] = dist_checkpointing.load(
            {"model": template},
            checkpoint_dir,
        )
        unprefixed = {
            name.removeprefix("draft."): tensor
            for name, tensor in loaded["model"].items()
        }
        incompatible = restored.load_state_dict(unprefixed, strict=True)

        assert not incompatible.missing_keys
        assert not incompatible.unexpected_keys
        for name, parameter in source.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], parameter)
    finally:
        if created_process_group:
            torch.distributed.destroy_process_group()
