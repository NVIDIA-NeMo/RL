# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared config builders for Megatron + SGLang weight-update / generation tests.

Exposes:

- ``MEGATRON_CFGS`` / ``SGLANG_CFGS`` — the parametrization matrix the user
  asked for ("ep2 pp2 / tp2 pp2 / tp2 ep2 pp2" Megatron × "tp4 ep4 dp4
  --enable-dp-attention / tp4 ep2 dp4 --enable-dp-attention / tp2 ep2 pp2"
  SGLang).
- ``make_policy_config(...)`` — produces a ``PolicyConfig`` dict suitable for
  ``nemo_rl.models.policy.lm_policy.Policy`` against the sliced Nemotron-3
  Nano model.
- ``make_sglang_cfg(...)`` — produces an SGLang generation config compatible
  with ``SGLangGeneration``.
- ``required_world_size(...)`` / ``min_dp_for_megatron(...)`` — helpers used
  by the test fixtures to size ``RayVirtualCluster`` and to skip cleanly when
  the host doesn't have enough GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# These are the model-specific ids we hard-code here. They are read from the
# Nemotron-3-Nano tokenizer once and cached; expressing them as constants keeps
# the test fixtures importable without a tokenizer.
PAD_TOKEN_ID = 11    # Nemotron tokenizer's pad
EOS_TOKEN_ID = 0     # Nemotron tokenizer's eos


# ---------------------------------------------------------------------------
# Parametrization matrix
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MegatronShape:
    """One Megatron parallelism shape under test.

    ``ep`` is the expert-model-parallel size; it must divide the data-parallel
    size, so the minimum DP equals ``max(ep, 1)``. World size is therefore
    ``tp * pp * cp * dp``.
    """

    id: str
    tp: int = 1
    pp: int = 1
    ep: int = 1
    cp: int = 1
    # Optional dp_size override; if None, we use ``max(ep, 1)``.
    min_dp: int | None = None


@dataclass(frozen=True)
class SGLangShape:
    """One SGLang engine shape under test.

    Mirrors SGLang ``ServerArgs`` knobs: ``tp_size`` controls the actual GPU
    count of one engine; ``dp_size`` is meaningful only with
    ``--enable-dp-attention`` and refers to attention DP within those
    ``tp_size`` GPUs (i.e. it does not change the GPU count). ``pp_size``
    multiplies the per-engine GPU count.
    """

    id: str
    tp_size: int
    ep_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    enable_dp_attention: bool = False
    # Some shapes (PP > 1, dp-attention) require piecewise CUDA graph off in
    # SGLang upstream; we just always disable both graph paths in tests.
    disable_cuda_graph: bool = True

    @property
    def num_gpus_per_engine(self) -> int:
        return self.tp_size * self.pp_size


MEGATRON_CFGS: tuple[MegatronShape, ...] = (
    MegatronShape(id="mcore_ep2_pp2", tp=1, pp=2, ep=2),
    MegatronShape(id="mcore_tp2_pp2", tp=2, pp=2, ep=1),
    MegatronShape(id="mcore_tp2_ep2_pp2", tp=2, pp=2, ep=2),
)

SGLANG_CFGS: tuple[SGLangShape, ...] = (
    SGLangShape(
        id="sgl_tp4_ep4_dp4_dpattn",
        tp_size=4,
        ep_size=4,
        dp_size=4,
        enable_dp_attention=True,
    ),
    SGLangShape(
        id="sgl_tp4_ep2_dp4_dpattn",
        tp_size=4,
        ep_size=2,
        dp_size=4,
        enable_dp_attention=True,
    ),
    SGLangShape(
        id="sgl_tp2_ep2_pp2",
        tp_size=2,
        ep_size=2,
        dp_size=1,
        pp_size=2,
        enable_dp_attention=False,
    ),
)

# Single-GPU shapes used for the smallest end-to-end variant: one Megatron
# trainer rank (DP=1, TP=PP=EP=1) feeding one single-GPU SGLang engine
# (TP=1). Kept out of ``MEGATRON_CFGS`` / ``SGLANG_CFGS`` so they don't
# multiply through the full cartesian product — they're paired explicitly
# in the test parametrization.
MEGATRON_DP1: MegatronShape = MegatronShape(id="mcore_dp1", tp=1, pp=1, ep=1)
SGLANG_TP1: SGLangShape = SGLangShape(
    id="sgl_tp1",
    tp_size=1,
    ep_size=1,
    dp_size=1,
    pp_size=1,
    enable_dp_attention=False,
)


# ---------------------------------------------------------------------------
# Sizing helpers
# ---------------------------------------------------------------------------
def min_dp_for_megatron(shape: MegatronShape) -> int:
    """Smallest data-parallel size that satisfies ``EP | DP``."""
    if shape.min_dp is not None:
        return shape.min_dp
    return max(shape.ep, 1)


def megatron_world_size(shape: MegatronShape) -> int:
    return shape.tp * shape.pp * shape.cp * min_dp_for_megatron(shape)


def required_world_size(
    *, megatron: MegatronShape, sglang: SGLangShape, colocated: bool
) -> int:
    """Total GPUs needed for one ``(megatron, sglang, mode)`` triple.

    Colocate: trainer and SGLang share the same physical GPUs, so the answer
    is ``max(megatron, sglang)``. Disaggregate: trainer + inference are on
    disjoint placement groups, so the answer is ``megatron + sglang``.
    """
    m = megatron_world_size(megatron)
    s = sglang.num_gpus_per_engine  # one engine; multi-engine clusters scale this
    return max(m, s) if colocated else m + s


# ---------------------------------------------------------------------------
# Policy config builder (Megatron training side)
# ---------------------------------------------------------------------------
def make_policy_config(
    *,
    model_path: str,
    megatron: MegatronShape,
    colocated: bool,
    max_seq_len: int = 1024,
    train_micro_batch_size: int = 1,
) -> dict[str, Any]:
    """Build a ``PolicyConfig`` dict for ``lm_policy.Policy`` (Megatron backend).

    The returned dict mirrors the keys ``Policy.__init__`` and the megatron
    worker's ``validate_and_set_config`` read. The ``generation`` block is
    populated only as a stub — the real generator is constructed separately
    via ``SGLangGeneration``; ``Policy`` only needs ``generation.colocated``
    set so the megatron setup honours the colocate/disaggregate mode.
    """
    # Megatron requires ``global_batch_size`` to be divisible by
    # ``micro_batch_size * data_parallel_size``. Our DP comes from EP (since
    # EP must divide DP) — see ``min_dp_for_megatron``. The roundtrip test
    # never runs an actual training step so the smallest legal global batch
    # size will do.
    dp_size = min_dp_for_megatron(megatron)
    train_global_batch_size = train_micro_batch_size * dp_size
    return {
        "model_name": model_path,
        "tokenizer": {"name": model_path},
        "train_global_batch_size": train_global_batch_size,
        "train_micro_batch_size": train_micro_batch_size,
        "logprob_batch_size": train_micro_batch_size,
        "precision": "bfloat16",
        "max_total_sequence_length": max_seq_len,
        "make_sequence_length_divisible_by": megatron.tp if megatron.tp > 1 else 1,
        "max_grad_norm": 1.0,
        "offload_optimizer_for_logprob": False,
        "refit_buffer_size_gb": 1,
        # No DTensor — pure Megatron path.
        "dtensor_cfg": {"enabled": False},
        "megatron_cfg": {
            "enabled": True,
            # ``train_iters`` is required by ``_validate_training_config``.
            # The roundtrip test never actually steps the optimizer (we only
            # exercise the refit path) so any positive integer works; pick a
            # small one and keep the lr_decay_iters below in sync.
            "train_iters": 10,
            "empty_unused_memory_level": 1,
            "activation_checkpointing": False,
            "converter_type": "NemotronHForCausalLM",
            "tensor_model_parallel_size": megatron.tp,
            "expert_tensor_parallel_size": 1,
            "expert_model_parallel_size": megatron.ep,
            "pipeline_model_parallel_size": megatron.pp,
            "num_layers_in_first_pipeline_stage": None,
            "num_layers_in_last_pipeline_stage": None,
            "context_parallel_size": megatron.cp,
            "pipeline_dtype": "bfloat16",
            "sequence_parallel": megatron.tp > 1,
            "freeze_moe_router": True,
            "moe_router_dtype": "fp64",
            "moe_router_load_balancing_type": "none",
            "moe_router_bias_update_rate": 0.0,
            "moe_permute_fusion": False,
            "apply_rope_fusion": True,
            # NemotronH uses ``silu``, which is incompatible with
            # ``bias_activation_fusion=True`` (Megatron-Core only fuses bias
            # for gelu/swiglu/quick_geglu).
            "bias_activation_fusion": False,
            "defer_fp32_logits": False,
            "moe_per_layer_logging": False,
            "moe_enable_deepep": False,
            "moe_token_dispatcher_type": "alltoall",
            "moe_shared_expert_overlap": False,
            "peft": {"enabled": False},
            "optimizer": {
                "optimizer": "adam",
                "lr": 5.0e-6,
                "min_lr": 5.0e-7,
                "weight_decay": 0.0,
                "bf16": True,
                "fp16": False,
                "params_dtype": "float32",
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1e-8,
                "sgd_momentum": 0.9,
                "use_distributed_optimizer": True,
                "use_precision_aware_optimizer": True,
                "clip_grad": 1.0,
                "optimizer_cpu_offload": False,
                "optimizer_offload_fraction": 0.0,
            },
            "scheduler": {
                "start_weight_decay": 0.0,
                "end_weight_decay": 0.0,
                "weight_decay_incr_style": "constant",
                "lr_decay_style": "constant",
                "lr_decay_iters": 10,
                "lr_warmup_iters": 0,
                "lr_warmup_init": 5.0e-7,
            },
            "distributed_data_parallel_config": {
                "grad_reduce_in_fp32": False,
                "overlap_grad_reduce": False,
                "overlap_param_gather": False,
                "use_custom_fsdp": False,
                "data_parallel_sharding_strategy": "optim_grads_params",
            },
            "fp8_cfg": {
                "enabled": False,
                "fp8": "e4m3",
                "fp8_recipe": "blockwise",
                "fp8_param": False,
            },
            "env_vars": None,
        },
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {
            "enabled": True,
            "train_mb_tokens": max_seq_len * train_micro_batch_size,
            "logprob_mb_tokens": max_seq_len * train_micro_batch_size,
            "algorithm": "modified_first_fit_decreasing",
            "sequence_length_round": 64,
        },
        # Stub generation block; ``Policy`` only reads ``colocated.enabled``.
        "generation": {
            "backend": "sglang",
            "max_new_tokens": 16,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "stop_token_ids": None,
            "stop_strings": None,
            "colocated": {
                "enabled": colocated,
                "resources": {"gpus_per_node": None, "num_nodes": None},
            },
        },
    }


# ---------------------------------------------------------------------------
# SGLang generation config builder
# ---------------------------------------------------------------------------
def make_sglang_cfg(
    *,
    model_path: str,
    sglang: SGLangShape,
    colocated: bool,
    max_seq_len: int = 1024,
    pad_token_id: int = PAD_TOKEN_ID,
    eos_token_id: int = EOS_TOKEN_ID,
) -> dict[str, Any]:
    """Build the SGLang generation config consumed by ``SGLangGeneration``.

    Field names track ``nemo_rl.models.generation.sglang.config.SGLangConfig``
    and ``SglangSpecificArgs``; SGLang-side flags (``enable_dp_attention``,
    ``enable_ep_moe``, ``ep_size``, ``dp_size``, ``pp_size``) match upstream
    ``ServerArgs`` (see ``sglang/python/sglang/srt/server_args.py``).
    """
    sglang_cfg: dict[str, Any] = {
        "model_path": model_path,
        "dtype": "bfloat16",
        "random_seed": 42,
        "context_length": max_seq_len,
        "log_level": "warning",
        "skip_server_warmup": True,
        "dp_size": sglang.dp_size,
        "pp_size": sglang.pp_size,
        "ep_size": sglang.ep_size,
        "disable_piecewise_cuda_graph": True,
        "disable_cuda_graph": sglang.disable_cuda_graph,
        # In colocate mode the trainer and engine share the same GPU, so
        # the engine's static memory fraction has to leave room for
        # Megatron's resident weights. In disaggregate the engine has the
        # GPU to itself and can claim a much larger slice for the KV
        # cache + cuda graph.
        "mem_fraction_static": 0.3 if colocated else 0.7,
    }
    if sglang.enable_dp_attention:
        sglang_cfg["enable_dp_attention"] = True
    # NOTE: ``enable_ep_moe`` was removed in newer sglang versions; EP MoE is
    # now activated implicitly when ``ep_size > 1`` (or explicitly via
    # ``moe_a2a_backend``). We rely on the implicit path here.

    weight_transfer_mode = "ipc" if colocated else "broadcast"

    return {
        "backend": "sglang",
        "model_name": model_path,
        "model_path": model_path,
        "tokenizer": {"name": model_path},
        "dtype": "bfloat16",
        "max_new_tokens": 16,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": [eos_token_id],
        "stop_strings": None,
        "_pad_token_id": pad_token_id,
        "sglang_cfg": sglang_cfg,
        "sglang_server": {
            # Total inference-side GPUs in this SGLang group. We always launch
            # exactly one engine per parametrize variant, so num_gpus equals
            # one engine's GPU count.
            "num_gpus": sglang.num_gpus_per_engine,
            "num_gpus_per_engine": sglang.num_gpus_per_engine,
            # Offload is only meaningful in colocate mode, where the trainer
            # and engine share the same GPU and the engine has to release
            # its weights / kv / cuda_graph to free room for Megatron.
            # In disaggregate the engine owns its GPU outright; turning
            # ``torch_memory_saver`` off here also avoids the side effect
            # where it forces NCCL to fall back to the ``P2P/IPC``
            # transport (which fails on hosts without inter-GPU P2P).
            "needs_offload": colocated,
            "cpu_weight_backup": False,
            "sglang_server_concurrency": 64,
            "pause_generation_mode": "retract",
            "weight_transfer_mode": weight_transfer_mode,
        },
        "sglang_router": {
            "sglang_router_ip": None,
            "sglang_router_port": None,
        },
        "sglang_kwargs": {},
    }


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
@dataclass
class TestTriple:
    """A single (megatron, sglang, mode) parametrize variant."""

    megatron: MegatronShape
    sglang: SGLangShape
    colocated: bool
    extra_marks: list[Any] = field(default_factory=list)

    @property
    def id(self) -> str:
        mode = "colo" if self.colocated else "disag"
        return f"{mode}-{self.megatron.id}-{self.sglang.id}"


def all_triples() -> list[TestTriple]:
    """Full Cartesian product the user asked for: 2 modes × 3 mcore × 3 sgl."""
    out: list[TestTriple] = []
    for colocated in (True, False):
        for m in MEGATRON_CFGS:
            for s in SGLANG_CFGS:
                out.append(TestTriple(megatron=m, sglang=s, colocated=colocated))
    return out
