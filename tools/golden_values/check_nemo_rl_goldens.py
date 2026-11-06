#!/usr/bin/env python3
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
"""Check NeMo-RL's Megatron training path against a golden file.

Rebuilds the golden file's token batch, trains NeMo-RL's Megatron policy on it
for the same number of steps (NLL loss, the recorded optimizer settings), and
compares loss, grad norm and weight updates with the file. The parallel layout
may differ from the one the file was recorded at. Exits 1 on a mismatch.

    uv run python tools/golden_values/check_nemo_rl_goldens.py \
        --golden goldens/model.json.gz --hf-model /path/to/hf_checkpoint --tp 2 --gpus-per-node 2
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nemo_rl.algorithms.loss.loss_functions import NLLLossFn  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.distributed.virtual_cluster import (  # noqa: E402
    RayVirtualCluster,
    init_ray,
)
from nemo_rl.models.megatron.golden_values import (  # noqa: E402
    DEFAULT_TOLERANCES,
    compare_goldens,
    diff_config_fields,
    read_golden_file,
    synthetic_token_batch,
    write_golden_file,
)
from nemo_rl.models.policy import MegatronConfig  # noqa: E402
from nemo_rl.models.policy.lm_policy import Policy  # noqa: E402
from nemo_rl.utils.config import (  # noqa: E402
    load_config,
    register_omegaconf_resolvers,
)

# NeMo-RL sets these provider fields from its own config, so they are copied
# from the reference run's model config rather than left at NeMo-RL's defaults.
_PROVIDER_FIELDS_SET_BY_NEMO_RL = (
    "moe_router_dtype",
    "moe_router_load_balancing_type",
    "moe_permute_fusion",
    "moe_token_dispatcher_type",
    "moe_shared_expert_overlap",
    "apply_rope_fusion",
    "bias_activation_fusion",
    "gradient_accumulation_fusion",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--golden", required=True, help="Golden file to check against")
    parser.add_argument("--hf-model", required=True, help="Same checkpoint as recorded")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")
    parser.add_argument(
        "--etp", type=int, default=1, help="Expert tensor parallel size"
    )
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--out", help="Also write this run's values here")
    parser.add_argument(
        "--tolerance",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Override a tolerance from the golden file (loss, grad_norm, weight_update)",
    )
    return parser.parse_args()


def _policy_config(args: argparse.Namespace, golden: dict[str, Any]) -> dict[str, Any]:
    """examples/configs/sft.yaml's policy, switched to Megatron and the golden settings."""
    settings = golden["settings"]
    optimizer = settings["optimizer"]
    register_omegaconf_resolvers()
    config = OmegaConf.to_container(
        load_config(REPO_ROOT / "examples/configs/sft.yaml").policy, resolve=True
    )
    config.update(
        model_name=args.hf_model,
        tokenizer={**config["tokenizer"], "name": args.hf_model},
        train_global_batch_size=settings["global_batch_size"],
        train_micro_batch_size=settings["micro_batch_size"],
        max_total_sequence_length=settings["seq_length"],
        precision="bfloat16",
        make_sequence_length_divisible_by=args.tp,
        max_grad_norm=optimizer["clip_grad"],
        dynamic_batching={**config["dynamic_batching"], "enabled": False},
    )
    config["automodel_cfg"]["enabled"] = False
    megatron = config["megatron_cfg"]
    megatron.update(
        enabled=True,
        tensor_model_parallel_size=args.tp,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=args.etp,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=args.tp > 1,
        activation_checkpointing=False,
        freeze_moe_router=False,
        train_iters=1000,
    )
    megatron["optimizer"].update(
        optimizer=optimizer["name"],
        lr=optimizer["lr"],
        min_lr=optimizer["lr"],
        weight_decay=optimizer["weight_decay"],
        adam_beta1=optimizer["adam_beta1"],
        adam_beta2=optimizer["adam_beta2"],
        adam_eps=optimizer["adam_eps"],
        clip_grad=optimizer["clip_grad"],
        use_distributed_optimizer=True,
    )
    megatron["scheduler"].update(
        lr_decay_style="constant",
        lr_warmup_iters=0,
        lr_warmup_init=optimizer["lr"],
        start_weight_decay=optimizer["weight_decay"],
        end_weight_decay=optimizer["weight_decay"],
        weight_decay_incr_style="constant",
    )
    # As recorded, and as the Super 3.5 VL Megatron recipes run. With overlap,
    # a text-only batch also trips Megatron's first-step check that every
    # parameter in a bucket got a gradient: the vision tower gets none.
    megatron["distributed_data_parallel_config"].update(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )

    reference_model_config = golden["model_config"]
    for field in _PROVIDER_FIELDS_SET_BY_NEMO_RL:
        if field in reference_model_config:
            megatron[field] = reference_model_config[field]
    model_overrides = dict(megatron.get("model_overrides") or {})
    for field, value in settings["model_overrides"].items():
        if field in MegatronConfig.__annotations__:
            megatron[field] = value
        else:
            model_overrides[field] = value
    megatron["model_overrides"] = model_overrides
    return config


def _snapshot(policy: Policy) -> dict[str, Any]:
    (snapshot,) = [
        result
        for result in policy.run_all_workers_single_data("get_golden_snapshot")
        if result is not None
    ]
    return snapshot


def main() -> None:
    args = _parse_args()
    golden = read_golden_file(args.golden)
    settings = golden["settings"]
    tokens = synthetic_token_batch(
        num_rows=settings["global_batch_size"],
        seq_length=settings["seq_length"],
        seed=settings["seed"],
        token_low=settings["token_low"],
        token_high=settings["token_high"],
        excluded_token_ids=settings["excluded_token_ids"],
    )
    if hashlib.sha256(tokens.numpy().tobytes()).hexdigest() != settings["batch_sha256"]:
        raise RuntimeError("Rebuilt token batch does not match the golden file's")

    init_ray()
    cluster = RayVirtualCluster(
        name="golden_check",
        bundle_ct_per_node_list=[args.gpus_per_node] * args.num_nodes,
        use_gpus=True,
        num_gpus_per_node=args.gpus_per_node,
        max_colocated_worker_groups=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    policy = Policy(
        cluster=cluster,
        config=_policy_config(args, golden),
        tokenizer=tokenizer,
        init_optimizer=True,
        init_reference_model=False,
    )

    num_rows, seq_length = tokens.shape
    # Every position but the last predicts the next token, as in the recording.
    data = BatchedDataDict(
        {
            "input_ids": tokens,
            "input_lengths": torch.full((num_rows,), seq_length, dtype=torch.int32),
            "token_mask": torch.ones(num_rows, seq_length),
            "sample_mask": torch.ones(num_rows),
        }
    )
    policy.prepare_for_training()
    snapshots = [_snapshot(policy)]
    steps = []
    for step in range(1, settings["steps"] + 1):
        results = policy.train(
            data,
            NLLLossFn(),
            gbs=settings["global_batch_size"],
            mbs=settings["micro_batch_size"],
        )
        steps.append(
            {"loss": float(results["loss"]), "grad_norm": float(results["grad_norm"])}
        )
        print(f"step {step}: {steps[-1]}", flush=True)
        snapshots.append(_snapshot(policy))
    policy.shutdown()

    actual = {
        "steps": steps,
        "weights": [snapshot["weights"] for snapshot in snapshots],
        "model_config": snapshots[0]["model_config"],
        "layout": {
            "tp": args.tp,
            "ep": args.ep,
            "etp": args.etp,
            "world_size": args.num_nodes * args.gpus_per_node,
        },
    }
    if args.out:
        write_golden_file(args.out, {**golden, **actual, "recorded_by": "nemo-rl"})

    tolerances = {**DEFAULT_TOLERANCES, **golden["tolerances"]}
    for item in args.tolerance:
        metric, _, value = item.partition("=")
        tolerances[metric] = float(value)
    mismatches, worst = compare_goldens(golden, actual, tolerances)
    config_diff = diff_config_fields(golden["model_config"], actual["model_config"])

    print(f"Golden: {args.golden} (recorded at {golden['layout']})")
    print(f"Check run layout: {actual['layout']}")
    for step, (expected, got) in enumerate(zip(golden["steps"], steps), start=1):
        print(
            f"  step {step}: loss {expected['loss']:.6f} vs {got['loss']:.6f}, "
            f"grad_norm {expected['grad_norm']:.6f} vs {got['grad_norm']:.6f}"
        )
    for metric, error in worst.items():
        print(
            f"  worst {metric} relative error {error:.3g} (tolerance {tolerances[metric]:g})"
        )
    if config_diff:
        print("Model config fields that differ from the reference run:")
        for name, (expected, got) in config_diff.items():
            print(f"  {name}: {expected!r} vs {got!r}")
    if mismatches:
        print(f"FAIL: {len(mismatches)} mismatches")
        if any(m.metric == "initial_weights" for m in mismatches):
            print(
                "  The check run started from different weights than the golden. NeMo-RL "
                "reuses its HF->Megatron conversion under $NRL_MEGATRON_CHECKPOINT_DIR "
                "(default $HF_HOME/nemo_rl), keyed by the checkpoint path, so a checkpoint "
                "rewritten at the same path loads stale weights."
            )
        for mismatch in mismatches[:50]:
            print(f"  {mismatch}")
        sys.exit(1)
    print("PASS")
    ray.shutdown()


if __name__ == "__main__":
    main()
