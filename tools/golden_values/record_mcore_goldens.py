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
"""Record golden values for a model from Megatron-Bridge and Megatron-Core alone.

Trains a HuggingFace checkpoint for a few optimizer steps on a fixed token batch
and writes, per step, the loss and grad norm, plus a fingerprint of every weight
before the first step and after each one. Nothing here goes through NeMo-RL's
training code: the model comes from the Megatron-Bridge provider (with the HF
weights), the gradient-sync callbacks are installed the way Megatron-Bridge's
own setup does, and each step is Megatron-Core's forward-backward schedule plus
the distributed optimizer. ``check_nemo_rl_goldens.py`` replays the same batch
through NeMo-RL and compares. See README.md in this directory.

Launch one process per GPU, for example:

    torchrun --nproc-per-node 4 tools/golden_values/record_mcore_goldens.py \
        --hf-model /path/to/hf_checkpoint --tp 2 --ep 2 --out goldens/model.json.gz
"""

import argparse
import functools
import hashlib
import inspect
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from megatron.bridge import AutoBridge  # noqa: E402
from megatron.bridge.training.setup import _update_model_config_funcs  # noqa: E402
from megatron.core import parallel_state  # noqa: E402
from megatron.core.distributed import DistributedDataParallelConfig  # noqa: E402
from megatron.core.optimizer import (  # noqa: E402
    OptimizerConfig,
    get_megatron_optimizer,
)
from megatron.core.pipeline_parallel import get_forward_backward_func  # noqa: E402
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.utils import get_model_config  # noqa: E402

from nemo_rl.models.megatron.golden_values import (  # noqa: E402
    DEFAULT_TOLERANCES,
    GOLDEN_FORMAT_VERSION,
    fingerprint_named_tensors,
    scalar_config_fields,
    synthetic_token_batch,
    write_golden_file,
)

# Provider settings every golden run uses unless overridden with --override.
# Dropout draws random numbers the two paths can't share; trainers scale the
# MTP and MoE auxiliary losses differently; and the expert-bias update would
# change step 2's routing for reasons unrelated to the gradient. MTP layers stay
# (some architectures, like Nemotron 3.5 Super VL, require them) but their loss
# is zeroed. The checker applies the same values.
GOLDEN_MODEL_OVERRIDES: dict[str, Any] = {
    "attention_dropout": 0.0,
    "hidden_dropout": 0.0,
    "mtp_loss_scaling_factor": 0.0,
    "moe_aux_loss_coeff": 0.0,
    "moe_z_loss_coeff": None,
    "moe_router_bias_update_rate": 0.0,
    # Needs APEX's fused weight-gradient extension; NeMo-RL turns it off too.
    "gradient_accumulation_fusion": False,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hf-model", required=True, help="HF checkpoint path or id")
    parser.add_argument("--out", required=True, help="Golden file (.json or .json.gz)")
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Optimizer steps. A second step re-checks the updated weights; on small "
        "random-weight models it amplifies numerical noise, so compare it loosely",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")
    parser.add_argument(
        "--etp", type=int, default=None, help="Expert tensor parallel size"
    )
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--token-range",
        type=int,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Synthetic token ids are drawn from [LOW, HIGH); default [1000, min(vocab, 32000))",
    )
    # Adam (the only optimizer Megatron's distributed optimizer supports) moves
    # each weight by lr * g / (|g| + eps) on its first step. With the usual tiny
    # eps that is lr * sign(g), so a near-zero gradient element whose sign differs
    # by numerical noise between layouts moves by a full +-lr. A large eps keeps
    # the update about (lr / eps) * g, proportional to the gradient.
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-2)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Default: chosen so that a clipped step moves the weights by --update-rms",
    )
    parser.add_argument(
        "--update-rms",
        type=float,
        default=2e-3,
        help="Target RMS weight change per step for the default --lr; well above "
        "bf16 rounding of typical weights",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="FIELD=JSON",
        help="Extra provider setting, value parsed as JSON (repeatable)",
    )
    return parser.parse_args()


def _git_sha(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _model_identity(hf_model: str) -> dict[str, Any]:
    config_path = Path(hf_model) / "config.json"
    identity: dict[str, Any] = {"hf_model": hf_model}
    if config_path.is_file():
        identity["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return identity


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for module in ("megatron.core", "megatron.bridge", "transformer_engine"):
        try:
            versions[module] = getattr(
                __import__(module, fromlist=["_"]), "__version__"
            )
        except (ImportError, AttributeError):
            versions[module] = "unknown"
    return versions


def _unwrapped(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


def _microbatches(
    tokens: torch.Tensor, micro_batch_size: int
) -> Iterator[dict[str, torch.Tensor]]:
    """Next-token batches: every position but the last predicts the next token."""
    device = torch.device("cuda", torch.cuda.current_device())
    for start in range(0, tokens.shape[0], micro_batch_size):
        rows = tokens[start : start + micro_batch_size].to(device)
        labels = torch.roll(rows, shifts=-1, dims=1)
        loss_mask = torch.ones_like(rows, dtype=torch.float32)
        loss_mask[:, -1] = 0.0
        position_ids = torch.arange(rows.shape[1], device=device).expand_as(rows)
        yield {
            "tokens": rows,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }


def _token_loss(
    loss_mask: torch.Tensor, output: torch.Tensor | tuple[torch.Tensor, Any]
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Megatron's per-token loss contract: (loss sum, token count, report)."""
    if isinstance(output, tuple):
        # Multimodal models may return the loss mask they used after expansion.
        output, model_loss_mask = output
        if model_loss_mask is not None:
            loss_mask = model_loss_mask
    mask = loss_mask.reshape(-1).float()
    loss_sum = (output.float().reshape(-1) * mask).sum()
    num_tokens = mask.sum().to(torch.int)
    report = torch.stack([loss_sum.detach(), num_tokens.detach().float()])
    return loss_sum, num_tokens, {"loss_and_tokens": report}


def _forward_step(data_iterator, model, *, forward_parameters: frozenset[str]):
    batch = next(data_iterator)
    kwargs = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": None,
        "labels": batch["labels"],
    }
    if "loss_mask" in forward_parameters:
        kwargs["loss_mask"] = batch["loss_mask"]
    if "images" in forward_parameters:
        # Text-only batch for a vision-language model: the empty-image sentinel
        # Megatron-Bridge's own multimodal step uses.
        kwargs["images"] = torch.empty(
            0, 0, 0, dtype=torch.bfloat16, device=batch["tokens"].device
        )
    output = model(**kwargs)
    return output, functools.partial(_token_loss, batch["loss_mask"])


def _export_fingerprints(bridge, model: list, is_writer: bool) -> dict[str, Any]:
    """Every rank joins the export's collectives; only the writer fingerprints."""
    weights = bridge.export_hf_weights(model, cpu=False, show_progress=False)
    if is_writer:
        return fingerprint_named_tensors(weights)
    for _ in weights:
        pass
    return {}


def main() -> None:
    args = _parse_args()
    overrides = dict(GOLDEN_MODEL_OVERRIDES)
    for item in args.override:
        field, _, value = item.partition("=")
        overrides[field] = json.loads(value)

    bridge = AutoBridge.from_hf_pretrained(args.hf_model, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=True)
    for field, value in overrides.items():
        if not hasattr(provider, field):
            raise ValueError(f"{type(provider).__name__} has no field {field!r}")
        setattr(provider, field, value)
    provider.tensor_model_parallel_size = args.tp
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.expert_model_parallel_size = args.ep
    provider.expert_tensor_parallel_size = args.etp
    provider.sequence_parallel = args.tp > 1
    # Losses below are per-token sums; finalization divides by the global token
    # count, so the gradient is that of the global token-mean loss.
    provider.calculate_per_token_loss = True
    provider.finalize()
    provider.initialize_model_parallel(seed=args.seed)

    rank = torch.distributed.get_rank()
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    if args.global_batch_size % (dp_size * args.micro_batch_size):
        raise ValueError(
            f"--global-batch-size {args.global_batch_size} is not a multiple of "
            f"data-parallel size {dp_size} x --micro-batch-size {args.micro_batch_size}"
        )

    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        average_in_collective=False,
        use_distributed_optimizer=True,
        check_for_nan_in_grad=True,
    )
    model = provider.provide_distributed_model(
        ddp_config=ddp_config, wrap_with_ddp=True
    )
    if len(model) != 1:
        raise NotImplementedError("Virtual pipeline stages are not supported")
    vocab_size = provider.vocab_size
    token_low, token_high = args.token_range or (1000, min(vocab_size, 32000))
    excluded = sorted(
        {
            getattr(provider, field)
            for field in ("image_token_index", "img_start_token_id", "img_end_token_id")
            if isinstance(getattr(provider, field, None), int)
        }
    )
    tokens = synthetic_token_batch(
        num_rows=args.global_batch_size,
        seq_length=args.seq_length,
        seed=args.seed,
        token_low=token_low,
        token_high=token_high,
        excluded_token_ids=excluded,
    )
    rows_per_rank = args.global_batch_size // dp_size
    local_tokens = tokens[dp_rank * rows_per_rank : (dp_rank + 1) * rows_per_rank]
    num_microbatches = rows_per_rank // args.micro_batch_size

    forward_parameters = frozenset(
        inspect.signature(_unwrapped(model[0]).forward).parameters
    )
    forward_backward_func = get_forward_backward_func()
    is_writer = rank == 0
    weights = [_export_fingerprints(bridge, model, is_writer)]
    lr = args.lr
    if lr is None:
        # A clipped gradient has norm --clip-grad, so its RMS over N weights is
        # clip / sqrt(N), and the step moves weights by (lr / eps) times that.
        num_weights = torch.tensor(
            [sum(math.prod(f["shape"]) for f in weights[0].values())],
            dtype=torch.float64,
            device=torch.cuda.current_device(),
        )
        torch.distributed.broadcast(num_weights, src=0)
        lr = (
            args.update_rms
            * args.adam_eps
            * math.sqrt(num_weights.item())
            / args.clip_grad
        )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=lr,
            weight_decay=0.0,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_eps=args.adam_eps,
            clip_grad=args.clip_grad,
            use_distributed_optimizer=True,
            bf16=provider.bf16,
            fp16=provider.fp16,
            params_dtype=provider.params_dtype,
        ),
        model,
    )
    # Megatron-Bridge's setup() binding since Megatron-Bridge #6067: the runtime
    # config the scheduler reads, not the provider.
    _update_model_config_funcs(
        model,
        get_model_config(model[0]),
        ddp_config,
        optimizer,
        align_grad_reduce=True,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    steps = []
    for step in range(1, args.steps + 1):
        for chunk in model:
            chunk.zero_grad_buffer()
        optimizer.zero_grad()
        losses = forward_backward_func(
            forward_step_func=functools.partial(
                _forward_step, forward_parameters=forward_parameters
            ),
            data_iterator=_microbatches(local_tokens, args.micro_batch_size),
            model=model,
            num_microbatches=num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=False,
        )
        update_successful, grad_norm, num_zeros = optimizer.step()
        if not update_successful:
            raise RuntimeError(f"Optimizer step {step} was skipped")
        totals = torch.zeros(2, device=torch.cuda.current_device())
        for loss in losses:
            totals += loss["loss_and_tokens"]
        torch.distributed.all_reduce(
            totals, group=parallel_state.get_data_parallel_group()
        )
        steps.append(
            {
                "loss": (totals[0] / totals[1]).item(),
                "grad_norm": float(grad_norm),
                "num_zeros_in_grad": None if num_zeros is None else int(num_zeros),
            }
        )
        weights.append(_export_fingerprints(bridge, model, is_writer))
        if is_writer:
            print(f"step {step}: {steps[-1]}", flush=True)

    if is_writer:
        golden = {
            "format_version": GOLDEN_FORMAT_VERSION,
            "recorded_by": "megatron-core",
            "created_unix": int(time.time()),
            "model": _model_identity(args.hf_model),
            "provider": type(provider).__name__,
            # The scheduler's config, for diffing against the check run's.
            "model_config": scalar_config_fields(get_model_config(model[0])),
            "layout": {
                "world_size": torch.distributed.get_world_size(),
                "tp": args.tp,
                "ep": args.ep,
                "etp": args.etp,
                "dp": dp_size,
            },
            "settings": {
                "steps": args.steps,
                "global_batch_size": args.global_batch_size,
                "micro_batch_size": args.micro_batch_size,
                "seq_length": args.seq_length,
                "seed": args.seed,
                "token_low": token_low,
                "token_high": token_high,
                "excluded_token_ids": excluded,
                "batch_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
                "optimizer": {
                    "name": "adam",
                    "lr": lr,
                    "adam_beta1": args.adam_beta1,
                    "adam_beta2": args.adam_beta2,
                    "adam_eps": args.adam_eps,
                    "clip_grad": args.clip_grad,
                    "weight_decay": 0.0,
                },
                "model_overrides": overrides,
                "precision": "bf16" if provider.bf16 else str(provider.params_dtype),
            },
            "tolerances": DEFAULT_TOLERANCES,
            "versions": _versions(),
            "git": {
                "nemo_rl": _git_sha(REPO_ROOT),
                "megatron_bridge": _git_sha(
                    REPO_ROOT / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
                ),
            },
            "gpu": torch.cuda.get_device_name(),
            "steps": steps,
            "weights": weights,
        }
        write_golden_file(args.out, golden)
        print(f"Wrote {args.out}: {len(weights[0])} weights, {len(steps)} steps")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
