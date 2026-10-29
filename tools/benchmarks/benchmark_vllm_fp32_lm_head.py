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

"""Compare native FP32 LM-head numerics and generation on a fixed workload.

Run each arm in a fresh process on the same otherwise idle GPU. TP=1 exercises
NeMo RL's real synchronous worker initialization; TP>1 uses vLLM's MP executor
with the same NeMo RL configuration adapter. --analyze captures hidden states
only AFTER the timed repetitions, then evaluates both head paths against FP64
using the same BF16 states and live checkpoint weight. It does not train a model.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from nemo_rl.models.generation.vllm.config import configure_fp32_lm_head
from nemo_rl.models.generation.vllm.patches import ensure_vllm_source_compat


def inspect_head(worker: Any) -> dict[str, Any]:
    model = worker.model_runner.model
    return {
        "model_class": type(model).__name__,
        "head_dtype": str(model.logits_processor.head_dtype),
        "weight_dtype": str(model.lm_head.weight.dtype),
        "weight_shape": list(model.lm_head.weight.shape),
    }


def install_capture(worker: Any) -> None:
    model = worker.model_runner.model
    original = model.compute_logits
    worker.fp32_benchmark_hidden = []
    worker.fp32_benchmark_remaining = 2048

    def capture(hidden: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        result = original(hidden, *args, **kwargs)
        count = min(worker.fp32_benchmark_remaining, hidden.shape[0])
        if count:
            worker.fp32_benchmark_hidden.append(hidden[:count].detach().cpu())
            worker.fp32_benchmark_remaining -= count
        return result

    model.compute_logits = capture


def error_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.double().cpu()
    return {
        "mean": values.mean().item(),
        "p99": values.quantile(0.99).item(),
        "max": values.max().item(),
    }


@torch.inference_mode()
def analyze_head(worker: Any) -> dict[str, Any]:
    model = worker.model_runner.model
    head = model.lm_head
    processor = model.logits_processor
    original_dtype = processor.head_dtype
    hidden = torch.cat(worker.fp32_benchmark_hidden).to(head.weight.device)
    weight = head.weight[: processor.org_vocab_size]
    reference_weight = weight.double()
    metrics = {
        arm: {
            "max_abs_logit_error": 0.0,
            "argmax_flips": 0,
            "logprob": [],
            "sampled_logprob": [],
            "kl": [],
        }
        for arm in ("bf16", "fp32")
    }
    torch.manual_seed(42)
    rounded_equal = 0
    logit_count = 0
    for chunk in hidden.split(32):
        reference = chunk.double() @ reference_weight.t()
        if processor.soft_cap is not None:
            reference = processor.soft_cap * torch.tanh(reference / processor.soft_cap)
        reference = reference * processor.scale
        ref_lp = reference.log_softmax(-1)
        tokens = reference.argmax(-1, keepdim=True)
        sampled = torch.multinomial(ref_lp.exp(), 8, replacement=True)
        outputs = {}
        for arm, dtype in (("bf16", chunk.dtype), ("fp32", torch.float32)):
            processor.head_dtype = dtype
            output = processor(head, chunk).float()
            outputs[arm] = output
            result = metrics[arm]
            result["max_abs_logit_error"] = max(
                result["max_abs_logit_error"],
                (output.double() - reference).abs().max().item(),
            )
            result["argmax_flips"] += (
                (output.argmax(-1) != tokens.squeeze(-1)).sum().item()
            )
            lp = output.log_softmax(-1).double()
            result["logprob"].append(
                (lp.gather(-1, tokens) - ref_lp.gather(-1, tokens))
                .abs()
                .flatten()
                .cpu()
            )
            result["sampled_logprob"].append(
                (lp.gather(-1, sampled) - ref_lp.gather(-1, sampled))
                .abs()
                .flatten()
                .cpu()
            )
            # FP64 normalization isolates projection error from float32 softmax rounding.
            result["kl"].append(
                (ref_lp.exp() * (ref_lp - output.double().log_softmax(-1)))
                .sum(-1)
                .cpu()
            )
        rounded_equal += (
            (outputs["bf16"] == outputs["fp32"].to(chunk.dtype).float()).sum().item()
        )
        logit_count += outputs["bf16"].numel()
    for result in metrics.values():
        for key in ("logprob", "sampled_logprob", "kl"):
            result[key] = error_stats(torch.cat(result[key]))
    del reference_weight
    timings = {}
    for rows in (1, 8, 64, 512, 4096):
        h = hidden.repeat((rows + hidden.shape[0] - 1) // hidden.shape[0], 1)[:rows]
        samples = {"bf16": [], "fp32": []}
        for iteration in range(25):
            arms = [("bf16", h.dtype), ("fp32", torch.float32)]
            if iteration % 2:
                arms.reverse()
            for arm, dtype in arms:
                processor.head_dtype = dtype
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                processor._apply_head(head, h, None).float()
                end.record()
                end.synchronize()
                if iteration >= 5:
                    samples[arm].append(start.elapsed_time(end))
        timings[rows] = {
            arm: {"median_ms": statistics.median(v), "samples_ms": v}
            for arm, v in samples.items()
        }
    processor.head_dtype = original_dtype
    return {
        "positions": hidden.shape[0],
        "logit_entries": logit_count,
        "bf16_equals_rounded_fp32_fraction": rounded_equal / logit_count,
        "metrics": metrics,
        "head_timings": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="JSON list with GSM8K question/answer fields",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    if args.analyze and args.tp != 1:
        parser.error("--analyze requires --tp=1")
    ensure_vllm_source_compat()
    # vLLM is optional and must be imported after source compatibility setup.
    import vllm

    records = json.loads(args.prompts.read_text())[: args.requests]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        {
            "prompt_token_ids": tokenizer.apply_chat_template(
                [{"role": "user", "content": record["question"]}],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=False,
            )
        }
        for record in records
    ]
    engine_kwargs = {
        "max_num_seqs": args.requests,
        "attention_config": {"backend": "FLASH_ATTN"},
        "max_num_batched_tokens": 4096,
        "logprobs_mode": "raw_logprobs",
    }
    cfg = {
        "precision": "bfloat16",
        "tensor_parallel_size": args.tp,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 1,
        "gpu_memory_utilization": 0.65,
        "max_model_len": 2048,
        "async_engine": False,
        "skip_tokenizer_init": False,
        "load_format": "auto",
        "enforce_eager": args.eager,
        "kv_cache_dtype": "auto",
        "fp32_lm_head": args.fp32,
        "enable_prefix_caching": False,
    }
    if args.tp == 1:
        # Exercise the same argument construction used by sync and async NeMo workers.
        from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

        worker = VllmGenerationWorkerImpl(
            {"model_name": args.model, "vllm_cfg": cfg, "vllm_kwargs": engine_kwargs},
            bundle_indices=[],
            seed=42,
        )
        llm = worker.llm
    else:
        configure_fp32_lm_head(cfg, engine_kwargs)
        llm = vllm.LLM(
            model=args.model,
            dtype="bfloat16",
            tensor_parallel_size=args.tp,
            distributed_executor_backend="mp",
            enable_prefix_caching=False,
            seed=42,
            max_model_len=2048,
            enforce_eager=args.eager,
            gpu_memory_utilization=0.65,
            **engine_kwargs,
        )
    params = vllm.SamplingParams(
        temperature=0, max_tokens=args.new_tokens, ignore_eos=True, logprobs=1
    )
    llm.generate(prompts, params, use_tqdm=False)  # Warmup is not timed.
    runs = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        seconds = time.perf_counter() - start
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        runs.append(
            {
                "seconds": seconds,
                "tokens": tokens,
                "tokens_per_second": tokens / seconds,
            }
        )
    heads = llm.collective_rpc(inspect_head)
    expected = "torch.float32" if args.fp32 else "torch.bfloat16"
    assert all(
        h["head_dtype"] == expected and h["weight_dtype"] == "torch.bfloat16"
        for h in heads
    ), heads
    result = {
        "args": vars(args) | {"prompts": str(args.prompts), "output": str(args.output)},
        "nemo_rl_base": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "generation_config": cfg,
        "vllm_kwargs": engine_kwargs,
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "host": platform.node(),
        "head_metadata": heads,
        "runs": runs,
        "median_seconds": statistics.median(r["seconds"] for r in runs),
        "outputs": [
            {
                "token_ids": list(o.outputs[0].token_ids),
                "text": o.outputs[0].text,
                "logprobs": [
                    lp[token].logprob
                    for token, lp in zip(o.outputs[0].token_ids, o.outputs[0].logprobs)
                ],
            }
            for o in outputs
        ],
    }
    if args.analyze:
        llm.collective_rpc(install_capture)
        llm.generate(
            prompts,
            vllm.SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
            use_tqdm=False,
        )
        result["analysis"] = llm.collective_rpc(analyze_head)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        f"RESULT {args.output}: median {result['median_seconds']:.4f}s; {heads}",
        flush=True,
    )
    # EngineCore and Ray resources are released by the enclosing fresh process.


if __name__ == "__main__":
    main()
