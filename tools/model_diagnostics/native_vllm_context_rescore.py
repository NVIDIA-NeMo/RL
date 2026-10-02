# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Compare native stochastic decode with fresh prefill at fixed weights.

Uses a captured Super VL prompt/pixels, without Gym, GRPO, or weight updates.
Checks every token since the previous probe, including the final short interval.
A new request and disabled prefix caching give each probe a fresh prefill.
Large differences are diagnostic failures, not proof that the cache is stale.
The optional V2 worker observer retains GPU input/sample snapshots; it adds
GPU copies and can affect timing. No production vLLM files are modified.
"""

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any


def compare_scores(
    decode: list[float], fresh: list[float], *, start: int, threshold: float
) -> tuple[float, int | None]:
    """Return largest checked difference and first failing response index."""
    if len(decode) != len(fresh) or not 0 <= start <= len(decode):
        raise ValueError("Score lengths or checked interval do not match")
    maximum = 0.0
    first = None
    for index in range(start, len(decode)):
        difference = abs(decode[index] - fresh[index])
        if not math.isfinite(decode[index]) or not math.isfinite(fresh[index]):
            difference = math.inf
        maximum = max(maximum, difference)
        if first is None and difference > threshold:
            first = index
    return maximum, first


def install_worker_trace(worker: Any, capacity: int) -> dict[str, Any]:
    """Retain bounded GPU snapshots of V2 inputs and pre-copy sampler output."""
    # GPU dependencies are optional until this callback executes in a worker.
    import torch

    rank = torch.distributed.get_rank()
    runner = worker.model_runner
    description = {"rank": rank, "runner": type(runner).__module__}
    if rank != 0:
        return description
    if not hasattr(runner, "prepare_inputs") or not hasattr(
        runner, "execute_model_state"
    ):
        raise RuntimeError("Worker input observer requires the current V2 model runner")
    history = deque(maxlen=capacity)
    original_prepare = runner.prepare_inputs
    original_sample = runner.sample_tokens
    pending = None
    sequence = 0

    def prepare(*args: Any, **kwargs: Any) -> Any:
        nonlocal pending, sequence
        batch = original_prepare(*args, **kwargs)
        pending = None
        if not any(request.startswith("native-main-") for request in batch.req_ids):
            return batch
        if batch.num_draft_tokens:
            raise RuntimeError("This diagnostic requires speculative decoding disabled")
        indices = batch.idx_mapping
        positions = batch.positions[batch.logits_indices].long()
        fields = torch.stack(
            [
                positions,
                indices.long(),
                batch.input_ids[batch.logits_indices].long(),
                runner.req_states.last_sampled_tokens[indices].flatten().long(),
                runner.req_states.all_token_ids.gpu[indices, positions].long(),
                batch.seq_lens[: batch.num_reqs].long(),
                runner.req_states.num_computed_tokens.gpu[indices].long(),
            ],
            dim=-1,
        )
        pending = {
            "step": sequence,
            "request_ids": list(batch.req_ids),
            "columns": [
                "forward_position",
                "request_slot",
                "input_token",
                "last_sampled_token",
                "all_token_ids_at_position",
                "sequence_length",
                "computed_tokens_before_forward",
            ],
            "inputs": fields.detach().clone(),
        }
        sequence += 1
        return batch

    def sample(*args: Any, **kwargs: Any) -> Any:
        nonlocal pending
        entry = pending
        if entry is not None:
            state_indices = getattr(runner.model_state, "_mamba_state_idx_gpu", None)
            if state_indices is not None:
                entry["mamba_state_indices_after_forward"] = (
                    state_indices[runner.execute_model_state.input_batch.idx_mapping]
                    .detach()
                    .clone()
                )
        output = original_sample(*args, **kwargs)
        if entry is not None and output is not None:
            sampled = output.sampler_output
            entry["sampled_token_ids"] = sampled.sampled_token_ids.detach().clone()
            entry["num_sampled_tokens"] = output.num_sampled_tokens.detach().clone()
            if sampled.logprobs_tensors is not None:
                entry["logprob_token_ids"] = (
                    sampled.logprobs_tensors.logprob_token_ids.detach().clone()
                )
                entry["logprobs"] = sampled.logprobs_tensors.logprobs.detach().clone()
            history.append(entry)
        pending = None
        return output

    runner.prepare_inputs = prepare
    runner.sample_tokens = sample
    worker._native_context_trace = history
    return {
        **description,
        "capacity": capacity,
        "observer": "GPU clones; no per-step CPU reads",
    }


def read_worker_trace(
    worker: Any, request_id: str, target_position: int
) -> dict[str, Any]:
    """Retrieve historical input/sample rows around a failed response token."""
    import torch

    rank = torch.distributed.get_rank()
    report = {"rank": rank, "records": []}
    if rank != 0:
        return report
    history = getattr(worker, "_native_context_trace", ())
    report["retained_steps"] = len(history)
    for entry in history:
        if request_id not in entry["request_ids"]:
            continue
        row = entry["request_ids"].index(request_id)
        values = entry["inputs"][row].cpu().tolist()
        # A forward at position p-1 produces the score/sample at position p.
        if not target_position - 17 <= values[0] <= target_position + 7:
            continue
        record = {
            "step": entry["step"],
            **dict(zip(entry["columns"], values, strict=True)),
        }
        for key in (
            "sampled_token_ids",
            "num_sampled_tokens",
            "logprob_token_ids",
            "logprobs",
            "mamba_state_indices_after_forward",
        ):
            if key in entry:
                record[key] = entry[key][row].cpu().tolist()
        report["records"].append(record)
    return report


def load_prompt(
    args: argparse.Namespace, model_path: str
) -> tuple[Any, dict[str, Any]]:
    """Prepare the exact captured images and original response prompt."""
    import torch
    from einops import rearrange
    from transformers import AutoTokenizer, BatchFeature
    from vllm.inputs.engine import mm_input
    from vllm.multimodal.inputs import (
        MultiModalFieldConfig,
        MultiModalKwargsItems,
        PlaceholderRange,
    )

    from nemo_rl.data.multimodal_utils import reassemble_packed_multimodal

    records = [
        json.loads(line)
        for line in args.sample.with_name("samples.jsonl").read_text().splitlines()
    ]
    record = next(row for row in records if row["sample_id"] == args.sample.stem)
    payload = torch.load(args.sample, map_location="cpu", weights_only=False)
    first_response = int((payload["token_mask"][0] > 0).nonzero()[0])
    ids = payload["input_ids"][0, :first_response].tolist()
    media = {key: payload[key] for key in ("pixel_values", "imgs_sizes", "num_frames")}
    reassemble_packed_multimodal(media, [record["tags"]])
    pixels, sizes = media["pixel_values"].tensors[0], media["imgs_sizes"].tensors[0]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    image_id = tokenizer.convert_tokens_to_ids("<image>")
    spans = []
    for position, token in enumerate(ids):
        if token == image_id:
            if spans and sum(spans[-1]) == position:
                spans[-1][1] += 1
            else:
                spans.append([position, 1])
    assert len(spans) == len(sizes)
    patch_size = math.isqrt(pixels.shape[1] // 3)
    assert 3 * patch_size**2 == pixels.shape[1]
    images, offset = [], 0
    for (height, width), (_, length) in zip(sizes.tolist(), spans, strict=True):
        patches = (height // patch_size) * (width // patch_size)
        assert patches == length * 4
        images.append(
            rearrange(
                pixels[offset : offset + patches],
                "(py px) (c yy xx) -> c (py yy) (px xx)",
                py=height // patch_size,
                px=width // patch_size,
                c=3,
                yy=patch_size,
                xx=patch_size,
            )
        )
        offset += patches
    assert offset == len(pixels)
    inputs = BatchFeature(
        {
            "pixel_values_flat": images,
            "imgs_sizes": sizes.tolist(),
            "num_tokens_per_image": [length for _, length in spans],
            "image_num_patches": torch.ones(len(images), dtype=torch.int64),
        }
    )
    fields = {
        key: MultiModalFieldConfig.batched(
            "image", keep_on_cpu=key != "pixel_values_flat"
        )
        for key in inputs
    }
    mm_kwargs = MultiModalKwargsItems.from_hf_inputs(inputs, fields)
    prompt = mm_input(
        ids,
        mm_kwargs,
        {"image": [f"native-fixed-image-{index}" for index in range(len(images))]},
        {
            "image": [
                PlaceholderRange(offset=start, length=length) for start, length in spans
            ]
        },
    )
    return prompt, {
        "tokenizer": tokenizer,
        "prompt_ids": ids,
        "prepared_images": images,
        "image_sizes": sizes.tolist(),
        "source_record": record,
    }


async def run(args: argparse.Namespace) -> int:
    """Run stochastic native requests and periodic independent prefix probes."""
    import torch
    from omegaconf import OmegaConf

    from nemo_rl.models.generation.vllm.patches import _apply_vllm_patches

    config = OmegaConf.to_container(
        OmegaConf.load(args.generation_config), resolve=True
    )
    generation = config["policy"]["generation"]
    model_path = config["policy"]["model_name"]
    os.environ.update(generation["vllm_cfg"]["env_vars"])
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    _apply_vllm_patches(
        sys.executable, nemotron_h_fp32_lm_head=generation["vllm_cfg"]["fp32_lm_head"]
    )
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    kwargs = dict(generation["vllm_kwargs"])
    kwargs.update(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        dtype="bfloat16",
        max_model_len=65536,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.8,
        logprobs_mode="raw_logprobs",
    )
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "initializing",
        "vllm_version": importlib.metadata.version("vllm"),
        "python": sys.executable,
        "sample_path": str(args.sample),
        "generation_config": str(args.generation_config),
        "engine_kwargs": kwargs,
        "requests": args.requests,
        "rounds": args.rounds,
        "check_every": args.check_every,
        "threshold": args.threshold,
        "worker_trace": not args.no_worker_trace,
        "fixed_weights": True,
        "probes": 0,
        "checked_tokens": 0,
        "completed_requests": 0,
        "maximum_abs_logprob_difference": 0.0,
    }

    def save_status() -> None:
        (args.output / "status.json").write_text(json.dumps(report, indent=2) + "\n")

    save_status()
    prompt, prepared = load_prompt(args, model_path)
    prompt_ids = prepared["prompt_ids"]
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
    active = set()
    failed = asyncio.Event()
    probe_lock = asyncio.Lock()
    started = time.monotonic()

    async def check(
        request_id: str, tokens: list[int], logprobs: list[float], start: int
    ) -> None:
        async with probe_lock:
            if failed.is_set():
                return
            probe_id = "native-rescore-" + uuid.uuid4().hex
            probe = dict(prompt)
            probe["prompt_token_ids"] = prompt_ids + tokens
            result = None
            params = SamplingParams(
                max_tokens=1, prompt_logprobs=1, temperature=1, top_p=1
            )
            async for output in engine.generate(probe, params, request_id=probe_id):
                result = output
            assert result is not None and result.prompt_logprobs is not None
            fresh = [
                float(result.prompt_logprobs[len(prompt_ids) + index][token].logprob)
                for index, token in enumerate(tokens)
            ]
            maximum, first = compare_scores(
                logprobs, fresh, start=start, threshold=args.threshold
            )
            report["probes"] += 1
            report["checked_tokens"] += len(tokens) - start
            report["maximum_abs_logprob_difference"] = max(
                report["maximum_abs_logprob_difference"], maximum
            )
            event = {
                "request_id": request_id,
                "probe_id": probe_id,
                "checked_response_interval": [start, len(tokens)],
                "maximum_abs_logprob_difference": maximum,
                "first_mismatch_response_index": first,
                "elapsed_seconds": time.monotonic() - started,
            }
            with (args.output / "checks.jsonl").open("a") as log:
                log.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            if first is not None:
                failed.set()
                position = len(prompt_ids) + first
                all_ids = prompt_ids + tokens
                diagnostic = {
                    **event,
                    "absolute_position": position,
                    "token_id": tokens[first],
                    "previous_token_id": all_ids[position - 1],
                    "decode_logprob": logprobs[first],
                    "fresh_prefill_logprob": fresh[first],
                    "context_before": prepared["tokenizer"].decode(
                        all_ids[max(0, position - 24) : position]
                    ),
                    "context_including_token": prepared["tokenizer"].decode(
                        all_ids[max(0, position - 24) : position + 1]
                    ),
                    "worker_trace": [],
                    "source_record": prepared["source_record"],
                    "model_state": "fixed original HF weights; no refit",
                }
                (args.output / "first-mismatch.json").write_text(
                    json.dumps(diagnostic, indent=2) + "\n"
                )
                torch.save(
                    {
                        "prompt_token_ids": prompt_ids,
                        "generated_token_ids": tokens,
                        "generation_logprobs": logprobs,
                        "fresh_prefill_logprobs": fresh,
                        "prepared_images": prepared["prepared_images"],
                        "image_sizes": prepared["image_sizes"],
                    },
                    args.output / "first-mismatch.pt",
                )
                report["status"] = "mismatch"
                save_status()
                # Persist the replay fixture before optional worker diagnostics:
                # an RPC failure must not discard the first mismatch evidence.
                if not args.no_worker_trace:
                    try:
                        diagnostic["worker_trace"] = await engine.collective_rpc(
                            read_worker_trace, args=(request_id, position)
                        )
                    except Exception as error:
                        diagnostic["worker_trace_error"] = (
                            f"{type(error).__name__}: {error}"
                        )
                    (args.output / "first-mismatch.json").write_text(
                        json.dumps(diagnostic, indent=2) + "\n"
                    )
                await asyncio.gather(
                    *(engine.abort(request) for request in list(active))
                )
            save_status()

    async def generate(index: int) -> None:
        request_id = f"native-main-{index}-{uuid.uuid4().hex}"
        tokens, logprobs, checked = [], [], 0
        params = SamplingParams(
            max_tokens=min(args.max_tokens, 65536 - len(prompt_ids)),
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            seed=args.seed + index,
            logprobs=1,
            stop=generation.get("stop_strings"),
            output_kind=RequestOutputKind.DELTA,
        )
        active.add(request_id)
        try:
            async for output in engine.generate(prompt, params, request_id=request_id):
                if failed.is_set():
                    return
                completion = output.outputs[0]
                assert completion.logprobs is not None
                assert len(completion.token_ids) == len(completion.logprobs)
                for token, scores in zip(
                    completion.token_ids, completion.logprobs, strict=True
                ):
                    tokens.append(int(token))
                    logprobs.append(float(scores[token].logprob))
                if failed.is_set():
                    return
                if len(tokens) - checked >= args.check_every:
                    await check(request_id, list(tokens), list(logprobs), checked)
                    checked = len(tokens)
                    if failed.is_set():
                        return
            if len(tokens) > checked and not failed.is_set():
                await check(request_id, tokens, logprobs, checked)
            report["completed_requests"] += 1
            save_status()
        finally:
            active.discard(request_id)

    try:
        if not args.no_worker_trace:
            report["trace_workers"] = await engine.collective_rpc(
                install_worker_trace, args=(args.trace_steps,)
            )
        report["status"] = "running"
        save_status()

        async def rounds() -> None:
            for round_index in range(args.rounds):
                await asyncio.gather(
                    *(
                        generate(round_index * args.requests + index)
                        for index in range(args.requests)
                    )
                )
                if failed.is_set():
                    return

        try:
            await asyncio.wait_for(rounds(), timeout=args.deadline_seconds)
        except asyncio.TimeoutError:
            report["status"] = "budget_exhausted_without_mismatch"
        else:
            if not failed.is_set():
                report["status"] = "no_mismatch_in_completed_requests"
        return 1 if failed.is_set() else 0
    except Exception as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save_status()
        await asyncio.gather(*(engine.abort(request) for request in list(active)))
        engine.shutdown()


def main() -> int:
    """Read explicit fixture/output paths and run the bounded diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--check-every", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadline-seconds", type=int, default=3600)
    parser.add_argument("--trace-steps", type=int, default=8192)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-worker-trace", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.requests,
            args.rounds,
            args.check_every,
            args.max_tokens,
            args.deadline_seconds,
            args.trace_steps,
        )
        < 1
        or args.threshold <= 0
    ):
        parser.error("Counts, deadline, and threshold must be positive")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
