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

"""Verify exact Qwen incremental tokenization and measure suffix-only work."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from transformers import AutoTokenizer

from nemo_rl.models.generation.vllm.incremental_tokenizer import (
    ExactIncrementalTokenizer,
    ExactIncrementalTokenizerSessionManager,
)

PROMPT_PREFIX = (
    "<|im_start|>system\nYou are a coding agent.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\nInspect files.\n</think>\n\n"
    '<tool_call>\n{"name": "execute_bash", "arguments": {"command": "tests"}}\n'
    "</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n"
)
PROMPT_SUFFIX = "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
TEMPLATE_PREFIX = PROMPT_PREFIX[: PROMPT_PREFIX.index("<|im_start|>user")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-chars", type=int, default=30_000)
    parser.add_argument("--chunk-chars", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_output(*, output_chars: int, seed: int) -> str:
    random_generator = random.Random(seed)
    lines = [
        "collecting tests/unit/test_example.py::test_case PASSED\n",
        "src/module.py:120: value = compute(alpha, beta)\n",
        "Unicode boundary: cafe\u0301 한글 <|im_end|> literal output\n",
        "progress [##########] 100% elapsed=12.345s\n",
    ]
    pieces = []
    while sum(map(len, pieces)) < output_chars:
        pieces.append(random_generator.choice(lines))
    return "".join(pieces)[:output_chars]


def prompt(tool_output: str) -> str:
    return PROMPT_PREFIX + tool_output + PROMPT_SUFFIX


def full_tokens(tokenizer, rendered_prompt: str) -> list[int]:
    return list(
        tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
    )


def verify_edge_cases(tokenizer) -> int:
    cases = [
        ["a", "a\u0301", "a\u0301bc"],
        ["word", "wording", "wording next"],
        ["<|im_", "<|im_end|>", "<|im_end|> literal"],
        ["x ", "x  ", "x  \n", "x  \nnext"],
        ["한", "한글", "한글 output"],
    ]
    verified_steps = 0
    for outputs in cases:
        initial_prompt = prompt(outputs[0])
        encoder = ExactIncrementalTokenizer(
            tokenizer=tokenizer,
            rendered_prompt=initial_prompt,
            authoritative_token_ids=full_tokens(tokenizer, initial_prompt),
        )
        for output in outputs[1:]:
            rendered_prompt = prompt(output)
            try:
                encoder.append(rendered_prompt)
            except Exception as error:
                raise AssertionError(
                    f"incremental update failed for {outputs[0]!r} -> {output!r}"
                ) from error
            expected = full_tokens(tokenizer, rendered_prompt)
            if encoder.token_ids != expected:
                raise AssertionError(f"incremental token mismatch for {output!r}")
            verified_steps += 1
    return verified_steps


def run_full(tokenizer, prompts: list[str]) -> tuple[float, int, int]:
    started = time.perf_counter()
    token_count = 0
    encoded_chars = 0
    for rendered_prompt in prompts:
        token_count += len(full_tokens(tokenizer, rendered_prompt))
        encoded_chars += len(rendered_prompt)
    return time.perf_counter() - started, encoded_chars, token_count


def run_incremental(tokenizer, prompts: list[str]) -> tuple[float, int, int]:
    started = time.perf_counter()
    initial_tokens = full_tokens(tokenizer, prompts[0])
    encoder = ExactIncrementalTokenizer(
        tokenizer=tokenizer,
        rendered_prompt=prompts[0],
        authoritative_token_ids=initial_tokens,
    )
    encoded_chars = len(prompts[0])
    encoded_tokens = len(initial_tokens)
    for rendered_prompt in prompts[1:]:
        step = encoder.append(rendered_prompt)
        encoded_chars += step.encoded_chars
        encoded_tokens += step.encoded_tokens
    return time.perf_counter() - started, encoded_chars, encoded_tokens


def verify_fast_final(tokenizer, prompts: list[str]) -> dict[str, int | bool]:
    """Verify periodic checkpoints followed by checkpoint-free finalization."""
    checkpoint_interval = 8
    manager = ExactIncrementalTokenizerSessionManager(
        tokenizer=tokenizer,
        max_sessions=1,
        session_ttl_seconds=60,
        checkpoint_interval=checkpoint_interval,
    )
    manager.start(
        session_id="fixed-trace",
        sequence_no=0,
        rendered_prompt=prompts[0],
        authoritative_token_ids=full_tokens(tokenizer, prompts[0]),
    )
    checkpoint_count = 1
    checkpoint_mismatches = 0
    for sequence_no, rendered_prompt in enumerate(prompts[1:-1], start=1):
        authoritative_token_ids = None
        if manager.requires_checkpoint(
            session_id="fixed-trace",
            sequence_no=sequence_no,
        ):
            authoritative_token_ids = full_tokens(tokenizer, rendered_prompt)
        result = manager.append(
            session_id="fixed-trace",
            sequence_no=sequence_no,
            rendered_prompt=rendered_prompt,
            authoritative_token_ids=authoritative_token_ids,
        )
        checkpoint_count += result.checkpoint_count
        checkpoint_mismatches += result.checkpoint_mismatches

    final_sequence_no = len(prompts) - 1
    if manager.requires_checkpoint(
        session_id="fixed-trace",
        sequence_no=final_sequence_no,
    ):
        raise AssertionError("fixed trace final unexpectedly requires a checkpoint")
    final_result = manager.finalize(
        session_id="fixed-trace",
        sequence_no=final_sequence_no,
        rendered_prompt=prompts[-1],
    )
    expected_tokens = full_tokens(tokenizer, prompts[-1])
    if final_result.tokens != expected_tokens:
        raise AssertionError(
            "checkpoint-free final tokens differ from full tokenization"
        )
    return {
        "checkpoint_count": checkpoint_count,
        "checkpoint_mismatches": checkpoint_mismatches,
        "final_checkpoint_count": final_result.checkpoint_count,
        "final_token_count": len(final_result.tokens),
        "final_tokens_match": True,
    }


def verify_final_only(tokenizer, tool_output: str) -> dict[str, int | bool]:
    """Verify the production final-only empty-output-to-final transition."""
    manager = ExactIncrementalTokenizerSessionManager(
        tokenizer=tokenizer,
        max_sessions=1,
        session_ttl_seconds=60,
        checkpoint_interval=8,
    )
    initial_prompt = prompt("")
    final_prompt = prompt(tool_output)
    start_result = manager.start(
        session_id="final-only",
        sequence_no=0,
        rendered_prompt=initial_prompt,
        authoritative_token_ids=full_tokens(tokenizer, initial_prompt),
    )
    final_result = manager.finalize(
        session_id="final-only",
        sequence_no=1,
        rendered_prompt=final_prompt,
    )
    expected_tokens = full_tokens(tokenizer, final_prompt)
    if final_result.tokens != expected_tokens:
        raise AssertionError("final-only tokens differ from full tokenization")
    return {
        "start_checkpoint_count": start_result.checkpoint_count,
        "final_checkpoint_count": final_result.checkpoint_count,
        "checkpoint_mismatches": final_result.checkpoint_mismatches,
        "encoded_chars": final_result.encoded_chars,
        "encoded_tokens": final_result.encoded_tokens,
        "reused_tokens": final_result.reused_tokens,
        "final_token_count": len(final_result.tokens),
        "final_tokens_match": True,
    }


def verify_authoritative_prefix_seed(
    tokenizer, tool_output: str
) -> dict[str, int | bool]:
    """Verify EOS-bounded seed tokens against full Qwen tokenization."""
    template_prefix_tokens = full_tokens(tokenizer, TEMPLATE_PREFIX)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise AssertionError("tokenizer has no EOS token ID")
    template_eos_index = max(
        index
        for index, token_id in enumerate(template_prefix_tokens)
        if token_id == eos_token_id
    )
    authoritative_prefix_tokens = template_prefix_tokens[: template_eos_index + 1]
    initial_prompt = prompt("")
    final_prompt = prompt(tool_output)
    manager = ExactIncrementalTokenizerSessionManager(
        tokenizer=tokenizer,
        max_sessions=1,
        session_ttl_seconds=60,
        checkpoint_interval=8,
    )
    start_result, start_tokens = manager.start_from_authoritative_prefix(
        session_id="authoritative-prefix-seed",
        sequence_no=0,
        rendered_prompt=initial_prompt,
        template_prefix_prompt=TEMPLATE_PREFIX,
        authoritative_prefix_token_ids=authoritative_prefix_tokens,
    )
    expected_start_tokens = full_tokens(tokenizer, initial_prompt)
    if start_tokens != expected_start_tokens:
        raise AssertionError("prefix-seeded start differs from full tokenization")
    final_result = manager.finalize(
        session_id="authoritative-prefix-seed",
        sequence_no=1,
        rendered_prompt=final_prompt,
    )
    expected_final_tokens = full_tokens(tokenizer, final_prompt)
    if final_result.tokens != expected_final_tokens:
        raise AssertionError("prefix-seeded final differs from full tokenization")

    mismatched_prefix_tokens = list(authoritative_prefix_tokens)
    replacement_token = full_tokens(tokenizer, " replacement")[0]
    mismatch_index = len(mismatched_prefix_tokens) - 2
    if mismatched_prefix_tokens[mismatch_index] == replacement_token:
        raise AssertionError("test replacement token unexpectedly matches")
    mismatched_prefix_tokens[mismatch_index] = replacement_token
    mismatch_manager = ExactIncrementalTokenizerSessionManager(
        tokenizer=tokenizer,
        max_sessions=1,
        session_ttl_seconds=60,
        checkpoint_interval=8,
    )
    _, mismatched_start_tokens = mismatch_manager.start_from_authoritative_prefix(
        session_id="mismatched-authoritative-prefix-seed",
        sequence_no=0,
        rendered_prompt=initial_prompt,
        template_prefix_prompt=TEMPLATE_PREFIX,
        authoritative_prefix_token_ids=mismatched_prefix_tokens,
    )
    expected_mismatched_start = (
        mismatched_prefix_tokens[:-1] + expected_start_tokens[template_eos_index:]
    )
    if mismatched_start_tokens != expected_mismatched_start:
        raise AssertionError("prefix seed did not preserve authoritative model tokens")
    mismatched_final_result = mismatch_manager.finalize(
        session_id="mismatched-authoritative-prefix-seed",
        sequence_no=1,
        rendered_prompt=final_prompt,
    )
    expected_mismatched_final = (
        mismatched_prefix_tokens[:-1] + expected_final_tokens[template_eos_index:]
    )
    if mismatched_final_result.tokens != expected_mismatched_final:
        raise AssertionError("prefix-seeded final changed authoritative model tokens")
    return {
        "start_checkpoint_count": start_result.checkpoint_count,
        "start_encoded_chars": start_result.encoded_chars,
        "start_encoded_tokens": start_result.encoded_tokens,
        "start_reused_tokens": start_result.reused_tokens,
        "start_token_count": len(start_tokens),
        "start_tokens_match": True,
        "final_tokens_match": True,
        "mismatched_authoritative_prefix_preserved": True,
    }


def main() -> None:
    args = parse_args()
    if args.output_chars <= 0 or args.chunk_chars <= 0 or args.repeats <= 0:
        raise ValueError("output chars, chunk chars, and repeats must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    tool_output = build_output(output_chars=args.output_chars, seed=args.seed)
    snapshots = [
        tool_output[:end]
        for end in range(args.chunk_chars, len(tool_output), args.chunk_chars)
    ] + [tool_output]
    prompts = [prompt(snapshot) for snapshot in snapshots]

    verified_steps = verify_edge_cases(tokenizer)
    correctness_encoder = ExactIncrementalTokenizer(
        tokenizer=tokenizer,
        rendered_prompt=prompts[0],
        authoritative_token_ids=full_tokens(tokenizer, prompts[0]),
    )
    for rendered_prompt in prompts[1:]:
        correctness_encoder.append(rendered_prompt)
        if correctness_encoder.token_ids != full_tokens(tokenizer, rendered_prompt):
            raise AssertionError("incremental tokens differ from full tokenization")
        verified_steps += 1
    fast_final = verify_fast_final(tokenizer, prompts)
    final_only = verify_final_only(tokenizer, tool_output)
    authoritative_prefix_seed = verify_authoritative_prefix_seed(tokenizer, tool_output)

    full_times = []
    incremental_times = []
    full_encoded_chars = full_encoded_tokens = 0
    incremental_encoded_chars = incremental_encoded_tokens = 0
    for _ in range(args.repeats):
        full_time, full_encoded_chars, full_encoded_tokens = run_full(
            tokenizer, prompts
        )
        incremental_time, incremental_encoded_chars, incremental_encoded_tokens = (
            run_incremental(tokenizer, prompts)
        )
        full_times.append(full_time)
        incremental_times.append(incremental_time)

    full_mean = sum(full_times) / len(full_times)
    incremental_mean = sum(incremental_times) / len(incremental_times)
    result = {
        "model": str(args.model),
        "snapshots": len(prompts),
        "verified_incremental_steps": verified_steps,
        "fast_final": fast_final,
        "final_only": final_only,
        "authoritative_prefix_seed": authoritative_prefix_seed,
        "final_prompt_tokens": len(full_tokens(tokenizer, prompts[-1])),
        "full": {
            "mean_seconds": full_mean,
            "encoded_chars": full_encoded_chars,
            "encoded_tokens": full_encoded_tokens,
        },
        "incremental": {
            "mean_seconds": incremental_mean,
            "encoded_chars": incremental_encoded_chars,
            "encoded_tokens": incremental_encoded_tokens,
        },
        "reduction": {
            "time_pct": 100 * (1 - incremental_mean / full_mean),
            "encoded_chars_pct": 100
            * (1 - incremental_encoded_chars / full_encoded_chars),
            "encoded_tokens_pct": 100
            * (1 - incremental_encoded_tokens / full_encoded_tokens),
        },
    }
    serialized_result = json.dumps(result, indent=2, sort_keys=True)
    print(serialized_result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized_result + "\n")


if __name__ == "__main__":
    main()
