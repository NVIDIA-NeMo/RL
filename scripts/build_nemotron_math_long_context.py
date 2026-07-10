#!/usr/bin/env python3
"""Build a small, diverse long-input GRPO dataset from Nemotron math SFT rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerFast


# Selected by assistant-response character length only. Each row has enough
# reasoning before its first boxed expression to form a 30K-token prompt.
SELECTED_ROW_IDS = (
    129,
    176,
    119,
    7,
    3,
    80,
    19,
    131,
    112,
    150,
    193,
    5,
    11,
    115,
    15,
    29,
    194,
    110,
    198,
    36,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repeat-count", type=int, required=True)
    parser.add_argument(
        "--prompt-template", type=Path, default=Path("examples/prompts/cot.txt")
    )
    parser.add_argument("--target-tokens", type=int, default=30_000)
    return parser.parse_args()


def load_selected_rows(paths: list[Path]) -> dict[int, dict[str, Any]]:
    selected = set(SELECTED_ROW_IDS)
    rows: dict[int, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for entry in payload["rows"]:
            row_id = entry["row_idx"]
            if row_id in selected:
                rows[row_id] = entry["row"]

    missing = selected - rows.keys()
    if missing:
        raise ValueError(f"Selected rows missing from sources: {sorted(missing)}")
    return rows


def load_tokenizer(model_path: Path) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(model_path / "tokenizer.json")
    )
    tokenizer.chat_template = (model_path / "chat_template.jinja").read_text()
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|endoftext|>"
    return tokenizer


def extract_last_boxed(text: str) -> str:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        raise ValueError("Assistant response has no boxed answer")

    content_start = start + len(marker)
    depth = 1
    for pos in range(content_start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:pos].strip()
    raise ValueError("Unbalanced final boxed answer")


def input_text(problem: str, reasoning: str, padding: str = "") -> str:
    return (
        "Original problem:\n"
        f"{problem}\n\n"
        "Draft reasoning:\n"
        f"{reasoning}{padding}\n\n"
        "Continue from the draft reasoning and return only the final answer "
        "inside \\boxed{}."
    )


def rendered_token_count(
    tokenizer: PreTrainedTokenizerFast, prompt_template: str, text: str
) -> int:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_template.format(text)}],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def fit_reasoning_to_tokens(
    tokenizer: PreTrainedTokenizerFast,
    prompt_template: str,
    problem: str,
    reasoning: str,
    target_tokens: int,
) -> tuple[str, int]:
    def count(prefix: str, padding: str = "") -> int:
        return rendered_token_count(
            tokenizer, prompt_template, input_text(problem, prefix, padding)
        )

    if count(reasoning) < target_tokens:
        raise ValueError("Reasoning before the first boxed answer is too short")

    low, high = 0, len(reasoning)
    while low < high:
        mid = (low + high + 1) // 2
        if count(reasoning[:mid]) <= target_tokens:
            low = mid
        else:
            high = mid - 1

    # BPE boundaries can skip a token count. Back off slightly, then add a
    # semantically inert trailing comment one token at a time to hit the target.
    prefix = reasoning[:low]
    while count(prefix) > target_tokens:
        prefix = prefix[:-1]

    padding = ""
    fillers = (" ", "\n", ".", " x", " note")
    while count(prefix, padding) < target_tokens:
        current = count(prefix, padding)
        choices = []
        for filler in fillers:
            candidate_count = count(prefix, padding + filler)
            if current < candidate_count <= target_tokens:
                choices.append((candidate_count, filler))
        if not choices:
            prefix = prefix[:-1]
            continue
        _, filler = max(choices)
        padding += filler

    fitted = input_text(problem, prefix, padding)
    final_count = rendered_token_count(tokenizer, prompt_template, fitted)
    if final_count != target_tokens:
        raise AssertionError(f"Expected {target_tokens} tokens, got {final_count}")
    return fitted, len(prefix)


def main() -> None:
    args = parse_args()
    if args.repeat_count < 1:
        raise ValueError("--repeat-count must be at least 1")

    rows = load_selected_rows(args.source)
    tokenizer = load_tokenizer(args.model)
    prompt_template = args.prompt_template.read_text()

    records = []
    metadata_rows = []
    for row_id in SELECTED_ROW_IDS:
        messages = rows[row_id]["messages"]
        problem = next(m["content"] for m in messages if m["role"] == "user")
        assistant = messages[-1]["content"]
        first_boxed = assistant.find("\\boxed{")
        if first_boxed < 0:
            raise ValueError(f"Row {row_id} has no boxed answer")

        reasoning = assistant[:first_boxed].removeprefix("<think>").lstrip()
        answer = extract_last_boxed(assistant)
        fitted_input, retained_reasoning_chars = fit_reasoning_to_tokens(
            tokenizer,
            prompt_template,
            problem,
            reasoning,
            args.target_tokens,
        )
        records.append({"input": fitted_input, "output": answer})
        metadata_rows.append(
            {
                "source_row_id": row_id,
                "source_user_chars": len(problem),
                "source_assistant_chars": len(assistant),
                "retained_reasoning_chars": retained_reasoning_chars,
                "rendered_input_tokens": args.target_tokens,
                "answer": answer,
                "input_sha256": hashlib.sha256(fitted_input.encode()).hexdigest(),
            }
        )

    hashes = {row["input_sha256"] for row in metadata_rows}
    if len(hashes) != len(SELECTED_ROW_IDS):
        raise AssertionError("Generated prompts are not unique")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output_file:
        for _ in range(args.repeat_count):
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=True) + "\n")

    metadata = {
        "source_dataset": "nvidia/Nemotron-Cascade-2-SFT-Data",
        "source_subset": "math",
        "model_tokenizer": str(args.model),
        "prompt_template": str(args.prompt_template),
        "target_rendered_input_tokens": args.target_tokens,
        "num_unique_prompts": len(records),
        "repeat_count": args.repeat_count,
        "num_records": len(records) * args.repeat_count,
        "rows": metadata_rows,
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
