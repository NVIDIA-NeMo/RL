#!/usr/bin/env python3
"""Re-key a length ordering onto gym dataset content hashes.

The gym data processor (`nemo_gym_data_processor`) emits a placeholder
message_log, so the token-hash join in `nemo_rl.data.length_ordering` can never
match a gym dataset (0% hit-rate). This tool decodes each traced first-turn
prompt (its `tokens` field) with the run's tokenizer, locates which dataset
row's first user-message content appears verbatim inside it, and rewrites the
ordering labels keyed by the SHA-256 of that content string — a key the loader
can recompute from `extra_env_info` without any tokenization.

Usage (inside the training container / venv, CPU only):
    python tools/rekey_length_ordering_gym.py \
        <ordering.json> <trace.jsonl> <train.jsonl> <output.json> \
        [--tokenizer Qwen/Qwen3-30B-A3B]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def first_user_content(responses_create_params: dict) -> str | None:
    for message in responses_create_params.get("input", []):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            # Responses-API style content lists
            if isinstance(content, list):
                parts = [
                    p.get("text", "") for p in content if isinstance(p, dict)
                ]
                return "".join(parts)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ordering_json", type=Path)
    parser.add_argument("trace_jsonl", type=Path)
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-30B-A3B")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    ordering = json.loads(args.ordering_json.read_text())
    labels = ordering["labels"]

    # Dataset rows: user content string per row.
    row_contents: list[str] = []
    for line in args.train_jsonl.open():
        row = json.loads(line)
        content = first_user_content(row.get("responses_create_params", {}))
        row_contents.append(content or "")

    # One decoded prompt text per unique prompt_token_hash (first turn only).
    decoded: dict[str, str] = {}
    with args.trace_jsonl.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("turn_idx") != 0:
                continue
            h = row.get("prompt_token_hash")
            if not h or h in decoded or h not in labels:
                continue
            decoded[h] = tokenizer.decode(row["tokens"])
            if len(decoded) == len(labels):
                break

    print(f"decoded {len(decoded)}/{len(labels)} labeled prompts")

    # Match each decoded prompt to the dataset row whose user content it embeds.
    new_labels: dict[str, dict] = {}
    ambiguous = unmatched = 0
    for h, text in decoded.items():
        matches = [c for c in set(row_contents) if c and c in text]
        if not matches:
            unmatched += 1
            continue
        # Longest match wins (guards against one query being a prefix of another).
        best = max(matches, key=len)
        others = [m for m in matches if m != best and m not in best]
        if others:
            ambiguous += 1
            continue
        key = content_hash(best)
        entry = dict(labels[h])
        entry["prompt_token_hash"] = h
        new_labels[key] = entry

    order = sorted(new_labels, key=lambda k: new_labels[k]["rank"])
    payload = {
        "meta": {
            **ordering.get("meta", {}),
            "rekeyed": "gym_first_user_content_sha256",
            "source_ordering": str(args.ordering_json),
            "matched": len(new_labels),
            "unmatched": unmatched,
            "ambiguous": ambiguous,
        },
        "order": order,
        "labels": new_labels,
    }
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(
        f"wrote {len(new_labels)} content-keyed labels to {args.output_json} "
        f"({unmatched} unmatched, {ambiguous} ambiguous)"
    )


if __name__ == "__main__":
    main()
