#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert ShareGPT-style VLM tool-use SFT data into PivotRL rows for NeMo-Gym.

Every assistant turn that emits exactly one well-formed image-tool call becomes
one pivot row:

    responses_create_params.input  = the trajectory prefix, verbatim (teacher-forced)
    expected_action                = the tool call demonstrated at that turn

The rollout is a single model call at that state; the verifier
(resources_servers/image_tools) scores the emitted tool call against
`expected_action`.

Input row schema (one JSON object per line):
    {"id": ..., "image": [rel_path, ...], "conversations": [{"from": ..., "value": ...}],
     "__metadata__": {"media_root": ...}}

Images are bound to `<image>` placeholders in order of appearance across the
whole conversation, which is how the source data is generated (verified: the
placeholder count equals len(image) for every row).

Usage:
    python tools/convert_sft_to_image_pivot.py \
        --input  /path/to/train.jsonl \
        --out-train resources_servers/image_tools/data/train.jsonl \
        --out-val   resources_servers/image_tools/data/validation.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys

# Tools the image-tools Gym environment actually implements. Anything else in
# the SFT data is a corrupted demonstration and is dropped.
KNOWN_TOOLS = frozenset(
    {
        "image_zoom_in_tool",
        "image_crop_tool",
        "image_rotate_tool",
        "image_flip_tool",
        "image_diff_tool",
        "image_overlay_tool",
        "image_side_by_side_tool",
        "color_at_tool",
        "find_color_tool",
        "count_objects_tool",
    }
)

# Mirrors resources_servers.image_tools so the converter and the
# verifier agree on what a tool call is.
_XML_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_XML_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def parse_parameter_value(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for match in _XML_TOOL_CALL_RE.finditer(text or ""):
        name = match.group(1).strip()
        args = {}
        for pm in _XML_PARAMETER_RE.finditer(match.group(2)):
            args[pm.group(1).strip()] = parse_parameter_value(pm.group(2).strip())
        calls.append({"name": name, "arguments": args})
    return calls


def split_on_images(text: str) -> list[str]:
    return (text or "").split("<image>")


def build_user_message(text: str, images: list[str], cursor: int) -> tuple[dict, int]:
    """Build a user message, consuming `<image>` placeholders from `images`."""
    segments = split_on_images(text)
    content: list[dict] = []
    for i, seg in enumerate(segments):
        if i > 0:
            if cursor < len(images):
                content.append(
                    {
                        "type": "input_image",
                        "image_url": images[cursor],
                        "detail": "auto",
                    }
                )
                cursor += 1
        seg = seg.strip()
        if seg:
            content.append({"type": "input_text", "text": seg})
    if not content:
        content = [{"type": "input_text", "text": ""}]
    return {"role": "user", "type": "message", "content": content}, cursor


def build_assistant_message(text: str, strip_think: bool) -> dict:
    """Prefix assistant turn as a NeMoGymEasyInputMessage.

    Content must be a plain string. The two alternatives both fail validation:
    NeMoGymResponseOutputMessage requires an `id` (no default, only the model
    server can mint one), and NeMoGymEasyInputMessage's list form accepts only
    *input* content parts - `output_text` is an output part, so a
    [{"type": "output_text", ...}] list matches neither branch of the union.
    """
    if strip_think:
        text = _THINK_RE.sub("", text)
    return {
        "role": "assistant",
        "type": "message",
        "content": text.strip(),
    }


def convert_row(
    row: dict,
    strip_think: bool,
    stats: collections.Counter,
    multi_call_turns: str = "skip",
    max_images_per_prefix: int = 0,
    system_prompt_override: str | None = None,
) -> list[dict]:
    media_root = (row.get("__metadata__") or {}).get("media_root", "")
    rel_images = row.get("image") or []
    images = [
        p if os.path.isabs(p) else os.path.join(media_root, p) for p in rel_images
    ]

    convs = row.get("conversations") or []
    built: list[dict] = []
    cursor = 0
    pivots: list[dict] = []

    for turn in convs:
        who = turn.get("from")
        value = turn.get("value", "")

        if who == "system":
            text = (
                system_prompt_override if system_prompt_override is not None else value
            )
            built.append({"role": "system", "type": "message", "content": text})
            continue

        if who == "human":
            msg, cursor = build_user_message(value, images, cursor)
            built.append(msg)
            continue

        if who != "gpt":
            stats["unknown_role"] += 1
            continue

        calls = parse_tool_calls(value)
        if len(calls) > 1 and multi_call_turns == "first":
            calls = calls[:1]
            stats["multi_call_turn_took_first"] += 1

        # `cursor` is exactly the number of images already bound into the
        # prefix. Each image costs 1-2k tokens on a VLM, so a deep trajectory
        # can blow the context budget on images alone before any text.
        if max_images_per_prefix and cursor > max_images_per_prefix:
            stats["skipped_too_many_images"] += 1
            built.append(build_assistant_message(value, strip_think))
            continue

        if (
            len(calls) == 1
            and calls[0]["name"] in KNOWN_TOOLS
            and calls[0]["arguments"]
        ):
            # Emit a pivot at this state, before the demonstrated turn is added.
            pivots.append(
                {
                    "agent_ref": {
                        "type": "responses_api_agents",
                        "name": "image_tools_pivot_agent",
                    },
                    "responses_create_params": {
                        "input": [dict(m) for m in built],
                        # A pivot ends at the decision. Stop the generation at
                        # </tool_call> exactly like the image-tools agent does,
                        # so the rollout is the action and nothing after it.
                        # vLLM options ride along in metadata.extra_body.
                        "metadata": {
                            "extra_body": json.dumps(
                                {
                                    "stop": ["</tool_call>"],
                                    "include_stop_str_in_output": True,
                                }
                            )
                        },
                    },
                    "expected_action": calls[0],
                    "metadata": {
                        "source_id": row.get("id"),
                        "turn_index": len(built),
                        "tool_name": calls[0]["name"],
                        "num_images_in_prefix": cursor,
                    },
                }
            )
            stats[f"tool::{calls[0]['name']}"] += 1
            stats["pivots_emitted"] += 1
        elif len(calls) > 1:
            stats["skipped_multi_call_turn"] += 1
        elif len(calls) == 1:
            stats["skipped_bad_tool_name_or_args"] += 1

        built.append(build_assistant_message(value, strip_think))

    if not pivots:
        stats["rows_with_no_pivot"] += 1
    return pivots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", required=True, nargs="+", help="One or more ShareGPT JSONL files."
    )
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-pivots",
        type=int,
        default=0,
        help="Stop after emitting N pivots (0 = no cap). NemoGymDataset reads the "
        "whole JSONL into memory, so an uncapped large corpus is a real cost.",
    )
    ap.add_argument(
        "--system-prompt-file",
        default=None,
        help="Replace every row's system turn with the contents of this file. "
        "For A/B-ing prompt variants on identical rows: the demonstrated actions "
        "were produced under the original prompt, so this measures the policy's "
        "response to the prompt, not a change in what is being asked of it.",
    )
    ap.add_argument(
        "--max-images-per-prefix",
        type=int,
        default=8,
        help="Drop pivots whose prefix carries more than N images (0 = no cap). "
        "Each image costs 1-2k tokens, so deep trajectories overflow the context "
        "on images alone. Keep this <= vllm limit_mm_per_prompt.image.",
    )
    ap.add_argument(
        "--multi-call-turns",
        choices=("skip", "first"),
        default="skip",
        help="Turns emitting >1 tool call: 'skip' (default, the target is "
        "ambiguous and the agent allows one call per turn) or 'first'.",
    )
    ap.add_argument(
        "--keep-think-in-prefix",
        action="store_true",
        help="Keep <think> blocks in prefix assistant turns (default: strip, "
        "matching multi-turn inference where prior reasoning is dropped).",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Debug: stop after N input rows."
    )
    args = ap.parse_args()

    strip_think = not args.keep_think_in_prefix
    system_prompt_override = None
    if args.system_prompt_file:
        system_prompt_override = open(args.system_prompt_file).read().rstrip()
        print(
            f"system prompt overridden from {args.system_prompt_file} "
            f"({len(system_prompt_override)} chars)",
            file=sys.stderr,
        )
    stats: collections.Counter = collections.Counter()
    rng = random.Random(args.seed)

    # Split by source trajectory so pivots from one trajectory never straddle
    # the train/val boundary.
    by_traj: dict[str, list[dict]] = collections.defaultdict(list)

    n_emitted = 0
    for src in args.input:
        if args.max_pivots and n_emitted >= args.max_pivots:
            break
        with open(src) as fh:
            for i, line in enumerate(fh):
                if args.limit and i >= args.limit:
                    break
                if args.max_pivots and n_emitted >= args.max_pivots:
                    stats["stopped_at_max_pivots"] = 1
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_json_lines"] += 1
                    continue
                stats["input_rows"] += 1
                # Namespace the trajectory id by source file so ids colliding
                # across corpora cannot merge two different trajectories (which
                # would also leak them across the train/val split).
                tid = f"{os.path.basename(os.path.dirname(src))}::{row.get('id')}"
                for pivot in convert_row(
                    row,
                    strip_think,
                    stats,
                    args.multi_call_turns,
                    args.max_images_per_prefix,
                    system_prompt_override,
                ):
                    by_traj[tid].append(pivot)
                    n_emitted += 1

    traj_ids = sorted(by_traj)
    rng.shuffle(traj_ids)
    n_val = max(1, int(len(traj_ids) * args.val_frac)) if traj_ids else 0
    val_ids = set(traj_ids[:n_val])

    os.makedirs(os.path.dirname(os.path.abspath(args.out_train)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_val)), exist_ok=True)

    n_train = n_valrows = 0
    with open(args.out_train, "w") as ftr, open(args.out_val, "w") as fva:
        for tid in traj_ids:
            out = fva if tid in val_ids else ftr
            for pivot in by_traj[tid]:
                out.write(json.dumps(pivot) + "\n")
                if tid in val_ids:
                    n_valrows += 1
                else:
                    n_train += 1

    print("=== conversion summary ===", file=sys.stderr)
    for key in sorted(stats):
        if not key.startswith("tool::"):
            print(f"  {key}: {stats[key]}", file=sys.stderr)
    print("  --- per-tool pivots ---", file=sys.stderr)
    for key in sorted(k for k in stats if k.startswith("tool::")):
        print(f"    {key[6:]}: {stats[key]}", file=sys.stderr)
    print(f"  trajectories: {len(traj_ids)} (val: {len(val_ids)})", file=sys.stderr)
    print(f"  train rows: {n_train}", file=sys.stderr)
    print(f"  val rows:   {n_valrows}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
