"""Reproduce the vLLM-vs-processor token-mismatch offline.

Loads the Nano-Omni processor only (no model weights, no vLLM engine) and
tokenizes a synthetic multi-turn multimodal trajectory under several variants
to isolate what makes the reconstruction path (NeMo-RL) diverge from what vLLM
tokenized during rollout.

Variants:
  (S0) baseline WITHOUT system prompt              — original bug reproducer
  (S1) baseline WITH    system prompt              — current reconstruction path
  (S2) WITH sys + empty content-part separator     — probes join separator
  (S3) WITH sys + chat_template_kwargs             — probes template kwargs
       (enable_thinking=True, truncate_history_thinking=False)
  (S4) WITH sys + empty sep + kwargs               — both fixes combined

Compares each variant's token count and first-mismatch to S1 (the current
production reconstruction). Whichever variant closes the gap to zero tokens
identifies the missing piece.

Run:
  uv run --locked --extra mcore examples/nemo_gym/debug_nano_omni_template.py
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"

# Same default as resources_servers/gym_v/app.py:DEFAULT_SYSTEM_PROMPT — copy
# it here so the script has no repo import path. If your recipe overrides
# `responses_api_agents.gymv_agent.system_prompt`, paste that value instead.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant playing a text/visual game. On each turn, look "
    "at the observation, reason briefly about the best action, then output your "
    "action as \\boxed{...}."
)

# From the recipe (examples/nemo_gym/grpo_nemotron_omni_30ba3b_gymv_smoke.yaml).
CHAT_TEMPLATE_KWARGS = {
    "enable_thinking": True,
    "truncate_history_thinking": False,
}


def build_messages(*, with_system: bool) -> list[dict[str, Any]]:
    """Fake a two-turn FrozenLake-style rollout: user(image+text) -> asst -> user -> asst."""
    img = Image.new("RGB", (128, 128), "white")
    messages: list[dict[str, Any]] = []
    if with_system:
        messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    messages += [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Board turn 1. Choose LEFT/RIGHT/UP/DOWN."},
            ],
        },
        {"role": "assistant", "content": "I'll go right. \\boxed{RIGHT}"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Board turn 2."},
            ],
        },
        {"role": "assistant", "content": "Now down. \\boxed{DOWN}"},
    ]
    return messages


def tokenize(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    part_sep: str = "\n",
    template_kwargs: dict[str, Any] | None = None,
) -> tuple[str, torch.Tensor]:
    """Mirror _PlaceholderMultimodalProcessorAdapter.process, with knobs.

    Args:
        processor: HF AutoProcessor for Nano-Omni.
        messages: chat-completion-style messages (content can be list of parts).
        part_sep: separator used to join multimodal content parts (`<image>`,
            text) into a single string per message. Production uses ``"\\n"``.
        template_kwargs: extra kwargs forwarded to ``apply_chat_template``
            (e.g. ``enable_thinking``, ``truncate_history_thinking``). None for
            production behavior (kwargs not threaded through).
    """
    image_token = getattr(processor, "image_token", "<image>")
    template_kwargs = template_kwargs or {}
    images = []
    text_msgs = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if p["type"] == "image":
                    parts.append(image_token)
                    images.append(p["image"])
                elif p["type"] == "text":
                    parts.append(p["text"])
            text_msgs.append({"role": m["role"], "content": part_sep.join(parts)})
        else:
            text_msgs.append(m)

    formatted = processor.apply_chat_template(
        text_msgs,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )
    out = processor(text=formatted, images=images, return_tensors="pt")
    return formatted, out["input_ids"][0]


def first_mismatch(a: torch.Tensor, b: torch.Tensor) -> int:
    n = min(len(a), len(b))
    diff = (a[:n] != b[:n]).nonzero(as_tuple=True)[0]
    return int(diff[0]) if len(diff) else n


def show_window(
    name: str,
    formatted: str,
    ids: torch.Tensor,
    *,
    center: int,
    radius: int = 6,
) -> None:
    lo = max(0, center - radius)
    hi = min(len(ids), center + radius + 1)
    print(f"  {name} tokens[{lo}:{hi}] = {ids[lo:hi].tolist()}")
    # Find character offset roughly proportional to position, so a human can
    # eyeball what's diverging. Character indexing on the rendered string is a
    # rough approximation of token position — enough to spot the diverging char.
    char_lo = max(0, int(center * len(formatted) / max(1, len(ids)) - 40))
    char_hi = min(len(formatted), char_lo + 120)
    print(f"  {name} text≈: {repr(formatted[char_lo:char_hi])}")


def main() -> None:
    print(f"Loading processor for {MODEL} (CPU-only; no weights) ...")
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    msgs_no_sys = build_messages(with_system=False)
    msgs_sys = build_messages(with_system=True)

    variants: list[tuple[str, str, torch.Tensor]] = []

    # S0: baseline WITHOUT system prompt (pre-fix).
    fmt, ids = tokenize(processor, msgs_no_sys)
    variants.append(("S0  no_sys | \\n sep | no kwargs   ", fmt, ids))

    # S1: WITH system prompt, current reconstruction path (post-fix as of now).
    fmt, ids = tokenize(processor, msgs_sys)
    variants.append(("S1  sys    | \\n sep | no kwargs   ", fmt, ids))

    # S2: WITH sys + empty content-part separator.
    fmt, ids = tokenize(processor, msgs_sys, part_sep="")
    variants.append(("S2  sys    | \"\" sep | no kwargs   ", fmt, ids))

    # S3: WITH sys + chat_template_kwargs from the recipe.
    fmt, ids = tokenize(processor, msgs_sys, template_kwargs=CHAT_TEMPLATE_KWARGS)
    variants.append(("S3  sys    | \\n sep | kwargs      ", fmt, ids))

    # S4: WITH sys + empty sep + template kwargs.
    fmt, ids = tokenize(
        processor,
        msgs_sys,
        part_sep="",
        template_kwargs=CHAT_TEMPLATE_KWARGS,
    )
    variants.append(("S4  sys    | \"\" sep | kwargs      ", fmt, ids))

    # Extras — quick sanity: space separator, and a bare-empty template kwargs
    # variant to detect any non-thinking-related template branch.
    fmt, ids = tokenize(processor, msgs_sys, part_sep=" ")
    variants.append(("S5  sys    | ' ' sep | no kwargs   ", fmt, ids))

    # Reference is S1 (what production currently does after the system-prompt fix).
    _, _, ref_ids = variants[1]
    ref_len = len(ref_ids)

    print()
    print("=" * 78)
    print(f"{'variant':<44}  {'#tok':>6}  {'Δ vs S1':>8}  {'first_mm(vs S1)':>15}")
    print("-" * 78)
    for name, _fmt, ids in variants:
        d = len(ids) - ref_len
        fm = first_mismatch(ref_ids, ids)
        print(f"{name}  {len(ids):>6}  {d:>+8}  {fm:>15}")
    print("=" * 78)

    # For each variant that differs from S1 by exactly 1 token, print a window
    # around the divergence to eyeball what changed.
    print()
    print("Windows around divergence (variants ≠ S1):")
    for name, fmt, ids in variants[2:]:
        fm = first_mismatch(ref_ids, ids)
        if fm >= min(len(ref_ids), len(ids)):
            print(f"\n[{name.strip()}] no divergence in overlap")
            continue
        print(f"\n[{name.strip()}] first_mismatch={fm}, len_delta={len(ids) - ref_len}")
        show_window("  S1  ", variants[1][1], ref_ids, center=fm)
        show_window("  var ", fmt, ids, center=fm)

    # Also: pretty-print the first ~150 chars of S1's rendered template — that
    # is the ground truth of what our reconstruction feeds Nano-Omni today.
    print()
    print("S1 rendered head (first 250 chars):")
    print(repr(variants[1][1][:250]))


if __name__ == "__main__":
    main()
