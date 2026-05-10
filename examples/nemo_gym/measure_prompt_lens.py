"""Measure tokenized prompt length distribution for a NeMo-Gym JSONL dataset.

Use the output to set the token-budget knobs in grpo_nanov3_24xH100.yaml so
the runtime invariant holds:

    max_prompt_tokens + max_new_tokens + 1  <=  policy.max_total_sequence_length
                                            =   vllm_cfg.max_model_len

Usage:
    python -u measure_prompt_lens.py \\
        --model /workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
        --data  /workspace/data/prime_intellect/acereason/acereason-math-mock-train.jsonl \\
        --max-total-sequence-length 8192 \\
        --gym  # set if Gym drives generation (then max_input_seq_length is null)
"""

import argparse
import json

import numpy as np
from transformers import AutoTokenizer

# ANSI colors — kept minimal: bold for headers, cyan for recommended values.
BOLD = "\033[1m"
CYAN = "\033[36m"
DIM = "\033[2m"
RESET = "\033[0m"

# Buffer applied to measured max prompt length when Gym is OFF.
# Small (32 tokens) — covers tokenizer quirks and minor template tweaks but
# does not waste sequence budget. Larger headroom only helps if the dataset
# is expected to grow; in that case raise this explicitly.
PROMPT_BUFFER = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    ap.add_argument(
        "--data",
        default="/workspace/data/prime_intellect/acereason/acereason-math-mock-train.jsonl",
    )
    ap.add_argument("--max-total-sequence-length", type=int, default=8192,
                    help="Master memory budget. Drives the max_new_tokens recommendation.")
    ap.add_argument("--gym", action=argparse.BooleanOptionalAction, default=True,
                    help="Gym drives generation (default). When true, max_input_seq_length is null.")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    lens = []
    with open(args.data) as f:
        for line in f:
            msgs = json.loads(line)["responses_create_params"]["input"]
            # Two-step: render template to string, then tokenize. Avoids quirks
            # where apply_chat_template(..., tokenize=True) returns a wrapped
            # Encoding object whose len() is not the token count.
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            lens.append(len(ids))

    lens = np.asarray(lens)
    p = lambda q: int(np.percentile(lens, q))
    prompt_max = int(lens.max())

    print(f"\n{BOLD}📏 Prompt length distribution (n={len(lens)}){RESET}")
    print(f"  min={lens.min()}  p50={p(50)}  p90={p(90)}  p95={p(95)}  p99={p(99)}  max={prompt_max}")

    # Compute recommendations.
    mtsl = args.max_total_sequence_length
    prompt_budget = prompt_max + PROMPT_BUFFER
    max_new = mtsl - prompt_budget - 1

    if args.gym:
        max_input = "null"
        max_input_note = "Gym drives the prompt; NeMo-RL data loader is not gating it."
    else:
        max_input = prompt_budget
        max_input_note = f"prompt_max ({prompt_max}) + {PROMPT_BUFFER}-token buffer for tokenizer / template drift."

    breakdown = (
        f"= {mtsl} (mtsl) − {prompt_max} (prompt_max) − {PROMPT_BUFFER} (buffer) − 1 (boundary)"
    )

    print(f"\n{BOLD}✅ Recommended values{RESET}  {DIM}(mode: {'Gym' if args.gym else 'NeMo-RL'}){RESET}")
    print(f"  measured prompt_max (max tokens after chat template) = {CYAN}{prompt_max}{RESET}")
    print(f"  policy.max_total_sequence_length         = {CYAN}{mtsl}{RESET}")
    print(f"  policy.generation.vllm_cfg.max_model_len = {CYAN}${{policy.max_total_sequence_length}}{RESET}")
    print(f"  data.max_input_seq_length                = {CYAN}{max_input}{RESET}   {DIM}# {max_input_note}{RESET}")
    print(f"  policy.generation.max_new_tokens         = {CYAN}{max_new}{RESET}   {DIM}# {breakdown}{RESET}")
    print(f"  verifiers_agent.max_tokens               = {CYAN}${{policy.generation.max_new_tokens}}{RESET}   {DIM}# alias{RESET}")
    print()


if __name__ == "__main__":
    main()
