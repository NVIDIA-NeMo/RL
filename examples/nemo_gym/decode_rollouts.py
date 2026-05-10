"""Decode the first N rollouts from a NeMo-RL train_data_step{N}.jsonl trace.

Strips trailing batch-padding (consecutive EOS/<|im_end|> tokens that NeMo-RL
adds so every row in the step matches the longest sequence) before decoding,
so the rendered text shows only the real prompt + completion.

Output is written to `decoded_<input_stem>_first<N>.txt` next to the input
file. Use `--stdout` to print instead.

Usage:
    python decode_rollouts.py [path/to/train_data_stepN.jsonl] [num_rows] [--stdout]
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

TOKENIZER_PATH = "/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_FILE = (
    "/workspace/260509_nemorl_prime_verifiers/results/nemotron_3_nano_grpo/"
    "logs/260510-0232-logs/training/exp_001/train_data_step1.jsonl"
)


def strip_trailing_pad(token_ids, pad_id):
    end = len(token_ids)
    while end > 0 and token_ids[end - 1] == pad_id:
        end -= 1
    return token_ids[:end], len(token_ids) - end


def decode_row(out, tokenizer, row, idx, pad_id):
    out.write(f"\n{'=' * 80}\n=== ROLLOUT {idx} ===\n{'=' * 80}\n")
    out.write(f"reward       : {row.get('rewards')}\n")
    out.write(f"input_lengths: {row.get('input_lengths')}\n")
    agent_ref = row.get("agent_ref")
    if agent_ref:
        out.write(f"agent_ref    : {agent_ref}\n")

    token_ids_list = row.get("token_ids") or []
    input_lengths = row.get("input_lengths") or []

    for t_idx, token_ids in enumerate(token_ids_list):
        if not token_ids:
            out.write(f"\n--- trajectory {t_idx}: <empty>\n")
            continue
        prompt_len = input_lengths[t_idx] if t_idx < len(input_lengths) else 0
        unpadded, n_pad = strip_trailing_pad(token_ids, pad_id)
        prompt_ids = unpadded[:prompt_len]
        gen_ids = unpadded[prompt_len:]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        out.write(
            f"\n--- trajectory {t_idx}: prompt_tokens={len(prompt_ids)} "
            f"generated_tokens={len(gen_ids)} stripped_pad={n_pad}\n"
        )
        out.write(f"\n[PROMPT]\n{prompt_text}\n")
        out.write(f"\n[GENERATION]\n{gen_text}\n")


def main():
    argv = [a for a in sys.argv[1:] if a]
    to_stdout = "--stdout" in argv
    argv = [a for a in argv if a != "--stdout"]
    path = Path(argv[0]) if len(argv) > 0 else Path(DEFAULT_FILE)
    n = int(argv[1]) if len(argv) > 1 else 10

    print(f"Loading tokenizer from {TOKENIZER_PATH} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH, trust_remote_code=True
    )
    pad_id = tokenizer.eos_token_id
    print(f"Reading {path} (first {n} rows). Pad id = {pad_id}", file=sys.stderr)

    if to_stdout:
        out = sys.stdout
        out_path = None
    else:
        out_path = path.parent / f"decoded_{path.stem}_first{n}.txt"
        out = out_path.open("w")

    try:
        with path.open("r") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                row = json.loads(line)
                decode_row(out, tokenizer, row, i, pad_id)
    finally:
        if out_path is not None:
            out.close()
            print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
