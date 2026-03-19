"""Filter out prompts that exceed a token limit after chat template expansion.

Keeps prompts with token count <= (max_model_len - safety_margin) to ensure
vLLM can always accept the request and generate at least safety_margin tokens.
"""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

MAX_MODEL_LEN = 8192
SAFETY_MARGIN = 512
MAX_PROMPT_TOKENS = MAX_MODEL_LEN - SAFETY_MARGIN

MODEL_PATH = "/lustre/fsw/portfolios/llmservice/users/mfathi/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

INPUT_PATH = Path("3rdparty/Gym-workspace/Gym/resources_servers/turing_vif/data/multichallenge_vanilla_8k.jsonl")
OUTPUT_PATH = INPUT_PATH.with_name("multichallenge_vanilla_8k_filtered.jsonl")


def main():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    kept = 0
    removed = 0
    removed_ids = []

    with open(INPUT_PATH) as fin, open(OUTPUT_PATH, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            sample = json.loads(line)
            messages = sample["responses_create_params"]["input"]

            token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            n_tokens = len(token_ids)

            if n_tokens <= MAX_PROMPT_TOKENS:
                fout.write(line)
                kept += 1
            else:
                removed += 1
                removed_ids.append((sample.get("id", line_num), n_tokens))

    print(f"\nResults (max_prompt_tokens={MAX_PROMPT_TOKENS}):")
    print(f"  Kept:    {kept}")
    print(f"  Removed: {removed}")
    print(f"  Output:  {OUTPUT_PATH}")

    if removed_ids:
        print(f"\nRemoved samples (id, token_count):")
        for sid, n in sorted(removed_ids, key=lambda x: -x[1]):
            print(f"  id={sid}  tokens={n}")


if __name__ == "__main__":
    main()
