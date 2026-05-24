"""Extract first-turn user prompts from OASST for GRPO training."""

import gzip
import json

from huggingface_hub import hf_hub_download


def main():
    filename = hf_hub_download(
        repo_id="OpenAssistant/oasst1",
        filename="2023-04-12_oasst_all.trees.jsonl.gz",
        repo_type="dataset",
    )

    with gzip.open(filename) as f:
        all_objs = [json.loads(dp.decode("utf-8")) for dp in f.readlines()]

    records = []
    seen = set()
    for obj in all_objs:
        if "prompt" not in obj:
            continue
        prompt_obj = obj["prompt"]
        if prompt_obj["role"] != "prompter" or not prompt_obj.get("replies"):
            continue
        text = prompt_obj["text"]
        if text in seen:
            continue
        seen.add(text)
        records.append(
            {
                "messages": [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": ""},
                ],
            }
        )

    out_path = "examples/data/oasst_prompts.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
