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
"""Export the Flow-GRPO OCR dataset as flow-GRPO prompt jsonl files.

Source: https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr
(19,653 train / 1,018 val prompts; the upstream txt files lack a trailing
newline, so wc -l reports one fewer). The quoted text in each prompt is the
OCR ground truth, stored in metadata for the "ocr" reward.

Usage:
  uv run python tools/export_ocr_prompts.py --out-dir examples/data/diffusion/ocr
  # or use a local flow_grpo checkout:
  uv run python tools/export_ocr_prompts.py --source-dir /path/to/flow_grpo/dataset/ocr --out-dir ...
"""

import argparse
import json
import os
import urllib.request

RAW_BASE = "https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/ocr"
SPLITS = {"train": "train_prompts.jsonl", "test": "val_prompts.jsonl"}


def ocr_line_to_record(line: str) -> dict:
    parts = line.split('"')
    if len(parts) < 3:
        raise ValueError(f"OCR prompt without quoted ground truth: {line!r}")
    return {"prompt": line, "metadata": {"ground_truth": parts[1]}}


def _read_split(split: str, source_dir: str | None) -> list[str]:
    if source_dir is not None:
        with open(os.path.join(source_dir, f"{split}.txt"), encoding="utf-8") as f:
            text = f.read()
    else:
        with urllib.request.urlopen(f"{RAW_BASE}/{split}.txt") as resp:
            text = resp.read().decode("utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-dir", default=None)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for split, out_name in SPLITS.items():
        lines = _read_split(split, args.source_dir)
        out_path = os.path.join(args.out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(ocr_line_to_record(line), ensure_ascii=False) + "\n")
        print(f"{split}: {len(lines)} prompts -> {out_path}")


if __name__ == "__main__":
    main()
