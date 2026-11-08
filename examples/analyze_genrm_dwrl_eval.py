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

"""Compare the BT verdict of a DWRL pointwise GenRM with the verdict of its own reasoning.

Reads the output files of examples/run_eval_genrm_dwrl_openai.py for one benchmark from a results directory, e.g. the
checkpoints along a training run, named `step_<N>_<benchmark>_results.json`, and reports per checkpoint, in step order:

- BT accuracy: how often P("yes") is higher for the preferred response than for the other one (the DWRL verdict).
- Parse rate: how often the reasoning's JSON verdict could be read for both responses, with a score_1 in 1-5.
- Reasoning accuracy: over the parsed pairs, how often the reasoning scores the preferred response higher. The scores are
  integers, so ties are counted separately: "strict" counts a tie as wrong, "decided" leaves ties out.
- Agreement: over the parsed pairs without a tie, how often the reasoning and the BT verdict pick the same response.
- Score error: when the data has score ground truth, the mean absolute error of the reasoning's score_1 per response.

If BT accuracy improves over training while reasoning accuracy stays flat or falls, the reasoning is not being trained
toward better judgments; DWRL's thought term should improve both.

Other files for the benchmark, `<name>_<benchmark>_results.json` (e.g. `base_hs3local_results.json`), are listed first.
Score error needs score ground truth, so it is only reported for hs3local / hs4local.

Usage:
    python examples/analyze_genrm_dwrl_eval.py <results_dir> <benchmark> [--json summary.json]
    e.g.  python examples/analyze_genrm_dwrl_eval.py /path/to/results hs3local
"""

import argparse
import json
import math
import pathlib
import re
from typing import Any, Optional

BENCHMARKS = ["hs3local", "hs4local", "judgebench", "rmbench", "rewardbench2"]


def _score(prediction: Any) -> Optional[float]:
    """score_1 of a parsed reasoning verdict, if it is a valid 1-5 score (the eval saves the raw text when it is not)."""
    if not isinstance(prediction, dict):
        return None
    value = prediction.get("score_1")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not (math.isfinite(value) and float(value).is_integer() and 1 <= value <= 5):
        return None
    return float(value)


def _bt(value: Any) -> Optional[float]:
    """P("yes"), or None for the eval's error codes (-127 no output, -255 request error, -999 token not in top logprobs)."""
    return (
        float(value)
        if isinstance(value, (int, float)) and 0.0 <= value <= 1.0
        else None
    )


def _fraction(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def find_eval_files(
    results_dir: pathlib.Path, benchmark: str
) -> list[tuple[str, pathlib.Path]]:
    """(label, path) of the benchmark's result files: other files first (by name), then step_<N> files by N."""
    pattern = re.compile(rf"^(.*)_{re.escape(benchmark)}_results\.json$")
    found = []
    for path in results_dir.iterdir():
        match = pattern.match(path.name)
        if not path.is_file() or match is None:
            continue
        label = match.group(1)
        step = re.fullmatch(r"step_(\d+)", label)
        found.append(
            (
                (0, label, 0) if step is None else (1, "", int(step.group(1))),
                label,
                path,
            )
        )
    return [(label, path) for _, label, path in sorted(found)]


def check_pointwise(records: list[dict], path: pathlib.Path) -> None:
    """The files of the pairwise eval scripts have a different layout (their score_1 / score_2 are 1-5 judge scores)."""
    if records and not all(
        key in records[0]
        for key in ("prediction_1", "prediction_2", "score_1", "score_2", "metadata")
    ):
        raise ValueError(
            f"{path} is not an output of run_eval_genrm_dwrl_openai.py (DWRL pointwise): its records have the keys "
            f"{sorted(records[0])}"
        )


def summarize(records: list[dict]) -> dict:
    bt_n = bt_correct = 0
    parsed = reasoning_correct = reasoning_ties = 0
    agree_n = agree = 0
    score_errors = []
    for record in records:
        preference = record["metadata"][
            "preference"
        ]  # 0: response_1 is preferred, 1: response_2 is preferred
        bt_1, bt_2 = _bt(record.get("score_1")), _bt(record.get("score_2"))
        bt_winner = None
        if bt_1 is not None and bt_2 is not None:
            bt_n += 1
            # As in training, a tie counts as wrong
            bt_correct += (bt_1 > bt_2) if preference == 0 else (bt_2 > bt_1)
            bt_winner = 0 if bt_1 > bt_2 else 1 if bt_2 > bt_1 else None

        s_1, s_2 = (
            _score(record.get("prediction_1")),
            _score(record.get("prediction_2")),
        )
        ground_truth = record["metadata"].get("ground_truth_overall") or {}
        for score, truth in (
            (s_1, ground_truth.get("score_1")),
            (s_2, ground_truth.get("score_2")),
        ):
            if score is not None and truth is not None:
                score_errors.append(abs(score - truth))
        if s_1 is None or s_2 is None:
            continue
        parsed += 1
        if s_1 == s_2:
            reasoning_ties += 1
            continue
        reasoning_winner = 0 if s_1 > s_2 else 1
        reasoning_correct += reasoning_winner == preference
        if bt_winner is not None:
            agree_n += 1
            agree += reasoning_winner == bt_winner

    return {
        "rows": len(records),
        "bt_valid": bt_n,
        "bt_accuracy": _fraction(bt_correct, bt_n),
        "parse_rate": _fraction(parsed, len(records)),
        "reasoning_accuracy_strict": _fraction(reasoning_correct, parsed),
        "reasoning_accuracy_decided": _fraction(
            reasoning_correct, parsed - reasoning_ties
        ),
        "reasoning_tie_rate": _fraction(reasoning_ties, parsed),
        "reasoning_bt_agreement": _fraction(agree, agree_n),
        "score_1_mean_abs_error": sum(score_errors) / len(score_errors)
        if score_errors
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results_dir",
        type=pathlib.Path,
        help="Directory with the output JSON files of run_eval_genrm_dwrl_openai.py",
    )
    parser.add_argument("benchmark", choices=BENCHMARKS, help="Benchmark to report on")
    parser.add_argument(
        "--json", default=None, help="Also write the summaries to this JSON file"
    )
    args = parser.parse_args()

    eval_files = find_eval_files(args.results_dir, args.benchmark)
    if not eval_files:
        raise SystemExit(
            f"No *_{args.benchmark}_results.json files in {args.results_dir}"
        )

    summaries = {}
    for label, path in eval_files:  # one file at a time: some are over 100 MB
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        check_pointwise(records, path)
        summaries[label] = summarize(records)
        del records

    columns = [
        ("rows", "rows"),
        ("bt_accuracy", "BT acc"),
        ("parse_rate", "parsed"),
        ("reasoning_accuracy_strict", "reason acc"),
        ("reasoning_accuracy_decided", "reason acc (no ties)"),
        ("reasoning_tie_rate", "ties"),
        ("reasoning_bt_agreement", "agree w/ BT"),
        ("score_1_mean_abs_error", "score MAE"),
    ]
    name_width = max(len("checkpoint"), *(len(label) for label in summaries))
    widths = [max(len(title), 6) for _, title in columns]
    print(f"{args.benchmark}: {len(summaries)} files in {args.results_dir}")
    print(
        f"{'checkpoint':<{name_width}}  "
        + "  ".join(f"{title:>{width}}" for (_, title), width in zip(columns, widths))
    )
    for label, summary in summaries.items():
        cells = []
        for (key, _), width in zip(columns, widths):
            value = summary[key]
            text = (
                "-"
                if value is None
                else str(value)
                if isinstance(value, int)
                else f"{value:.4f}"
            )
            cells.append(f"{text:>{width}}")
        print(f"{label:<{name_width}}  " + "  ".join(cells))

    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
