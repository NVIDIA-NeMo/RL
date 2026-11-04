"""Locate the first observed difference, with clean/clean as the attribution control."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from precision_probe_common import (
    canonical_request_records,
    first_boundary_difference,
    save_json,
    semantic_model_state,
)

STAGES = [
    "trainer_before_refit",
    "wire_send",
    "wire_receive",
    "misc_send",
    "misc_receive",
    "sampler_after_refit",
    "request_inputs",
    "sampling_parameters",
    "sampler_logits",
    "generated_outputs",
]


def read_phase(root, phase):
    stages = {}
    counts = {}
    for stage in STAGES:
        ranked = defaultdict(list)
        for folder in sorted(
            (Path(root) / "boundaries" / phase).glob(stage + "-rank-*-pid-*")
        ):
            rank = folder.name.split("-rank-", 1)[1].split("-pid-", 1)[0]
            for path in sorted(folder.glob("*.json")):
                record = json.loads(path.read_text())
                if (
                    stage
                    in [
                        "trainer_before_refit",
                        "wire_send",
                        "wire_receive",
                        "misc_send",
                        "misc_receive",
                        "sampler_after_refit",
                        "sampler_logits",
                    ]
                    and record.get("refit_index", 1) != 1
                ):
                    continue
                if stage in ["trainer_before_refit", "sampler_after_refit"]:
                    # RNG/config are recorded separately, rather than confusing a
                    # global RNG offset with a difference in model parameters.
                    record = semantic_model_state(record["model"])
                elif stage == "sampler_logits":
                    if record.get("forward_context_complete", False):
                        counts["sampler_logits_with_forward_context"] = (
                            counts.get("sampler_logits_with_forward_context", 0) + 1
                        )
                    record = {
                        **{k: record[k] for k in ["shape", "dtype", "sha256"]},
                        "forward_context": record.get("forward_context"),
                        "forward_context_complete": record.get(
                            "forward_context_complete", False
                        ),
                        "lm_head_input": record.get("lm_head_input"),
                    }
                ranked[rank].append(record)
        counts[stage] = sum(map(len, ranked.values()))
        if ranked:
            # Initial refit + initial request cohort; later policy versions are
            # retained in the raw records, not mixed into the primary comparison.
            limit = (
                1
                if stage in ["trainer_before_refit", "sampler_after_refit"]
                else (
                    8
                    if stage
                    in ["request_inputs", "sampling_parameters", "generated_outputs"]
                    else 1000000
                )
            )
            stages[stage] = {
                rank: canonical_request_records(values[:limit])
                if stage
                in ["request_inputs", "sampling_parameters", "generated_outputs"]
                else values[:limit]
                for rank, values in ranked.items()
            }
    return stages, counts


def first_value_difference(left, right, path=""):
    if type(left) is not type(right):
        return {"path": path, "left": str(left)[:200], "right": str(right)[:200]}
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return {"path": path + "." + str(key), "reason": "missing"}
            result = first_value_difference(
                left[key], right[key], path + "." + str(key)
            )
            if result:
                return result
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "left_length": len(left), "right_length": len(right)}
        for i, (a, b) in enumerate(zip(left, right)):
            result = first_value_difference(a, b, path + f"[{i}]")
            if result:
                return result
        return None
    if left != right:
        return {"path": path, "left": left, "right": right}
    return None


def attribution(clean_repeat, poisoned):
    if clean_repeat is not None:
        return {
            "conclusion": "baseline_not_repeatable_or_incomplete",
            "baseline_first_difference": clean_repeat,
            "poison_first_difference": poisoned,
            "causal_claim": "Cannot attribute a difference to checkpoint poisoning before resolving the clean/clean control.",
        }
    if poisoned is not None:
        return {
            "conclusion": "poison_specific_candidate",
            "poison_first_difference": poisoned,
            "causal_claim": "First observed boundary differs only in this clean/poison pair; inspect raw tensors and repeat before assigning a cause.",
        }
    return {
        "conclusion": "no_observed_difference",
        "causal_claim": "Only the recorded boundaries and request cohort were compared.",
    }


def restrict_sequential_logit_window(stages, requests):
    outputs = [
        record
        for values in stages.get("generated_outputs", {}).values()
        for record in values
    ]
    if len(outputs) != requests or any(
        len(record.get("lengths", [])) != 1 for record in outputs
    ):
        return {
            "pass": False,
            "reason": "Initial sequential request coverage is incomplete",
        }
    tokens = sum(record["lengths"][0] for record in outputs)
    logits = stages.get("sampler_logits", {})
    if (
        tokens <= 0
        or not logits
        or any(
            len(values) < tokens
            or any(record["shape"][0] != 1 for record in values[:tokens])
            for values in logits.values()
        )
    ):
        return {
            "pass": False,
            "reason": "One-logit-row-per-token coverage is incomplete",
        }
    stages["sampler_logits"] = {
        rank: values[:tokens] for rank, values in logits.items()
    }
    return {"pass": True, "requests": requests, "logit_calls_per_rank": tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--require-forward-context", action="store_true")
    parser.add_argument("--initial-sequential-requests", type=int, default=0)
    args = parser.parse_args()
    records = {}
    counts = {}
    windows = {}
    for phase in ["clean_a", "clean_b", "poison"]:
        records[phase], counts[phase] = read_phase(args.root, phase)
        if args.initial_sequential_requests:
            windows[phase] = restrict_sequential_logit_window(
                records[phase], args.initial_sequential_requests
            )
    aa = first_boundary_difference(records["clean_a"], records["clean_b"], STAGES)
    ap = first_boundary_difference(records["clean_a"], records["poison"], STAGES)
    details = {}
    for label, difference, other in [
        ("clean_clean", aa, "clean_b"),
        ("clean_poison", ap, "poison"),
    ]:
        if difference and difference["reason"] == "different":
            stage = difference["stage"]
            details[label] = first_value_difference(
                records["clean_a"][stage], records[other][stage]
            )
    result = {
        "coverage": counts,
        "sequential_logit_window": windows,
        "complete_logit_context": {
            phase: phase_counts.get("sampler_logits_with_forward_context", 0)
            == phase_counts.get("sampler_logits", 0)
            and phase_counts.get("sampler_logits", 0) > 0
            for phase, phase_counts in counts.items()
        },
        "stage_comparisons": {
            label: {
                stage: first_boundary_difference(
                    records["clean_a"], records[other], [stage]
                )
                for stage in STAGES
            }
            for label, other in [("clean_clean", "clean_b"), ("clean_poison", "poison")]
        },
        "first_differing_values": details,
        **attribution(aa, ap),
    }
    missing_context = args.require_forward_context and not all(
        result["complete_logit_context"].values()
    )
    if missing_context:
        result["conclusion"] = "incomplete_forward_context"
        result["causal_claim"] = (
            "Model token/position capture is incomplete; identical conditioning has not been established."
        )
    incomplete_window = any(not value["pass"] for value in windows.values())
    if incomplete_window:
        result["conclusion"] = "incomplete_sequential_logit_window"
        result["causal_claim"] = (
            "The first serial request cohort could not be matched to logit calls."
        )
    save_json(Path(args.root) / "boundary-result.json", result)
    print(json.dumps(result, indent=2))
    raise SystemExit(
        aa is not None or ap is not None or missing_context or incomplete_window
    )


if __name__ == "__main__":
    main()
