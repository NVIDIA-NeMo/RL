"""Compare logical refit weights and full logits only at identical token contexts."""

import json
from pathlib import Path

import torch

from precision_probe_common import tensor_error


def records(root, mode, stage, named=False):
    result = {}
    for folder in sorted((Path(root) / mode).glob(stage + "-rank-*-pid-*")):
        rank = int(folder.name.split("-rank-", 1)[1].split("-pid-", 1)[0])
        for path in sorted(folder.glob("*.json")):
            item = json.loads(path.read_text())
            key = (rank, item["layer"] if named else path.stem)
            assert key not in result, (stage, "Ambiguous duplicate capture", key)
            result[key] = (item, path.with_suffix(".pt"))
    return result


def compare_refit_boundaries(root, mode, reference):
    result = {}
    for stage in ["sampler_logical_mxfp8", "sampler_execution_mxfp8"]:
        a, b = [records(root, label, stage, named=True) for label in [mode, reference]]
        coverage = bool(a and b) and a.keys() == b.keys() and {k[0] for k in a} == set(range(4))
        differences = [list(k) for k in a.keys() & b.keys() if a[k][0] != b[k][0]]
        result[stage] = {"coverage": coverage, "equal": coverage and not differences,
                         "layers_per_rank": {str(rank): sum(k[0] == rank for k in a) for rank in range(4)},
                         "different_layers": differences}
        if stage == "sampler_logical_mxfp8":
            effective_coverage = coverage and all(
                set(v[0].get("effective_weights", {})) == {"w13_weight", "w2_weight"}
                for values in [a, b] for v in values.values())
            effective_differences = [list(k) for k in a.keys() & b.keys()
                if a[k][0].get("effective_weights") != b[k][0].get("effective_weights")]
            result[stage]["effective_weight_coverage"] = effective_coverage
            result[stage]["effective_weights_equal"] = effective_coverage and not effective_differences
            result[stage]["different_effective_weight_layers"] = effective_differences
    a, b = [records(root, label, "sampler_logits") for label in [mode, reference]]
    coverage = bool(a and b) and a.keys() == b.keys() and {k[0] for k in a} == set(range(4))
    logits = {"coverage": coverage, "equal": coverage, "calls": len(a), "first_difference": None}
    for key in sorted(a.keys() & b.keys()):
        x, y = a[key][0], b[key][0]
        if not x["forward_context_complete"] or not y["forward_context_complete"]:
            reason = {"reason": "missing_forward_context"}
        elif x["forward_context"] != y["forward_context"]:
            reason = {"reason": "different_token_context"}
        elif not x["finite"] or not y["finite"]:
            reason = {"reason": "nonfinite_logits"}
        elif x["sha256"] != y["sha256"]:
            reason = {"reason": "different_logits_at_same_context", "error": tensor_error(
                torch.load(a[key][1], weights_only=True, map_location="cpu"),
                torch.load(b[key][1], weights_only=True, map_location="cpu"), atol=0, rtol=0)}
        else:
            continue
        logits["equal"] = False
        if logits["first_difference"] is None:
            logits["first_difference"] = {"rank": key[0], "call": key[1], **reason}
            break
    result["sampler_logits"] = logits
    return result
