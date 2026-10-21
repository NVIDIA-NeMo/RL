#!/usr/bin/env python3
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

"""Audit the 50-step Flint DAPO run using read-only CPU checkpoint inspection.

The original smoke auditor is imported unchanged. Only its exact zero-weight-
decay requirement is replaced with verification of this run's explicit 0.1.
Payload checks cover the complete output head and all 49 attention sinks;
ordinary parameter values and optimizer moments are outside this audit's scope.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile


TRAIN_METRIC_KEYS = ("loss", "grad_norm", "reward", "gen_kl_error", "policy_kl_error", "lr")
EXPECTED_VALIDATION_STEPS = (0, *range(5, 51, 5))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load audit helper: {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses in the unchanged base auditor resolve their defining module.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def metric_records(run_root: Path) -> tuple[dict, dict, list, list]:
    """Select the latest complete event per step across checkpoint resumes."""
    train, validation, malformed = {}, {}, []
    files = sorted((run_root / "evidence" / "jobs").rglob("*.metrics.jsonl"),
                   key=lambda path: (path.stat().st_mtime_ns, str(path)))
    for path in files:
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    step, prefix, metrics = record["step"], record["prefix"], record["metrics"]
                    if not isinstance(step, int) or isinstance(step, bool) or not isinstance(metrics, dict):
                        raise ValueError("Invalid metric event")
                except (ValueError, KeyError, TypeError) as error:
                    malformed.append({"file": str(path), "line": line_number, "error": type(error).__name__})
                    continue
                item = {"step": step, "metrics": metrics, "file": str(path), "line": line_number}
                # Rollout-only train events precede the completed optimizer event.
                if prefix == "train" and "loss" in metrics:
                    train[step] = item
                elif prefix == "validation" and "accuracy" in metrics:
                    validation[step] = item
    return train, validation, [str(path) for path in files], malformed


def inspect_training_metrics(train: dict) -> dict:
    errors = []
    expected = set(range(1, 51))
    if set(train) != expected:
        errors.append(f"Completed training steps differ from 1..50; missing={sorted(expected-set(train))}, unexpected={sorted(set(train)-expected)}")
    for step, item in sorted(train.items()):
        metrics = item["metrics"]
        for key in TRAIN_METRIC_KEYS:
            if not finite_number(metrics.get(key)):
                errors.append(f"Step {step} lacks a finite {key}")
        if finite_number(metrics.get("grad_norm")) and metrics["grad_norm"] < 0:
            errors.append(f"Step {step} has a negative gradient norm")
        if finite_number(metrics.get("total_turns")) and metrics["total_turns"] != 256:
            errors.append(f"Step {step} reported {metrics['total_turns']} rollout turns instead of 256")
    nonzero_gradient_steps = [step for step, item in sorted(train.items())
                              if finite_number(item["metrics"].get("grad_norm")) and item["metrics"]["grad_norm"] > 0]
    if not nonzero_gradient_steps:
        errors.append("No completed training step reported a positive gradient norm")
    return {"status": "pass" if not errors else "fail", "errors": errors,
            "completed_step_count": len(train), "completed_steps": sorted(train),
            "nonzero_gradient_steps": nonzero_gradient_steps,
            "records": [item for _, item in sorted(train.items())]}


def scalar_reward(value) -> float:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not finite_number(value) or value not in (0.0, 1.0):
        raise ValueError("AIME reward is not one finite binary DAPO score")
    return float(value)


def prompt_fingerprint(content) -> str:
    while isinstance(content, list) and len(content) == 1 and isinstance(content[0], list):
        content = content[0]
    if not isinstance(content, list):
        raise ValueError("Validation content is not a conversation")
    users = [message["content"] for message in content
             if isinstance(message, dict) and message.get("role") == "user"]
    if not users or not all(isinstance(value, str) for value in users):
        raise ValueError("Validation conversation has no textual user prompt")
    return hashlib.sha256(json.dumps(users, ensure_ascii=False).encode()).hexdigest()


def inspect_validation(run_root: Path, validation: dict) -> dict:
    errors, records = [], []
    for step in EXPECTED_VALIDATION_STEPS:
        item = validation.get(step)
        if item is None:
            errors.append(f"No completed AIME validation metric at step {step}")
            continue
        accuracy = item["metrics"].get("accuracy")
        if not finite_number(accuracy) or not 0.0 <= accuracy <= 1.0:
            errors.append(f"Step {step} validation accuracy is invalid")
        candidates = sorted((run_root / "logs" / "dapo50").rglob(f"val_data_step{step}.jsonl"),
                            key=lambda path: (path.stat().st_mtime_ns, str(path)))
        if not candidates:
            errors.append(f"No raw AIME validation records at step {step}")
            continue
        path = candidates[-1]
        rewards, prompts, indices = [], Counter(), []
        try:
            with path.open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rewards.append(scalar_reward(row["rewards"]))
                    prompts[prompt_fingerprint(row["content"])] += 1
                    indices.append(row["idx"])
            if len(rewards) != 960:
                errors.append(f"Step {step} processed {len(rewards)} AIME rows instead of 960")
            if indices != list(range(960)):
                errors.append(f"Step {step} raw validation row indices do not cover 0..959 exactly")
            if len(prompts) != 30 or set(prompts.values()) != {32}:
                errors.append(f"Step {step} validation does not cover 30 distinct prompts 32 times each")
            empirical_accuracy = sum(rewards) / len(rewards) if rewards else None
            if finite_number(accuracy) and (empirical_accuracy is None or abs(empirical_accuracy-accuracy) > 1e-6):
                errors.append(f"Step {step} logged accuracy does not match its raw rewards")
            records.append({"step": step, "metric_file": item["file"], "raw_file": str(path),
                            "samples_processed": len(rewards), "distinct_prompts": len(prompts),
                            "samples_per_distinct_prompt": sorted(set(prompts.values())),
                            "accuracy": accuracy, "raw_reward_mean": empirical_accuracy})
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"Step {step} validation records are invalid: {type(error).__name__}: {error}")
    return {"status": "pass" if not errors else "fail", "errors": errors,
            "expected_steps": list(EXPECTED_VALIDATION_STEPS), "records": records,
            "protocol": "Unaltered DAPO AIME-2024 split: 960 rows, 30 prompts repeated 32 times; one response per row"}


def adapt_protected_audit(report: dict) -> dict:
    """Replace only the known smoke weight-decay gate, preserving other errors."""
    weight_decay = report.get("weight_decay")
    expected = "Training config must explicitly evidence weight_decay=0; found 0.1"
    original_errors = list(report.get("errors", []))
    if not finite_number(weight_decay) or weight_decay != 0.1:
        report.setdefault("errors", []).append(f"DAPO training config must evidence weight_decay=0.1; found {weight_decay!r}")
    else:
        report["errors"] = [error for error in original_errors if error != expected]
        if original_errors.count(expected) != 1:
            report["errors"].append("Unchanged smoke auditor did not emit exactly its expected weight_decay=0.1 rejection")
    report["weight_decay_policy_adjustment"] = {
        "expected_weight_decay": 0.1, "actual_weight_decay": weight_decay,
        "expected_weight_decay_verified": finite_number(weight_decay) and weight_decay == 0.1,
        "removed_exact_error": expected if original_errors.count(expected) == 1 and weight_decay == 0.1 else None,
        "original_errors": original_errors,
        "change_attribution": "Parameter changes may include AdamW weight decay and are not attributed solely to gradients",
    }
    summary = report.get("summary", {})
    requirements = {
        "source_tensor_count": 36734, "metadata_audited_tensor_count": 36734,
        "audited_tensor_count": 50, "sink_count": 49, "payload_scope": "protected",
        "audited_saved_parameters_finite": True, "dtype_protection_passed": True,
        "all_source_parameter_metadata_covered": True, "all_selected_payload_parameters_covered": True,
        "all_saved_parameters_finite": None, "weight_decay_zero_verified": False,
    }
    for key, value in requirements.items():
        if key not in summary or summary[key] != value:
            report.setdefault("errors", []).append(f"Protected audit summary {key}={summary.get(key)!r}; expected {value!r}")
    if (report.get("checkpoint_config") or {}).get("num_hidden_layers") != 49:
        report.setdefault("errors", []).append("Saved checkpoint configuration does not evidence all 49 layers")
    protected = report.get("tensors", [])
    names = [item.get("name") for item in protected]
    sinks = [name for name in names if isinstance(name, str) and name.endswith("self_attn.softmax_offset")]
    if len(set(names)) != 50 or "lm_head.weight" not in names or len(set(sinks)) != 49:
        report.setdefault("errors", []).append("Protected payloads do not contain exactly one head and 49 distinct sinks")
    for item in protected:
        if item.get("saved_dtypes") != ["F32"] or item.get("finite") is not True or item.get("covered_elements") != item.get("parameter_count"):
            report.setdefault("errors", []).append(f"Protected payload {item.get('name')!r} is not complete, finite F32")
    report["status"] = "pass" if not report.get("errors") else "fail"
    return report


def run_audit(run_root: Path, output: Path) -> dict:
    checkpoint = run_root / "results" / "dapo50" / "step_50"
    training_config = checkpoint / "config.yaml"
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "run_root": str(run_root), "checkpoint": str(checkpoint), "errors": [],
              "scope": {"model_metadata": "All 36,734 source tensor names and saved regions",
                        "model_payload": "Complete output head and all 49 attention sinks only",
                        "ordinary_parameter_payload_finiteness_verified": False,
                        "optimizer": "Selected saved LM-head Adam step tensors only; no moments read",
                        "parameter_change_attribution": "Weight decay is 0.1; changes are not attributed solely to gradients"}}
    try:
        helper = load_module("flint_base_training_audit", Path("/flint-work/tools/audit_training_result.py"))
        protected = helper.audit(Path("/flint-model"), checkpoint, training_config, "float32", 1_000_000, True, "protected")
        protected = adapt_protected_audit(protected)
        protected_path = output.with_name(output.stem + ".protected.json")
        write_json(protected_path, protected)
        report["protected_checkpoint_audit"] = {"status": protected["status"], "path": str(protected_path), "summary": protected.get("summary"), "errors": protected.get("errors")}
        if protected["status"] != "pass":
            report["errors"].append("Protected checkpoint or full metadata audit failed")
    except Exception as error:
        report["errors"].append(f"Protected checkpoint audit failed: {type(error).__name__}: {error}")
    try:
        helper = load_module("flint_base_optimizer_audit", Path("/flint-work/tools/check_resume_optimizer.py"))
        optimizer = helper.optimizer_step(checkpoint, "lm_head")
        optimizer.update(expected_step=100, passed=optimizer["value"] == 100)
        optimizer_path = output.with_name(output.stem + ".optimizer.json")
        write_json(optimizer_path, optimizer)
        report["selected_adam_counter"] = {"status": "pass" if optimizer["passed"] else "fail", "path": str(optimizer_path), **optimizer}
        if not optimizer["passed"]:
            report["errors"].append(f"Selected Adam counter is {optimizer['value']}, expected 100")
    except Exception as error:
        report["errors"].append(f"Selected Adam counter audit failed: {type(error).__name__}: {error}")
    train, validation, files, malformed = metric_records(run_root)
    report["metric_files"] = files
    report["malformed_historical_metric_events"] = malformed
    report["metric_selection"] = "Latest completed event per step, ordered by file modification time, path, and event line; rollout-only events excluded"
    report["training_metrics"] = inspect_training_metrics(train)
    report["aime_validation"] = inspect_validation(run_root, validation)
    for key in ("training_metrics", "aime_validation"):
        if report[key]["status"] != "pass":
            report["errors"].append(f"{key} failed: {report[key]['errors']}")
    report["status"] = "pass" if not report["errors"] else "fail"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    run_root, output = args.run_root.resolve(strict=True), args.output.resolve()
    if run_root / "results" in output.parents or Path("/flint-model") in output.parents:
        raise ValueError("Write the audit outside read-only checkpoint and source model directories")
    try:
        report = run_audit(run_root, output)
    except Exception as error:
        report = {"status": "error", "error": f"{type(error).__name__}: {error}", "run_root": str(run_root)}
    write_json(output, report)
    print(json.dumps({"status": report["status"], "output": str(output), "errors": report.get("errors", report.get("error"))}), flush=True)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
