#!/usr/bin/env python3
"""Generate ClippedPGLossFn numerical goldens from a pinned repository commit.

This script is intentionally separate from the unit test. Run it against a clean
worktree containing the source commit recorded in the output, then review and
check in the generated JSON. CI consumes the JSON and never regenerates it.
"""

import argparse
import copy
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


BASE_INPUTS: dict[str, Any] = {
    "input_ids": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    "token_mask": [[0.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0]],
    "sample_mask": [1.0, 1.0],
    "advantages": [
        [0.0, 1.2, -0.7, 0.4, 2.0],
        [0.0, -1.1, 0.6, 1.5, -0.3],
    ],
    "prev_logprobs": [
        [0.0, -1.0, -1.1, -0.8, -1.3],
        [0.0, -0.9, -1.0, -1.2, -0.8],
    ],
    "generation_logprobs": [
        [0.0, -1.2, -1.5, -0.7, -1.1],
        [0.0, -1.4, -0.7, -1.3, -1.0],
    ],
    "reference_policy_logprobs": [
        [0.0, -1.1, -0.7, -1.4, -1.0],
        [0.0, -1.3, -0.8, -1.0, -1.4],
    ],
    "rewards": [1.0, 0.0],
    "next_token_logprobs": [
        [-1.6, -0.8, -1.1, -1.4],
        [-0.7, -1.4, -0.9, -1.2],
    ],
}


CASE_CONFIGS: dict[str, dict[str, Any]] = {
    "ppo_token": {"reference_policy_kl_penalty": 0.0},
    "force_on_policy": {
        "reference_policy_kl_penalty": 0.0,
        "force_on_policy_ratio": True,
    },
    "reinforce": {
        "reference_policy_kl_penalty": 0.0,
        "disable_ppo_ratio": True,
    },
    "dual_clip": {
        "reference_policy_kl_penalty": 0.0,
        "ratio_clip_c": 3.0,
    },
    "cispo_with_token_is": {
        "reference_policy_kl_penalty": 0.0,
        "use_cispo": True,
        "use_importance_sampling_correction": True,
    },
    "kl_on_policy_unfiltered": {
        "reference_policy_kl_penalty": 0.17,
        "use_on_policy_kl_approximation": True,
    },
    "tis_token": {
        "reference_policy_kl_penalty": 0.0,
        "use_importance_sampling_correction": True,
        "truncated_importance_sampling_type": "tis",
        "truncated_importance_sampling_ratio": 1.3,
        "truncated_importance_sampling_ratio_min": 0.7,
    },
    "icepop_token": {
        "reference_policy_kl_penalty": 0.0,
        "use_importance_sampling_correction": True,
        "truncated_importance_sampling_type": "icepop",
        "truncated_importance_sampling_ratio": 1.25,
        "truncated_importance_sampling_ratio_min": 0.75,
    },
    "seq_mask_tis": {
        "reference_policy_kl_penalty": 0.0,
        "use_importance_sampling_correction": True,
        "truncated_importance_sampling_type": "seq-mask-tis",
        "truncated_importance_sampling_ratio": 1.15,
        "truncated_importance_sampling_ratio_min": 0.95,
    },
    "gspo_sequence_is": {
        "reference_policy_kl_penalty": 0.0,
        "use_importance_sampling_correction": True,
        "sequence_level_importance_ratios": True,
        "token_level_loss": False,
    },
    "vapo_mixed_rewards": {
        "reference_policy_kl_penalty": 0.0,
        "positive_example_nll_weight": 0.3,
    },
    "vapo_no_positive_rewards": {
        "reference_policy_kl_penalty": 0.0,
        "positive_example_nll_weight": 0.3,
    },
}


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _case_inputs(case_name: str) -> dict[str, Any]:
    inputs = copy.deepcopy(BASE_INPUTS)
    if case_name == "dual_clip":
        # Make the second valid token's negative-advantage ratio large enough for
        # the c=3 dual bound to be selected.
        inputs["prev_logprobs"][0][2] = -3.0
    elif case_name == "kl_on_policy_unfiltered":
        inputs["curr_logprobs_unfiltered"] = [
            [-1.3, -0.9, -1.0, -1.2],
            [-0.8, -1.2, -1.1, -1.0],
        ]
    elif case_name == "vapo_no_positive_rewards":
        inputs["rewards"] = [-1.0, 0.0]
    return inputs


def _tensor(value: Any, *, integer: bool = False) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.int64 if integer else torch.float64)


def _run_case(
    *,
    config_cls: type,
    loss_cls: type,
    case_name: str,
    config: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    next_token_logprobs = _tensor(inputs["next_token_logprobs"]).requires_grad_(True)
    data = {
        "input_ids": _tensor(inputs["input_ids"], integer=True),
        "token_mask": _tensor(inputs["token_mask"]),
        "sample_mask": _tensor(inputs["sample_mask"]),
        "advantages": _tensor(inputs["advantages"]),
        "prev_logprobs": _tensor(inputs["prev_logprobs"]),
        "generation_logprobs": _tensor(inputs["generation_logprobs"]),
        "reference_policy_logprobs": _tensor(inputs["reference_policy_logprobs"]),
        "rewards": _tensor(inputs["rewards"]),
    }
    gradient_inputs = [next_token_logprobs]
    gradient_names = ["next_token_logprobs"]
    if "curr_logprobs_unfiltered" in inputs:
        curr_logprobs_unfiltered = _tensor(
            inputs["curr_logprobs_unfiltered"]
        ).requires_grad_(True)
        data["curr_logprobs_unfiltered"] = curr_logprobs_unfiltered
        gradient_inputs.append(curr_logprobs_unfiltered)
        gradient_names.append("curr_logprobs_unfiltered")

    loss_fn = loss_cls(config_cls(**config))
    global_valid_seqs = data["sample_mask"].sum()
    global_valid_toks = (
        data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)
    ).sum()
    loss, metrics = loss_fn(
        next_token_logprobs=next_token_logprobs,
        data=data,
        global_valid_seqs=global_valid_seqs,
        global_valid_toks=global_valid_toks,
    )
    gradients = torch.autograd.grad(loss, gradient_inputs)

    return {
        "name": case_name,
        "config": config,
        "input_overrides": {
            name: value
            for name, value in inputs.items()
            if name not in BASE_INPUTS or value != BASE_INPUTS[name]
        },
        "expected": {
            "loss": loss.detach().item(),
            "gradients": {
                name: gradient.detach().tolist()
                for name, gradient in zip(gradient_names, gradients, strict=True)
            },
            "metrics": {name: float(value) for name, value in metrics.items()},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    if source_commit != args.expected_commit:
        raise RuntimeError(
            f"Expected source commit {args.expected_commit}, got {source_commit}"
        )
    dirty_state = _git(repo_root, "status", "--porcelain")
    if dirty_state:
        raise RuntimeError(f"Golden source worktree must be clean:\n{dirty_state}")

    sys.path.insert(0, str(repo_root))
    # Import only after selecting the source worktree so the recorded commit,
    # rather than the generator's checkout, supplies the implementation.
    from nemo_rl.algorithms.loss.loss_functions import (  # noqa: PLC0415
        ClippedPGLossConfig,
        ClippedPGLossFn,
    )

    imported_source = Path(inspect.getfile(ClippedPGLossFn)).resolve()
    if not imported_source.is_relative_to(repo_root):
        raise RuntimeError(
            f"Imported {imported_source}, expected a module beneath {repo_root}"
        )

    cases = [
        _run_case(
            config_cls=ClippedPGLossConfig,
            loss_cls=ClippedPGLossFn,
            case_name=case_name,
            config=config,
            inputs=_case_inputs(case_name),
        )
        for case_name, config in CASE_CONFIGS.items()
    ]
    output = {
        "schema_version": 1,
        "compatibility_contract": {
            "scope": (
                "Preserve loss, gradients, and metrics from the pinned legacy "
                "implementation except for the intentional corrections listed here."
            ),
            "intentional_corrections": [
                {
                    "case": "gspo_sequence_is",
                    "metric": "sampling_importance_ratio",
                    "reason": (
                        "Reduce one sequence weight per valid sample instead of "
                        "preserving the legacy [B, 1] by [B] broadcast."
                    ),
                }
            ],
        },
        "provenance": {
            "source_commit": source_commit,
            "source_file": str(imported_source.relative_to(repo_root)),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": "cpu",
            "dtype": "float64",
        },
        "base_inputs": BASE_INPUTS,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
