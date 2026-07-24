#!/usr/bin/env python3
"""Fail closed if NeMo-RL Multi-LoRA depends on an external nousnet checkout."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_PATHS = (
    "nousnet_pre_multilora",
    "Automodel_container_base",
    "RL_super_v3",
    "/workspace/nousnet",
)
IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+nousnet(?:\.|\s|$)")
TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".slurm",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".md",
    ".txt",
}
SKIP_PREFIXES = ("results/", "charts_native100/")


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    failures: list[str] = []
    for rel in tracked_files():
        if rel.startswith(SKIP_PREFIXES):
            continue
        path = ROOT / rel
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            if any(token in line for token in FORBIDDEN_PATHS):
                failures.append(f"external path: {rel}:{line_no}: {line.strip()}")
            if path.suffix == ".py" and IMPORT_RE.search(line):
                failures.append(f"external import: {rel}:{line_no}: {line.strip()}")

    required = (
        "ray.sub",
        "examples/configs/recipes/multi_lora/ray.sub",
        "examples/configs/recipes/multi_lora/sft_8gpu_native.slurm",
        "examples/run_sft_multi_lora.py",
        "nemo_rl/models/multi_lora/adapter.py",
        "patches/automodel/nemo_automodel_components__peft_lora.py",
        "patches/automodel/nemo_automodel_components_distributed_parallelizer.py",
        "patches/automodel/nemo_automodel_components_distributed_parallelizer_utils.py",
        "patches/automodel/nemo_automodel_components_checkpoint_checkpointing.py",
    )
    for rel in required:
        if not (ROOT / rel).is_file():
            failures.append(f"missing repo-contained runtime file: {rel}")

    launcher = (
        ROOT / "examples/configs/recipes/multi_lora/sft_8gpu_native.slurm"
    ).read_text(encoding="utf-8")
    for token in (
        "examples/configs/recipes/multi_lora/ray.sub",
        "STANDALONE_OK: nousnet not importable",
        "AUTOMODEL_INTEGRATION_OK",
    ):
        if token not in launcher:
            failures.append(f"launcher missing standalone assertion/wiring: {token}")

    recipe_ray_sub = (
        ROOT / "examples/configs/recipes/multi_lora/ray.sub"
    ).read_text(encoding="utf-8")
    if "--mpi=pmi2" not in recipe_ray_sub:
        failures.append("recipe ray.sub is not the cluster-adapted pmi2 variant")

    if failures:
        print("STANDALONE_AUDIT=FAIL")
        print("\n".join(failures))
        return 1

    print("STANDALONE_AUDIT=PASS")
    print("external_nousnet_imports=0")
    print("external_source_paths=0")
    print("repo_contained_ray_sub=1")
    print("repo_contained_automodel_integration_files=4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
