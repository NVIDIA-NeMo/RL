"""Record the installed dependency sources without constructing a training model."""

import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

import torch
from nemo_automodel.components.moe.layers import Gate
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from torch.distributed.fsdp import FSDPModule


def main():
    versions = {}
    for package in (
        "torch",
        "transformers",
        "nemo-automodel",
        "transformer-engine",
        "nemo-lens",
        "transferqueue",
        "multi-storage-client",
        "ray",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    sources = {}
    for name, implementation in (
        ("Gate.forward", Gate.forward),
        ("Gate.update_bias", Gate.update_bias),
        ("MoEAuxLossAutoScaler.backward", MoEAuxLossAutoScaler.backward),
        (
            "FSDPModule.set_requires_gradient_sync",
            FSDPModule.set_requires_gradient_sync,
        ),
    ):
        filename = inspect.getsourcefile(implementation)
        sources[name] = {
            "path": filename,
            "line": inspect.getsourcelines(implementation)[1],
            "file_sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
        }

    result = {
        "kind": "dependency_introspection_only",
        "training_or_parity_executed": False,
        "user": os.environ.get("USER"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_git_version": torch.version.git_version,
        "versions": versions,
        "sources": sources,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
