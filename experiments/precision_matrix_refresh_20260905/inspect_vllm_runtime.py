"""Record installed vLLM source without importing a model or initializing CUDA."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys


def main() -> None:
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    dist = importlib.metadata.distribution("vllm")
    package = Path(dist.locate_file("vllm")).resolve()
    files = (
        "v1/worker/gpu_worker.py",
        "v1/worker/gpu_model_runner.py",
        "utils/mem_utils.py",
        "model_executor/layers/fused_moe/routed_experts.py",
        "model_executor/model_loader/reload/layerwise.py",
        "model_executor/model_loader/reload/meta.py",
    )
    metadata = {"python": sys.version, "executable": sys.executable,
                "vllm_version": dist.version, "package": str(package), "files": {}}
    for name in files:
        source = package / name
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        metadata["files"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
