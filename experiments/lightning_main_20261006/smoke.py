import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main() -> None:
    role = sys.argv[1]
    root = Path.cwd().resolve()
    expected = json.loads(Path(os.environ["EXPECTED_FINGERPRINT"]).read_text())
    baked = json.loads(Path("/opt/nemo_rl_container_fingerprint").read_text())
    # Git metadata is not in the source archive. Validate the committed manifest
    # directly; submodule Python imports below must resolve into its source tree.
    for key in (
        "pyproject.toml",
        "uv.lock",
        "nemo_rl/distributed/actor_environments.py",
    ):
        if baked.get(key) != expected[key]:
            raise RuntimeError(f"Nightly dependency fingerprint mismatch: {key}")
    print(
        json.dumps({"role": role, "executable": sys.executable, "fingerprint": baked})
    )
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("Expected one GB200 node with four visible GPUs")
    for device in range(4):
        x = torch.arange(16, device=f"cuda:{device}", dtype=torch.float32)
        torch.testing.assert_close(x.sum().cpu(), torch.tensor(120.0))
    modules = ["nemo_rl", "ray"]
    if role == "policy":
        modules += [
            "megatron.core",
            "megatron.bridge",
            "transformer_engine.pytorch",
            "mamba_ssm",
        ]
    elif role.startswith("vllm"):
        modules += ["vllm", "flashinfer"]
        if importlib.metadata.version("vllm").split("+")[0] != "0.29.0":
            raise RuntimeError("Expected vLLM 0.29.0")
        from vllm.platforms import current_platform

        if not current_platform.is_cuda():
            raise RuntimeError("vLLM did not resolve the CUDA platform")
    for name in modules:
        module = importlib.import_module(name)
        print(f"{name}: {module.__file__}")
        if name in ("nemo_rl", "megatron.core", "megatron.bridge"):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError(f"Stale source import: {name}")
    print(f"PASS: {role}")


if __name__ == "__main__":
    main()
