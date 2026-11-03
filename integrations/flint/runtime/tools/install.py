#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Install reviewed overlays into a disposable copy of the pinned Flint image."""

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from verify_runtime import AOT_PATH, SOURCE_PATH, verify_bundle

RUNTIME = Path(__file__).resolve().parents[1]
PATCHES = RUNTIME.parent / "patches"
SITE = Path("/opt/flint/envs/sglang/lib/python3.13/site-packages")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_patch(
    root: Path, patch: Path, pins: dict[str, tuple[str, str]], strip: int = 1
) -> dict[Path, bytes]:
    """Produce exact patched bytes in a temporary tree, rejecting other baselines."""
    observed = {name: digest(root / name) for name in pins}
    if all(observed[name] == after for name, (_, after) in pins.items()):
        return {}
    if not all(observed[name] == before for name, (before, _) in pins.items()):
        raise RuntimeError(f"Unknown or partially patched source group: {patch.name}")
    result = {}
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        for name in pins:
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, destination)
        subprocess.run(
            [
                "patch",
                "--batch",
                "--forward",
                "--fuzz=0",
                f"-p{strip}",
                "-d",
                str(staging),
            ],
            input=patch.read_bytes(),
            check=True,
            capture_output=True,
        )
        for name, (_, after) in pins.items():
            if digest(staging / name) != after:
                raise RuntimeError(f"Patched source hash differs: {name}")
            result[root / name] = (staging / name).read_bytes()
    return result


def main() -> None:
    stack = json.loads((RUNTIME / "stack.json").read_text())
    for profile in ["driver", "training", "sglang"]:
        expected = {
            "torch": stack["torch"],
            "ray": stack["ray"],
            "transformers": stack["transformers"][profile],
        }
        if profile == "training":
            expected["transformer-engine"] = stack["transformer_engine"]
        probe = subprocess.run(
            [
                f"/opt/flint/bin/python-{profile}",
                "-c",
                "import importlib.metadata,json,platform,sys; "
                "print(json.dumps({'python':platform.python_version(), "
                "'packages':{n:importlib.metadata.version(n) for n in sys.argv[1:]}}))",
                *expected,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        observed = json.loads(probe.stdout)
        if observed != {"python": stack["python"], "packages": expected}:
            raise RuntimeError(f"Package pins differ in the {profile} profile")
    bundle = Path("/flint-flashinfer-artifact")
    verify_bundle(bundle, stack["flashinfer"]["manifest_sha256"])
    # The private model implementation is an explicit, independently pinned input.
    plugin = Path("/opt/flint/flint-sglang/src/flint_sglang")
    for name, expected in stack["flint_sglang_sources_sha256"].items():
        if digest(plugin / name) != expected:
            raise RuntimeError(
                f"Flint plugin differs from the tested implementation: {name}"
            )
    plans = prepare_patch(
        Path("/opt/flint/Automodel"),
        PATCHES / "automodel-sink-checkpoint.patch",
        {
            "nemo_automodel/components/models/flint/layers.py": (
                "dbe9b8eb7585cd01c28feaf3172ef886b76d90bcd95f4593f7bae08051a55593",
                "e57d036664e889d4ac639d1aeec198c5e127d2926de00c0656ede4b233ed4a4c",
            ),
        },
    )
    kv = json.loads((PATCHES / "hybrid-kv-offload.json").read_text())
    plans.update(
        prepare_patch(
            SITE,
            PATCHES / "sglang-hybrid-kv-offload.patch",
            {
                entry["relative_path"]: (
                    entry["baseline_sha256"],
                    entry["candidate_sha256"],
                )
                for entry in kv["changes"]
            },
        )
    )
    plans.update(
        prepare_patch(
            SITE / "sglang",
            PATCHES / "fa4-learned-sink-fix.patch",
            {
                "jit_kernel/flash_attn/cute/flash_fwd_sm100.py": (
                    "9b27db3b1d1b64bcd6e15bd47648bb3f22fe5614f79794f850c3808612efd949",
                    stack["fa4_interface_sha256"],
                ),
            },
            strip=3,
        )
    )
    router = SITE / "sglang_router/router_args.py"
    router_before = "189b46ce5ed15067c7e97605d402e15767d64f5b2810b6fb2db77db22cbf8016"
    router_after = "09cd8984a1f693cc2efe40b963befd6d5bc578e972272e994a046f183807f3e8"
    if digest(router) == router_before:
        candidate = router.read_bytes() + (
            b"\n\n# Compatibility for NeMo RL's managed-router configuration field.\n"
            b"RouterArgs.router_policy = property(lambda self: self.policy, lambda self, value: setattr(self, 'policy', value))\n"
        )
        if hashlib.sha256(candidate).hexdigest() != router_after:
            raise RuntimeError("Router alias candidate differs")
        plans[router] = candidate
    elif digest(router) != router_after:
        raise RuntimeError("Unknown router source version")
    # All source and bundle checks finish before any installed file is changed.
    for target, payload in plans.items():
        target.write_bytes(payload)
    shutil.copyfile(bundle / AOT_PATH.name, AOT_PATH)
    shutil.copyfile(bundle / SOURCE_PATH.name, SOURCE_PATH)
    print(
        json.dumps(
            {
                "status": "installed",
                "source_files_updated": len(plans),
                "library_sha256": stack["flashinfer"]["library_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
