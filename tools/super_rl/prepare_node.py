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

"""Link image-owned Gym environments and check each configured native import."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

from omegaconf import OmegaConf

NATIVE_IMPORT_CHECK = """
import importlib
from pathlib import Path
import sys

module = importlib.import_module(sys.argv[1])
actual = Path(module.__file__).resolve()
expected = Path(sys.argv[2]).resolve()
if actual != expected:
    raise RuntimeError(f"Native Gym import source differs: {actual} != {expected}")
print(f"{module.__name__}: {actual}")
"""


def prepare(config: Path, gym: Path, image_venvs: Path, runtime_venvs: Path) -> dict:
    gym = gym.resolve()
    graph = OmegaConf.load(config).env.nemo_gym
    # Local model config fragments add services absent from the main recipe.
    graphs = [graph]
    for fragment in graph.config_paths:
        path = Path(fragment)
        graphs.append(OmegaConf.load(path if path.is_absolute() else gym / path))
    identities = set()
    imports = []
    for services in graphs:
        for service in services.values():
            if not OmegaConf.is_dict(service):
                continue
            for kind, components in service.items():
                if kind not in {
                    "resources_servers",
                    "responses_api_models",
                    "responses_api_agents",
                }:
                    continue
                for component in components:
                    identity = (kind, component)
                    if identity in identities:
                        continue
                    identities.add(identity)
                    source = image_venvs / kind / component / ".venv"
                    destination = runtime_venvs / kind / component / ".venv"
                    interpreter = source / "bin/python"
                    if not interpreter.is_file():
                        raise FileNotFoundError(interpreter)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.is_symlink() or destination.exists():
                        if destination.resolve() != source.resolve():
                            raise ValueError(
                                f"Existing environment differs: {destination}"
                            )
                    else:
                        destination.symlink_to(source, target_is_directory=True)
                    module = ".".join((*identity, "app"))
                    env = os.environ | {
                        "PYTHONPATH": f"{gym / kind / component}:{gym}",
                        # Gym reorders sys.path. A staged namespace component
                        # must take precedence over the image's editable tree.
                        "NEMO_GYM_EXTRA_ROOTS": str(gym),
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "1",
                    }
                    subprocess.run(
                        [
                            str(interpreter),
                            "-c",
                            NATIVE_IMPORT_CHECK,
                            module,
                            str(gym / kind / component / "app.py"),
                        ],
                        cwd=gym,
                        env=env,
                        check=True,
                        timeout=180,
                    )
                    imports.append(module)
    return {
        "complete": True,
        "hostname": socket.gethostname(),
        "components": sorted(imports),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gym", type=Path, required=True)
    parser.add_argument("--image-venvs", type=Path, required=True)
    parser.add_argument("--runtime-venvs", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.config, args.gym, args.image_venvs, args.runtime_venvs)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
