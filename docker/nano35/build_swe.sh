#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd /opt/nemo-rl
source docker/nano35/environment.sh
export UV_LINK_MODE=hardlink
export UV_HTTP_TIMEOUT=180 UV_HTTP_RETRIES=5
export UV_TORCH_BACKEND=cu130
export POETRY_VIRTUALENVS_IN_PROJECT=true
export MAKE_BUILD_TIMEOUT_SECONDS=1200
GYM_ROOT=/opt/nemo-rl/3rdparty/Gym-workspace/Gym
SWE_ROOT="$GYM_ROOT/responses_api_agents/swe_agents"
OH_SETUP="$SWE_ROOT/swe_openhands_setup"
mkdir -p "$OH_SETUP" /opt/nano35-metadata/swe

# Pin the installer and seed the ARM64 jq binary before Gym's setup script.
curl --retry 3 -fsSL \
    https://github.com/conda-forge/miniforge/releases/download/26.7.2-0/Miniforge3-26.7.2-0-Linux-aarch64.sh \
    -o /tmp/nano35-miniforge.sh
printf '%s  %s\n' 89b786c8d2c8b0fda7553914c1314ae4ddaa094503802f279377b19ac4463cb2 /tmp/nano35-miniforge.sh \
    | sha256sum -c -
if [[ ! -x "$OH_SETUP/miniforge3/bin/conda" ]]; then
    bash /tmp/nano35-miniforge.sh -b -p "$OH_SETUP/miniforge3"
fi
rm /tmp/nano35-miniforge.sh
curl --retry 3 -fsSL https://github.com/jqlang/jq/releases/download/jq-1.8.1/jq-linux-arm64 \
    -o "$OH_SETUP/miniforge3/bin/jq"
chmod 755 "$OH_SETUP/miniforge3/bin/jq"
"$OH_SETUP/miniforge3/bin/jq" --version

python - <<'PY'
import json
import os
import shutil
import subprocess
from pathlib import Path

repo = Path('/opt/nemo-rl')
swe = repo / '3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents'
pins = json.loads((repo / 'docker/nano35/swe_harness_pins.json').read_text())

def checkout(url, commit, directory):
    if (directory/'.git').exists():
        head = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
        if head != commit:
            raise RuntimeError(f'Unexpected existing checkout at {directory}: {head}')
        return
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-q', str(directory)], check=True)
    subprocess.run(['git', '-C', str(directory), 'remote', 'add', 'origin', url], check=True)
    subprocess.run(['git', '-C', str(directory), 'fetch', '--depth=1', 'origin', commit], check=True)
    subprocess.run(['git', '-C', str(directory), 'checkout', '--detach', 'FETCH_HEAD'], check=True)

pin = pins['openhands']
setup = swe / 'swe_openhands_setup'
openhands = setup / 'OpenHands'
checkout(pin['url'], pin['commit'], openhands)
patch = str(repo / pin['patch'])
already_applied = subprocess.run(['git', '-C', str(openhands), 'apply', '--reverse', '--check', patch],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
if not already_applied:
    subprocess.run(['git', '-C', str(openhands), 'apply', '--check', patch], check=True)
    subprocess.run(['git', '-C', str(openhands), 'apply', patch], check=True)
script = (swe / 'setup_scripts/openhands.sh').read_text()
old = "$miniforge_dir/bin/python -m pip install -q 'packaging==26.0'"
assert old in script
script = script.replace(old, "uv pip install --python \"$miniforge_dir/bin/python\" -q 'packaging==26.0'")
build_script = Path('/tmp/nano35-openhands-setup.sh')
build_script.write_text(script)
env = dict(os.environ, SETUP_DIR=str(setup), MINIFORGE_DIR=str(setup/'miniforge3'),
           OPENHANDS_DIR=str(openhands), AGENT_FRAMEWORK_REPO=pin['url'],
           AGENT_FRAMEWORK_COMMIT=pin['commit'])
subprocess.run(['bash', str(build_script)], env=env, check=True)

for key in ('swebench', 'multilingual', 'r2e', 'rebench'):
    pin = pins[key]
    setup = swe / pin['directory']
    directory = setup / pin['repository_directory']
    checkout(pin['url'], pin['commit'], directory)
    if key == 'rebench':
        # This frozen harness stores parsers below lib/. Gym's import path
        # already supports it; its prebuilt-tree check expects the old alias.
        assert (directory/'lib/agent/log_parsers.py').is_file()
        if not (directory/'agent').exists():
            (directory/'agent').symlink_to('lib/agent', target_is_directory=True)
        continue
    uv_dir = setup/'uv'
    uv_dir.mkdir(exist_ok=True)
    # The evaluator container binds this setup tree; keep its interpreter and
    # uv executable inside that tree instead of referring to the outer image.
    shutil.copy2('/usr/local/bin/uv', uv_dir/'uv')
    env = dict(os.environ, SETUP_DIR=str(setup), UV_DIR=str(uv_dir), PYTHON_DIR=str(setup/'python'),
               SWEBENCH_DIR=str(directory), SWEBENCH_REPO=pin['url'], SWEBENCH_COMMIT=pin['commit'],
               R2E_GYM_DIR=str(directory), EVAL_HARNESS_REPO=pin['url'], EVAL_HARNESS_COMMIT=pin['commit'])
    subprocess.run(['bash', str(swe/'setup_scripts'/pin['script'])], env=env, check=True)
PY

bash docker/nano35/finish_swe.sh
