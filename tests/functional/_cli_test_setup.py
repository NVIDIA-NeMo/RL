# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#!/usr/bin/env python3

"""Creates all checkpoints needed for CLI entry-point tests.

Usage:
    python tests/functional/_cli_test_setup.py <tmpdir> <model_name>

Writes <tmpdir>/paths.env with shell variable assignments for DCP_PATH,
MEG_PATH, LORA_PATH, and CONFIG_YAML so the calling shell script can
source it and pass exact paths to each converter CLI invocation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tests.functional.converter_test_utils import (
    create_dcp_checkpoint,
    create_megatron_checkpoint,
    create_megatron_lora_checkpoint,
    create_test_config,
    write_config_yaml,
)

tmp, model = sys.argv[1], sys.argv[2]

config = create_test_config()
# Override checkpoint_dir to keep all artifacts inside the temp directory
# so nothing is left behind after the trap cleanup in the shell script.
config["checkpointing"]["checkpoint_dir"] = os.path.join(tmp, "ckpts")

dcp_path = create_dcp_checkpoint(model, config, tmp)

# create_megatron_checkpoint returns the iter_0000000 subdirectory path,
# which is what --megatron-ckpt-path and --base-ckpt both expect.
meg_path = create_megatron_checkpoint(model, tmp)

lora_path = create_megatron_lora_checkpoint(model, meg_path, tmp)

config_yaml = os.path.join(tmp, "config.yaml")
write_config_yaml(config, config_yaml)

with open(os.path.join(tmp, "paths.env"), "w") as f:
    f.write(f'DCP_PATH="{dcp_path}"\n')
    f.write(f'MEG_PATH="{meg_path}"\n')
    f.write(f'LORA_PATH="{lora_path}"\n')
    f.write(f'CONFIG_YAML="{config_yaml}"\n')

print("✓ All checkpoints created and paths written to paths.env")
