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
import logging
import os
import sys
from pathlib import Path

"""
This is a work around to ensure whenever NeMo RL is imported, that we
add Megatron-LM to the python path. This is because the only sub-package
that's officially installed is megatron.core. So we add the whole repo into
the path so we can access megatron.{training,legacy,inference,...}

Since users may pip install NeMo RL, this is a convenience so they do not
have to manually run with PYTHONPATH=3rdparty/Megatron-LM-workspace/Megatron-LM.
"""
megatron_path = (
    Path(__file__).parent.parent / "3rdparty" / "Megatron-LM-workspace" / "Megatron-LM"
)
if megatron_path.exists() and str(megatron_path) not in sys.path:
    sys.path.append(str(megatron_path))

from nemo_rl.package_info import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __shortversion__,
    __version__,
)

os.environ["RAY_USAGE_STATS_ENABLED"] = "0"


def _patch_nsight_file():
    """Patch the nsight.py file to fix the context.py_executable assignment.

    Until this fix is upstreamed, we will maintain this patch here. This patching
    logic is only applied if the user intends to use nsys profiling which they enable with
    NRL_NSYS_WORKER_PATTERNS.

    If enabled, will effectively apply the following patch in an idempotent manner:

    https://github.com/ray-project/ray/compare/master...terrykong:ray:tk/nsight-py-exeutable-fix?expand=1

    This hack works b/c the nsight plugin is not called from the main driver process, so
    as soon as nemo_rl is imported, the patch is applied and the source of the nsight.py module
    is up to date before the nsight.py is actually needed.
    """
    # Only apply patch if user intends to use nsys profiling

    # Don't rely on nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS since nemo_rl may not be available
    # on the node that imports nemo_rl.
    if not os.environ.get("NRL_NSYS_WORKER_PATTERNS"):
        return

    try:
        from ray._private.runtime_env import nsight

        file_to_patch = nsight.__file__

        # Read the current file content
        with open(file_to_patch, "r") as f:
            content = f.read()

        # The line we want to replace
        old_line = 'context.py_executable = " ".join(self.nsight_cmd) + " python"'
        new_line = 'context.py_executable = " ".join(self.nsight_cmd) + f" {context.py_executable}"'

        # Check if patch has already been applied (idempotent check)
        if new_line in content:
            # Already patched
            logging.info(f"Ray nsight plugin already patched at {file_to_patch}")
            return

        # Check if the old line exists to patch
        if old_line not in content:
            # Nothing to patch or file structure has changed
            logging.warning(
                f"Expected line not found in {file_to_patch} - Ray version may have changed"
            )
            return

        # Apply the patch
        patched_content = content.replace(old_line, new_line)

        # Write back the patched content
        with open(file_to_patch, "w") as f:
            f.write(patched_content)

        logging.info(f"Successfully patched Ray nsight plugin at {file_to_patch}")

    except (ImportError, FileNotFoundError, PermissionError) as e:
        # Allow failures gracefully - Ray might not be installed or file might be read-only
        pass


# Apply the patch
_patch_nsight_file()
