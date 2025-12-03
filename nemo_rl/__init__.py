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

# Configure logging to show file location
logging.basicConfig(
    format="%(levelname)s:%(name)s:%(filename)s:%(lineno)d: %(message)s",
)

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
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"


def _is_build_isolation():
    """Detect if we're running in a uv build isolation environment.

    When running uv lock/sync, uv creates a temporary isolated environment
    in ~/.cache/uv/builds-v*/ to build packages and introspect metadata.
    We skip the fingerprint check in this context since the user is updating dependencies.

    Returns True if in build isolation, False otherwise.
    """
    # Check if we're in uv's build isolation directory
    # uv always uses paths like: /root/.cache/uv/builds-v0/.tmp*/
    return "/builds-v" in sys.prefix


def _check_container_fingerprint():
    """Check if container dependencies match the current code (container-only).

    This check only runs when NRL_CONTAINER=1 is set (inside containers).
    It compares the container's fingerprint (computed at build time) with
    the current code's fingerprint to detect dependency drift.

    This check is also skipped entirely if NRL_FORCE_REBUILD_VENVS=true is set,
    since environment rebuilding will ensure dependencies are consistent regardless
    of a mismatch.

    If there's a mismatch, raises RuntimeError unless NRL_IGNORE_VERSION_MISMATCH is set.
    """
    # Skip check if not in container or if we're going to force venv rebuild anyway
    if not os.environ.get("NRL_CONTAINER"):
        return
    if os.environ.get("NRL_FORCE_REBUILD_VENVS", "").lower() == "true":
        logging.info(
            "Skipping container fingerprint check because NRL_FORCE_REBUILD_VENVS=true (venvs will be rebuilt anyway)"
        )
        return

    # Skip check if we're in a build isolation environment (e.g., during uv lock/sync)
    if _is_build_isolation():
        logging.debug(
            "Skipping container fingerprint check because we're in a build isolation environment"
        )
        return

    try:
        import json
        import runpy
        import sys
        from io import StringIO

        # Get repo root (relative to this module)
        repo_root = Path(__file__).parent.parent
        fingerprint_script = repo_root / "tools" / "generate_fingerprint.py"

        # Check if script exists
        if not fingerprint_script.exists():
            logging.warning(
                f"Fingerprint script not found at {fingerprint_script}, skipping version check"
            )
            return

        # Compute current code fingerprint using runpy (cleaner than subprocess)
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        try:
            runpy.run_path(str(fingerprint_script), run_name="__main__")
            current_fingerprint_json = captured_output.getvalue().strip()
        finally:
            sys.stdout = old_stdout

        if not current_fingerprint_json:
            logging.warning("Failed to compute code fingerprint: empty output")
            return

        current_fingerprint = json.loads(current_fingerprint_json)

        # Read container fingerprint
        container_fingerprint_file = Path("/opt/nemo_rl_container_fingerprint")
        if not container_fingerprint_file.exists():
            logging.warning(
                "Container fingerprint file not found, skipping version check"
            )
            return

        container_fingerprint = json.loads(
            container_fingerprint_file.read_text().strip()
        )

        # Compare fingerprints and find differences
        all_keys = set(current_fingerprint.keys()) | set(container_fingerprint.keys())
        differences = []

        for key in sorted(all_keys):
            current_val = current_fingerprint.get(key, "missing")
            container_val = container_fingerprint.get(key, "missing")

            if current_val != container_val:
                differences.append(f"  - {key}:")
                differences.append(f"      Container: {container_val}")
                differences.append(f"      Current:   {current_val}")

        if differences:
            diff_text = "\n".join(differences)
            sep_line = "\n" + ("-" * 80)
            warning_msg = (
                f"{sep_line}\n"
                "WARNING: Container/Code Version Mismatch Detected!\n"
                f"{sep_line}\n"
                "Your container's dependencies do not match your current code.\n"
                "\n"
                "Differences found:\n"
                f"{diff_text}\n"
                "\n"
                "This can lead to unexpected behavior or errors.\n"
                "\n"
                "Solutions:\n"
                "  1. Rebuild the container to match your code\n"
                "  2. Set NRL_FORCE_REBUILD_VENVS=true to rebuild virtual environments\n"
                "     (This forces Ray workers to recreate their venvs with updated dependencies)\n"
                "  3. Update the container fingerprint to match your current code (for local dev):\n"
                "     python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint\n"
                "  4. Set NRL_IGNORE_VERSION_MISMATCH=1 to bypass this check (not recommended)\n"
                "\n"
                "Learn more about dependency management:\n"
                "  https://github.com/NVIDIA-NeMo/RL/blob/main/docs/design-docs/dependency-management.md\n"
                f"{sep_line}\n"
            )

            # Check if user wants to ignore the mismatch
            if not bool(os.environ.get("NRL_IGNORE_VERSION_MISMATCH")):
                logging.warning(
                    warning_msg
                    + "Proceeding anyway (NRL_IGNORE_VERSION_MISMATCH is set)..."
                )
        else:
            logging.debug("Container fingerprint matches code fingerprint")

    except RuntimeError:
        # Re-raise RuntimeError for version mismatches (user should see this)
        raise
    except Exception as e:
        # Log other errors but don't crash on version check failures
        logging.debug(f"Version check failed (non-fatal): {e}")


# Perform container version check
_check_container_fingerprint()


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


# Need to set PYTHONPATH to include transformers downloaded modules.
# Assuming the cache directory is the same cross venvs.
def patch_transformers_module_dir(env_vars: dict[str, str]):
    hf_home = os.environ.get("HF_HOME", None)
    if hf_home is None:
        return env_vars

    module_dir = os.path.join(hf_home, "modules")
    if not os.path.isdir(module_dir):
        return env_vars

    if "PYTHONPATH" not in env_vars:
        env_vars["PYTHONPATH"] = module_dir
    else:
        env_vars["PYTHONPATH"] = f"{module_dir}:{env_vars['PYTHONPATH']}"

    return env_vars


patch_transformers_module_dir(os.environ)
