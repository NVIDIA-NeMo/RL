# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Avoid redundant APT network requests in the pinned native OpenCode installer.

Harbor 0.6.6 always updates package indexes before installing curl, including in
images where curl already exists. Keep the native installer and its NVM/npm
commands intact; only make this prerequisite installation conditional.
"""

import hashlib
import json
import os
import stat
import tempfile
from importlib.metadata import distribution
from pathlib import Path

ORIGINAL = 'command="apt-get update && apt-get install -y curl",'
CONDITIONAL = 'command="command -v curl >/dev/null 2>&1 || (apt-get update && apt-get install -y curl)",'


def patch_installer(source: Path) -> bool:
    """Patch one known native command atomically, preserving its original file."""
    before = source.read_text()
    if before.count(CONDITIONAL) == 1 and ORIGINAL not in before:
        return False
    if before.count(ORIGINAL) != 1 or CONDITIONAL in before:
        raise ValueError(
            f"Unrecognized Harbor OpenCode installer; refusing to patch {source}"
        )
    backup = source.with_name("opencode.py.before-curl-prerequisite-check")
    if not backup.exists():
        backup.write_text(before)
    after = before.replace(ORIGINAL, CONDITIONAL, 1)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=source.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(after)
            output.flush()
            os.fsync(output.fileno())
            os.fchmod(output.fileno(), stat.S_IMODE(source.stat().st_mode))
        temporary.replace(source)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "patch": "Harbor OpenCode conditional curl prerequisite",
                "source": str(source),
                "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
                "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
                "native_agent_and_NVM_npm_commands_unchanged": True,
            }
        )
    )
    return True


def main() -> None:
    harbor = distribution("harbor")
    if harbor.version != "0.6.6":
        raise ValueError(f"Expected pinned harbor==0.6.6; found {harbor.version}")
    source = Path(harbor.locate_file("harbor/agents/installed/opencode.py"))
    changed = patch_installer(source)
    print(f"Harbor native OpenCode curl prerequisite verified; changed={changed}")


if __name__ == "__main__":
    main()
