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

"""Helpers for tests that exercise NeMo-RL's vLLM source patches.

``_apply_vllm_patches`` rewrites the *installed* vLLM in site-packages, in
place, the first time a generation worker is constructed. Any test that runs
after one of those in the same session therefore reads an already-patched file
-- so a fixture that copies the installed source and calls it "pristine" is
silently wrong, and the negative-control tests built on it stop testing
anything. Reversing the patch here makes those fixtures order-independent.
"""

import ast
from pathlib import Path

from nemo_rl.models.generation.vllm import patches


def patch_snippets(patch_fn_name: str) -> tuple[str, str]:
    """Return ``(old_snippet, new_snippet)`` for a patch function in patches.py.

    Read out of the source with ``ast`` rather than duplicated here, so the
    snippets cannot drift from the patch they are meant to reverse.
    """
    tree = ast.parse(Path(patches.__file__).read_text())
    try:
        func = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == patch_fn_name
        )
    except StopIteration:
        raise AssertionError(
            f"{patch_fn_name} not found in patches.py; the test helper needs updating"
        ) from None

    snippets = {}
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("old_snippet", "new_snippet")
        ):
            snippets[node.targets[0].id] = ast.literal_eval(node.value)

    missing = {"old_snippet", "new_snippet"} - snippets.keys()
    if missing:
        raise AssertionError(
            f"{patch_fn_name} no longer defines {sorted(missing)}; the test "
            "helper can no longer reverse its patch"
        )
    return snippets["old_snippet"], snippets["new_snippet"]


def write_unpatched_copy(
    relative_source: str, patch_fn_name: str, destination: Path
) -> Path:
    """Copy an installed vLLM file to `destination` with its patch reversed.

    Args:
        relative_source: Path under the vLLM package, as passed to
            ``patches._get_vllm_file``.
        patch_fn_name: Name of the patch function in ``patches.py`` whose edit
            should be undone.
        destination: File to write.

    Returns:
        ``destination``.
    """
    old_snippet, new_snippet = patch_snippets(patch_fn_name)
    content = Path(patches._get_vllm_file(relative_source)).read_text()

    if new_snippet in content:
        content = content.replace(new_snippet, old_snippet, 1)
    assert new_snippet not in content, (
        f"reversing {patch_fn_name} left its replacement behind in "
        f"{relative_source}; the patch may have been applied more than once"
    )
    assert old_snippet in content, (
        f"{relative_source} contains neither the patched nor the original form "
        f"of the {patch_fn_name} anchor; vLLM has probably changed upstream"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content)
    return destination
