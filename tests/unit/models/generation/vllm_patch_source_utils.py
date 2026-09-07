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


def patch_replacements(patch_fn_name: str) -> tuple[tuple[str, str], ...]:
    """Return all ``(old_snippet, new_snippet)`` pairs for a source patch.

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
    named_pairs: dict[str, dict[str, str]] = {}
    replacements = None
    for node in ast.walk(func):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue

        target_name = node.targets[0].id
        if target_name in ("old_snippet", "new_snippet"):
            snippets[target_name] = ast.literal_eval(node.value)
        elif target_name == "replacements":
            replacements = ast.literal_eval(node.value)
        elif target_name.endswith(("_old", "_new")):
            pair_name, side = target_name.rsplit("_", 1)
            named_pairs.setdefault(pair_name, {})[side] = ast.literal_eval(node.value)

    if replacements is not None:
        return tuple(replacements)

    if named_pairs:
        incomplete = [
            name for name, pair in named_pairs.items() if pair.keys() != {"old", "new"}
        ]
        if incomplete:
            raise AssertionError(
                f"{patch_fn_name} defines incomplete replacement pairs: {incomplete}"
            )
        return tuple((pair["old"], pair["new"]) for pair in named_pairs.values())

    missing = {"old_snippet", "new_snippet"} - snippets.keys()
    if missing:
        raise AssertionError(
            f"{patch_fn_name} no longer defines {sorted(missing)}; the test "
            "helper can no longer reverse its patch"
        )
    return ((snippets["old_snippet"], snippets["new_snippet"]),)


def patch_snippets(patch_fn_name: str) -> tuple[str, str]:
    """Return the single replacement pair used by a one-anchor patch."""
    replacements = patch_replacements(patch_fn_name)
    if len(replacements) != 1:
        raise AssertionError(
            f"{patch_fn_name} has {len(replacements)} replacement pairs; use "
            "patch_replacements()"
        )
    return replacements[0]


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
    replacements = patch_replacements(patch_fn_name)
    content = Path(patches._get_vllm_file(relative_source)).read_text()

    for old_snippet, new_snippet in reversed(replacements):
        # A deletion-only optional edit cannot be reconstructed from the
        # patched source. Leaving it deleted is sufficient to restore and
        # exercise every required anchor in the patch.
        if not new_snippet:
            continue
        if new_snippet in content:
            content = content.replace(new_snippet, old_snippet, 1)
        assert new_snippet not in content, (
            f"reversing {patch_fn_name} left its replacement behind in "
            f"{relative_source}; the patch may have been applied more than once"
        )
        assert old_snippet in content, (
            f"{relative_source} contains neither the patched nor the original "
            f"form of a {patch_fn_name} anchor; vLLM has probably changed upstream"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content)
    return destination
