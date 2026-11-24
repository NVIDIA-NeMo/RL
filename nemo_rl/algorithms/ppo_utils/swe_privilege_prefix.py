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
"""SWE reference block for the privileged value: the accepted fix and the grading tests.

:func:`build_privilege_prefix` is the prefix builder
(``value.privileged_value.prefix_builder_fqn``) for SWE agent rollouts. The block
holds the instance's golden patch, test patch, ``PASS_TO_PASS`` and
``FAIL_TO_PASS``, within per-field caps given as
``prefix_builder_kwargs.caps``.

The fix is read from a patch string under one of ``GOLD_KEYS``, or reassembled
from R2E-Gym's structured ``parsed_commit_content`` when no patch string is
present.
"""

import json
from typing import Any

from pydantic import BaseModel, PositiveInt

from nemo_rl.algorithms.ppo_utils.privileged_value import (
    OMITTED_MARKER,
    cap_tokens,
    count_tokens,
)
from nemo_rl.models.value.config import PrivilegedValueConfig

# The accepted fix, in priority order.
GOLD_KEYS: tuple[str, ...] = ("golden_patch", "gold_patch", "patch")

# R2E-Gym carries no patch string; its structured parsed_commit_content is reassembled
# into a unified diff so the critic sees one schema.
R2E_COMMIT_KEY = "parsed_commit_content"
# R2E-Gym states its acceptance criterion as expected_output_json; emitted in the
# fail_to_pass slot.
R2E_EXPECTED_KEY = "expected_output_json"
_DIFF_LINE_PREFIX = {"context": " ", "deleted": "-", "added": "+"}

# Emission order: least discriminative first, so fail_to_pass sits nearest the
# trajectory.
SECTION_ORDER: tuple[str, ...] = (
    "golden_patch",
    "test_patch",
    "pass_to_pass",
    "fail_to_pass",
)

# Allocation order, deliberately not the emission order: the least useful field
# loses budget first.
ALLOCATION_PRIORITY: tuple[str, ...] = (
    "fail_to_pass",
    "golden_patch",
    "test_patch",
    "pass_to_pass",
)


class SwePrivilegeCaps(BaseModel, extra="forbid"):
    """Per-field token ceilings within the reference block's total budget.

    They stop one oversized field from consuming the budget. They may sum to more
    than ``max_total_tokens``; the total stays the binding constraint.
    """

    golden_patch: PositiveInt
    test_patch: PositiveInt
    pass_to_pass: PositiveInt
    fail_to_pass: PositiveInt


class SwePrefixBuilderKwargs(BaseModel, extra="forbid"):
    """``value.privileged_value.prefix_builder_kwargs`` of the SWE builder."""

    caps: SwePrivilegeCaps


def _as_lines(value: Any) -> str:
    """Normalise a FAIL_TO_PASS / PASS_TO_PASS field to one entry per line.

    Sources store these as whitespace-separated strings or JSON lists, and some
    entries are themselves JSON-encoded lists. One matchable item per line
    makes the critic's job at token ``t`` ("how many reference items has the
    agent hit so far?") a per-item lookup.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("["):
            try:
                value = json.loads(s)
            except json.JSONDecodeError:
                return s
        else:
            return s
    if isinstance(value, (list, tuple)):
        return "\n".join(str(x) for x in value)
    return str(value)


def _is_test_path(path: str) -> bool:
    """Path heuristic for the fix/tests split, used only for a record without relevant_files.

    Deliberately does not treat bare ``test*.py`` names as tests.
    """
    low = path.lower()
    base = low.rsplit("/", 1)[-1]
    return (
        base.startswith("test_")
        or base.endswith("_test.py")
        or base.endswith("_test.go")
        or "/tests/" in f"/{low}"
        or "/test/" in f"/{low}"
    )


def _unified_diff_from_file_diff(fd: dict[str, Any]) -> str:
    """Rebuild one file's unified diff from R2E-Gym's structured hunks.

    Emits the ``diff --git`` / ``index`` / ``---`` / ``+++`` / ``@@`` shape a
    patch string has, so the critic sees a single patch format everywhere.
    ``modified_entities`` (whole function bodies, which dwarf the hunks) is
    deliberately dropped.
    """
    path = ((fd.get("header") or {}).get("file") or {}).get("path") or ""
    if not path:
        return ""
    minus = (fd.get("minus_file") or {}).get("path") or f"a/{path}"
    plus = (fd.get("plus_file") or {}).get("path") or f"b/{path}"
    out = [f"diff --git a/{path} b/{path}"]
    idx = fd.get("index_line") or {}
    if idx.get("old_commit_hash") and idx.get("new_commit_hash"):
        mode = f" {idx['mode']}" if idx.get("mode") else ""
        out.append(f"index {idx['old_commit_hash']}..{idx['new_commit_hash']}{mode}")
    if fd.get("is_binary_file"):
        out.append(fd.get("binary_line") or f"Binary files {minus} and {plus} differ")
        return "\n".join(out)
    out += [f"--- {minus}", f"+++ {plus}"]
    for hunk in fd.get("hunks") or []:
        d = hunk.get("descriptor") or {}
        o, n = d.get("old_range") or {}, d.get("new_range") or {}
        section = d.get("section") or ""
        out.append(
            f"@@ -{o.get('start', 0)},{o.get('length', 0)} "
            f"+{n.get('start', 0)},{n.get('length', 0)} @@"
            + (f" {section}" if section else "")
        )
        for line in (hunk.get("line_group") or {}).get("all_lines") or []:
            prefix = _DIFF_LINE_PREFIX.get(line.get("type"), " ")
            out.append(prefix + (line.get("content") or ""))
    return "\n".join(out)


def _resolve_r2e_gym(src: dict[str, Any]) -> dict[str, str]:
    """Privileged fields for an R2E-Gym instance (no patch string in metadata).

    Returns ``{}`` when this is not an R2E-Gym-shaped record, so the caller can
    keep failing loudly on a genuinely broken data path rather than papering
    over it with empty strings.
    """
    try:
        commit = json.loads(src.get(R2E_COMMIT_KEY) or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    file_diffs = commit.get("file_diffs") if isinstance(commit, dict) else None
    if not file_diffs:
        return {}

    # A file is part of the fix unless it looks like a test and relevant_files
    # does not name it; neither signal alone is right.
    relevant = src.get("relevant_files")
    relevant = set(relevant) if isinstance(relevant, list) else set()

    gold_parts, test_parts = [], []
    for fd in file_diffs:
        if not isinstance(fd, dict):
            continue
        text = _unified_diff_from_file_diff(fd)
        if not text:
            continue
        path = ((fd.get("header") or {}).get("file") or {}).get("path") or ""
        is_test = _is_test_path(path) and path not in relevant
        (test_parts if is_test else gold_parts).append(text)

    # "name: STATUS" per line, the one-item-per-line shape _as_lines() gives.
    expected = ""
    try:
        eo = json.loads(src.get(R2E_EXPECTED_KEY) or "{}")
        if isinstance(eo, dict):
            expected = "\n".join(f"{k}: {v}" for k, v in eo.items())
    except (json.JSONDecodeError, TypeError):
        expected = ""

    return {
        "golden_patch": "\n".join(gold_parts),
        "test_patch": "\n".join(test_parts),
        "fail_to_pass": expected,
        "pass_to_pass": "",
    }


def resolve_privilege_fields(env_info: dict[str, Any]) -> dict[str, str]:
    """Pull the privileged fields out of one rollout's ``extra_env_info``.

    Top-level ``metadata`` wins over ``instance_dict`` when both carry a key.
    Rows in the Gym sandboxed-agent format (e.g. ``swe_rebench``) carry the
    instance fields at the row's top level instead; those are the fallback.
    """
    env_info = env_info or {}
    md = (env_info.get("responses_create_params") or {}).get("metadata") or {}
    try:
        idict = json.loads(md.get("instance_dict") or "{}")
    except (json.JSONDecodeError, TypeError):
        idict = {}
    if not isinstance(idict, dict):
        idict = {}
    src: dict[str, Any] = {
        **{k: v for k, v in env_info.items() if v},
        **idict,
        **{k: v for k, v in md.items() if v},
    }

    gold = ""
    for key in GOLD_KEYS:
        v = src.get(key)
        if isinstance(v, str) and v.strip():
            gold = v
            break
    instance_id = str(src.get("instance_id") or md.get("instance_id") or "")

    # R2E-Gym stores the fix structurally; consulted only when no patch string
    # resolved.
    if not gold:
        r2e = _resolve_r2e_gym(src)
        if r2e.get("golden_patch"):
            return {"instance_id": instance_id, **r2e}

    test_patch = src.get("test_patch")
    return {
        "instance_id": instance_id,
        "golden_patch": gold,
        "test_patch": test_patch if isinstance(test_patch, str) else "",
        "fail_to_pass": _as_lines(src.get("FAIL_TO_PASS")),
        "pass_to_pass": _as_lines(src.get("PASS_TO_PASS")),
    }


def build_reference_block(
    fields: dict[str, str],
    tokenizer: Any,
    max_total_tokens: int,
    caps: SwePrivilegeCaps,
) -> tuple[str, dict[str, Any]]:
    """Assemble the reference block for one instance, within a fixed token budget.

    Fixed section order and fixed markup on every instance, so the critic trains
    on one byte-stable schema. Fields are emitted verbatim, with no instructions:
    a scalar value head does not follow them.

    Budget is spent in ALLOCATION_PRIORITY order, not emission order: what the
    budget cannot cover is dropped from the least useful field first. A truncated
    instance therefore still carries fail_to_pass and as much golden_patch as
    fits, and loses pass_to_pass.

    Returns ``(block_text, stats)``; see
    ``nemo_rl.algorithms.ppo_utils.privileged_value`` for the stats keys.
    """
    remaining = max_total_tokens
    kept: dict[str, str] = {}
    stats: dict[str, Any] = {
        "truncated_fields": [],
        "wanted_tokens": 0,
        "kept_tokens": 0,
    }

    for name in ALLOCATION_PRIORITY:
        raw = fields.get(name, "") or ""
        if not raw:
            continue
        cap = getattr(caps, name)
        want = count_tokens(raw, tokenizer, cap)
        stats["wanted_tokens"] += want
        budget = min(cap, max(remaining, 0))
        body, truncated = (
            cap_tokens(raw, budget, tokenizer) if budget > 0 else ("", True)
        )
        got = count_tokens(body, tokenizer, budget) if body else 0
        if truncated:
            stats["truncated_fields"].append(name)
        kept[name] = body if body else OMITTED_MARKER
        stats["kept_tokens"] += got
        remaining -= got

    parts = ["<reference>"]
    for name in SECTION_ORDER:
        body = kept.get(name, "")
        if body:
            parts.append(f"<{name}>\n{body}\n</{name}>")
    if stats["truncated_fields"]:
        # Block-level note so the critic can tell a complete reference from a
        # partial one, instead of inferring it from a missing section.
        parts.append(
            "[truncated: " + ", ".join(sorted(stats["truncated_fields"])) + "]"
        )
    parts.append("</reference>")
    return "\n".join(parts), stats


def build_privilege_prefix(
    extra_env_info: dict[str, Any],
    tokenizer: Any,
    cfg: PrivilegedValueConfig,
) -> tuple[str, dict[str, Any]]:
    """Prefix builder for SWE rollouts: the instance's reference block and its stats.

    Raises:
        ValueError: No golden patch resolves for the instance. Training on would
            silently produce a blind critic under a privileged label.
    """
    fields = resolve_privilege_fields(extra_env_info)
    if not fields["golden_patch"]:
        raise ValueError(
            "SWE privilege prefix: no golden patch resolved "
            f"(instance_id={fields['instance_id']!r}). Expected one of {GOLD_KEYS} "
            "in extra_env_info's responses_create_params.metadata / its "
            "instance_dict / the row's top level, or an R2E-Gym-style "
            f"{R2E_COMMIT_KEY!r}."
        )
    kwargs = SwePrefixBuilderKwargs.model_validate(cfg.prefix_builder_kwargs)
    return build_reference_block(fields, tokenizer, cfg.max_total_tokens, kwargs.caps)
