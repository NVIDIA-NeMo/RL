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
"""Privileged critic inputs for SWE agentic rollouts (SingleController path).

The critic sees the accepted fix and the grading tests; the policy never does.
This is the asymmetric actor-critic setup from sim2real robotics, adapted to a
terminal-reward, ~150-turn agentic workload. Ported from the legacy async-PPO
implementation; the reference-block format is byte-identical so critics
pretrained there warm-start here.

Placement is a hard correctness constraint, not a tuning knob. The value head is
causal, so ``V(s_t)`` attends only to tokens before ``t``. A SWE rollout
interleaves ~150 assistant turns across the whole context, so the ONLY placement
that reaches every supervised position is BEFORE the first assistant token.

Because the block is a pure prefix, the critic sequence of a row is exactly
``prefix + input_ids[:input_length]`` and its response mask is the policy's
shifted by ``len(prefix)``. The SingleController therefore stores nothing extra
in the data plane: :class:`SwePrivilegePrefixStore` caches one tokenized prefix
per instance on the controller and stamps its key and length on each row's tag,
and the value workers splice it in after fetch (:func:`prepend_privilege_prefix`)
and slice the values back out (:func:`values_to_policy_layout`).

Unbiasedness is preserved: the policy cannot see the reference, so
``a_t ⊥ z | s_t`` and ``E[∇log π(a_t|s_t) · V(s_t, z)] = 0``. The privileged
inputs must never reach the policy worker.

Field availability: the curriculum draws on FIVE source datasets, and the fix is
recoverable for 100% of them -- but not uniformly.

  * swe-bench-ext, SWE-rebench-V2, nv-internal-1, SWE-Gym carry a patch STRING,
    under three different key names, usually nested inside the ``instance_dict``
    JSON string rather than at the top level of ``metadata``.
  * R2E-Gym carries no patch string at all. It stores the commit structurally
    under ``parsed_commit_content``, which :func:`_resolve_r2e_gym` reassembles
    into a unified diff so the critic sees one schema everywhere.
"""

import asyncio
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import torch
from pydantic import BaseModel, Field

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

if TYPE_CHECKING:
    from nemo_rl.experience.interfaces import PromptGroupRecord

# The accepted fix, in priority order. swe-bench-ext and SWE-rebench use
# ``patch``, nv-internal-1 uses ``gold_patch``, SWE-Gym carries ``golden_patch``
# alongside ``patch``.
GOLD_KEYS: tuple[str, ...] = ("golden_patch", "gold_patch", "patch")

# R2E-Gym is the fifth dataset in this curriculum and the one exception to
# "every instance carries a patch string": it stores NO unified diff at all.
# 456 of the 7394 collected groups (6.2%) are R2E-Gym, and the original audit
# missed them, which is what made the first privileged launch die with
# "no golden patch resolved for 48/512 rollouts".
#
# The reference fix IS present, just structured rather than textual, under
# ``parsed_commit_content`` -- a JSON blob of per-file hunks that we reassemble
# into a real unified diff below. Two other sources were considered and
# rejected:
#   * the ``prompt`` field embeds a ```diff block, but measured over all 456
#     instances it covers only PART of the non-test files in 38% of them (it is
#     the issue-writer's prompt, not the patch of record) -- so it silently
#     under-reports the fix;
#   * ``old_file_content``/``new_file_content`` are whole files (median ~190 KB),
#     which would blow the token budget on unchanged lines.
# Reassembling the hunks gives 100% coverage AND keeps the critic on ONE schema
# across all five datasets, which is the thing that actually has to be learnable.
R2E_COMMIT_KEY = "parsed_commit_content"
# R2E-Gym leaves FAIL_TO_PASS/PASS_TO_PASS empty (0/456) and states its
# acceptance criterion as ``expected_output_json``: test name -> expected status
# (PASSED / ERROR / FAILED). That is the same role FAIL_TO_PASS plays for the
# other four datasets, so it is emitted in that slot.
R2E_EXPECTED_KEY = "expected_output_json"
_DIFF_LINE_PREFIX = {"context": " ", "deleted": "-", "added": "+"}

# Emitted in this order. Least discriminative first: the block sits ~100k tokens
# before the late values that need it, and this model is a 52-layer hybrid with
# only 6 attention layers (MEMEM*EMEMEM*...), so mamba state retains RECENT
# context best. FAIL_TO_PASS is the most compact and most discriminative field,
# so it goes last -- nearest the trajectory.
SECTION_ORDER: tuple[str, ...] = (
    "golden_patch",
    "test_patch",
    "pass_to_pass",
    "fail_to_pass",
)

# Single TOTAL budget for the reference block. Fixed by construction, so the
# value model's sequence budget is exactly policy_len + this + slack -- no
# dependence on a per-field cap sum staying in sync with the seqlen bump.
#
# Measured over 400 random instances (real tokenizer): the untruncated block is
# median 5467 / p90 17683 / p99 77911 / max 326605 tokens, so 32768 truncates
# ~4.8% of instances. Watch privilege/frac_truncated: it is the exact measure of
# how much privileged information the budget is discarding.
DEFAULT_MAX_TOTAL_TOKENS = 32768

# Per-field ceilings, applied WITHIN the total budget so one pathological field
# cannot consume it (one corpus instance carries a 323k-token golden patch, and
# pass_to_pass reaches 125k).
# Deliberately sum to MORE than the total budget, so the single total is the
# binding constraint and these only stop one pathological field from eating it.
DEFAULT_CAPS: dict[str, int] = {
    "golden_patch": 24576,
    "test_patch": 16384,
    "pass_to_pass": 4096,
    "fail_to_pass": 4096,
}

# Budget is allocated in THIS order, which is deliberately not the emission
# order. fail_to_pass is tiny (median 56 tokens) and states the acceptance
# criterion; golden_patch says which files should change, the signal that is
# 93% unknown at t=0; test_patch is expensive and aimed at the late region the
# blind critic already handles; pass_to_pass is regression noise with a brutal
# tail (p90 9798). Emission order stays least->most discriminative for recency.
ALLOCATION_PRIORITY: tuple[str, ...] = (
    "fail_to_pass",
    "golden_patch",
    "test_patch",
    "pass_to_pass",
)

TRUNCATION_MARKER = "\n... [truncated]"
# A field that had content but got no budget is marked rather than silently
# omitted: the critic trains on a fixed schema, so "this exists but was cut" and
# "this instance has none" must not look identical.
OMITTED_MARKER = "... [omitted: token budget]"


CONFIG_KEY = "swe_privileged_critic"


def _as_lines(value: Any) -> str:
    """Normalise a FAIL_TO_PASS / PASS_TO_PASS field to one entry per line.

    nv-internal-1 stores these as whitespace-separated strings, the other three
    datasets as JSON lists, and some entries are themselves JSON-encoded lists.
    One matchable item per line is what makes the critic's job at token ``t``
    ("how many reference items has the agent hit so far?") a per-item lookup.
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
    """Path heuristic for the fix/tests split -- the FALLBACK only.

    ``relevant_files`` (see :func:`_resolve_r2e_gym`) states the split
    authoritatively and covers 100% of this corpus, so this only runs on a
    record that lacks it. Deliberately conservative: an earlier version also
    treated ``test*.py`` as a test file and thereby swallowed pandas'
    ``pandas/util/testing.py`` -- a source module -- leaving that instance with
    an empty golden patch.
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

    Emits exactly the ``diff --git`` / ``index`` / ``---`` / ``+++`` / ``@@``
    shape the other four datasets supply verbatim, so the critic sees a single
    patch format everywhere. ``modified_entities`` (whole function bodies, which
    dwarf the hunks) is deliberately dropped.
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

    # Gold/test split from BOTH available signals, because neither alone is
    # right. ``relevant_files`` is R2E-Gym's "primary" file(s) and is narrower
    # than the fix -- on one Pillow instance it names ImagePalette.py while the
    # commit also fixes Image.py, and R2E-Gym's OWN diff rendering includes
    # both. The path heuristic is broader but misfires on source modules that
    # merely look test-shaped (pandas/util/testing.py). Union of the two: a file
    # is part of the fix unless it looks like a test AND R2E-Gym did not call it
    # relevant. Cross-checked against R2E-Gym's own rendering over all 456
    # instances (see the docstring's note on _unified_diff_from_file_diff).
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

    # expected_output_json is the acceptance criterion; "name: STATUS" per line
    # matches the one-item-per-line shape _as_lines() gives the other datasets.
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

    The shards already carry the full instance metadata per rollout under
    ``extra_env_info[i]["responses_create_params"]["metadata"]``, so nothing has
    to be re-joined against the source JSONL at critic-build time.

    Top-level ``metadata`` wins over ``instance_dict`` when both carry a key.
    """
    md = (env_info or {}).get("responses_create_params", {}).get("metadata", {})
    try:
        idict = json.loads(md.get("instance_dict") or "{}")
    except (json.JSONDecodeError, TypeError):
        idict = {}
    if not isinstance(idict, dict):
        idict = {}
    src: dict[str, Any] = {**idict, **{k: v for k, v in md.items() if v}}

    gold = ""
    for key in GOLD_KEYS:
        v = src.get(key)
        if isinstance(v, str) and v.strip():
            gold = v
            break
    instance_id = str(src.get("instance_id") or md.get("instance_id") or "")

    # R2E-Gym: no patch string anywhere, but the commit is present structurally.
    # Only consulted when the textual keys came up empty, so the other four
    # datasets take exactly the path they always did.
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


def _cap_tokens(text: str, max_tokens: int, tokenizer: Any) -> str:
    """Truncate ``text`` to ``max_tokens``, deterministically.

    Truncation must be a pure function of the instance and never of the rollout,
    or sibling rollouts in a group would receive different reference blocks and
    the privilege signal would vary WITHIN a task -- manufacturing exactly the
    within-group length confound that already cripples the blind critic
    (Spearman(value, length) = -0.82).
    """
    if not text:
        return ""
    # Cheap char pre-cut so a 4MB patch is never fully tokenized. 20 chars/token
    # is far above any observed ratio (diffs measure 4-9), so this cannot cut
    # anything the token cap would have kept.
    ids = tokenizer.encode(text[: max_tokens * 20], add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text[: max_tokens * 20]
    # Reserve room for the marker so the RESULT respects max_tokens; otherwise
    # every truncated field overshoots the total budget by the marker length.
    marker_len = len(tokenizer.encode(TRUNCATION_MARKER, add_special_tokens=False))
    keep = max(max_tokens - marker_len, 0)
    return tokenizer.decode(ids[:keep]) + TRUNCATION_MARKER


def _count_tokens(text: str, tokenizer: Any, hint: int) -> int:
    """Token count, with a char pre-cut so a multi-MB field is never fully encoded."""
    if not text:
        return 0
    return len(tokenizer.encode(text[: max(hint, 1) * 20], add_special_tokens=False))


def build_reference_block(
    fields: dict[str, str],
    tokenizer: Any,
    caps: Optional[dict[str, int]] = None,
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
) -> tuple[str, dict[str, Any]]:
    """Assemble the reference block for one instance, within a FIXED token budget.

    Fixed section order and fixed markup on every instance: the critic trains on
    this format for thousands of steps, so a learnable, byte-stable schema
    matters far more than prose. Deliberately carries no instructions or
    roleplay -- a scalar value head does not follow them.

    Fields are emitted VERBATIM (v1). Diff compression -- stripping index lines,
    hunk headers and context -- is a deliberate follow-up, kept out of the first
    experiment so it cannot confound the privileged-vs-blind comparison.

    Budget is spent in ALLOCATION_PRIORITY order, NOT emission order: what the
    budget cannot cover is dropped from the least useful field first. A truncated
    instance therefore still carries fail_to_pass and as much golden_patch as
    fits, and loses pass_to_pass -- rather than losing the acceptance criterion
    because it happened to be emitted last.

    Returns ``(block_text, stats)``; stats feed the ``privilege/*`` metrics so
    the information being discarded is measured rather than assumed.
    """
    caps = {**DEFAULT_CAPS, **(caps or {})}
    remaining = int(max_total_tokens)
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
        want = _count_tokens(raw, tokenizer, caps[name])
        stats["wanted_tokens"] += want
        budget = min(caps[name], max(remaining, 0))
        body = _cap_tokens(raw, budget, tokenizer) if budget > 0 else ""
        got = _count_tokens(body, tokenizer, budget) if body else 0
        if got < want:
            stats["truncated_fields"].append(name)
        kept[name] = body if body else OMITTED_MARKER
        stats["kept_tokens"] += got
        remaining -= got

    parts = ["<reference>"]
    for name in SECTION_ORDER:  # emission order stays least->most discriminative
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
    stats["truncated"] = bool(stats["truncated_fields"])
    stats["dropped_tokens"] = max(stats["wanted_tokens"] - stats["kept_tokens"], 0)
    return "\n".join(parts), stats


# ── SingleController integration ─────────────────────────────────────────────

# Row tags stamped at commit: which cached prefix a row uses, and its length.
PRIVILEGE_KEY_TAG = "swe_privilege_key"
PRIVILEGE_PREFIX_LEN_TAG = "swe_privilege_prefix_len"
# Markup / chat-template slack on top of the block's token budget.
_TEMPLATE_SLACK_TOKENS = 256


class SwePrivilegedCriticConfig(BaseModel, extra="forbid"):
    """``value.swe_privileged_critic``: prefix the critic input with a reference block.

    The block carries the accepted fix and the grading tests of the rollout's
    SWE instance. Keep ``max_total_tokens`` and ``caps`` identical to the
    critic-pretrain run a critic is warm-started from: they change which content
    is truncated, a distribution shift the warm start was not trained on.
    """

    # Build privileged critic inputs. False leaves the critic blind.
    enabled: bool = False
    # Token budget for the whole reference block; fields are truncated to fit
    # in ALLOCATION_PRIORITY order.
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS
    # Per-field ceilings within the budget. Keys omitted here keep DEFAULT_CAPS.
    caps: dict[str, int] = Field(default_factory=dict)


def resolve_config(
    value_config: Optional[Mapping[str, Any]],
) -> Optional[SwePrivilegedCriticConfig]:
    """The parsed ``value.swe_privileged_critic`` block, or None when disabled/absent."""
    if value_config is None or CONFIG_KEY not in value_config:
        return None
    cfg = SwePrivilegedCriticConfig.model_validate(value_config[CONFIG_KEY])
    return cfg if cfg.enabled else None


def privilege_budget_tokens(cfg: SwePrivilegedCriticConfig) -> int:
    """Upper bound on the tokens the reference block adds to a critic sequence."""
    return cfg.max_total_tokens + _TEMPLATE_SLACK_TOKENS


def render_reference_prefix(
    fields: dict[str, str],
    tokenizer: Any,
    cfg: SwePrivilegedCriticConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Tokenize one instance's reference block as a system-role chat turn.

    Returns the prefix token ids (int32, 1-D) and the block's truncation stats.
    """
    block, stats = build_reference_block(
        fields, tokenizer, cfg.caps, cfg.max_total_tokens
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": block}],
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]
    return ids.to(dtype=torch.int32), stats


def _group_id_of(sample_id: str) -> str:
    return sample_id.rsplit("_g", 1)[0]


class SwePrivilegePrefixStore:
    """Controller-side cache of one tokenized reference block per SWE instance.

    Installed as the replay buffer's post-write enricher, so every group is
    stamped before it becomes selectable. Sibling rollouts of an instance share
    one byte-identical prefix: the privilege is constant within a task, so it
    cannot introduce a within-group confound. A prefix is dropped once no
    committed-but-untrained group references it.
    """

    def __init__(self, tokenizer: Any, cfg: SwePrivilegedCriticConfig) -> None:
        self._tokenizer = tokenizer
        self._cfg = cfg
        self._prefixes: dict[str, torch.Tensor] = {}
        self._stats: dict[str, dict[str, Any]] = {}
        self._group_keys: dict[str, str] = {}
        self._refcounts: dict[str, int] = {}

    async def enrich(
        self, meta: KVBatchMeta, record: "PromptGroupRecord"
    ) -> KVBatchMeta:
        """Resolve the group's instance, cache its prefix, and stamp every row tag.

        Raises:
            ValueError: No golden patch resolves for the instance. Training on
                would silently produce a blind critic under a privileged label.
        """
        fields = resolve_privilege_fields(record.extra_env_info or {})
        if not fields["golden_patch"]:
            raise ValueError(
                "SWE privileged critic: no golden patch resolved for prompt "
                f"{record.prompt_idx} (instance_id={fields['instance_id']!r}). "
                f"Expected one of {GOLD_KEYS} in extra_env_info's "
                "responses_create_params.metadata / its instance_dict, or an "
                f"R2E-Gym-style {R2E_COMMIT_KEY!r}."
            )
        key = fields["instance_id"] or f"__prompt{record.prompt_idx}"
        if key not in self._prefixes:
            prefix, stats = await asyncio.to_thread(
                render_reference_prefix, fields, self._tokenizer, self._cfg
            )
            self._prefixes[key] = prefix
            self._stats[key] = stats
        group_id = _group_id_of(meta.sample_ids[0])
        if group_id not in self._group_keys:
            self._group_keys[group_id] = key
            self._refcounts[key] = self._refcounts.get(key, 0) + 1
        prefix_len = int(self._prefixes[key].numel())
        tags = [
            {**tag, PRIVILEGE_KEY_TAG: key, PRIVILEGE_PREFIX_LEN_TAG: prefix_len}
            for tag in (meta.tags or [{} for _ in meta.sample_ids])
        ]
        return replace(meta, tags=tags)

    def prefixes_for(self, meta: KVBatchMeta) -> dict[str, torch.Tensor]:
        """The prefixes the rows of ``meta`` reference, keyed as in their tags."""
        return {key: self._prefixes[key] for key in _row_keys(meta)}

    def step_metrics(self, metas: Sequence[KVBatchMeta]) -> dict[str, float]:
        """``privilege/*`` metrics over the unique instances a step trained on.

        Reported per instance, not per rollout: the block is byte-identical
        across a group's siblings, so rollout-weighting would restate group size.
        """
        keys = dict.fromkeys(key for meta in metas for key in _row_keys(meta))
        stats = [self._stats[key] for key in keys]
        if not stats:
            return {}
        n = len(stats)
        metrics = {
            "privilege/frac_truncated": sum(s["truncated"] for s in stats) / n,
            "privilege/block_tokens_mean": sum(s["kept_tokens"] for s in stats) / n,
            "privilege/block_tokens_max": float(max(s["kept_tokens"] for s in stats)),
            "privilege/dropped_tokens_mean": sum(s["dropped_tokens"] for s in stats)
            / n,
            "privilege/wanted_tokens_mean": sum(s["wanted_tokens"] for s in stats) / n,
            "privilege/n_instances": float(n),
        }
        for name in ALLOCATION_PRIORITY:
            metrics[f"privilege/frac_truncated_{name}"] = (
                sum(name in s["truncated_fields"] for s in stats) / n
            )
        return metrics

    def release(self, metas: Sequence[KVBatchMeta]) -> None:
        """Drop the prefixes that no committed-but-untrained group still needs."""
        sample_ids = (sample_id for meta in metas for sample_id in meta.sample_ids)
        for group_id in dict.fromkeys(_group_id_of(s) for s in sample_ids):
            key = self._group_keys.pop(group_id, None)
            if key is None:
                continue
            self._refcounts[key] -= 1
            if self._refcounts[key] == 0:
                del self._refcounts[key]
                del self._prefixes[key]
                del self._stats[key]

    def state_dict(self) -> dict[str, Any]:
        """The cached prefixes and their stats, saved next to the replay buffer."""
        return {
            "prefixes": dict(self._prefixes),
            "stats": {key: dict(stats) for key, stats in self._stats.items()},
        }

    def restore(self, state: dict[str, Any], metas: Sequence[KVBatchMeta]) -> None:
        """Re-cache the prefixes the restored replay groups reference.

        References are rebuilt from the restored groups' row tags rather than
        saved: whatever the checkpoint caught mid-release, every restored group
        holds exactly one reference and nothing else does.

        Raises:
            ValueError: A restored group references a prefix the checkpoint lacks.
        """
        self._prefixes.clear()
        self._stats.clear()
        self._group_keys.clear()
        self._refcounts.clear()
        for meta in metas:
            keys = set(_row_keys(meta))
            if len(keys) != 1:
                raise ValueError(
                    "SWE privileged critic: a restored group references "
                    f"{len(keys)} prefixes; expected one per group."
                )
            (key,) = keys
            if key not in state["prefixes"]:
                raise ValueError(
                    f"SWE privileged critic: restored group references prefix {key!r}, "
                    "which the checkpoint does not contain."
                )
            self._prefixes[key] = state["prefixes"][key]
            self._stats[key] = state["stats"][key]
            group_id = _group_id_of(meta.sample_ids[0])
            if group_id not in self._group_keys:
                self._group_keys[group_id] = key
                self._refcounts[key] = self._refcounts.get(key, 0) + 1

    def __len__(self) -> int:
        return len(self._prefixes)


def _row_keys(meta: KVBatchMeta) -> list[str]:
    if not meta.tags or any(PRIVILEGE_KEY_TAG not in tag for tag in meta.tags):
        raise ValueError(
            "SWE privileged critic: batch rows carry no privilege tag. Every "
            "group must pass through SwePrivilegePrefixStore.enrich at commit."
        )
    return [tag[PRIVILEGE_KEY_TAG] for tag in meta.tags]


def _prefix_lengths(meta: KVBatchMeta) -> list[int]:
    _row_keys(meta)
    return [int(tag[PRIVILEGE_PREFIX_LEN_TAG]) for tag in meta.tags]


def critic_view_meta(meta: KVBatchMeta) -> KVBatchMeta:
    """``meta`` with each row's length extended by its prefix (critic layout)."""
    lengths = [
        length + prefix_len
        for length, prefix_len in zip(meta.sequence_lengths, _prefix_lengths(meta))
    ]
    return replace(meta, sequence_lengths=lengths)


def policy_view_meta(meta: KVBatchMeta) -> KVBatchMeta:
    """Inverse of :func:`critic_view_meta`: lengths back in the policy layout."""
    lengths = [
        length - prefix_len
        for length, prefix_len in zip(meta.sequence_lengths, _prefix_lengths(meta))
    ]
    return replace(meta, sequence_lengths=lengths)


def prepend_privilege_prefix(
    data: BatchedDataDict[Any],
    meta: KVBatchMeta,
    prefixes: Mapping[str, torch.Tensor],
    shift_fields: tuple[str, ...] = (),
) -> BatchedDataDict[Any]:
    """Turn a fetched policy-layout batch into the critic layout, in place.

    Each row becomes ``prefix + input_ids[:L]``. ``token_mask`` and every field in
    ``shift_fields`` (per-token, e.g. returns / old values) are shifted right by
    the prefix length and zero over the prefix, so the value loss and clipping
    see exactly the policy's response positions. The fetched tensors must already
    be padded to at least the longest critic row: the driver mints the forward
    pad target from :func:`critic_view_meta` lengths.
    """
    keys = _row_keys(meta)
    input_ids = data["input_ids"]
    lengths = data["input_lengths"]
    width = input_ids.shape[1]
    new_ids = input_ids.clone()
    new_token_mask = torch.zeros_like(data["token_mask"])
    shifted = {name: torch.zeros_like(data[name]) for name in shift_fields}
    for row, key in enumerate(keys):
        prefix = prefixes[key].to(device=input_ids.device, dtype=input_ids.dtype)
        p, n = int(prefix.numel()), int(lengths[row])
        if p + n > width:
            raise ValueError(
                f"privileged critic row {row} needs {p + n} tokens but the batch "
                f"is padded to {width}; the forward pad target was not minted "
                "from critic lengths."
            )
        new_ids[row, p : p + n] = input_ids[row, :n]
        new_ids[row, :p] = prefix
        new_token_mask[row, p : p + n] = data["token_mask"][row, :n]
        for name in shift_fields:
            shifted[name][row, p : p + n] = data[name][row, :n]
    data["input_ids"] = new_ids
    data["token_mask"] = new_token_mask
    data["input_lengths"] = lengths + torch.as_tensor(
        _prefix_lengths(meta), dtype=lengths.dtype, device=lengths.device
    )
    for name, tensor in shifted.items():
        data[name] = tensor
    return data


def values_to_policy_layout(
    values: torch.Tensor,
    meta: KVBatchMeta,
    policy_lengths: torch.Tensor,
) -> torch.Tensor:
    """Slice critic-layout values back onto the policy's token positions."""
    out = torch.zeros_like(values)
    for row, p in enumerate(_prefix_lengths(meta)):
        n = int(policy_lengths[row])
        out[row, :n] = values[row, p : p + n]
    return out
