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
"""Privileged critic inputs for SWE agentic rollouts.

The critic sees the accepted fix and the grading tests; the policy never does.
This is the asymmetric actor-critic setup from sim2real robotics, adapted to a
terminal-reward, ~150-turn agentic workload.

Why this exists separately from :mod:`nemo_rl.algorithms.privileged_critic`
(which targets single-turn math RLVR and cannot be reused here):

  * it reads ``extra_env_info["ground_truth"]`` -- a key SWE rollouts do not
    have -- so it would silently inject an empty block;
  * it asserts single-turn and rejects interleaved multi-turn, which is every
    SWE rollout (~296 messages);
  * ``max_answer_tokens`` defaults to 256, sized for a math answer, not a patch.

Only :func:`nemo_rl.algorithms.privileged_critic.remap_by_response_mask` is
shared, and it is row-order preserving, so per-sample tensors stay aligned.

Placement is a hard correctness constraint, not a tuning knob. The value head is
causal, so ``V(s_t)`` attends only to tokens before ``t``. A SWE rollout
interleaves ~150 assistant turns across the whole context, so the ONLY placement
that reaches every supervised position is BEFORE the first assistant token.
Appending would be a silent no-op that looks like a negative result.

Unbiasedness is preserved: the policy cannot see the reference, so
``a_t ⊥ z | s_t`` and ``E[∇log π(a_t|s_t) · V(s_t, z)] = 0``. The privileged
batch must never reach the policy worker.

Field availability: the curriculum draws on FIVE source datasets, and the fix is
recoverable for 100% of them -- but not uniformly.

  * swe-bench-ext, SWE-rebench-V2, nv-internal-1, SWE-Gym carry a patch STRING,
    under three different key names, usually nested inside the ``instance_dict``
    JSON string rather than at the top level of ``metadata``.
  * R2E-Gym carries no patch string at all. It stores the commit structurally
    under ``parsed_commit_content``, which :func:`_resolve_r2e_gym` reassembles
    into a unified diff so the critic sees one schema everywhere.

An earlier audit covered only the first four and concluded "100%", which is why
the first privileged launch died with "no golden patch resolved for 48/512
rollouts". Re-measured directly over the 7394 collected rollout groups:
SWE-rebench-V2 4769, swe-bench-ext 1613, R2E-Gym 456, nv-internal-1 377,
SWE-Gym 179 -- gold now resolves for all 7394.

``rubric``/``requirements``/``interface`` exist for only 6.9% (and ``interface``
is already in the agent's prompt), so nothing here depends on them.
"""

import json
from typing import Any, Optional

import torch

from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

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


def resolve_config(master_config: Any) -> Optional[dict[str, Any]]:
    """The ``value.swe_privileged_critic`` block, or None when disabled/absent.

    Mirrors how ``value.privileged_critic`` is resolved so the two features read
    identically at every call site. Absence is the default: an existing config
    that has never heard of this key behaves exactly as before.
    """
    value_config = getattr(master_config, "value", None) or {}
    cfg = value_config.get(CONFIG_KEY)
    if cfg is None or not cfg.get("enabled"):
        return None
    other = value_config.get("privileged_critic")
    if other is not None and other.get("enabled"):
        raise ValueError(
            "value.swe_privileged_critic and value.privileged_critic are both "
            "enabled. They build different augmented critic layouts; enable "
            "exactly one. (privileged_critic targets single-turn math RLVR and "
            "cannot handle SWE rollouts anyway.)"
        )
    return cfg


def privilege_budget_tokens(cfg: dict[str, Any]) -> int:
    """Upper bound on the tokens the reference block can add to a sequence.

    Used at setup to raise the VALUE model's sequence budget, so the augmented
    sequences fit its packing bins. Bounded by construction: every field is
    capped, so this is a true worst case rather than an estimate.
    """
    total = int((cfg or {}).get("max_total_tokens") or DEFAULT_MAX_TOTAL_TOKENS)
    return total + 256  # + markup / chat-template slack


def build_turn_value_batch_augmented(
    critic_batch: BatchedDataDict,
    train_data: BatchedDataDict,
    turn_spans: Any,
) -> BatchedDataDict:
    """Turn-level anchor batch expressed in the AUGMENTED (privileged) layout.

    ``build_turn_value_batch`` builds the anchor batch from ``train_data``, i.e.
    the POLICY layout. With a privileged critic the input ids differ, so using it
    would train the critic on non-privileged sequences while its values were
    computed from privileged ones. This is the reason the older
    ``privileged_critic`` is hard-rejected in combination with ``turn_gae``.

    Both the anchor mask and the anchor-layout returns are carried across with
    :func:`remap_by_response_mask`, which is valid because anchors are a subset
    of the response tokens and the response tokens are preserved verbatim.

    Order matters: ``token_mask`` is the response mask that keys every remap, so
    it is replaced by the anchor mask only after the returns have been moved.
    """
    from nemo_rl.algorithms.privileged_critic import remap_by_response_mask

    resp_aug = critic_batch["token_mask"]
    resp_pol = train_data["token_mask"]

    critic_batch["returns"] = remap_by_response_mask(
        train_data["returns"], resp_pol, resp_aug
    )
    anchor_aug = remap_by_response_mask(
        turn_spans.anchor_mask.to(torch.float32), resp_pol, resp_aug
    )
    critic_batch["token_mask"] = anchor_aug.to(resp_aug.dtype)
    critic_batch["sample_mask"] = train_data["sample_mask"]
    return critic_batch


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
        for line in ((hunk.get("line_group") or {}).get("all_lines") or []):
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


def build_swe_privileged_value_inputs(
    repeated_batch: BatchedDataDict,
    tokenizer: Any,
    pcfg: dict[str, Any],
    make_seq_len_divisible_by: int = 1,
    metrics_out: Optional[dict[str, float]] = None,
) -> BatchedDataDict:
    """Critic input batch: reference block prefixed to each verbatim rollout.

    Row-aligned with ``train_data``. ``token_mask`` marks exactly the same
    response tokens, so :func:`remap_by_response_mask` can carry values back to
    the policy layout (it asserts equal per-row response counts, which is the
    construction check that the rollout was preserved verbatim).

    Unlike the math implementation this does NOT split prompt from response --
    SWE rollouts interleave ~150 assistant turns with tool output, and every one
    of those turns is a supervised position. The whole message log is kept
    untouched and only a prefix is added.
    """
    caps = {**DEFAULT_CAPS, **(pcfg.get("caps") or {})}
    max_total = int(pcfg.get("max_total_tokens") or DEFAULT_MAX_TOTAL_TOKENS)
    message_logs = repeated_batch["message_log"]
    env_infos = repeated_batch.get("extra_env_info", None)
    if env_infos is None:
        raise ValueError(
            "SWE privileged critic: extra_env_info is absent from the batch, so "
            "there is no privileged information to inject. The run would "
            "silently train a blind critic while labelled privileged."
        )

    # Cache by instance: a step has ~32 unique tasks but ~512 rollouts, and all
    # 16 siblings of a group MUST receive a byte-identical block (that is what
    # keeps the privilege constant within a group, so it cannot introduce a
    # within-task confound).
    block_cache: dict[str, torch.Tensor] = {}
    block_stats: dict[str, dict[str, Any]] = {}
    missing: list[int] = []
    critic_message_logs: list[list[dict[str, Any]]] = []

    for i, (msgs, info) in enumerate(zip(message_logs, env_infos)):
        fields = resolve_privilege_fields(info)
        if not fields["golden_patch"]:
            missing.append(i)
        key = fields["instance_id"] or f"__row{i}"
        if key not in block_cache:
            block, bstats = build_reference_block(fields, tokenizer, caps, max_total)
            block_stats[key] = bstats
            rendered = tokenizer.apply_chat_template(
                [{"role": "system", "content": block}],
                tokenize=False,
                add_generation_prompt=False,
                add_special_tokens=False,
            )
            block_cache[key] = tokenizer(
                rendered, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0].to(dtype=torch.long)
        prefix = block_cache[key]

        critic_msgs: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "",  # token_ids provided; content is unused by the flattener
                "token_ids": prefix,
                "token_loss_mask": torch.zeros_like(prefix),
            }
        ]
        for m in msgs:
            tid = torch.as_tensor(m["token_ids"], dtype=torch.long).flatten()
            # Preserve the caller's mask when it set one (both PPO loops and
            # critic pretraining unmask all assistant messages before this
            # point); fall back to the same role rule if not.
            if "token_loss_mask" in m:
                mask = torch.as_tensor(m["token_loss_mask"], dtype=tid.dtype).flatten()
            else:
                mask = (
                    torch.ones_like(tid)
                    if m["role"] == "assistant"
                    else torch.zeros_like(tid)
                )
            critic_msgs.append(
                {
                    "role": m["role"],
                    "content": "",
                    "token_ids": tid,
                    "token_loss_mask": mask,
                }
            )
        critic_message_logs.append(critic_msgs)

    # The fix resolves for 100% of the five source datasets, so any miss is a
    # data-path or new-dataset bug, not a straggler. Fail loudly rather than
    # degrade to a blind critic under a privileged label.
    if missing:
        raise ValueError(
            f"SWE privileged critic: no golden patch resolved for {len(missing)}/"
            f"{len(message_logs)} rollouts (rows {missing[:8]}...). Expected one of "
            f"{GOLD_KEYS} in extra_env_info metadata / its instance_dict, or an "
            f"R2E-Gym-style {R2E_COMMIT_KEY!r}. Either the metadata did not "
            "survive the data path, or the campaign mixes in a SIXTH dataset with "
            "yet another schema — check dataset_name on the failing rows."
        )

    if metrics_out is not None and block_stats:
        # Reported per INSTANCE (not per rollout): the block is byte-identical
        # across a group's 16 siblings, so rollout-weighting would just restate
        # the group size. frac_truncated is the honest measure of how much
        # privileged information the fixed budget is throwing away.
        s = list(block_stats.values())
        n = len(s)
        metrics_out["privilege/frac_truncated"] = sum(x["truncated"] for x in s) / n
        metrics_out["privilege/block_tokens_mean"] = (
            sum(x["kept_tokens"] for x in s) / n
        )
        metrics_out["privilege/block_tokens_max"] = max(x["kept_tokens"] for x in s)
        metrics_out["privilege/dropped_tokens_mean"] = (
            sum(x["dropped_tokens"] for x in s) / n
        )
        metrics_out["privilege/wanted_tokens_mean"] = (
            sum(x["wanted_tokens"] for x in s) / n
        )
        metrics_out["privilege/n_instances"] = float(n)
        for _f in ALLOCATION_PRIORITY:
            metrics_out[f"privilege/frac_truncated_{_f}"] = (
                sum(_f in x["truncated_fields"] for x in s) / n
            )

    flat, input_lengths = batched_message_log_to_flat_message(
        critic_message_logs,
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
        make_sequence_length_divisible_by=make_seq_len_divisible_by,
    )
    return BatchedDataDict(
        {
            "input_ids": flat["token_ids"],
            "input_lengths": input_lengths,
            "token_mask": flat["token_loss_mask"],
        }
    )
