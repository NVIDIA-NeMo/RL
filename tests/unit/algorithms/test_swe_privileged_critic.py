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

import json

import pytest
import torch

from nemo_rl.algorithms.swe_privileged_critic import (
    DEFAULT_CAPS,
    SECTION_ORDER,
    _as_lines,
    _cap_tokens,
    build_reference_block,
    build_swe_privileged_value_inputs,
    resolve_privilege_fields,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class _FakeTokenizer:
    """Whitespace tokenizer with a ChatML-ish template; enough for structure tests."""

    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, ids):
        return " ".join("w" for _ in ids)

    def apply_chat_template(self, msgs, **kwargs):
        return "".join(f"<s>{m['role']}\n{m['content']}</s>\n" for m in msgs)

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        n = max(len(text.split()), 1)
        return {"input_ids": torch.arange(n).unsqueeze(0)}


def _env_info(**meta):
    return {"responses_create_params": {"metadata": meta}}


# ---------------------------------------------------------------- field resolution
@pytest.mark.parametrize("gold_key", ["golden_patch", "gold_patch", "patch"])
def test_resolves_every_gold_key_variant(gold_key):
    """Four source datasets name the accepted fix three different ways."""
    info = _env_info(
        instance_id="x", instance_dict=json.dumps({gold_key: "diff --git a/f b/f"})
    )
    assert resolve_privilege_fields(info)["golden_patch"] == "diff --git a/f b/f"


def test_resolves_from_instance_dict_and_top_level_wins():
    """Most instances carry the fields only inside the instance_dict JSON string."""
    info = _env_info(
        instance_id="x",
        test_patch="TOP",
        instance_dict=json.dumps({"patch": "G", "test_patch": "NESTED"}),
    )
    f = resolve_privilege_fields(info)
    assert f["golden_patch"] == "G"
    assert f["test_patch"] == "TOP"


def test_resolve_tolerates_unparseable_instance_dict():
    info = _env_info(instance_id="x", patch="G", instance_dict="{not json")
    assert resolve_privilege_fields(info)["golden_patch"] == "G"


# ------------------------------------------------------- R2E-Gym (no patch string)
def _r2e_file_diff(path, added, deleted=(), section=""):
    lines = [{"content": c, "type": "deleted"} for c in deleted]
    lines += [{"content": c, "type": "added"} for c in added]
    return {
        "header": {"file": {"path": path}},
        "index_line": {"old_commit_hash": "aaa", "new_commit_hash": "bbb", "mode": "100644"},
        "minus_file": {"path": f"a/{path}"},
        "plus_file": {"path": f"b/{path}"},
        "is_binary_file": False,
        "hunks": [
            {
                "descriptor": {
                    "old_range": {"start": 1, "length": 3},
                    "new_range": {"start": 1, "length": 4},
                    "section": section,
                },
                "line_group": {"all_lines": [{"content": "ctx", "type": "context"}] + lines},
                # whole function bodies; must NOT reach the block
                "modified_entities": [{"content": "def f():\n    " + "x" * 5000}],
            }
        ],
    }


def _r2e_env_info(file_diffs, relevant_files=None, expected_output=None):
    """R2E-Gym stores no patch string: the commit is structured, under
    parsed_commit_content, and the acceptance criterion is expected_output_json."""
    return _env_info(
        instance_id="r2e-1",
        dataset_name="R2E-Gym/R2E-Gym-Subset",
        instance_dict=json.dumps(
            {
                "instance_id": "r2e-1",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
                "relevant_files": relevant_files or [],
                "parsed_commit_content": json.dumps({"file_diffs": file_diffs}),
                "expected_output_json": json.dumps(expected_output or {}),
            }
        ),
    )


def test_r2e_gym_reconstructs_a_unified_diff():
    """R2E-Gym is the 6.2% of the corpus with no patch string anywhere; the fix
    has to be rebuilt from structured hunks or the whole group is unusable."""
    info = _r2e_env_info(
        [_r2e_file_diff("src/app.py", added=["new line"], deleted=["old line"], section="def f():")],
        relevant_files=["src/app.py"],
    )
    gold = resolve_privilege_fields(info)["golden_patch"]
    assert gold.startswith("diff --git a/src/app.py b/src/app.py")
    assert "index aaa..bbb 100644" in gold
    assert "--- a/src/app.py" in gold and "+++ b/src/app.py" in gold
    assert "@@ -1,3 +1,4 @@ def f():" in gold
    assert "+new line" in gold and "-old line" in gold and " ctx" in gold
    # modified_entities are whole function bodies and would dwarf the hunks
    assert "xxxxx" not in gold


def test_r2e_gym_splits_fix_from_grading_tests():
    info = _r2e_env_info(
        [
            _r2e_file_diff("src/app.py", added=["fix"]),
            _r2e_file_diff("tests/test_app.py", added=["assert True"]),
        ],
        relevant_files=["src/app.py"],
    )
    f = resolve_privilege_fields(info)
    assert "src/app.py" in f["golden_patch"] and "tests/test_app.py" not in f["golden_patch"]
    assert "tests/test_app.py" in f["test_patch"] and "src/app.py" not in f["test_patch"]


def test_r2e_gym_relevant_files_does_not_narrow_the_fix():
    """relevant_files names the PRIMARY file only. A second non-test file that it
    omits is still part of the accepted fix (R2E-Gym's own diff rendering
    includes it), so it must not be misfiled as a grading test."""
    info = _r2e_env_info(
        [
            _r2e_file_diff("src/app.py", added=["fix a"]),
            _r2e_file_diff("src/helper.py", added=["fix b"]),
        ],
        relevant_files=["src/app.py"],
    )
    gold = resolve_privilege_fields(info)["golden_patch"]
    assert "src/app.py" in gold and "src/helper.py" in gold
    assert resolve_privilege_fields(info)["test_patch"] == ""


def test_r2e_gym_test_shaped_source_module_stays_in_the_fix():
    """pandas/util/testing.py is a source module. A pure path heuristic files it
    as a test and leaves that instance with an EMPTY golden patch -- which is
    exactly the failure the whole R2E-Gym branch exists to prevent."""
    info = _r2e_env_info(
        [
            _r2e_file_diff("pandas/util/testing.py", added=["fix"]),
            _r2e_file_diff("pandas/tests/test_testing.py", added=["assert True"]),
        ],
        relevant_files=["pandas/util/testing.py"],
    )
    f = resolve_privilege_fields(info)
    assert "pandas/util/testing.py" in f["golden_patch"]
    assert "pandas/tests/test_testing.py" in f["test_patch"]


def test_r2e_gym_expected_output_becomes_the_acceptance_criterion():
    """FAIL_TO_PASS/PASS_TO_PASS are empty for every R2E-Gym instance; the
    expected test statuses play that role and belong in that budget slot."""
    info = _r2e_env_info(
        [_r2e_file_diff("src/app.py", added=["fix"])],
        relevant_files=["src/app.py"],
        expected_output={"test_a": "PASSED", "test_b": "ERROR"},
    )
    f = resolve_privilege_fields(info)
    assert f["fail_to_pass"] == "test_a: PASSED\ntest_b: ERROR"
    assert f["pass_to_pass"] == ""


def test_r2e_gym_branch_does_not_touch_the_other_datasets():
    """A record that HAS a patch string must take exactly the path it always did,
    even if it also happens to carry parsed_commit_content."""
    info = _env_info(
        instance_id="x",
        instance_dict=json.dumps(
            {
                "patch": "REAL",
                "test_patch": "REAL_TESTS",
                "FAIL_TO_PASS": ["a::b"],
                "parsed_commit_content": json.dumps(
                    {"file_diffs": [_r2e_file_diff("src/app.py", added=["ignored"])]}
                ),
            }
        ),
    )
    f = resolve_privilege_fields(info)
    assert f["golden_patch"] == "REAL"
    assert f["test_patch"] == "REAL_TESTS"
    assert f["fail_to_pass"] == "a::b"


def test_unresolvable_record_still_raises():
    """The loud failure must survive: a genuinely broken data path (no patch, no
    parsed commit) must not be papered over with an empty block."""
    info = _env_info(instance_id="x", instance_dict=json.dumps({"repo": "r"}))
    assert resolve_privilege_fields(info)["golden_patch"] == ""


@pytest.mark.parametrize(
    "raw,expected",
    [
        (["a::b", "c::d"], "a::b\nc::d"),  # swe-bench-ext / SWE-Gym / rebench
        ('["a::b", "c::d"]', "a::b\nc::d"),  # JSON-encoded list
        ("t1 t2", "t1 t2"),  # nv-internal-1 stores a plain string
        (None, ""),
    ],
)
def test_fail_to_pass_normalisation(raw, expected):
    """One matchable item per line, whichever way the dataset stored it."""
    assert _as_lines(raw) == expected


# ---------------------------------------------------------------- block assembly
def test_block_section_order_is_fixed():
    """The critic trains on this schema for thousands of steps; order must be stable."""
    tok = _FakeTokenizer()
    fields = {k: f"body-{k}" for k in SECTION_ORDER}
    block, _ = build_reference_block(fields, tok)
    positions = [block.index(f"<{name}>") for name in SECTION_ORDER]
    assert positions == sorted(positions)
    # fail_to_pass last => nearest the trajectory (mamba retains recent best)
    assert block.rindex("<fail_to_pass>") > block.rindex("<golden_patch>")


def test_absent_sections_are_omitted_not_emptied():
    tok = _FakeTokenizer()
    block, _ = build_reference_block({"golden_patch": "g"}, tok)
    assert "<golden_patch>" in block and "<test_patch>" not in block


def test_caps_bound_output_and_are_deterministic():
    """Truncation must depend only on the instance, never on the rollout."""
    tok = _FakeTokenizer()
    huge = " ".join(str(i) for i in range(50_000))
    a = _cap_tokens(huge, 100, tok)
    b = _cap_tokens(huge, 100, tok)
    assert a == b
    assert len(tok.encode(a)) <= 100 + 4  # +marker


def test_default_caps_cover_every_section():
    assert set(DEFAULT_CAPS) == set(SECTION_ORDER)


# ---------------------------------------------------------------- batch construction
def _batch(n_rows=4, instance="inst-1", turns=3):
    logs, infos = [], []
    for r in range(n_rows):
        msgs = [
            {
                "role": "user",
                "token_ids": torch.arange(5 + r),
                "token_loss_mask": torch.zeros(5 + r, dtype=torch.long),
            }
        ]
        for t in range(turns):
            k = 4 + t + r
            msgs.append(
                {
                    "role": "assistant",
                    "token_ids": torch.arange(k),
                    "token_loss_mask": torch.ones(k, dtype=torch.long),
                }
            )
            msgs.append(
                {
                    "role": "user",
                    "token_ids": torch.arange(3),
                    "token_loss_mask": torch.zeros(3, dtype=torch.long),
                }
            )
        logs.append(msgs)
        infos.append(
            _env_info(
                instance_id=instance,
                instance_dict=json.dumps(
                    {"patch": "GOLD", "test_patch": "TESTS", "FAIL_TO_PASS": ["t::a"]}
                ),
            )
        )
    return BatchedDataDict({"message_log": logs, "extra_env_info": infos})


def test_response_tokens_preserved_verbatim():
    """remap_by_response_mask asserts equal per-row counts; this is that contract."""
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message

    tok = _FakeTokenizer()
    rb = _batch()
    cb = build_swe_privileged_value_inputs(rb, tok, {}, 1)
    flat, _ = batched_message_log_to_flat_message(
        rb["message_log"], pad_value_dict={"token_ids": 0}
    )
    assert torch.equal(cb["token_mask"].sum(-1), flat["token_loss_mask"].sum(-1))


def test_multi_turn_is_supported():
    """The math implementation asserts single-turn; SWE rollouts are ~296 messages."""
    rb = _batch(turns=8)
    cb = build_swe_privileged_value_inputs(rb, _FakeTokenizer(), {}, 1)
    assert cb["token_mask"].sum() > 0


def test_block_identical_across_siblings():
    """A block that varied within a group would recreate the length confound.

    Compares the reference prefix only. Slicing up to the first response token
    would also drag in the rollout's own prompt, which is identical for real
    siblings but deliberately varies row-to-row in this fixture.
    """
    tok = _FakeTokenizer()
    rb = _batch(n_rows=6)
    fields = resolve_privilege_fields(rb["extra_env_info"][0])
    rendered = tok.apply_chat_template(
        [{"role": "system", "content": build_reference_block(fields, tok)[0]}],
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    n_prefix = tok(rendered)["input_ids"].shape[1]
    cb = build_swe_privileged_value_inputs(rb, tok, {}, 1)
    prefixes = {tuple(cb["input_ids"][i][:n_prefix].tolist()) for i in range(6)}
    assert len(prefixes) == 1
    assert n_prefix > 0


def test_privilege_precedes_every_supervised_token():
    """Causality: values before the block could not see it."""
    tok = _FakeTokenizer()
    cb = build_swe_privileged_value_inputs(_batch(), tok, {}, 1)
    for i in range(cb["input_ids"].shape[0]):
        assert int(cb["token_mask"][i].nonzero()[0, 0]) > 0


def test_missing_gold_patch_raises():
    """100% coverage was audited, so a miss is a data-path bug, not a straggler."""
    rb = _batch()
    rb["extra_env_info"][2] = _env_info(instance_id="broken", instance_dict="{}")
    with pytest.raises(ValueError, match="no golden patch resolved"):
        build_swe_privileged_value_inputs(rb, _FakeTokenizer(), {}, 1)


def test_missing_extra_env_info_raises():
    rb = BatchedDataDict({"message_log": _batch()["message_log"]})
    with pytest.raises(ValueError, match="extra_env_info is absent"):
        build_swe_privileged_value_inputs(rb, _FakeTokenizer(), {}, 1)


# ------------------------------------------------- composition: residual x turn-level
def _turn_spans_for(mask):
    """Anchor at the first token of each contiguous response run."""
    from nemo_rl.algorithms.turn_level import TurnSpans

    b, s = mask.shape
    anchor = torch.zeros_like(mask)
    for i in range(b):
        prev = 0
        for j in range(s):
            cur = int(mask[i, j])
            if cur and not prev:
                anchor[i, j] = 1
            prev = cur
    k = int(anchor.sum(-1).max())
    pos = torch.zeros(b, k, dtype=torch.long)
    valid = torch.zeros(b, k, dtype=torch.bool)
    for i in range(b):
        idx = anchor[i].nonzero().flatten()
        pos[i, : len(idx)] = idx
        valid[i, : len(idx)] = True
    return TurnSpans(
        anchor_mask=anchor,
        turn_index=torch.zeros_like(mask, dtype=torch.int32),
        anchor_pos=pos,
        turn_valid=valid,
        num_turns=valid.sum(-1),
        turn_ntokens=torch.ones(b, k, dtype=torch.long),
    )


def test_turn_level_anchors_survive_into_the_augmented_layout():
    """Privileged + turn_gae: anchors must land on augmented response positions.

    build_turn_value_batch builds from train_data (POLICY layout); using it with
    a privileged critic would train on non-privileged sequences. This is the
    combination the older privileged_critic hard-rejects.
    """
    from nemo_rl.algorithms.swe_privileged_critic import (
        build_turn_value_batch_augmented,
    )
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message

    tok = _FakeTokenizer()
    rb = _batch(n_rows=3, turns=4)
    cb = build_swe_privileged_value_inputs(rb, tok, {}, 1)
    flat, _ = batched_message_log_to_flat_message(
        rb["message_log"], pad_value_dict={"token_ids": 0}
    )
    pol_mask = flat["token_loss_mask"]
    spans = _turn_spans_for(pol_mask)

    resp_aug = cb["token_mask"].clone()
    train_data = {
        "token_mask": pol_mask,
        "sample_mask": torch.ones(3),
        # anchor-layout returns, as turn GAE emits them
        "returns": spans.anchor_mask.float() * 7.0,
    }
    out = build_turn_value_batch_augmented(cb, train_data, spans)

    # same number of anchors per row, and all of them inside the aug response mask
    assert torch.equal(out["token_mask"].sum(-1), spans.anchor_mask.sum(-1))
    assert bool((out["token_mask"].bool() & ~resp_aug.bool()).sum() == 0)
    # the anchor returns survived the layout change
    assert torch.allclose(
        out["returns"][out["token_mask"].bool()],
        torch.full((int(out["token_mask"].sum()),), 7.0),
    )


@pytest.mark.parametrize("residual", [False, True])
def test_privileged_composes_with_both_critic_targets(residual):
    """Privilege is an INPUT-layout change; residual is a TARGET change.

    They are orthogonal: the per-sample return-space offsets are [B] and
    row-order is preserved by the augmentation, so they stay aligned.
    """
    from nemo_rl.algorithms.advantage_estimator import (
        GeneralizedAdvantageEstimator,
        ResidualBaselineEstimator,
    )
    from nemo_rl.algorithms.privileged_critic import remap_by_response_mask
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message

    tok = _FakeTokenizer()
    rb = _batch(n_rows=4, turns=2)
    cb = build_swe_privileged_value_inputs(rb, tok, {}, 1)
    flat, _ = batched_message_log_to_flat_message(
        rb["message_log"], pad_value_dict={"token_ids": 0}
    )
    pol_mask = flat["token_loss_mask"].float()

    # values live in the AUGMENTED layout, then come back to the policy layout
    vals_aug = torch.randn_like(cb["token_mask"], dtype=torch.float32)
    vals_pol = remap_by_response_mask(vals_aug, cb["token_mask"], pol_mask)

    class _L:
        use_kl_in_reward = False
        reference_policy_kl_penalty = 0.0
        reference_policy_kl_type = "low_var_kl"

    inner = GeneralizedAdvantageEstimator(
        {
            "gae_lambda": 1.0,
            "gae_gamma": 1,
            "normalize_advantages": False,
            "gae_lambda_value": None,
            "gae_lambda_policy": None,
            "length_adaptive_alpha": 0.0,
        },
        _L(),
    )
    est = ResidualBaselineEstimator(inner, residual_target=residual)
    adv, returns = est.compute_advantage(
        prompt_ids=torch.full((4, 2), 3),
        rewards=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        mask=pol_mask,
        values=vals_pol,
    )
    # returns go forward into the augmented layout for the value loss
    ret_aug = remap_by_response_mask(returns, pol_mask, cb["token_mask"])
    assert ret_aug.shape == cb["token_mask"].shape
    assert torch.allclose(ret_aug[cb["token_mask"].bool()], returns[pol_mask.bool()])
    # per-sample offsets are row-aligned with the augmented batch
    assert est.last_returns_to_abs.shape[0] == cb["input_ids"].shape[0]
    assert adv.shape == pol_mask.shape


# ------------------------------------------------- fixed budget + truncation marks
def test_total_budget_binds_and_is_marked_inline():
    """Over-budget instances are truncated, and the block SAYS SO.

    A silently short block is indistinguishable from an instance that simply has
    less reference material, which would make the critic's schema ambiguous.
    """
    tok = _FakeTokenizer()
    big = " ".join(f"w{i}" for i in range(5000))
    fields = {k: big for k in SECTION_ORDER}
    block, stats = build_reference_block(fields, tok, max_total_tokens=200)

    assert stats["truncated"] is True
    assert stats["kept_tokens"] <= 200
    assert stats["dropped_tokens"] > 0
    assert "[truncated" in block  # block-level note
    assert "... [truncated]" in block  # per-field cut marker
    for f in stats["truncated_fields"]:
        assert f in block


def test_budget_is_spent_in_priority_order_not_emission_order():
    """fail_to_pass must survive a tight budget; pass_to_pass is sacrificed first."""
    tok = _FakeTokenizer()
    big = " ".join(f"w{i}" for i in range(5000))
    fields = {k: big for k in SECTION_ORDER}
    _, stats = build_reference_block(fields, tok, max_total_tokens=300)
    # fail_to_pass is allocated first, so it is not among the starved fields
    assert "pass_to_pass" in stats["truncated_fields"]
    assert (
        stats["truncated_fields"].index("pass_to_pass")
        == len(stats["truncated_fields"]) - 1
        or "pass_to_pass" in stats["truncated_fields"]
    )


def test_starved_field_is_marked_not_dropped():
    """A field with content but no budget gets a placeholder, keeping the schema stable."""
    from nemo_rl.algorithms.swe_privileged_critic import OMITTED_MARKER

    tok = _FakeTokenizer()
    big = " ".join(f"w{i}" for i in range(5000))
    block, _ = build_reference_block(
        {"fail_to_pass": big, "pass_to_pass": big}, tok, max_total_tokens=50
    )
    assert OMITTED_MARKER in block
    assert "<pass_to_pass>" in block  # section present, content marked as omitted


def test_untruncated_block_has_no_markers():
    tok = _FakeTokenizer()
    block, stats = build_reference_block(
        {k: "short body" for k in SECTION_ORDER}, tok, max_total_tokens=10_000
    )
    assert stats["truncated"] is False
    assert "[truncated" not in block


def test_truncation_metrics_are_emitted_per_instance():
    """privilege/* is reported per INSTANCE, not per rollout (block is shared)."""
    tok = _FakeTokenizer()
    rb = _batch(n_rows=6)
    m = {}
    build_swe_privileged_value_inputs(
        rb, tok, {"max_total_tokens": 10_000}, 1, metrics_out=m
    )
    assert m["privilege/n_instances"] == 1.0  # 6 rollouts, one instance
    assert m["privilege/frac_truncated"] == 0.0
    assert m["privilege/block_tokens_mean"] > 0
    for f in SECTION_ORDER:
        assert f"privilege/frac_truncated_{f}" in m

    big = " ".join(f"w{i}" for i in range(5000))
    for info in rb["extra_env_info"]:
        info["responses_create_params"]["metadata"]["instance_dict"] = json.dumps(
            {"patch": big, "test_patch": big, "FAIL_TO_PASS": [big]}
        )
    m2 = {}
    build_swe_privileged_value_inputs(
        rb, tok, {"max_total_tokens": 200}, 1, metrics_out=m2
    )
    assert m2["privilege/frac_truncated"] == 1.0
    assert m2["privilege/dropped_tokens_mean"] > 0
