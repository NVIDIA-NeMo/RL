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
"""Legacy async-PPO rollout metrics ported to the SingleController path.

The first classes are the legacy unit tests (RL jiaqiz/ppo-dev,
tests/unit/experience/test_rollouts.py) run against the port unchanged except
for the function names; the rest cover the SC-specific adapters.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from nemo_rl.algorithms.multi_trace_metrics import finalize_sum_count_metrics
from nemo_rl.experience.interfaces import TRACE_METADATA_KEY, Completion
from nemo_rl.experience.legacy_rollout_metrics import (
    ROLLOUT_DEBUG_TAG,
    aggregate_rollout_metrics_with_sum_counts,
    compaction_rollout_metrics,
    decode_rollout_debug_tag,
    legacy_rollout_group_metrics,
    legacy_rollout_timing_aliases,
    rollout_debug_info,
    rollout_debug_tags,
    rollout_has_think_tag_string_violation,
    rollout_traces,
    termination_kind,
    think_tag_violation_metrics,
    trace_has_think_tag_token_violation,
    trace_kind,
    trace_kind_metrics,
)
from nemo_rl.utils.timer import Timer


class TestRolloutDebugInfo:
    """Unit tests for rollout_debug_info (rollout_debug_step*.jsonl provenance)."""

    @staticmethod
    def _segment(session_id, segment_index, reason, parent=""):
        return {
            "metadata": {
                "session_id": session_id,
                "parent_session_id": parent,
                "segment_index": str(segment_index),
                "segment_boundary_reason": reason,
            }
        }

    def test_counts_compactions_and_subagents_from_segment_metadata(self):
        full_result = {
            "reward": 1.0,
            "resolved": True,
            "patch_exists": True,
            "agent_error_kind": None,
            "agent_timed_out": False,
            "eval_timed_out": False,
            "oom_killed": False,
            "eval_oom_killed": False,
            "openhands_run_time": 1234.5,
            "final_eval_time": 60.0,
            "instance_config": {
                "name": "inst",
                "mask_sample": True,
                "problem_info": {
                    "instance_id": "django__django-1",
                    "dataset_name": "swegym",
                },
            },
            "responses": [
                self._segment("ses_main", 0, ""),
                self._segment("ses_main", 1, "compaction"),
                self._segment("ses_main", 2, "post_compaction"),
                self._segment("ses_sub", 0, "", parent="ses_main"),
                {"metadata": None},  # tolerated: legacy/empty metadata
            ],
        }

        info = rollout_debug_info(full_result)

        # Must be JSON-serializable: it is written to jsonl every step.
        json.dumps(info)
        assert info["instance_id"] == "django__django-1"
        assert info["dataset_name"] == "swegym"
        assert info["reward"] == 1.0
        assert info["mask_sample"] is True
        assert info["num_segments"] == 5
        assert info["num_compactions"] == 1
        assert info["num_root_compactions"] == 1
        assert info["num_subagent_compactions"] == 0
        assert info["num_subagent_sessions"] == 1
        assert info["segments"][1]["segment_boundary_reason"] == "compaction"
        assert info["segments"][3]["parent_session_id"] == "ses_main"
        assert info["segments"][4] == {
            "session_id": None,
            "parent_session_id": "",
            "segment_index": None,
            "segment_boundary_reason": "",
        }

    def test_splits_compactions_by_root_and_subagent_session(self):
        # Root compacts once. The subagent compacts once too, but its summary
        # segment arrives WITHOUT parent_session_id (the fork's compaction
        # call does not forward the parent header), so a per-segment split
        # would misattribute it to the root. Classification is per session.
        full_result = {
            "reward": 0.0,
            "instance_config": {"name": "inst", "problem_info": {}},
            "responses": [
                self._segment("ses_main", 0, ""),
                self._segment("ses_main", 1, "compaction"),
                self._segment("ses_main", 2, "post_compaction"),
                self._segment("ses_sub", 0, "", parent="ses_main"),
                self._segment("ses_sub", 1, "compaction"),  # no parent stamp
                self._segment("ses_sub", 2, "post_compaction", parent="ses_main"),
            ],
        }

        info = rollout_debug_info(full_result)

        assert info["num_compactions"] == 2
        assert info["num_root_compactions"] == 1
        assert info["num_subagent_compactions"] == 1
        assert info["num_subagent_sessions"] == 1

    def test_degrades_gracefully_without_swe_fields(self):
        # Non-SWE Gym agents / legacy single-response results.
        info = rollout_debug_info(
            {
                "reward": 0.0,
                "response": {"output": []},
                "instance_config": {"name": "x"},
            }
        )
        json.dumps(info)
        assert info["instance_id"] == "x"
        assert info["num_segments"] == 0
        assert info["num_compactions"] == 0
        assert info["num_subagent_sessions"] == 0
        assert info["mask_sample"] is False

        empty = rollout_debug_info({})
        json.dumps(empty)
        assert empty["instance_id"] is None
        assert empty["segments"] == []


class TestTerminationKind:
    """Unit tests for termination_kind (one-hot terminal reason of a rollout)."""

    def test_priority_order(self):
        # agent OOM beats everything, then agent timeout, then agent_error_kind.
        assert (
            termination_kind(
                {
                    "oom_killed": True,
                    "agent_timed_out": True,
                    "agent_error_kind": "max_iteration",
                }
            )
            == "agent_oom"
        )
        assert (
            termination_kind(
                {"agent_timed_out": True, "agent_error_kind": "context_window"}
            )
            == "agent_timeout"
        )
        # agent_error_kind beats eval flags.
        assert (
            termination_kind(
                {"agent_error_kind": "stuck_in_loop", "eval_timed_out": True}
            )
            == "stuck_in_loop"
        )
        assert (
            termination_kind({"eval_oom_killed": True, "eval_timed_out": True})
            == "eval_oom"
        )

    def test_named_and_other_error_kinds(self):
        for kind in ("max_iteration", "context_window", "stuck_in_loop"):
            assert termination_kind({"agent_error_kind": kind}) == kind
        assert termination_kind({"agent_error_kind": "bench_crashed"}) == "other_error"

    def test_eval_flags_and_completed(self):
        assert termination_kind({"eval_timed_out": True}) == "eval_timeout"
        assert termination_kind({}) == "completed"
        assert (
            termination_kind(
                {
                    "agent_error_kind": None,
                    "agent_timed_out": False,
                    "oom_killed": False,
                    "eval_timed_out": False,
                    "eval_oom_killed": None,
                }
            )
            == "completed"
        )


class TestTraceKind:
    """Unit tests for _trace_kind (segment kind of a trace)."""

    @staticmethod
    def _md(segment_index="0", reason="", parent=""):
        return {
            "session_id": "ses_main",
            "parent_session_id": parent,
            "segment_index": segment_index,
            "segment_boundary_reason": reason,
        }

    def test_subagent_wins_over_everything(self):
        assert (
            trace_kind(
                self._md("1", "compaction", parent="ses_main"), {"num_compactions": 1}
            )
            == "subagent"
        )

    def test_compaction_segments(self):
        info = {"num_compactions": 1}
        assert trace_kind(self._md("1", "compaction"), info) == "compaction_summary"
        assert trace_kind(self._md("2", "post_compaction"), info) == "post_compaction"
        # segment 0 of a rollout that compacted is the pre-compaction head.
        assert trace_kind(self._md("0"), info) == "pre_compaction"
        assert trace_kind(self._md(0), info) == "pre_compaction"  # int index tolerated

    def test_uncompacted_and_legacy_metadata(self):
        assert trace_kind(self._md("0"), {"num_compactions": 0}) == "uncompacted"
        assert trace_kind({}, {"num_compactions": 0}) == "uncompacted"
        assert trace_kind(None, None) == "uncompacted"
        assert trace_kind({}, {}) == "uncompacted"

    def test_empty_rollout_dummy(self):
        assert trace_kind({}, {"num_compactions": 0}, is_empty_rollout=True) == "empty"
        # The flag wins even if stale metadata says otherwise.
        assert (
            trace_kind(
                self._md("1", "compaction"),
                {"num_compactions": 1},
                is_empty_rollout=True,
            )
            == "empty"
        )


class TestCompactionRolloutMetrics:
    """Unit tests for compaction_rollout_metrics (per-group /sum + /count pairs)."""

    TERMINATION_KINDS = (
        "completed",
        "max_iteration",
        "context_window",
        "stuck_in_loop",
        "other_error",
        "agent_timeout",
        "agent_oom",
        "eval_timeout",
        "eval_oom",
    )

    def test_sums_and_counts(self):
        infos = [
            {"num_compactions": 0, "mask_sample": False, "resolved": True},
            # masked AND resolved: agent timed out after a good patch.
            {
                "num_compactions": 1,
                "mask_sample": True,
                "resolved": True,
                "agent_timed_out": True,
            },
            {
                "num_compactions": 2,
                "mask_sample": False,
                "resolved": False,
                "agent_error_kind": "max_iteration",
            },
            {
                "num_compactions": 3,
                "mask_sample": True,
                "resolved": False,
                "oom_killed": True,
            },
        ]
        rewards = [1.0, 1.0, 0.0, 0.0]
        m = compaction_rollout_metrics(infos, rewards)

        assert m["rollouts/count"] == 4
        assert (
            m["reward/by_num_compactions/0/sum"],
            m["reward/by_num_compactions/0/count"],
        ) == (1.0, 1)
        assert (
            m["reward/by_num_compactions/1/sum"],
            m["reward/by_num_compactions/1/count"],
        ) == (1.0, 1)
        assert (
            m["reward/by_num_compactions/2plus/sum"],
            m["reward/by_num_compactions/2plus/count"],
        ) == (0.0, 2)

        assert m["termination/completed/count"] == 1
        assert m["termination/agent_timeout/count"] == 1
        assert m["termination/max_iteration/count"] == 1
        assert m["termination/agent_oom/count"] == 1
        assert m["reward/by_termination/agent_timeout/sum"] == 1.0
        assert m["reward/by_termination/agent_timeout/count"] == 1
        assert m["reward/by_termination/completed/sum"] == 1.0

        assert m["mask_sample/by_kind/agent_timeout/count"] == 1
        assert m["mask_sample/by_kind/agent_oom/count"] == 1
        assert m["mask_sample/by_kind/completed/count"] == 0
        assert (m["reward/masked/sum"], m["reward/masked/count"]) == (1.0, 2)
        assert (m["reward/unmasked/sum"], m["reward/unmasked/count"]) == (1.0, 2)
        assert m["reward/masked_resolved/count"] == 1

        # Stable key set: every termination kind is present even when absent.
        for kind in self.TERMINATION_KINDS:
            assert f"termination/{kind}/count" in m
            assert f"reward/by_termination/{kind}/sum" in m
            assert f"reward/by_termination/{kind}/count" in m
            assert f"mask_sample/by_kind/{kind}/count" in m
        assert all(math.isfinite(v) for v in m.values())
        json.dumps(m)

    def test_delegation_split(self):
        # Two delegating rollouts (one solved, one env-masked and unsolved) and
        # two non-delegating (one solved, one not).
        infos = [
            {"num_subagent_sessions": 2, "mask_sample": False, "resolved": True},
            {
                "num_subagent_sessions": 1,
                "mask_sample": True,
                "resolved": False,
                "agent_timed_out": True,
            },
            {"num_subagent_sessions": 0, "mask_sample": False, "resolved": True},
            {"num_subagent_sessions": 0, "mask_sample": False, "resolved": False},
        ]
        rewards = [1.0, 0.0, 1.0, 0.0]
        m = compaction_rollout_metrics(infos, rewards)

        assert (
            m["reward/by_delegation/delegated/sum"],
            m["reward/by_delegation/delegated/count"],
        ) == (1.0, 2)
        assert (
            m["reward/by_delegation/none/sum"],
            m["reward/by_delegation/none/count"],
        ) == (1.0, 2)
        # Solve rate over ALL rollouts: 1/2 both arms — the env-masked rollout
        # drags the delegating arm down through no fault of the policy.
        assert (
            m["resolved/by_delegation/delegated/sum"],
            m["resolved/by_delegation/delegated/count"],
        ) == (1, 2)
        assert (
            m["resolved/by_delegation/none/sum"],
            m["resolved/by_delegation/none/count"],
        ) == (1, 2)
        # Dropping it: delegating solves 1/1, non-delegating 1/2.
        assert (
            m["resolved/by_delegation_unmasked/delegated/sum"],
            m["resolved/by_delegation_unmasked/delegated/count"],
        ) == (1, 1)
        assert (
            m["resolved/by_delegation_unmasked/none/sum"],
            m["resolved/by_delegation_unmasked/none/count"],
        ) == (1, 2)
        assert (
            m["reward/by_delegation_unmasked/delegated/sum"],
            m["reward/by_delegation_unmasked/delegated/count"],
        ) == (1.0, 1)
        # The yield gap that explains the difference.
        assert m["mask_sample/by_delegation/delegated/count"] == 1
        assert m["mask_sample/by_delegation/none/count"] == 0

        # Delegation buckets partition the rollouts, exactly like the mask split.
        assert (
            m["reward/by_delegation/delegated/count"]
            + m["reward/by_delegation/none/count"]
            == m["rollouts/count"]
        )
        # Stable key set even when a bucket is empty.
        none_only = compaction_rollout_metrics([{"num_subagent_sessions": 0}], [1.0])
        for bucket in ("delegated", "none"):
            for key in (
                "reward/by_delegation",
                "resolved/by_delegation",
                "reward/by_delegation_unmasked",
                "resolved/by_delegation_unmasked",
            ):
                assert f"{key}/{bucket}/sum" in none_only
                assert f"{key}/{bucket}/count" in none_only
            assert f"mask_sample/by_delegation/{bucket}/count" in none_only
        assert none_only["reward/by_delegation/delegated/count"] == 0
        assert all(math.isfinite(v) for v in m.values())
        json.dumps(m)

    def test_delegation_means_survive_finalize(self):
        """/sum + /count -> /mean through the trainer's aggregation step."""
        infos = [
            {"num_subagent_sessions": 1, "mask_sample": False, "resolved": True},
            {"num_subagent_sessions": 1, "mask_sample": False, "resolved": False},
            {"num_subagent_sessions": 0, "mask_sample": False, "resolved": False},
        ]
        final = finalize_sum_count_metrics(
            compaction_rollout_metrics(infos, [1.0, 0.0, 0.0])
        )
        assert final["resolved/by_delegation/delegated/mean"] == 0.5
        assert final["resolved/by_delegation/none/mean"] == 0.0
        assert final["reward/by_delegation/delegated/mean"] == 0.5
        # Empty bucket: no /mean (no rollouts to average), and no NaN emitted.
        empty = finalize_sum_count_metrics(
            compaction_rollout_metrics([{"num_subagent_sessions": 0}], [1.0])
        )
        assert "resolved/by_delegation/delegated/mean" not in empty
        assert all(math.isfinite(v) for v in empty.values())

    def test_empty_group(self):
        m = compaction_rollout_metrics([], [])
        assert m["rollouts/count"] == 0
        assert all(v == 0 for v in m.values())

    def test_length_mismatch_is_an_error(self):
        with pytest.raises(AssertionError):
            compaction_rollout_metrics([{}], [1.0, 0.0])


class TestTraceKindMetrics:
    """Unit tests for trace_kind_metrics (per-trace sizes by segment kind)."""

    def test_compaction_sizes_and_turns_by_kind(self):
        kinds = [
            "pre_compaction",
            "compaction_summary",
            "post_compaction",
            "subagent",
            "uncompacted",
            "empty",
        ]
        turns = [10, 1, 5, 3, 7, 1]
        prompt = [50000, 90000, 3000, 2000, 1000, 1]
        gen = [4000, 1500, 2000, 500, 800, 0]
        m = trace_kind_metrics(kinds, turns, prompt, gen)

        assert (
            m["compaction/summary_gen_tokens/sum"],
            m["compaction/summary_gen_tokens/count"],
            m["compaction/summary_gen_tokens/max"],
        ) == (1500, 1, 1500)
        assert (
            m["compaction/trigger_prompt_tokens/sum"],
            m["compaction/trigger_prompt_tokens/count"],
            m["compaction/trigger_prompt_tokens/max"],
        ) == (90000, 1, 90000)
        assert (
            m["compaction/post_compaction_prompt_tokens/sum"],
            m["compaction/post_compaction_prompt_tokens/count"],
            m["compaction/post_compaction_prompt_tokens/max"],
        ) == (3000, 1, 3000)
        assert m["turns_per_trace/by_kind/pre_compaction/sum"] == 10
        assert m["turns_per_trace/by_kind/post_compaction/sum"] == 5
        assert m["turns_per_trace/by_kind/subagent/max"] == 3
        assert m["turns_per_trace/by_kind/uncompacted/count"] == 1
        # The forced single-call summary and the dummy are not turn-comparable.
        assert not any(
            k.startswith("turns_per_trace/by_kind/compaction_summary") for k in m
        )
        assert not any(k.startswith("turns_per_trace/by_kind/empty") for k in m)

    def test_absent_kinds_emit_zeros(self):
        m = trace_kind_metrics(
            ["uncompacted", "uncompacted"], [4, 6], [100, 200], [10, 20]
        )
        assert (
            m["compaction/summary_gen_tokens/sum"],
            m["compaction/summary_gen_tokens/count"],
            m["compaction/summary_gen_tokens/max"],
        ) == (0, 0, 0)
        assert m["compaction/trigger_prompt_tokens/count"] == 0
        assert m["compaction/post_compaction_prompt_tokens/count"] == 0
        assert (
            m["turns_per_trace/by_kind/uncompacted/sum"],
            m["turns_per_trace/by_kind/uncompacted/count"],
            m["turns_per_trace/by_kind/uncompacted/max"],
        ) == (10, 2, 6)
        assert m["turns_per_trace/by_kind/subagent/count"] == 0


class TestThinkTagViolation:
    """Unit tests for the think-tag detection helpers (think_open=12, think_close=13)."""

    TOKEN_IDS = {"think_open": 12, "think_close": 13}

    @staticmethod
    def _msg(role, ids):
        return {"role": role, "token_ids": torch.tensor(ids, dtype=torch.long)}

    @classmethod
    def _trace(cls, message_log, full_result=None, **extra):
        return {
            "message_log": message_log,
            "input_message_log": message_log[:1],
            "full_result": full_result
            if full_result is not None
            else {"reward": 1.0, "responses": []},
            **extra,
        }

    # -- token-ID check -----------------------------------------------------

    def test_token_check_accepts_well_formed_turns(self):
        # thinking enabled: prompt ends with <think>, generation closes it once.
        log = [self._msg("user", [1, 2, 12]), self._msg("assistant", [5, 6, 13, 7])]
        assert trace_has_think_tag_token_violation(log, 12, 13) is False
        # thinking disabled: balanced prompt, no tags generated.
        log = [self._msg("user", [1, 12, 13]), self._msg("assistant", [5, 6])]
        assert trace_has_think_tag_token_violation(log, 12, 13) is False
        # multi-turn: later prompts include earlier <think>...</think> pairs.
        log = [
            self._msg("user", [1, 12]),
            self._msg("assistant", [5, 13, 7]),
            self._msg("user", [1, 12, 5, 13, 7, 8, 12]),
            self._msg("assistant", [9, 13, 10]),
        ]
        assert trace_has_think_tag_token_violation(log, 12, 13) is False

    def test_token_check_flags_bad_generations_and_prompts(self):
        # generated <think>
        log = [self._msg("user", [1, 12]), self._msg("assistant", [12, 5, 13])]
        assert trace_has_think_tag_token_violation(log, 12, 13) is True
        # thinking enabled but never closed
        log = [self._msg("user", [1, 12]), self._msg("assistant", [5, 6])]
        assert trace_has_think_tag_token_violation(log, 12, 13) is True
        # two closes
        log = [self._msg("user", [1, 12]), self._msg("assistant", [5, 13, 6, 13])]
        assert trace_has_think_tag_token_violation(log, 12, 13) is True
        # close generated with thinking disabled
        log = [self._msg("user", [1, 12, 13]), self._msg("assistant", [5, 13])]
        assert trace_has_think_tag_token_violation(log, 12, 13) is True
        # unexpected prompt pattern (two opens)
        log = [self._msg("user", [12, 12]), self._msg("assistant", [5, 13])]
        assert trace_has_think_tag_token_violation(log, 12, 13) is True
        # violation only in a later pair is still caught
        log = [
            self._msg("user", [1, 12]),
            self._msg("assistant", [5, 13]),
            self._msg("user", [1, 12, 5, 13, 12]),
            self._msg("assistant", [5]),
        ]
        assert trace_has_think_tag_token_violation(log, 12, 13) is True

    def test_token_check_honours_configured_ids_and_list_token_ids(self):
        log = [self._msg("user", [1, 100]), self._msg("assistant", [5, 101])]
        assert trace_has_think_tag_token_violation(log, 100, 101) is False
        assert trace_has_think_tag_token_violation(log, 12, 13) is False  # balanced 0/0
        log = [
            {"role": "user", "token_ids": [1, 12]},
            {"role": "assistant", "token_ids": [12, 13]},
        ]
        assert trace_has_think_tag_token_violation(log, 12, 13) is True

    # -- string check -------------------------------------------------------

    def test_string_check_reads_responses_or_response(self):
        def full_result_with(gen_str, legacy=False):
            resp = {"output": [{"type": "message", "generation_str": gen_str}]}
            if legacy:
                return {"reward": 1.0, "response": resp}
            return {"reward": 1.0, "responses": [resp]}

        assert (
            rollout_has_think_tag_string_violation(full_result_with("plan</think>done"))
            is False
        )
        assert (
            rollout_has_think_tag_string_violation(full_result_with("<think>x</think>"))
            is True
        )
        assert (
            rollout_has_think_tag_string_violation(
                full_result_with("a</think>b</think>")
            )
            is True
        )
        assert (
            rollout_has_think_tag_string_violation(
                full_result_with("a</think>b</think>", legacy=True)
            )
            is True
        )
        # No decoded text at all -> no string violation.
        assert rollout_has_think_tag_string_violation({"reward": 1.0}) is False

    # -- metrics + penalty wiring -------------------------------------------

    def test_violation_metrics_count_rollouts_and_traces_without_touching_reward(self):
        ok_log = [self._msg("user", [1, 12]), self._msg("assistant", [5, 13])]
        bad_log = [self._msg("user", [1, 12]), self._msg("assistant", [5, 6])]
        shared_a = {"reward": 1.0, "responses": []}
        shared_b = {
            "reward": 1.0,
            "responses": [{"output": [{"generation_str": "<think>oops"}]}],
        }
        shared_c = {"reward": 0.0, "responses": []}
        shared_d = {"reward": 0.0, "responses": []}
        rollout_results = [
            # A: token violation on the second trace only
            [self._trace(ok_log, shared_a), self._trace(bad_log, shared_a)],
            # B: clean tokens, string violation
            [self._trace(ok_log, shared_b)],
            # C: clean
            [
                self._trace(ok_log, shared_c),
                self._trace(ok_log, shared_c),
                self._trace(ok_log, shared_c),
            ],
            # D: empty dummy, ignored
            [self._trace(ok_log, shared_d, is_empty_rollout=True)],
        ]
        m = think_tag_violation_metrics(rollout_results, self.TOKEN_IDS)
        assert m == {
            "format/think_tag_violation/count": 2,
            "format/think_tag_violation/by_trace_in_rollout_idx/0/count": 0,
            "format/think_tag_violation/by_trace_in_rollout_idx/1/count": 1,
            "format/think_tag_violation/by_trace_in_rollout_idx/2plus/count": 0,
        }
        # Count-only: rewards are untouched.
        assert [r["reward"] for r in (shared_a, shared_b, shared_c, shared_d)] == [
            1.0,
            1.0,
            0.0,
            0.0,
        ]


# ---------------------------------------------------------------------------
# SC adapters
# ---------------------------------------------------------------------------


def _msg(role, ids, **extra):
    return {"role": role, "token_ids": torch.tensor(ids, dtype=torch.long), **extra}


def _ok_turns(n_turns=1, gen=(5, 13)):
    """n well-formed (user, assistant) pairs; thinking enabled (prompt ends <think>)."""
    log = []
    for _ in range(n_turns):
        log += [_msg("user", [1, 12]), _msg("assistant", list(gen))]
    return log


def _segment(session, index, reason="", parent=""):
    return {
        "metadata": {
            "session_id": session,
            "parent_session_id": parent,
            "segment_index": str(index),
            "segment_boundary_reason": reason,
        },
        "output": [],
    }


def _sc_results():
    """Three SC NeMo-Gym rollout results of one prompt group.

    R0: multi-trace (main + one subagent session), resolved.
    R1: single trace, env-masked agent timeout, malformed thinking (no </think>).
    R2: the masked empty-generation placeholder (OOM-killed before any call).
    """
    r0_full = {
        "reward": 1.0,
        "resolved": True,
        "instance_config": {"problem_info": {"instance_id": "repo__x-1"}},
        "responses": [_segment("ses_a", 0), _segment("ses_b", 0, parent="ses_a")],
        "SECRET_LIKE_FIELD": "must-not-be-copied",
    }
    r0 = {
        "message_log": _ok_turns(3),
        "full_result": r0_full,
        "session_traces": [
            {
                "message_log": _ok_turns(3),
                "trace_metadata": {
                    "trace_in_rollout_idx": 0,
                    "session_id": "ses_a",
                    "parent_session_id": "",
                    "segment_index": "0",
                    "segment_boundary_reason": "",
                },
            },
            {
                "message_log": _ok_turns(2),
                "trace_metadata": {
                    "trace_in_rollout_idx": 1,
                    "session_id": "ses_b",
                    "parent_session_id": "ses_a",
                    "segment_index": "0",
                    "segment_boundary_reason": "",
                },
            },
        ],
    }
    r1 = {
        "message_log": [_msg("user", [1, 12]), _msg("assistant", [5, 6])],
        "full_result": {
            "reward": 0.0,
            "resolved": False,
            "agent_timed_out": True,
            "instance_config": {"mask_sample": True},
        },
    }
    r2 = {
        "message_log": [_msg("user", [0])],
        "full_result": {
            "reward": 0.0,
            "oom_killed": True,
            "is_empty_rollout": True,
            "instance_config": {"mask_sample": True},
        },
    }
    return [r0, r1, r2]


def test_rollout_traces_views_sc_results_as_legacy_traces():
    r0, r1, r2 = _sc_results()
    traces = rollout_traces(r0)
    assert len(traces) == 2
    # SC's index column is dropped from the legacy-shaped metadata.
    assert "trace_in_rollout_idx" not in traces[1]["trace_metadata"]
    assert traces[1]["trace_metadata"]["parent_session_id"] == "ses_a"
    assert traces[0]["full_result"] is r0["full_result"]
    assert [t["is_empty_rollout"] for t in rollout_traces(r1)] == [False]
    assert [t["is_empty_rollout"] for t in rollout_traces(r2)] == [True]


def test_legacy_rollout_group_metrics_on_sc_results():
    timer = Timer()
    m = legacy_rollout_group_metrics(
        _sc_results(),
        {"token_ids": {"think_open": 12, "think_close": 13}},
        timer=timer,
        timer_prefix="timing/rollout",
    )
    assert m["rollouts/count"] == 3
    assert m["traces_per_sample/mean"] == pytest.approx(4 / 3)
    # Per-trace user-turn counts: 3, 2 (R0), 1 (R1), 1 (R2 placeholder).
    assert m["turns_per_trace/histogram"] == [3, 2, 1, 1]
    assert m["turns_per_trace/mean"] == pytest.approx(7 / 4)
    assert m["subagent_traces_per_sample/mean"] == pytest.approx(1 / 3)
    assert m["subagent_sessions_per_sample/max"] == 1
    assert m["delegation_rate"] == pytest.approx(1 / 3)
    assert m["mask_sample_rate"] == pytest.approx(2 / 3)
    assert m["compaction_rate"] == 0.0
    assert m["termination/completed/count"] == 1
    assert m["termination/agent_timeout/count"] == 1
    assert m["termination/agent_oom/count"] == 1
    assert m["mask_sample/by_kind/agent_oom/count"] == 1
    assert m["reward/by_delegation/delegated/sum"] == 1.0
    assert m["resolved/by_delegation/delegated/sum"] == 1
    assert (m["reward/masked/sum"], m["reward/masked/count"]) == (0.0, 2)
    assert m["turns_per_trace/by_kind/subagent/sum"] == 2
    assert m["turns_per_trace/by_kind/uncompacted/count"] == 2
    assert m["empty_rollout_count"] == 1
    # Legacy per-rollout sizes sum every trace of the rollout.
    assert m["turns_per_sample/histogram"] == [5, 1, 1]
    assert m["gen_tokens_per_sample/histogram"] == [10, 2, 0]
    assert m["total_tokens_per_sample/histogram"] == [20, 4, 1]
    assert m["mean_gen_tokens_per_sample"] == pytest.approx(4.0)
    # R1 has the token violation; the empty placeholder is skipped.
    assert m["format/think_tag_violation/count"] == 1
    assert m["format/think_tag_violation/by_trace_in_rollout_idx/0/count"] == 1
    timing = timer.get_timing_metrics("sum")
    assert "timing/rollout/prepare_for_metrics_calculation" in timing
    assert "timing/rollout/aggregate_metrics" in timing


def test_group_metrics_skip_think_tags_without_token_ids_and_empty_groups():
    m = legacy_rollout_group_metrics(_sc_results(), {"penalize_unwanted_tokens": False})
    assert not any(k.startswith("format/") for k in m)
    assert legacy_rollout_group_metrics([], None) == {}


def _completion(message_log, env_extras, reward=0.0):
    return Completion(
        message_log=message_log, env_extras=env_extras, truncated=False, reward=reward
    )


def test_rollout_debug_tags_carry_legacy_provenance_only():
    r0, r1, _ = _sc_results()
    completions = [
        _completion(
            t["message_log"],
            {**r0["full_result"], TRACE_METADATA_KEY: t["trace_metadata"]},
            reward=1.0,
        )
        for t in r0["session_traces"]
    ] + [_completion(r1["message_log"], r1["full_result"])]

    tags = rollout_debug_tags(completions)
    assert tags is not None and len(tags) == 3
    assert all(isinstance(t, str) for t in tags)
    rows = [decode_rollout_debug_tag({ROLLOUT_DEBUG_TAG: t}) for t in tags]
    assert [r["rollout_local_idx"] for r in rows] == [0, 0, 1]
    assert [r["trace_in_rollout_idx"] for r in rows] == [0, 1, 0]
    assert [r["is_empty_rollout"] for r in rows] == [False, False, False]
    assert [r["trace_metadata"]["kind"] for r in rows] == [
        "uncompacted",
        "subagent",
        "uncompacted",
    ]
    assert rows[0]["trace_metadata"]["turns"] == 3
    assert rows[1]["trace_metadata"]["prompt_tokens"] == 2
    assert rows[1]["trace_metadata"]["gen_tokens"] == 4
    assert "trace_in_rollout_idx" not in rows[1]["trace_metadata"]
    assert rows[0]["rollout_info"]["instance_id"] == "repo__x-1"
    assert rows[0]["rollout_info"]["num_subagent_sessions"] == 1
    assert rows[2]["rollout_info"]["mask_sample"] is True
    # Fixed-field whitelist: nothing else from full_result reaches the jsonl.
    assert all("must-not-be-copied" not in t for t in tags)


def test_rollout_debug_tags_none_for_non_gym_completions():
    completions = [_completion(_ok_turns(), {"reward": 1.0, "status": "ok"})]
    assert rollout_debug_tags(completions) is None
    assert rollout_debug_tags([]) is None
    assert decode_rollout_debug_tag({"weight_version": 3}) is None


def test_record_to_train_batch_and_pack_payload_stamp_rollout_debug_tags():
    from nemo_rl.experience.interfaces import PromptGroupRecord
    from nemo_rl.experience.payload import pack_payload, record_to_train_batch

    def gen_log():
        return [
            _msg("user", [10, 11]),
            _msg(
                "assistant",
                [20, 21],
                generation_logprobs=torch.tensor([-0.1, -0.2]),
            ),
        ]

    full = {"reward": 1.0, "instance_config": {"name": "inst-7"}}
    record = PromptGroupRecord(
        prompt_idx=4,
        prompt=[_msg("user", [10, 11])],
        extra_env_info=None,
        metadata={"task_name": "nemo_gym"},
        completions=[_completion(gen_log(), dict(full), 1.0) for _ in range(2)],
        rollout_metrics={},
        loss_multiplier=1.0,
    )
    batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
        include_message_violation_fields=False,
    )
    _, fields, tags = pack_payload(
        batch, weight_version=2, group_id="grp", prompt_idx=4
    )
    assert "rollout_debug_tags" not in fields
    rows = [decode_rollout_debug_tag(t) for t in tags]
    assert [r["rollout_local_idx"] for r in rows] == [0, 1]
    assert rows[1]["rollout_info"]["instance_id"] == "inst-7"
    assert tags[0]["weight_version"] == 2 and tags[0]["prompt_idx"] == 4


def test_aggregate_rollout_metrics_with_sum_counts_is_exact():
    g1 = compaction_rollout_metrics(
        [{"mask_sample": True, "agent_timed_out": True}, {}], [0.0, 1.0]
    )
    g2 = compaction_rollout_metrics([{}, {}, {}, {}], [1.0, 1.0, 0.0, 0.0])
    per_group: dict[str, list] = {}
    for g in (g1, g2):
        g = {**g, "empty_rollout_count": 1, "delegation_rate": 0.5}
        g["turns_per_trace/histogram"] = [1, 2]
        for k, v in g.items():
            per_group.setdefault(k, []).append(v)

    seen: dict = {}

    def mean_fn(d):
        seen.update(d)
        return {
            k: (sum(v) / len(v) if isinstance(v[0], (int, float)) else v)
            for k, v in d.items()
        }

    out = aggregate_rollout_metrics_with_sum_counts(per_group, mean_fn)
    # Exact over all 6 rollouts, not a mean of per-group means.
    assert out["rollouts/count"] == 6
    assert out["reward/unmasked/mean"] == pytest.approx(3 / 5)
    assert out["termination/agent_timeout/rate"] == pytest.approx(1 / 6)
    assert out["mask_sample/by_kind/agent_timeout/rate"] == pytest.approx(1 / 6)
    assert out["empty_rollout_count"] == 2
    assert "reward/unmasked/sum" not in out
    # Everything else still goes through the caller's aggregator.
    assert out["delegation_rate"] == 0.5
    assert "delegation_rate" in seen and "rollouts/count" not in seen
    assert all(math.isfinite(v) for v in out.values() if isinstance(v, (int, float)))


def test_legacy_rollout_timing_aliases():
    env = {
        "timing/rollout/shard/nemo_gym/await_results": 3.0,
        "timing/rollout/shard/nemo_gym/postprocess_results": 1.0,
        "timing/rollout/shard/nemo_gym/postprocess_results_pct": 25.0,
    }
    out = legacy_rollout_timing_aliases(
        env, "timing/rollout/shard/nemo_gym", "timing/rollout"
    )
    assert out == {
        "timing/rollout/await_results": 3.0,
        "timing/rollout/postprocess_results": 1.0,
        "timing/rollout/postprocess_results_pct": 25.0,
    }
    assert legacy_rollout_timing_aliases({}, "x", "y") == {}
