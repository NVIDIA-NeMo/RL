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
import unittest
from copy import deepcopy

from nemo_rl.environments.nemo_gym_task import (
    bind_nemo_gym_harness_group,
    get_nemo_gym_harness_routing,
    serialize_nemo_gym_harness_routing,
    validate_nemo_gym_harness_result,
    validate_nemo_gym_harness_routing,
)
from nemo_rl.experience.metric_utils import (
    aggregate_harness_routing_metrics,
    harness_routing_metrics,
)


def routing(*, executed="codex", task_type="agentic"):
    return {
        "benchmark": "avb",
        "dataset": "avb-dss-v1",
        "env": "harbor-avb",
        "task_type": task_type,
        "selected_harness": "codex" if task_type == "agentic" else None,
        "executed_harness": executed if task_type == "agentic" else None,
        "routing_version": "benchmark-harness-v1",
        "routing_seed": 42,
        "group_id": "stable-group",
        "group_index": 1 if task_type == "agentic" else None,
    }


class HarnessRoutingTests(unittest.TestCase):
    def test_bind_group_preserves_prepared_ordinal_selection_and_sample_order(self):
        metadata = routing(executed=None)
        metadata["group_id"] = None
        row = {
            "task_id": {"taskset": "avb:train", "task_id": "prompt"},
            "harness_routing": metadata,
            "task_input": {
                "harness_routing": metadata,
                "responses_create_params": {"input": ["frame-0", "frame-1"]},
            },
        }
        original = deepcopy(row)
        first = bind_nemo_gym_harness_group(row, group_id="stable-group")
        retry = bind_nemo_gym_harness_group(row, group_id="stable-group")
        self.assertEqual(first["harness_routing"], retry["harness_routing"])
        self.assertEqual(first["harness_routing"]["group_index"], 1)
        self.assertEqual(first["harness_routing"]["selected_harness"], "codex")
        self.assertEqual(
            first["harness_routing"], first["task_input"]["harness_routing"]
        )
        self.assertEqual(
            first["task_input"]["responses_create_params"]["input"],
            ["frame-0", "frame-1"],
        )
        self.assertEqual(row, original)

    def test_runtime_result_is_required_and_cannot_change_selection_or_identity(self):
        request = {"harness_routing": routing(executed=None)}
        actual = {"full_result": {"harness_routing": routing()}}
        self.assertEqual(
            validate_nemo_gym_harness_result(actual, request)["executed_harness"],
            "codex",
        )
        with self.assertRaisesRegex(ValueError, "omitted"):
            validate_nemo_gym_harness_result({}, request)
        for key, value in [
            ("selected_harness", "hermes"),
            ("group_id", "another"),
            ("routing_seed", 7),
        ]:
            changed = routing()
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_nemo_gym_harness_result({"harness_routing": changed}, request)

    def test_agentic_missing_ordinal_and_wrong_executed_harness_are_rejected(self):
        value = routing()
        value["group_index"] = None
        with self.assertRaisesRegex(ValueError, "prepared benchmark group_index"):
            validate_nemo_gym_harness_routing(value)
        value = routing()
        value["executed_harness"] = "hermes"
        with self.assertRaisesRegex(ValueError, "executed_harness"):
            validate_nemo_gym_harness_routing(value)

    def test_zero_rewards_and_unequal_groups_use_pooled_counts(self):
        groups = []
        for rewards in ([0.0, 0.0, 0.0], [1.0]):
            group = {}
            for reward in rewards:
                for name, value in harness_routing_metrics(
                    routing(),
                    reward=reward,
                    valid=True,
                    masked=False,
                    generation_tokens=2,
                    total_tokens=5,
                ).items():
                    group[name] = group.get(name, 0.0) + value
            groups.append(group)
        pooled = aggregate_harness_routing_metrics(
            {name: [g[name] for g in groups] for name in groups[0]}
        )
        prefix = "nemo_gym/routing/benchmark/avb/harness/codex"
        self.assertEqual(pooled[prefix + "/reward_sum"], 1.0)
        self.assertEqual(pooled[prefix + "/reward_count"], 4.0)
        self.assertEqual(pooled[prefix + "/reward_mean"], 0.25)
        self.assertEqual(pooled[prefix + "/generation_token_count"], 8.0)

    def test_setup_failure_is_not_attributed_to_selected_but_unexecuted_harness(self):
        metadata = routing(executed=None)
        counters = harness_routing_metrics(
            metadata,
            reward=0.0,
            valid=False,
            masked=True,
            generation_tokens=0,
            total_tokens=0,
        )
        self.assertFalse(any("/harness/" in name for name in counters))
        prefix = "nemo_gym/routing/benchmark/avb"
        self.assertEqual(counters[prefix + "/rollout_count"], 1)
        self.assertEqual(counters[prefix + "/valid_rollout_count"], 0)
        self.assertEqual(counters[prefix + "/invalid_rollout_count"], 1)
        self.assertEqual(counters[prefix + "/masked_rollout_count"], 1)
        self.assertEqual(counters[prefix + "/reward_count"], 0)

    def test_overall_actual_harness_pools_benchmarks_and_unequal_groups(self):
        metrics = {}
        for benchmark, rewards in (("avb", [0.0, 0.0, 0.0]), ("claw", [1.0])):
            metadata = routing()
            metadata["benchmark"] = benchmark
            group = {}
            for reward in rewards:
                for name, value in harness_routing_metrics(
                    metadata,
                    reward=reward,
                    valid=True,
                    masked=False,
                    generation_tokens=2,
                    total_tokens=5,
                ).items():
                    group[name] = group.get(name, 0.0) + value
            for name, value in group.items():
                metrics.setdefault(name, []).append(value)
        pooled = aggregate_harness_routing_metrics(metrics)
        prefix = "nemo_gym/routing/harness/codex"
        self.assertEqual(pooled[prefix + "/rollout_count"], 4)
        self.assertEqual(pooled[prefix + "/reward_sum"], 1)
        self.assertEqual(pooled[prefix + "/reward_count"], 4)
        self.assertEqual(pooled[prefix + "/reward_mean"], 0.25)
        self.assertEqual(pooled[prefix + "/generation_token_count"], 8)
        self.assertEqual(pooled["nemo_gym/routing/benchmark/avb/reward_mean"], 0)
        self.assertEqual(pooled["nemo_gym/routing/benchmark/claw/reward_mean"], 1)

    def test_direct_null_harness_keeps_benchmark_env_totals_and_masked_tokens(self):
        metadata = routing(task_type="non_agentic")
        original = deepcopy(metadata)
        counters = harness_routing_metrics(
            metadata,
            reward=0.0,
            valid=True,
            masked=True,
            generation_tokens=8,
            total_tokens=20,
        )
        self.assertFalse(any("/harness/" in name for name in counters))
        self.assertEqual(
            counters["nemo_gym/routing/env/harbor-avb/total_token_count"], 20
        )
        self.assertEqual(
            counters["nemo_gym/routing/benchmark/avb/masked_generation_token_count"], 8
        )
        self.assertEqual(metadata, original)
        self.assertEqual(
            json.loads(serialize_nemo_gym_harness_routing(metadata)), original
        )

    def test_legacy_unrouted_rows_are_unchanged(self):
        row = {"responses_create_params": {"input": "ordinary task"}}
        self.assertIs(bind_nemo_gym_harness_group(row, group_id="group"), row)
        self.assertIsNone(
            get_nemo_gym_harness_routing({"full_result": {"reward": 0.0}})
        )


if __name__ == "__main__":
    unittest.main()
