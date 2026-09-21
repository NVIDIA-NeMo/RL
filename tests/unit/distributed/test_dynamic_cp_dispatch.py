# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from random import Random
from unittest.mock import patch

import numpy as np
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.dynamic_context_parallel import (
    cp_loss_multiplier,
    plan_cp_phases,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.policy.dynamic_cp import (
    _enabled_global_aux_loss,
    _minimum_cp_size_for_experts,
    build_cp_dispatch,
    collect_cp_outputs,
    cp_schedule_matches,
    owned_real_task_count,
    real_task_participation_count,
    validate_dynamic_cp,
)


class TestDynamicCPDispatch(unittest.TestCase):
    def setUp(self):
        self.data = BatchedDataDict(
            input_ids=torch.arange(5 * 32).reshape(5, 32),
            input_lengths=torch.tensor([5, 11, 17, 25, 29]),
            sample_mask=torch.tensor([1, 1, 0, 1, 1]),
            sample_ids=torch.arange(5),
        )
        self.data["token_mask"] = (
            torch.arange(32)[None, :] < self.data["input_lengths"][:, None]
        ).long()
        self.cfg = {
            "make_sequence_length_divisible_by": 1,
            "megatron_cfg": {
                "tensor_model_parallel_size": 2,
                "expert_model_parallel_size": 1,
                "sequence_parallel": False,
                "dynamic_context_parallel": {
                    "enabled": True,
                    "tokens_per_rank": 8,
                    "max_size": 4,
                },
            },
        }
        self.mesh = NamedSharding(
            np.arange(8).reshape(1, 2, 2, 2),
            [
                "pipeline_parallel",
                "data_parallel",
                "context_parallel",
                "tensor_parallel",
            ],
        )

    def test_moe_minimum_contains_complete_expert_group(self):
        self.assertEqual(
            _minimum_cp_size_for_experts(
                {
                    "tensor_model_parallel_size": 1,
                    "expert_tensor_parallel_size": 1,
                    "expert_model_parallel_size": 8,
                },
                1,
            ),
            8,
        )
        self.assertEqual(
            _minimum_cp_size_for_experts(
                {
                    "tensor_model_parallel_size": 2,
                    "expert_tensor_parallel_size": 1,
                    "expert_model_parallel_size": 16,
                },
                1,
            ),
            8,
        )
        self.assertEqual(
            _minimum_cp_size_for_experts(
                {
                    "tensor_model_parallel_size": 2,
                    "expert_tensor_parallel_size": 2,
                    "expert_model_parallel_size": 8,
                },
                1,
            ),
            8,
        )
        self.assertEqual(
            _minimum_cp_size_for_experts(
                {
                    "tensor_model_parallel_size": 2,
                    "expert_tensor_parallel_size": 2,
                    "expert_model_parallel_size": 1,
                },
                1,
            ),
            1,
        )
        with self.assertRaisesRegex(ValueError, "to divide"):
            _minimum_cp_size_for_experts(
                {
                    "tensor_model_parallel_size": 2,
                    "expert_tensor_parallel_size": 1,
                    "expert_model_parallel_size": 12,
                },
                1,
            )

    def test_global_aux_loss_is_detected_even_with_provider_coefficient(self):
        self.assertTrue(
            _enabled_global_aux_loss(
                {"moe_router_load_balancing_type": "global_aux_loss"}
            )
        )
        self.assertTrue(
            _enabled_global_aux_loss(
                {
                    "moe_router_load_balancing_type": [
                        "aux_loss",
                        "global_aux_loss",
                    ]
                }
            )
        )
        self.assertTrue(
            _enabled_global_aux_loss(
                {
                    "model_overrides": {
                        "moe_router_load_balancing_type": "global_aux_loss"
                    }
                }
            )
        )
        self.assertFalse(
            _enabled_global_aux_loss({"moe_router_load_balancing_type": "seq_aux_loss"})
        )

    def test_global_aux_loss_aligns_full_domain_collective_rounds(self):
        data = BatchedDataDict(
            input_ids=torch.zeros(5, 12, dtype=torch.long),
            input_lengths=torch.full((5,), 10),
            sample_mask=torch.ones(5, dtype=torch.long),
            token_mask=torch.ones(5, 12, dtype=torch.long),
        )
        cfg = self._validation_cfg()
        cfg["megatron_cfg"].update(
            {
                "moe_router_load_balancing_type": "global_aux_loss",
                "dynamic_context_parallel": {
                    "enabled": True,
                    "tokens_per_rank": 10,
                    "max_size": 1,
                },
            }
        )

        validate_dynamic_cp(cfg, lanes=4)
        dispatch = build_cp_dispatch(data, cfg, self.mesh, batch_size=5, training=True)
        local_counts = [
            len(plan.steps[0].assignments) for plans in dispatch.plans for plan in plans
        ]
        assert local_counts == [2, 2, 2, 2]
        assert any(
            not task.sample_indices
            for plans in dispatch.plans
            for plan in plans
            for task in plan.steps[0].assignments
        )

    def _validation_cfg(self) -> dict:
        return {
            "make_sequence_length_divisible_by": 1,
            "sequence_packing": {"enabled": True, "pair_grouping_key": None},
            "dynamic_batching": {"enabled": False},
            "draft": {"enabled": False},
            "megatron_cfg": {
                "enabled": True,
                "pipeline_model_parallel_size": 1,
                "tensor_model_parallel_size": 1,
                "expert_model_parallel_size": 1,
                "expert_tensor_parallel_size": 1,
                "sequence_parallel": False,
                "dynamic_context_parallel": {
                    "enabled": True,
                    "tokens_per_rank": 8,
                    "max_size": 4,
                },
            },
        }

    def test_dynamic_cp_validation_requires_pp_one(self):
        cfg = self._validation_cfg()
        cfg["megatron_cfg"]["pipeline_model_parallel_size"] = 2
        with self.assertRaisesRegex(ValueError, "PP=1"):
            validate_dynamic_cp(cfg, lanes=4)

    def test_dynamic_cp_validation_rejects_quantile_router(self):
        cfg = self._validation_cfg()
        cfg["megatron_cfg"]["model_overrides"] = {
            "moe_router_load_balancing_type": "quantile_balancing"
        }
        with self.assertRaisesRegex(ValueError, "quantile_balancing"):
            validate_dynamic_cp(cfg, lanes=4)

    def test_dynamic_cp_validation_rejects_moe_microbatch_overlap(self):
        cfg = self._validation_cfg()
        cfg["megatron_cfg"]["model_overrides"] = {
            "overlap_moe_expert_parallel_comm": True
        }
        with self.assertRaisesRegex(ValueError, "overlap_moe_expert_parallel_comm"):
            validate_dynamic_cp(cfg, lanes=4)

    def test_dynamic_cp_validation_allows_mtp_and_hybridep_prepad_separately(self):
        mtp_cfg = self._validation_cfg()
        mtp_cfg["megatron_cfg"]["mtp_num_layers"] = 1
        validate_dynamic_cp(mtp_cfg, lanes=4)

        hybridep_cfg = self._validation_cfg()
        hybridep_cfg["megatron_cfg"].update(
            {
                "moe_token_dispatcher_type": "flex",
                "moe_flex_dispatcher_backend": "hybridep",
                "moe_hybridep_prepad_packed_inputs": True,
            }
        )
        validate_dynamic_cp(hybridep_cfg, lanes=4)

    def test_dynamic_cp_validation_allows_mtp_with_hybridep_prepad(self):
        cfg = self._validation_cfg()
        cfg["megatron_cfg"].update(
            {
                "mtp_num_layers": 1,
                "moe_hybridep_prepad_packed_inputs": True,
            }
        )
        validate_dynamic_cp(cfg, lanes=4)

    def test_dynamic_cp_keeps_preference_pairs_atomic(self):
        data = BatchedDataDict(
            input_ids=torch.arange(4 * 16).reshape(4, 16),
            input_lengths=torch.tensor([9, 3, 8, 4]),
            pair_index=torch.tensor([10, 10, 20, 20]),
            sample_mask=torch.ones(4, dtype=torch.long),
            token_mask=torch.ones(4, 16, dtype=torch.long),
        )
        cfg = self._validation_cfg()
        cfg["sequence_packing"]["pair_grouping_key"] = "pair_index"

        validate_dynamic_cp(cfg, lanes=4)
        dispatch = build_cp_dispatch(data, cfg, self.mesh, batch_size=4, training=True)

        real_assignments = [
            task.sample_indices
            for group in dispatch.schedule.groups_by_batch[0]
            for task in group.assignments
            if task.sample_indices
        ]
        for pair in ((0, 1), (2, 3)):
            assert any(
                set(pair).issubset(assignment) for assignment in real_assignments
            )

    def test_dynamic_cp_rejects_atomic_pair_split_across_steps(self):
        data = BatchedDataDict(
            input_ids=torch.arange(4 * 16).reshape(4, 16),
            input_lengths=torch.tensor([9, 3, 8, 4]),
            pair_index=torch.tensor([10, 20, 10, 20]),
            sample_mask=torch.ones(4, dtype=torch.long),
            token_mask=torch.ones(4, 16, dtype=torch.long),
        )
        cfg = self._validation_cfg()
        cfg["sequence_packing"]["pair_grouping_key"] = "pair_index"

        with self.assertRaisesRegex(ValueError, "cannot cross"):
            build_cp_dispatch(data, cfg, self.mesh, batch_size=2, training=True)

    def test_outputs_from_nonzero_static_cp_are_preserved(self):
        dispatch = build_cp_dispatch(
            self.data, self.cfg, self.mesh, batch_size=None, training=False
        )
        results = []
        for lane, plan in enumerate(p for dp in dispatch.plans for p in dp):
            payload = dispatch.data[lane // 2][lane % 2]
            values = []
            for task in plan.steps[0].assignments:
                values.extend(
                    payload["sample_ids"][list(task.sample_indices)].tolist()
                    if task.sample_indices
                    else [-1]
                )
            results.append(BatchedDataDict(logprobs=torch.tensor(values)[:, None]))
        restored = collect_cp_outputs(results, dispatch, self.data.size)
        torch.testing.assert_close(restored["logprobs"].flatten(), torch.arange(5))
        with self.assertRaises(ValueError):
            collect_cp_outputs(results[::2], dispatch, self.data.size)

    def test_moe_dispatch_never_schedules_less_than_ep_over_tp(self):
        cfg = {**self.cfg, "megatron_cfg": dict(self.cfg["megatron_cfg"])}
        cfg["megatron_cfg"].update(
            {
                "expert_tensor_parallel_size": 1,
                "expert_model_parallel_size": 8,
                "dynamic_context_parallel": {
                    "enabled": True,
                    "tokens_per_rank": 8,
                    "max_size": 4,
                },
            }
        )
        dispatch = build_cp_dispatch(
            self.data, cfg, self.mesh, batch_size=None, training=False
        )
        assert all(
            task.cp_size == 4
            for dp_plans in dispatch.plans
            for plan in dp_plans
            for task in plan.steps[0].assignments
        )

    def test_global_normalizer_and_autograd_do_not_count_cp_copies(self):
        dispatch = build_cp_dispatch(
            self.data, self.cfg, self.mesh, batch_size=5, training=True
        )
        mask = self.data["token_mask"][:, 1:] * self.data["sample_mask"][:, None]
        normalizer = mask.sum()
        baseline_weight = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        inputs = self.data["input_ids"][:, 1:].double() / 100
        baseline = (
            torch.nn.functional.softplus(inputs * baseline_weight) * mask
        ).sum() / normalizer
        baseline.backward()
        weight = baseline_weight.detach().clone().requires_grad_()
        total = weight * 0
        for lane, plan in enumerate(p for dp in dispatch.plans for p in dp):
            payload = dispatch.data[lane // 2][lane % 2]
            step = plan.steps[0]
            self.assertEqual(step.valid_tokens, normalizer.item())
            self.assertEqual(step.valid_sequences, 4)
            nmb = len(step.assignments)
            for task in step.assignments:
                if not task.sample_indices:
                    continue
                local = payload.select_indices(list(task.sample_indices))
                local_mask = local["token_mask"][:, 1:] * local["sample_mask"][:, None]
                loss = (
                    torch.nn.functional.softplus(
                        local["input_ids"][:, 1:].double() / 100 * weight
                    )
                    * local_mask
                ).sum() / step.valid_tokens
                multiplier = cp_loss_multiplier(
                    active_cp_size=task.cp_size,
                    schedule_cp_size=2,
                    num_microbatches=nmb,
                    replicated_cp_loss=True,
                )
                total = total + loss * multiplier * 2 / nmb
        total.backward()
        torch.testing.assert_close(total, baseline)
        torch.testing.assert_close(weight.grad, baseline_weight.grad)

    def test_arbitrary_sample_permutations_restore_every_output(self):
        # A five-row reverse permutation is self-inverse and cannot detect an
        # accidental second argsort in BatchedDataDict.reorder_data.
        rng = Random(2026)
        for count in (9, 17, 64):
            for static_cp in (1, 2, 4):
                with self.subTest(count=count, static_cp=static_cp):
                    data = BatchedDataDict(
                        input_ids=torch.zeros(count, 32, dtype=torch.long),
                        input_lengths=torch.tensor(
                            [rng.randint(2, 32) for _ in range(count)]
                        ),
                        sample_ids=torch.arange(count),
                    )
                    mesh = NamedSharding(
                        np.arange(8).reshape(1, 4 // static_cp, static_cp, 2),
                        self.mesh.names,
                    )
                    dispatch = build_cp_dispatch(
                        data, self.cfg, mesh, batch_size=None, training=False
                    )
                    results = []
                    for payloads, plans in zip(dispatch.data, dispatch.plans):
                        for payload, plan in zip(payloads, plans):
                            values = []
                            for task in plan.steps[0].assignments:
                                values.extend(
                                    payload["sample_ids"][
                                        list(task.sample_indices)
                                    ].tolist()
                                    if task.sample_indices
                                    else [-1]
                                )
                            results.append(
                                BatchedDataDict(logprobs=torch.tensor(values)[:, None])
                            )
                    restored = collect_cp_outputs(results, dispatch, count)
                    torch.testing.assert_close(
                        restored["logprobs"].flatten(), torch.arange(count)
                    )

    def test_steps_keep_separate_denominators(self):
        data = BatchedDataDict.from_batches([self.data, self.data])
        data["sample_mask"][5:] = 0
        dispatch = build_cp_dispatch(
            data, self.cfg, self.mesh, batch_size=5, training=True
        )
        for dp in dispatch.plans:
            for plan in dp:
                self.assertGreater(plan.steps[0].valid_tokens, 0)
                self.assertEqual(plan.steps[1].valid_tokens, 0)

    def test_dispatch_preserves_uneven_packed_task_lists(self):
        data = BatchedDataDict(
            input_ids=torch.arange(5 * 12).reshape(5, 12),
            input_lengths=torch.tensor([10, 10, 10, 10, 10]),
            sample_ids=torch.arange(5),
        )
        data["sample_mask"] = torch.ones(5, dtype=torch.long)
        data["token_mask"] = torch.ones(5, 12, dtype=torch.long)
        cfg = {**self.cfg, "megatron_cfg": dict(self.cfg["megatron_cfg"])}
        cfg["megatron_cfg"]["dynamic_context_parallel"] = {
            "enabled": True,
            "tokens_per_rank": 10,
            "max_size": 1,
        }
        dispatch = build_cp_dispatch(data, cfg, self.mesh, batch_size=5, training=True)
        local_counts = [
            len(plan.steps[0].assignments)
            for dp_plans in dispatch.plans
            for plan in dp_plans
        ]
        self.assertEqual(sorted(local_counts), [1, 1, 1, 2])
        unique_real_tasks = sum(
            1
            for group in dispatch.schedule.groups_by_batch[0]
            for task in group.assignments
            if task.sample_indices
        )
        self.assertEqual(
            sum(
                owned_real_task_count(plan)
                for dp_plans in dispatch.plans
                for plan in dp_plans
            ),
            unique_real_tasks,
        )
        real_task_participations = sum(
            task.cp_size
            for group in dispatch.schedule.groups_by_batch[0]
            for task in group.assignments
            if task.sample_indices
        )
        self.assertEqual(
            sum(
                real_task_participation_count(plan)
                for dp_plans in dispatch.plans
                for plan in dp_plans
            ),
            real_task_participations,
        )
        self.assertTrue(
            all(
                len(plan.steps[0].groups) == 1
                for dp_plans in dispatch.plans
                for plan in dp_plans
            )
        )
        worker_outputs = []
        for payloads, plans in zip(dispatch.data, dispatch.plans):
            for payload, plan in zip(payloads, plans):
                values = []
                for task in plan.steps[0].assignments:
                    values.extend(
                        payload["sample_ids"][list(task.sample_indices)].tolist()
                        if task.sample_indices
                        else [-1]
                    )
                worker_outputs.append(
                    BatchedDataDict(logprobs=torch.tensor(values)[:, None])
                )
        restored = collect_cp_outputs(worker_outputs, dispatch, data.size)
        torch.testing.assert_close(restored["logprobs"].flatten(), torch.arange(5))

        baseline_weight = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        inputs = data["input_ids"][:, 1:].double() / 100
        baseline = torch.nn.functional.softplus(inputs * baseline_weight).sum() / 55
        baseline.backward()

        uneven_weight = baseline_weight.detach().clone().requires_grad_()
        uneven_loss = uneven_weight * 0
        for payloads, plans in zip(dispatch.data, dispatch.plans):
            for payload, plan in zip(payloads, plans):
                step = plan.steps[0]
                num_local_tasks = len(step.assignments)
                for task in step.assignments:
                    if not task.sample_indices:
                        continue
                    local = payload.select_indices(list(task.sample_indices))
                    local_loss = (
                        torch.nn.functional.softplus(
                            local["input_ids"][:, 1:].double() / 100 * uneven_weight
                        ).sum()
                        / step.valid_tokens
                    )
                    multiplier = cp_loss_multiplier(
                        active_cp_size=task.cp_size,
                        schedule_cp_size=2,
                        num_microbatches=num_local_tasks,
                        replicated_cp_loss=True,
                    )
                    # MCore PP=1 applies static_CP / local_microbatch_count.
                    uneven_loss = (
                        uneven_loss + local_loss * multiplier * 2 / num_local_tasks
                    )
        uneven_loss.backward()
        torch.testing.assert_close(uneven_loss, baseline)
        torch.testing.assert_close(uneven_weight.grad, baseline_weight.grad)

    def test_score_and_train_reuse_one_schedule(self):
        with patch(
            "nemo_rl.models.policy.dynamic_cp.plan_cp_phases",
            wraps=plan_cp_phases,
        ) as planner:
            score = build_cp_dispatch(
                self.data,
                self.cfg,
                self.mesh,
                batch_size=5,
                training=False,
            )
            train = build_cp_dispatch(
                self.data,
                self.cfg,
                self.mesh,
                batch_size=5,
                training=True,
                schedule=score.schedule,
            )

        self.assertEqual(planner.call_count, 1)
        self.assertIs(train.schedule, score.schedule)
        self.assertTrue(
            cp_schedule_matches(
                score.schedule,
                self.data,
                self.cfg,
                self.mesh,
                batch_size=5,
            )
        )
        for score_dp, train_dp in zip(score.plans, train.plans):
            for score_plan, train_plan in zip(score_dp, train_dp):
                self.assertEqual(
                    score_plan.steps[0].assignments,
                    train_plan.steps[0].assignments,
                )
                self.assertEqual(score_plan.steps[0].valid_tokens, 0)
                self.assertGreater(train_plan.steps[0].valid_tokens, 0)

    def test_schedule_reuse_rejects_different_batch_boundaries(self):
        score = build_cp_dispatch(
            self.data,
            self.cfg,
            self.mesh,
            batch_size=5,
            training=False,
        )
        self.assertFalse(
            cp_schedule_matches(
                score.schedule,
                self.data,
                self.cfg,
                self.mesh,
                batch_size=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
