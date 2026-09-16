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
    build_cp_dispatch,
    collect_cp_outputs,
    cp_schedule_matches,
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
