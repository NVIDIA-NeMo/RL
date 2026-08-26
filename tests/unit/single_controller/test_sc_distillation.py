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
"""CPU tests for distillation on the SingleController path."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_rl.algorithms.distillation import DistillationConfig
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    algo_config,
    is_distillation_run,
    is_ppo_run,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import (
    DP_DISTILLATION_TRAIN_FIELDS,
    DP_TRAIN_FIELDS,
    TEACHER_TOPK_FIELDS,
)


def _meta() -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["s0", "s1"],
    )


def _teacher_ctrl(topk_logits_k: int = 8):
    """Bare actor carrying only what the teacher stage reads."""
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._is_distillation = True
    ctrl._algo_cfg = SimpleNamespace(topk_logits_k=topk_logits_k)
    ctrl._teacher = MagicMock()
    return ctrl


class TestTeacherStage:
    def test_loads_scores_then_offloads_and_records_both_columns(self):
        ctrl = _teacher_ctrl(topk_logits_k=16)
        meta = _meta()

        out = asyncio.run(ctrl._teacher_stage(meta))

        teacher = ctrl._teacher
        teacher.prepare_for_lp_inference.assert_called_once_with()
        teacher.get_topk_logits_from_meta.assert_called_once_with(meta, 16)
        teacher.offload_after_refit.assert_called_once_with()
        for field in TEACHER_TOPK_FIELDS:
            assert field in out.fields

    def test_offloads_after_scoring_not_before(self):
        """The teacher shares the training GPUs with the student. Offloading
        before the forward would score with a model that is not resident."""
        ctrl = _teacher_ctrl()
        calls: list[str] = []
        ctrl._teacher = SimpleNamespace(
            prepare_for_lp_inference=lambda: calls.append("load"),
            get_topk_logits_from_meta=lambda *a: calls.append("score"),
            offload_after_refit=lambda: calls.append("offload"),
        )

        asyncio.run(ctrl._teacher_stage(_meta()))

        assert calls == ["load", "score", "offload"]

    def test_without_a_teacher_it_says_so(self):
        ctrl = _teacher_ctrl()
        ctrl._teacher = None

        with pytest.raises(AssertionError, match="requires a teacher"):
            asyncio.run(ctrl._teacher_stage(_meta()))


class TestAlgorithmBlockValidation:
    """MasterConfig admits exactly one algorithm block, and `teacher` pairs
    with exactly one of them.

    Built from the shipped GRPO exemplar rather than a hand-rolled dict, so
    these keep testing the validators and not the other 30 required fields.
    """

    @staticmethod
    def _resolved() -> dict:
        from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

        register_omegaconf_resolvers()
        repo_root = Path(__file__).parents[3]
        raw = load_config(
            repo_root
            / "examples/configs"
            / "grpo_math_1B_megatron_single_controller.yaml"
        )
        resolved = OmegaConf.to_container(raw, resolve=True)
        assert isinstance(resolved, dict)
        return resolved

    def _as_distillation(self) -> dict:
        resolved = self._resolved()
        grpo = resolved.pop("grpo")
        resolved["distillation"] = {
            k: v
            for k, v in grpo.items()
            if k in DistillationConfig.model_fields or k == "seed"
        }
        resolved["distillation"]["topk_logits_k"] = 8
        resolved["teacher"] = resolved["policy"]
        resolved["loss_fn"] = {"kl_type": "mixed", "mixed_kl_weight": 0.5}
        return resolved

    def test_a_distillation_config_validates(self):
        cfg = MasterConfig(**self._as_distillation())
        assert is_distillation_run(cfg)
        assert not is_ppo_run(cfg)
        assert algo_config(cfg) is cfg.distillation

    def test_distillation_without_a_teacher_is_rejected(self):
        resolved = self._as_distillation()
        del resolved["teacher"]

        with pytest.raises(ValidationError, match="requires a `teacher`"):
            MasterConfig(**resolved)

    def test_a_teacher_without_distillation_is_rejected(self):
        """Quietly ignoring it is the failure mode this path already rejects
        for unsupported algorithm knobs."""
        resolved = self._resolved()
        resolved["teacher"] = resolved["policy"]

        with pytest.raises(ValidationError, match="only used by a `distillation` run"):
            MasterConfig(**resolved)

    def test_grpo_plus_distillation_is_rejected(self):
        resolved = self._as_distillation()
        resolved["grpo"] = self._resolved()["grpo"]

        with pytest.raises(ValidationError, match="Only one algorithm block"):
            MasterConfig(**resolved)


class TestTrainFieldSelection:
    def test_distillation_narrows_the_fetched_columns(self):
        """A distillation run writes no advantages and no logprobs, and
        fetching a column nobody wrote errors rather than reading zeros."""
        assert "advantages" in DP_TRAIN_FIELDS
        assert "advantages" not in DP_DISTILLATION_TRAIN_FIELDS
        assert "prev_logprobs" not in DP_DISTILLATION_TRAIN_FIELDS
        assert "reference_policy_logprobs" not in DP_DISTILLATION_TRAIN_FIELDS

    def test_the_teacher_columns_stay_out_of_the_default_schema(self):
        """Same reason PPO_VALUE_FIELDS are kept out: a GRPO run writes
        neither, so a worker fetching them would error."""
        for field in TEACHER_TOPK_FIELDS:
            assert field not in DP_TRAIN_FIELDS
            assert field in DP_DISTILLATION_TRAIN_FIELDS

    def test_the_loss_inputs_are_all_present(self):
        """DistillationLossDataDict names exactly what the loss reads."""
        from nemo_rl.algorithms.loss.loss_functions import DistillationLossDataDict

        for field in DistillationLossDataDict.__annotations__:
            assert field in DP_DISTILLATION_TRAIN_FIELDS, field
