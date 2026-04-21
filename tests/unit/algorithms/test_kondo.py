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

import torch

from nemo_rl.algorithms.kondo import (
    apply_kondo_mode,
    compute_required_row_multiple,
    resolve_kondo_config,
    screen_kondo_tokens,
)
from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _make_train_data() -> BatchedDataDict:
    return BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [
                    [1, 11, 12, 13],
                    [1, 21, 22, 23],
                    [1, 31, 32,  0],
                ],
                dtype=torch.long,
            ),
            "token_mask": torch.tensor(
                [
                    [0.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0],
                ]
            ),
            "sample_mask": torch.tensor([1.0, 1.0, 1.0]),
            "advantages": torch.tensor(
                [
                    [0.0, 3.0, 3.0, 3.0],
                    [0.0, 1.0, 1.0, 1.0],
                    [0.0, 2.0, 2.0, 0.0],
                ]
            ),
            "prev_logprobs": torch.tensor(
                [
                    [0.0, -4.0, -3.0, -2.0],
                    [0.0, -1.0, -1.0, -1.0],
                    [0.0, -2.0, -2.0,  0.0],
                ]
            ),
            "generation_logprobs": torch.tensor(
                [
                    [0.0, -4.0, -3.0, -2.0],
                    [0.0, -1.0, -1.0, -1.0],
                    [0.0, -2.0, -2.0,  0.0],
                ]
            ),
        }
    )


def test_compute_required_row_multiple_respects_batching_mode():
    assert compute_required_row_multiple(2, 4, False, False) == 8
    assert compute_required_row_multiple(2, 4, True, False) == 2
    assert compute_required_row_multiple(2, 4, False, True) == 2


def test_screen_kondo_tokens_builds_exact_dense_reference_gate():
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "dense_reference",
            "target_backward_token_fraction": 0.5,
            "priority_mode": "delight",
        }
    )
    screening = screen_kondo_tokens(_make_train_data(), cfg)

    assert screening is not None
    assert screening.total_valid_tokens == 8
    assert screening.screenable_token_count == 8
    assert screening.reference_token_gate.sum().item() == 4.0
    assert screening.loss_token_mask.shape == torch.Size([3, 4])
    assert screening.loss_token_mask[:, 0].sum().item() == 0.0


def test_apply_kondo_mode_dense_reference_preserves_full_batch():
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "dense_reference",
            "target_backward_token_fraction": 0.5,
        }
    )
    train_data = _make_train_data()
    screening = screen_kondo_tokens(train_data, cfg)
    policy_train_data, metrics = apply_kondo_mode(
        train_data=train_data,
        screening=screening,
        cfg=cfg,
        row_multiple=1,
    )

    assert policy_train_data["input_ids"].shape[0] == 3
    assert metrics["kondo_rows_total"] == 3.0
    assert metrics["kondo_rows_kept"] == 3.0
    assert metrics["kondo_selected_token_recall"] == 1.0
    assert metrics["kondo_actual_backward_token_fraction"] == 1.0
    assert metrics["kondo_kept_row_token_fraction"] == 1.0
    assert "loss_token_mask" in train_data


def test_apply_kondo_mode_routed_compacts_rows_under_alignment_constraints():
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "routed",
            "target_backward_token_fraction": 0.25,
            "priority_mode": "delight",
            "min_selected_rows": 1,
        }
    )
    train_data = _make_train_data()
    screening = screen_kondo_tokens(train_data, cfg)
    policy_train_data, metrics = apply_kondo_mode(
        train_data=train_data,
        screening=screening,
        cfg=cfg,
        row_multiple=2,
    )

    assert policy_train_data["input_ids"].shape[0] == 2
    assert metrics["kondo_rows_kept"] == 2.0
    assert metrics["kondo_rows_total"] == 3.0
    assert metrics["kondo_selected_token_recall"] == 1.0
    assert metrics["kondo_actual_backward_token_fraction"] == 0.25
    assert metrics["kondo_kept_row_token_fraction"] == 0.75
    assert train_data["kondo_row_selected"].sum().item() == 2.0


def test_apply_kondo_mode_response_dense_rows_v2_preserves_teacher_recall():
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "response_dense_rows_v2",
            "target_backward_token_fraction": 0.25,
            "recall_floor": 1.0,
            "priority_mode": "delight",
            "min_selected_rows": 1,
        }
    )
    train_data = _make_train_data()
    screening = screen_kondo_tokens(train_data, cfg)
    policy_train_data, metrics = apply_kondo_mode(
        train_data=train_data,
        screening=screening,
        cfg=cfg,
        row_multiple=2,
    )

    assert screening is not None
    assert policy_train_data["input_ids"].shape[0] == 2
    assert metrics["kondo_rows_kept"] == 2.0
    assert metrics["kondo_selected_token_recall"] == 1.0
    assert metrics["kondo_selected_token_fraction"] == 0.25
    assert metrics["kondo_actual_backward_token_fraction"] == 0.625
    assert metrics["kondo_kept_row_token_fraction"] == 0.625
    assert metrics["kondo_recall_floor"] == 1.0
    assert metrics["kondo_recall_floor_satisfied"] == 1.0
    torch.testing.assert_close(
        policy_train_data["loss_token_mask"][:, 1:].sum(dim=-1),
        torch.tensor([2.0, 0.0]),
    )


def test_response_dense_rows_v2_zero_cost_alignment_filler_does_not_force_dense():
    train_data = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [
                    [1, 11, 12],
                    [1,  0,  0],
                ],
                dtype=torch.long,
            ),
            "token_mask": torch.tensor(
                [
                    [0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "sample_mask": torch.tensor([1.0, 1.0]),
            "advantages": torch.tensor(
                [
                    [0.0, 3.0, 3.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "prev_logprobs": torch.tensor(
                [
                    [0.0, -4.0, -3.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "generation_logprobs": torch.tensor(
                [
                    [0.0, -4.0, -3.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        }
    )
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "response_dense_rows_v2",
            "target_backward_token_fraction": 0.5,
            "recall_floor": 1.0,
            "priority_mode": "delight",
            "min_selected_rows": 1,
        }
    )
    screening = screen_kondo_tokens(train_data, cfg)
    policy_train_data, metrics = apply_kondo_mode(
        train_data=train_data,
        screening=screening,
        cfg=cfg,
        row_multiple=2,
    )

    assert policy_train_data["input_ids"].shape[0] == 2
    assert metrics["kondo_bypass_step"] == 0.0
    assert metrics["kondo_selected_token_recall"] == 1.0
    assert metrics["kondo_actual_backward_token_fraction"] == 1.0
    assert train_data["kondo_row_selected"].tolist() == [1.0, 1.0]


def test_kondo_bypasses_when_nonfinite_screen_tokens_are_present():
    cfg = resolve_kondo_config(
        {
            "enabled": True,
            "mode": "routed",
            "target_backward_token_fraction": 0.5,
            "bypass_on_nonfinite": True,
        }
    )
    train_data = _make_train_data()
    train_data["prev_logprobs"][0, 1] = float("-inf")
    screening = screen_kondo_tokens(train_data, cfg)
    policy_train_data, metrics = apply_kondo_mode(
        train_data=train_data,
        screening=screening,
        cfg=cfg,
        row_multiple=1,
    )

    assert screening is not None
    assert screening.should_bypass is True
    assert policy_train_data["input_ids"].shape[0] == 3
    assert metrics["kondo_bypass_step"] == 1.0
    assert metrics["kondo_nonfinite_screen_token_count"] == 1.0


def test_clipped_pg_loss_respects_loss_token_mask_and_normalizer():
    cfg = {
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.2,
        "ratio_clip_c": None,
        "disable_ppo_ratio": True,
        "reference_policy_kl_penalty": 0.0,
        "reference_policy_kl_type": "k3",
        "kl_input_clamp_value": 20.0,
        "kl_output_clamp_value": 10.0,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "truncated_importance_sampling_ratio": None,
        "sequence_level_importance_ratios": False,
        "token_level_loss": True,
        "force_on_policy_ratio": False,
    }
    loss_fn = ClippedPGLossFn(cfg)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "advantages": torch.tensor([[0.0, 2.0, 2.0, 2.0]]),
            "prev_logprobs": torch.tensor([[0.0, -1.0, -2.0, -3.0]]),
            "generation_logprobs": torch.tensor([[0.0, -1.0, -2.0, -3.0]]),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
            "loss_token_mask": torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
            "loss_normalizer": torch.tensor([4.0]),
            "sample_mask": torch.tensor([1.0]),
        }
    )
    next_token_logprobs = torch.tensor([[-1.0, -2.0, -3.0]])

    loss, _ = loss_fn(
        next_token_logprobs=next_token_logprobs,
        data=data,
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(3.0),
    )

    torch.testing.assert_close(loss, torch.tensor(2.0))
