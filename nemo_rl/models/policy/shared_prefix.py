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
"""Config rules of the ``policy.sequence_packing.share_prefixes`` packing option.

With the option on, a forward runs a prompt once for all the rows that share it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from nemo_rl.algorithms import opd as opd_module
from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    need_top_k_or_top_p_filtering,
)
from nemo_rl.models.policy.draft_config import coerce_draft_config

if TYPE_CHECKING:
    from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
    from nemo_rl.models.policy import MegatronConfig, PolicyConfig

_OPTION = "policy.sequence_packing.share_prefixes"


def validate_shared_prefix_config(master_config: MasterConfig) -> None:
    """Reject a config that ``policy.sequence_packing.share_prefixes`` cannot run with.

    Does nothing when the option is off.

    Args:
        master_config: The single-controller config.

    Raises:
        ValueError: If the option is on and another setting does not meet its requirements.
    """
    policy_config = master_config.policy
    sequence_packing_config = policy_config.get("sequence_packing")
    if sequence_packing_config is None or not sequence_packing_config.get(
        "share_prefixes"
    ):
        return

    if not sequence_packing_config["enabled"]:
        raise ValueError(f"{_OPTION} requires policy.sequence_packing.enabled=true.")
    for key in ("pair_grouping_key", "max_sequences_per_bin"):
        if sequence_packing_config.get(key) is not None:
            raise ValueError(
                f"{_OPTION} requires policy.sequence_packing.{key} to be unset."
            )
    if policy_config["dynamic_batching"]["enabled"]:
        raise ValueError(f"{_OPTION} requires policy.dynamic_batching.enabled=false.")

    megatron_config = policy_config.get("megatron_cfg")
    if megatron_config is None or not megatron_config["enabled"]:
        raise ValueError(f"{_OPTION} requires policy.megatron_cfg.enabled=true.")
    megatron_config = cast("MegatronConfig", megatron_config)
    if megatron_config["pipeline_model_parallel_size"] != 1:
        raise ValueError(
            f"{_OPTION} requires policy.megatron_cfg.pipeline_model_parallel_size=1."
        )
    peft_config = megatron_config.get("peft")
    if peft_config is not None and peft_config["enabled"]:
        raise ValueError(f"{_OPTION} requires policy.megatron_cfg.peft.enabled=false.")
    for key in (
        "use_fused_linear_logprobs",
        "moe_hybridep_prepad_packed_inputs",
        "zero_train_gen_mismatch",
        "batch_invariant_mode",
    ):
        if megatron_config.get(key):
            raise ValueError(f"{_OPTION} requires policy.megatron_cfg.{key}=false.")

    generation_config = policy_config["generation"]
    if need_top_k_or_top_p_filtering(
        TrainingSamplingParams(
            top_k=generation_config["top_k"], top_p=generation_config["top_p"]
        )
    ):
        raise ValueError(
            f"{_OPTION} requires policy.generation.top_k=null and "
            "policy.generation.top_p=1.0."
        )
    # Deferred: router_replay imports PolicyConfig from this package (circular import).
    from nemo_rl.models.megatron.router_replay import router_replay_enabled

    if router_replay_enabled(policy_config):
        raise ValueError(f"{_OPTION} requires policy.router_replay.enabled=false.")
    if policy_config.get("is_vlm"):
        raise ValueError(f"{_OPTION} requires policy.is_vlm=false.")
    draft_config = coerce_draft_config(policy_config.get("draft"))
    if draft_config is not None and draft_config.enabled:
        raise ValueError(f"{_OPTION} requires policy.draft.enabled=false.")

    if not master_config.token_capture.enabled:
        raise ValueError(
            f"{_OPTION} requires token_capture.enabled=true: the rollout reassembler "
            "records the prompt length and the prompt hash of every row."
        )
    if master_config.loss_fn.score_centering:
        raise ValueError(f"{_OPTION} requires loss_fn.score_centering=false.")
    if master_config.loss_fn.seq_logprob_error_in_loss:
        raise ValueError(f"{_OPTION} requires loss_fn.seq_logprob_error_in_loss=false.")
    if master_config.loss_fn.positive_example_nll_weight != 0:
        raise ValueError(f"{_OPTION} requires loss_fn.positive_example_nll_weight=0.")
    if opd_module.get_opd_full_config(master_config) is not None:
        raise ValueError(
            f"{_OPTION} requires on_policy_distillation.full.enabled=false."
        )


def bound_by_full_length(policy_config: PolicyConfig, *, budget_key: str) -> bool:
    """Whether forwards packed with ``budget_key`` must fit the budget at full length.

    Rows that share a prompt have a packed length, which counts the prompt once, and a
    full length, which counts it in every row. In a forward with shared prompts,
    Megatron's multi-token prediction (MTP) runs only in training, and it processes
    every row at its full length.

    Args:
        policy_config: The policy config, with the Megatron backend enabled.
        budget_key: The ``policy.sequence_packing`` token budget the caller packs with:
            ``"train_mb_tokens"`` or ``"logprob_mb_tokens"``.

    Returns:
        False for the logprob budget. For the train budget, False only when
        ``policy.megatron_cfg.mtp_num_layers`` is 0.
    """
    if budget_key == "logprob_mb_tokens":
        return False
    # An absent key leaves the number of MTP layers to the checkpoint.
    return policy_config["megatron_cfg"].get("mtp_num_layers") != 0
