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

import fnmatch

import pytest

# draft.utils imports megatron at module top, so every test here needs mcore.
pytestmark = pytest.mark.mcore


def test_build_provider_returns_none_when_no_draft_optimizer_settings():
    from nemo_rl.models.megatron.draft.utils import (
        build_draft_optimizer_override_provider,
    )

    assert (
        build_draft_optimizer_override_provider(
            {"enabled": True, "model_name": None, "lr": None}
        )
        is None
    )
    assert build_draft_optimizer_override_provider({"enabled": True}) is None


def test_draft_override_group_values_and_standard_overrides_preserved():
    from megatron.bridge.training.config import (
        OptimizerConfigOverrideProviderContext,
    )
    from megatron.core.optimizer import OptimizerConfig, ParamKey

    from nemo_rl.models.megatron.draft.utils import (
        build_draft_optimizer_override_provider,
    )

    provider = build_draft_optimizer_override_provider(
        {"enabled": True, "lr": 5e-5, "min_lr": None, "weight_decay": 0.0}
    )
    assert provider is not None

    optimizer_config = OptimizerConfig(optimizer="adam", lr=1e-6, min_lr=1e-7)
    context = OptimizerConfigOverrideProviderContext(
        scheduler_config=None, optimizer_config=optimizer_config, model=None
    )
    overrides = provider.build_config_overrides(context)

    draft_key = ParamKey(name="*draft_model.*")
    assert draft_key in overrides
    draft_override = overrides[draft_key]
    assert draft_override["max_lr"] == 5e-5
    # min_lr defaults to the draft lr when unset.
    assert draft_override["min_lr"] == 5e-5
    assert draft_override["start_wd"] == 0.0
    assert draft_override["end_wd"] == 0.0

    # The standard megatron-bridge overrides must survive the merge.
    from megatron.core.optimizer import get_standard_config_overrides

    standard = get_standard_config_overrides(config=optimizer_config) or {}
    for key in standard:
        assert key in overrides

    # The glob must match DDP-prefixed draft param names but not policy params.
    assert fnmatch.fnmatch(
        "module.module.draft_model.eagle_module.fc.weight", "*draft_model.*"
    )
    assert not fnmatch.fnmatch(
        "module.module.decoder.layers.0.mlp.linear_fc1.weight", "*draft_model.*"
    )


def _make_param_group(*, max_lr, min_lr, params):
    """A param group shaped like mcore's ``_get_param_groups`` output."""
    return {
        "wd_mult": 1.0,
        "lr_mult": 1.0,
        "is_expert_parallel": False,
        "is_decoupled_lr": False,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "params": params,
    }


def test_identifier_keys_extended_and_idempotent():
    from megatron.core import optimizer as mcore_optimizer_pkg
    from megatron.core.optimizer import distrib_optimizer as mcore_distrib_optimizer
    from megatron.core.optimizer import optimizer as mcore_optimizer

    from nemo_rl.models.megatron.draft.utils import (
        extend_param_group_identifier_keys_for_resume,
    )

    extend_param_group_identifier_keys_for_resume()
    extend_param_group_identifier_keys_for_resume()

    expected = (
        "wd_mult",
        "lr_mult",
        "is_expert_parallel",
        "is_decoupled_lr",
        "max_lr",
        "min_lr",
    )
    # Every module that consumes the tuple at load time must see the extension;
    # distrib_optimizer holds its own binding via ``from ... import``.
    for module in (mcore_optimizer, mcore_distrib_optimizer, mcore_optimizer_pkg):
        assert module.param_group_identifier_keys == expected


def test_resume_group_matching_keeps_policy_and_draft_hyperparams():
    """Regression test: resume must not conflate the policy and draft groups.

    The draft param group differs from the policy group only in
    max_lr/min_lr/wd endpoints, so under mcore's stock 4-key identity both
    groups hash to the same key and the policy group resumes with the draft
    hyperparameters (policy at 50x LR). ``DistributedOptimizer.
    load_state_dict`` builds an equivalent dict keyed by the same tuple, so
    this exercises the shared failure mechanism for both load paths.
    """
    from megatron.core.optimizer import optimizer as mcore_optimizer
    from megatron.core.optimizer.optimizer import MegatronOptimizer

    from nemo_rl.models.megatron.draft.utils import (
        extend_param_group_identifier_keys_for_resume,
    )

    extend_param_group_identifier_keys_for_resume()

    policy_group = _make_param_group(max_lr=1e-6, min_lr=1e-7, params=[0, 1])
    draft_group = _make_param_group(max_lr=5e-5, min_lr=5e-5, params=[2])
    current_groups = [policy_group, draft_group]
    saved_groups = [dict(policy_group), dict(draft_group)]

    # Sanity: under the stock 4-key identity the two groups collide.
    stock_keys = ("wd_mult", "lr_mult", "is_expert_parallel", "is_decoupled_lr")
    assert tuple(policy_group[k] for k in stock_keys) == tuple(
        draft_group[k] for k in stock_keys
    )

    reordered = MegatronOptimizer._filter_and_reorder_param_groups(
        current_groups, saved_groups
    )
    assert reordered[0]["max_lr"] == 1e-6
    assert reordered[0]["min_lr"] == 1e-7
    assert reordered[1]["max_lr"] == 5e-5
    assert reordered[1]["min_lr"] == 5e-5
    assert reordered[0]["params"] == [0, 1]
    assert reordered[1]["params"] == [2]

    # The extended identity must be collision-free over these groups, which is
    # what DistributedOptimizer.load_state_dict's dict-based mapping needs.
    extended_keys = mcore_optimizer.param_group_identifier_keys
    identity_tuples = {tuple(g[k] for k in extended_keys) for g in saved_groups}
    assert len(identity_tuples) == len(saved_groups)


def test_draft_override_indistinct_from_policy_is_rejected():
    from megatron.bridge.training.config import (
        OptimizerConfigOverrideProviderContext,
    )
    from megatron.core.optimizer import OptimizerConfig

    from nemo_rl.models.megatron.draft.utils import (
        build_draft_optimizer_override_provider,
    )

    optimizer_config = OptimizerConfig(optimizer="adam", lr=1e-6, min_lr=1e-7)
    context = OptimizerConfigOverrideProviderContext(
        scheduler_config=None, optimizer_config=optimizer_config, model=None
    )

    # weight_decay-only override: the draft group would keep the policy's
    # (max_lr, min_lr) and be indistinguishable at resume.
    wd_only_provider = build_draft_optimizer_override_provider(
        {"enabled": True, "lr": None, "min_lr": None, "weight_decay": 0.01}
    )
    assert wd_only_provider is not None
    with pytest.raises(ValueError, match="distinct from the policy"):
        wd_only_provider.build_config_overrides(context)

    # Draft lr/min_lr numerically equal to the policy's: same problem.
    same_lr_provider = build_draft_optimizer_override_provider(
        {"enabled": True, "lr": 1e-6, "min_lr": 1e-7, "weight_decay": 0.01}
    )
    assert same_lr_provider is not None
    with pytest.raises(ValueError, match="distinct from the policy"):
        same_lr_provider.build_config_overrides(context)

    # Draft lr equal to the policy lr but min_lr defaulting to the draft lr
    # (!= policy min_lr) keeps the pair distinct and is allowed.
    distinct_min_lr_provider = build_draft_optimizer_override_provider(
        {"enabled": True, "lr": 1e-6, "min_lr": None, "weight_decay": None}
    )
    overrides = distinct_min_lr_provider.build_config_overrides(context)
    assert overrides is not None
