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
import pytest
import torch
from pydantic import ValidationError

from nemo_rl.algorithms.single_controller_utils.pass_rate_filter import (
    PassRateFilterConfig,
    PassRateFilterStats,
    pass_rate_filter_config,
    pass_rate_filter_weights,
)


def _batch(passes_per_group, group_size=16, prompt_len=5):
    """Canonical rows only: group g has passes_per_group[g] rewards of 1, the rest 0."""
    prompts, rewards = [], []
    for g, k in enumerate(passes_per_group):
        for i in range(group_size):
            prompts.append([g + 1] * prompt_len)
            rewards.append(1.0 if i < k else 0.0)
    n = len(rewards)
    return dict(
        prompt_ids=torch.tensor(prompts),
        rewards=torch.tensor(rewards),
        vote_mask=torch.ones(n),
        canonical_of_row=torch.arange(n),
        train_mask=torch.ones(n, 4),
    )


def test_default_band_keeps_3_to_9_of_16():
    cfg = PassRateFilterConfig(enabled=True)
    w, stats = pass_rate_filter_weights(**_batch([0, 2, 3, 9, 10, 16]), config=cfg)
    per_group = w.reshape(6, 16)
    assert torch.equal(per_group[:, 0], torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))
    assert (per_group == per_group[:, :1]).all()  # whole groups
    assert stats.groups == 6 and stats.groups_below == 2 and stats.groups_above == 2
    m = stats.to_metrics()
    assert m["pass_rate_filter/group_share_in_band"] == pytest.approx(2 / 6)
    assert m["pass_rate_filter/trained_token_share_in_band"] == pytest.approx(2 / 6)


def test_out_of_band_weight_and_bounds_inclusive():
    cfg = PassRateFilterConfig(enabled=True, lower=3 / 16, upper=9 / 16, out_of_band_weight=0.3)
    w, _ = pass_rate_filter_weights(**_batch([2, 3, 9, 10]), config=cfg)
    assert torch.allclose(w.reshape(4, 16)[:, 0], torch.tensor([0.3, 1.0, 1.0, 0.3]))


def test_pass_threshold_counts_share_at_or_above():
    b = _batch([0, 0])
    b["rewards"] = torch.tensor([0.4] * 12 + [0.6] * 4 + [0.9] * 16)  # group 0: mean 0.45, 4/16 >= 0.5
    w_mean, _ = pass_rate_filter_weights(**b, config=PassRateFilterConfig(enabled=True))
    w_thr, _ = pass_rate_filter_weights(**b, config=PassRateFilterConfig(enabled=True, pass_threshold=0.5))
    assert w_mean[0] == 1.0 and w_mean[16] == 0.0  # 0.45 in band; 0.9 above
    assert w_thr[0] == 1.0 and w_thr[16] == 0.0  # 0.25 in band; 1.0 above


def test_non_voting_rollouts_do_not_count():
    b = _batch([5])  # 5 / 16 passes
    b["vote_mask"][5:13] = 0  # 8 failing rollouts do not vote -> 5 / 8 = 0.625 > 0.6
    w, stats = pass_rate_filter_weights(**b, config=PassRateFilterConfig(enabled=True))
    assert (w == 0.0).all() and stats.groups_above == 1


def test_segment_rows_follow_their_canonical_row_and_pad_rows_are_neutral():
    b = _batch([0, 5])  # group 0 out of band, group 1 in band
    n = b["rewards"].shape[0]
    # one segment row for rollout 0 (group 0) and rollout 16 (group 1), plus one DP pad row
    b["prompt_ids"] = torch.cat([b["prompt_ids"], b["prompt_ids"][[0, 16]], b["prompt_ids"][[0]]])
    b["rewards"] = torch.cat([b["rewards"], b["rewards"][[0, 16]], torch.zeros(1)])
    b["vote_mask"] = torch.cat([b["vote_mask"], torch.zeros(3)])
    b["canonical_of_row"] = torch.cat([b["canonical_of_row"], torch.tensor([0, 16, -1])])
    b["train_mask"] = torch.cat([b["train_mask"], torch.ones(2, 4), torch.zeros(1, 4)])
    w, stats = pass_rate_filter_weights(**b, config=PassRateFilterConfig(enabled=True))
    assert w[n] == 0.0 and w[n + 1] == 1.0 and w[n + 2] == 1.0
    assert stats.groups == 2


def test_config_parsing_and_validation():
    class Algo:
        pass

    a = Algo()
    assert pass_rate_filter_config(a) is None
    a.pass_rate_filter = {"enabled": False}
    assert pass_rate_filter_config(a) is None
    a.pass_rate_filter = {"enabled": True}
    cfg = pass_rate_filter_config(a)
    assert cfg.lower == 0.15 and cfg.upper == 0.6 and cfg.out_of_band_weight == 0.0
    with pytest.raises(ValidationError):
        PassRateFilterConfig(enabled=True, lower=0.7, upper=0.6)
    with pytest.raises(ValidationError):
        pass_rate_filter_config(type("A", (), {"pass_rate_filter": {"enabled": True, "uper": 0.5}})())


def test_stats_merge():
    s = PassRateFilterStats(groups=2, groups_below=1, pass_rate_sum=0.5, trained_tokens=10, trained_tokens_in_band=4)
    s.merge(PassRateFilterStats(groups=2, groups_above=1, pass_rate_sum=1.0, trained_tokens=10, trained_tokens_in_band=6))
    m = s.to_metrics()
    assert m["pass_rate_filter/groups"] == 4 and m["pass_rate_filter/group_share_in_band"] == 0.5
    assert m["pass_rate_filter/trained_token_share_in_band"] == 0.5
    assert PassRateFilterStats().to_metrics() == {}
