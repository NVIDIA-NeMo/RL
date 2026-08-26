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
"""finalize_draft_ratio_metrics must reproduce the correct token-weighted
global ratio from num/den pairs summed across microbatches/DP ranks (the
training loop's default per-key metric reduction sums everything not on its
small mean-reduction allowlist) — not the wrong value a pre-divided
per-microbatch ratio would sum into.
"""

from nemo_rl.models.automodel.draft.integration import finalize_draft_ratio_metrics


def test_finalize_ratio_metrics_reproduces_token_weighted_global_ratio():
    # Two microbatches with very different token counts: a naive mean of the
    # two per-microbatch ratios (0.9, 0.1) would give 0.5, but the correct
    # token-weighted global ratio is (90 + 10) / (100 + 100) = 0.5 here by
    # coincidence -- use unequal totals below to distinguish the two.
    mb1_num, mb1_den = 90.0, 100.0  # ratio 0.9, small microbatch
    mb2_num, mb2_den = 10.0, 900.0  # ratio ~0.011, large microbatch
    naive_mean_of_ratios = (mb1_num / mb1_den + mb2_num / mb2_den) / 2
    correct_global_ratio = (mb1_num + mb2_num) / (mb1_den + mb2_den)
    assert abs(naive_mean_of_ratios - correct_global_ratio) > 0.1

    # Simulates grpo.py's per-key np.sum over all_mb_metrics after two
    # microbatches contributed draft_full_acc_num@0/draft_full_acc_den@0.
    metrics = {
        "draft_full_acc_num@0": mb1_num + mb2_num,
        "draft_full_acc_den@0": mb1_den + mb2_den,
        "draft_loss": 7.5,  # untouched: no matching _num/_den pair
    }
    finalize_draft_ratio_metrics(metrics)

    assert "draft_full_acc_num@0" not in metrics
    assert "draft_full_acc_den@0" not in metrics
    assert metrics["draft_full_acc@0"] == correct_global_ratio
    assert metrics["draft_loss"] == 7.5


def test_finalize_ratio_metrics_zero_den_yields_zero_not_nan():
    metrics = {"draft_tau_num": 0.0, "draft_tau_den": 0.0}
    finalize_draft_ratio_metrics(metrics)
    assert metrics["draft_tau"] == 0.0


def test_finalize_ratio_metrics_leaves_unpaired_num_alone():
    # A _num with no matching _den (e.g. a metric that legitimately ends in
    # "_num" for other reasons) must not be touched.
    metrics = {"some_other_num": 3.0}
    finalize_draft_ratio_metrics(metrics)
    assert metrics == {"some_other_num": 3.0}
