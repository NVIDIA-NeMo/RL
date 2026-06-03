# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Advantage-collapse demonstration: GRPO vs GDPO on multi-reward signals.

This is a tiny, CPU-only, no-training illustration of *why* GDPO exists. It feeds
crafted reward components through the real ``GRPOAdvantageEstimator`` and
``GDPOAdvantageEstimator`` and shows that GRPO maps responses with the same total
reward to identical advantages (the "advantage collapse" the GDPO paper describes,
https://arxiv.org/abs/2601.05242), while GDPO recovers their distinct composition.

Run:
    uv run examples/run_gdpo_grpo_advantage_demo.py
"""

import torch

from nemo_rl.algorithms.advantage_estimator import (
    GDPOAdvantageEstimator,
    GRPOAdvantageEstimator,
)


def main() -> None:
    # Four responses to ONE prompt, scored by two reward components with different
    # base rates (correctness is common, format is rarer -> different variance).
    #   A=(1,1)  B=(1,0)  C=(0,1)  D=(1,0)
    # B (correct, unformatted) and C (wrong, formatted) share the SAME total (1.0)
    # but are genuinely different responses.
    names = ["A=(1,1)", "B=(1,0)", "C=(0,1)", "D=(1,0)"]
    reward1 = torch.tensor([1.0, 1.0, 0.0, 1.0])  # correctness
    reward2 = torch.tensor([1.0, 0.0, 1.0, 0.0])  # format
    total_reward = reward1 + reward2  # what GRPO collapses everything into

    prompt_ids = torch.zeros(4, 1, dtype=torch.long)  # all same prompt group
    mask = torch.ones(4, 1)  # one response token each (advantages are per-sample here)

    cfg = {"use_leave_one_out_baseline": False, "normalize_rewards": True}

    grpo = GRPOAdvantageEstimator(cfg, loss_config=None)
    gdpo = GDPOAdvantageEstimator(cfg, loss_config=None)

    grpo_adv = grpo.compute_advantage(prompt_ids, total_reward, mask)[:, 0]
    gdpo_adv = gdpo.compute_advantage(
        prompt_ids,
        rewards=None,  # GDPO reads per-component keys instead
        mask=mask,
        repeated_batch={"reward1": reward1, "reward2": reward2},
    )[:, 0]

    print(f"total reward (what GRPO sees): {total_reward.tolist()}  <- B,C,D tie at 1.0\n")
    print(f"{'response':10} {'GRPO adv':>10} {'GDPO adv':>10}")
    for name, g, d in zip(names, grpo_adv.tolist(), gdpo_adv.tolist()):
        print(f"{name:10} {g:>10.3f} {d:>10.3f}")

    tied = [1, 2, 3]  # B, C, D all have total_reward == 1.0
    grpo_distinct = len({round(grpo_adv[i].item(), 4) for i in tied})
    gdpo_distinct = len({round(gdpo_adv[i].item(), 4) for i in tied})
    print(
        f"\ndistinct advantages among tied {{B,C,D}}  ->  GRPO: {grpo_distinct}  GDPO: {gdpo_distinct}"
    )

    # Assertions so this doubles as a regression test of the collapse behavior.
    assert torch.isclose(grpo_adv[1], grpo_adv[2]), "GRPO should collapse B and C"
    assert not torch.isclose(gdpo_adv[1], gdpo_adv[2]), "GDPO should separate B and C"
    assert torch.isclose(gdpo_adv[1], gdpo_adv[3]), "GDPO should keep identical B and D equal"
    print("\nOK: GRPO collapses B/C; GDPO separates them while keeping identical B/D equal.")


if __name__ == "__main__":
    main()
