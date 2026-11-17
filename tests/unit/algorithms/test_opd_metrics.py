import math

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import OPDAdvantageEstimator
from nemo_rl.algorithms.opd import OnPolicyDistillationConfig
from nemo_rl.algorithms.opd_metrics import compute_opd_metrics, teacher_labels

P = "on_policy_distillation"


def test_teacher_labels():
    mapping = {
        "math_agent": "/t/teachers/math/s200",
        "if_agent": "/t/teachers/if-fc/step_340",
        "fc_agent": "/t/teachers/if-fc/step_340",
        "brand_agent": "/t/teachers/branded",
    }
    assert teacher_labels(mapping) == {
        "math_agent": "math",
        "if_agent": "if-fc",
        "fc_agent": "if-fc",
        "brand_agent": "branded",
    }
    assert teacher_labels({"a": "/t/x/s1", "b": "/t/x/s1"}) == {"a": "s1", "b": "s1"}


def _batch():
    """3 samples: row 0 (teacher a, correct), row 1 (teacher b, wrong), row 2 (teacher a, overlong-filtered)."""
    token_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1]],
        dtype=torch.float32,
    )
    adv = torch.zeros(3, 6)
    adv[0, 2:5] = torch.tensor([0.0, -2.0, 0.005])
    adv[1, 1:3] = torch.tensor([1.0, -1.0])
    adv[2, 1:] = 3.0  # must not reach the KL metrics: sample_mask is 0
    return dict(
        teacher_lp=adv,  # student_lp = 0, so A = teacher_lp
        student_lp=torch.zeros(3, 6),
        token_mask=token_mask,
        sample_mask=torch.tensor([1.0, 1.0, 0.0]),
        teachers=["a", "b", "a"],
        rewards=torch.tensor([1.0, 0.0, 0.0]),
        truncated=torch.tensor([False, False, True]),
        gen_lp=torch.full((3, 6), -0.5),
        agents=["x", "y", "x"],
    )


def test_compute_opd_metrics_values():
    m = compute_opd_metrics(**_batch())
    a_adv = [0.0, -2.0, 0.005]
    assert m[f"{P}/all/k1_tok"] == pytest.approx(-(sum(a_adv) + 0.0) / 5)
    assert m[f"{P}/a/k1_tok"] == pytest.approx(-sum(a_adv) / 3)
    assert m[f"{P}/a/k1_seq"] == pytest.approx(-sum(a_adv) / 3)
    assert m[f"{P}/a/k3_tok"] == pytest.approx(
        sum(math.expm1(x) - x for x in a_adv) / 3
    )
    assert m[f"{P}/a/frac_tok_small"] == pytest.approx(2 / 3)
    assert m[f"{P}/a/frac_tok_neg"] == pytest.approx(1 / 3)
    assert m[f"{P}/a/frac_tok_big"] == 0.0
    assert m[f"{P}/a/tok_share"] == pytest.approx(3 / 5)
    # sum|A|: teacher a 2.005, teacher b 2.0
    assert m[f"{P}/a/budget_share"] == pytest.approx(2.005 / 4.005)
    assert m[f"{P}/b/budget_share"] == pytest.approx(2.0 / 4.005)
    assert m[f"{P}/all/budget_share"] == pytest.approx(1.0)
    assert m[f"{P}/a/student_nll"] == pytest.approx(0.5)
    assert m[f"{P}/a/teacher_xent"] == pytest.approx(-sum(a_adv) / 3)
    # length / truncation count the filtered sample, KL metrics don't
    assert m[f"{P}/a/n_samples"] == 2
    assert m[f"{P}/a/gen_len_mean"] == pytest.approx(4.0)
    assert m[f"{P}/a/trunc_rate"] == pytest.approx(0.5)
    # correctness: row 0 correct, row 2 wrong but has no loss tokens
    assert m[f"{P}/a/acc"] == pytest.approx(0.5)
    assert m[f"{P}/a/k1_tok_correct"] == pytest.approx(-sum(a_adv) / 3)
    assert f"{P}/a/k1_tok_wrong" not in m
    assert m[f"{P}/b/k1_tok_wrong"] == pytest.approx(0.0)
    assert m[f"{P}/a/k1_tok_pos_0-1k"] == pytest.approx(-sum(a_adv) / 3)
    # per agent: the light set only
    assert m[f"{P}/agent/x/k1_tok"] == pytest.approx(-sum(a_adv) / 3)
    assert m[f"{P}/agent/x/trunc_rate"] == pytest.approx(0.5)
    assert f"{P}/agent/x/k3_tok" not in m


def test_compute_opd_metrics_position_buckets():
    n = 5000
    adv = torch.full((1, n), -3.0)
    adv[0, :1024], adv[0, 1024:4096] = -1.0, -2.0
    m = compute_opd_metrics(
        teacher_lp=adv,
        student_lp=torch.zeros(1, n),
        token_mask=torch.ones(1, n),
        sample_mask=torch.ones(1),
        teachers=["t"],
    )
    assert m[f"{P}/t/k1_tok_pos_0-1k"] == pytest.approx(1.0)
    assert m[f"{P}/t/k1_tok_pos_1k-4k"] == pytest.approx(2.0)
    assert m[f"{P}/t/k1_tok_pos_4k-16k"] == pytest.approx(3.0)
    assert f"{P}/t/k1_tok_pos_16k-64k" not in m
    assert f"{P}/t/acc" not in m  # no rewards given


@pytest.mark.parametrize("enabled", [False, True])
def test_opd_estimator_detailed_metrics(enabled):
    b = _batch()
    opd_cfg = OnPolicyDistillationConfig(
        teacher_model_by_agent_name={
            "x": "/t/teachers/math/s1",
            "y": "/t/teachers/code/s2",
        },
        default_teacher_alias="x",
        detailed_metrics={"enabled": enabled},
    )
    estimator = OPDAdvantageEstimator({"name": "opd"}, {}, opd_config=opd_cfg)
    estimator.compute_advantage(
        prompt_ids=None,
        rewards=b["rewards"],
        mask=b["token_mask"] * b["sample_mask"].unsqueeze(-1),
        teacher_logprobs=b["teacher_lp"],
        prev_logprobs=b["student_lp"],
        repeated_batch={
            "agent_ref": [{"name": a} for a in b["agents"]],
            "truncated": b["truncated"],
        },
        token_mask=b["token_mask"],
        sample_mask=b["sample_mask"],
        generation_logprobs=b["gen_lp"],
    )
    keys = estimator.last_metrics
    assert f"{P}/teacher_student_logprob_gap_mean" in keys
    assert (f"{P}/math/k1_tok" in keys) is enabled
    assert (f"{P}/code/k1_tok" in keys) is enabled
    assert (f"{P}/agent/x/k1_tok" in keys) is enabled
