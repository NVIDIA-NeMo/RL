"""Tests for the RL metrics -> nemo-lens tee."""

import pytest

from nemo_rl.telemetry.metrics import map_rl_metrics


def test_map_basic_keys():
    mapped = map_rl_metrics(
        {
            "reward": 0.5,
            "loss": 1.0,
            "grad_norm": 2.0,
            "lr": 1e-4,
            "valid_tokens_per_sec_per_gpu": 100.0,
            "mean_gen_tokens_per_sample": 64,
        }
    )
    assert mapped["reward_mean"] == 0.5
    assert mapped["policy_loss"] == 1.0
    assert mapped["grad_norm"] == 2.0
    assert mapped["learning_rate"] == 1e-4
    assert mapped["tokens_per_sec"] == 100.0
    assert mapped["response_length_mean"] == 64.0


def test_map_first_candidate_wins():
    # "loss" is the first candidate for policy_loss.
    mapped = map_rl_metrics({"loss": 1.0, "policy_loss": 2.0})
    assert mapped["policy_loss"] == 1.0


def test_map_skips_bool_and_non_numeric():
    mapped = map_rl_metrics({"reward": True, "loss": "nan", "entropy": 0.1})
    assert "reward_mean" not in mapped
    assert "policy_loss" not in mapped
    assert mapped["entropy"] == 0.1


def test_map_empty_for_unknown_keys():
    assert map_rl_metrics({"unrelated_metric": 1.0}) == {}


def test_tee_emits_only_for_train_prefix_when_exporting():
    pytest.importorskip("nemo.lens")
    import nemo_rl.telemetry.setup as setup_mod
    from nemo.lens import NemoLensConfig, setup_telemetry
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from nemo_rl.telemetry.metrics import tee_rl_metrics_to_otel
    from nemo_rl.telemetry.span_groups import RLSpanGroup

    reader = InMemoryMetricReader()
    cfg = NemoLensConfig(enabled=True, _span_group_cls=RLSpanGroup)
    setup_mod._TELEMETRY_HANDLE = setup_telemetry(
        cfg, rank=0, world_size=1, metric_reader=reader
    )

    tee_rl_metrics_to_otel({"reward": 0.5, "grad_norm": 1.0}, "train")
    tee_rl_metrics_to_otel({"reward": 9.0}, "validation")  # wrong prefix -> ignored

    names = {
        metric.name
        for rm in reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
    }
    assert "rl.reward.mean" in names
    assert "rl.grad_norm" in names


def test_tee_noop_when_not_exporting():
    # No telemetry handle set -> must be a silent no-op (no exception).
    from nemo_rl.telemetry.metrics import tee_rl_metrics_to_otel

    tee_rl_metrics_to_otel({"reward": 0.5}, "train")
