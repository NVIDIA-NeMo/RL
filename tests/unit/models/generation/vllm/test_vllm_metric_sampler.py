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

prometheus_client = pytest.importorskip("prometheus_client")

from prometheus_client import CollectorRegistry, Counter, Gauge
from prometheus_client.core import GaugeMetricFamily

from nemo_rl.models.generation.vllm.vllm_metric_sampler import (
    VLLM_METRIC_FIELDS,
    metric_capture_timing,
    read_restricted_vllm_metrics,
)


def populated_registry() -> CollectorRegistry:
    registry = CollectorRegistry()
    labels = ["model_name", "engine"]
    label_values = {"model_name": "Qwen3-30B-A3B", "engine": "0"}
    Gauge(
        "vllm:num_requests_running",
        "running",
        labels,
        registry=registry,
    ).labels(**label_values).set(128)
    Gauge(
        "vllm:num_requests_waiting",
        "waiting",
        labels,
        registry=registry,
    ).labels(**label_values).set(7)
    Gauge(
        "vllm:kv_cache_usage_perc",
        "kv",
        labels,
        registry=registry,
    ).labels(**label_values).set(0.42)
    Counter(
        "vllm:generation_tokens",
        "tokens",
        labels,
        registry=registry,
    ).labels(**label_values).inc(1234)
    Counter(
        "vllm:num_preemptions",
        "preemptions",
        labels,
        registry=registry,
    ).labels(**label_values).inc(2)
    return registry


class UnrelatedExplodingCollector:
    """A full-registry scan would collect this unrelated expensive metric."""

    def describe(self):
        yield GaugeMetricFamily("unrelated_expensive_metric", "unrelated")

    def collect(self):
        raise AssertionError("unrelated collector must not be collected")


def test_restricted_reader_reads_only_required_series_with_source_labels() -> None:
    registry = populated_registry()
    registry.register(UnrelatedExplodingCollector())

    values, sources = read_restricted_vllm_metrics(registry)

    assert values == {
        "inflight_batch_size": 128,
        "num_pending": 7,
        "kv_cache_usage_perc": pytest.approx(0.42),
        "generation_tokens": 1234,
        "num_preemptions": 2,
    }
    assert set(sources) == VLLM_METRIC_FIELDS
    for source in sources.values():
        assert source["labels"] == {
            "model_name": "Qwen3-30B-A3B",
            "engine": "0",
        }
    assert sources["generation_tokens"]["sample_name"] == (
        "vllm:generation_tokens_total"
    )
    assert sources["num_preemptions"]["sample_name"] == (
        "vllm:num_preemptions_total"
    )


def test_restricted_reader_rejects_missing_required_series() -> None:
    registry = populated_registry()
    collector = registry._names_to_collectors["vllm:num_preemptions_total"]
    registry.unregister(collector)

    with pytest.raises(
        RuntimeError,
        match=r"fields=\['num_preemptions'\].*"
        r"vllm:num_preemptions_total",
    ):
        read_restricted_vllm_metrics(registry)


def test_restricted_reader_rejects_multiple_label_series() -> None:
    registry = populated_registry()
    running = registry._names_to_collectors["vllm:num_requests_running"]
    running.labels(model_name="Qwen3-30B-A3B", engine="1").set(64)

    with pytest.raises(
        RuntimeError,
        match="Multiple Prometheus series matched.*inflight_batch_size",
    ):
        read_restricted_vllm_metrics(registry)


def test_restricted_reader_has_no_full_registry_fallback() -> None:
    class RegistryWithoutRestrictedCollection:
        def collect(self):
            raise AssertionError("full registry must not be collected")

    with pytest.raises(RuntimeError, match="refusing to fall back"):
        read_restricted_vllm_metrics(RegistryWithoutRestrictedCollection())


def test_capture_timing_separates_wake_lock_and_snapshot_delay() -> None:
    timing = metric_capture_timing(
        interval_s=0.25,
        scheduled_at_monotonic_s=10.0,
        attempted_at_monotonic_s=10.5,
        started_at_monotonic_s=10.6,
        finished_at_monotonic_s=10.8,
    )

    assert timing["capture_duration_s"] == pytest.approx(0.2)
    assert timing["capture_lock_wait_s"] == pytest.approx(0.1)
    assert timing["capture_wake_lateness_s"] == pytest.approx(0.5)
    assert timing["capture_deadline_lateness_s"] == pytest.approx(0.6)
    assert timing["capture_finish_lateness_s"] == pytest.approx(0.8)
    assert timing["capture_missed_periods"] == 3


def test_synchronous_capture_has_no_deadline_or_missed_periods() -> None:
    timing = metric_capture_timing(
        interval_s=0.25,
        scheduled_at_monotonic_s=None,
        attempted_at_monotonic_s=20.0,
        started_at_monotonic_s=20.0,
        finished_at_monotonic_s=20.01,
    )

    assert timing["capture_scheduled_at_monotonic_s"] is None
    assert timing["capture_wake_lateness_s"] is None
    assert timing["capture_deadline_lateness_s"] is None
    assert timing["capture_finish_lateness_s"] is None
    assert timing["capture_missed_periods"] == 0
