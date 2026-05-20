import json

from nemo_rl.utils.dynamo_prometheus import (
    DynamoPrometheusMonitor,
    _resolve_metric_endpoints,
)


class _FailingRun:
    def log(self, *args, **kwargs):
        raise AssertionError("Dynamo Prometheus should not log to W&B")

    def define_metric(self, *args, **kwargs):
        raise AssertionError("Dynamo Prometheus should not define W&B metrics")

    def finish(self):
        raise AssertionError("Dynamo Prometheus should not own a W&B run")


class _FailingWandbLogger:
    run = _FailingRun()

    def define_metric(self, *args, **kwargs):
        raise AssertionError("Dynamo Prometheus should not define W&B metrics")


class _DummyLogger:
    def __init__(self):
        self.wandb_logger = _FailingWandbLogger()

    def log_metrics(self, *args, **kwargs):
        raise AssertionError("Dynamo Prometheus should not use logger.log_metrics")


def test_resolve_metric_endpoint_from_dgd_name():
    endpoints = _resolve_metric_endpoints(
        {"dgd_name": "jonas-swe", "namespace": "default"},
        {"service_names": ["vllmdecodeworker", "vllmprefillworker"]},
    )

    assert endpoints == [
        (
            "vllmdecodeworker",
            "http://jonas-swe-vllmdecodeworker.default.svc.cluster.local:9090/metrics",
        ),
        (
            "vllmprefillworker",
            "http://jonas-swe-vllmprefillworker.default.svc.cluster.local:9090/metrics",
        ),
    ]


def test_parse_dynamo_metrics_filters_and_adds_deltas():
    monitor = DynamoPrometheusMonitor(
        logger=_DummyLogger(),
        dynamo_cfg={"dgd_name": "jonas-swe"},
        prometheus_cfg={"endpoints": {"decode": "http://example/metrics"}},
    )
    first_scrape = """
# HELP dynamo_component_request_duration_seconds request duration
# TYPE dynamo_component_request_duration_seconds histogram
dynamo_component_request_duration_seconds_sum{dynamo_component="VllmDecodeWorker",dynamo_endpoint="generate",model="Qwen/Qwen2.5"} 4
dynamo_component_request_duration_seconds_count{dynamo_component="VllmDecodeWorker",dynamo_endpoint="generate",model="Qwen/Qwen2.5"} 2
dynamo_component_request_duration_seconds_bucket{dynamo_component="VllmDecodeWorker",dynamo_endpoint="generate",model="Qwen/Qwen2.5",le="1"} 1
dynamo_component_request_bytes_total{dynamo_component="VllmDecodeWorker",dynamo_endpoint="generate",model="Qwen/Qwen2.5"} 10
python_info{version="3.11"} 1
"""
    second_scrape = first_scrape.replace(" 4\n", " 7\n").replace(
        " 2\n", " 3\n"
    ).replace(" 10\n", " 15\n")

    first = monitor._parse_prometheus_text("decode", first_scrape)
    second = monitor._parse_prometheus_text("decode", second_scrape)

    assert not any("python_info" in key for key in first)
    assert not any("_bucket" in key for key in first)

    bytes_key = (
        "decode/dynamo_component_request_bytes_total/"
        "dynamo_component.VllmDecodeWorker/dynamo_endpoint.generate/model.Qwen_Qwen2.5"
    )
    assert first[bytes_key] == 10
    assert second[f"{bytes_key}_delta"] == 5

    mean_keys = [key for key in second if "request_duration_seconds_mean_seconds" in key]
    assert len(mean_keys) == 1
    assert second[mean_keys[0]] == 3


def test_start_skips_when_local_export_is_disabled():
    logger = _DummyLogger()
    monitor = DynamoPrometheusMonitor(
        logger=logger,
        dynamo_cfg={"dgd_name": "jonas-swe"},
        prometheus_cfg={"endpoints": {"decode": "http://example/metrics"}},
    )

    monitor.start()

    assert monitor.is_running is False


def test_collect_writes_export_sample_on_elapsed_time(tmp_path, monkeypatch):
    logger = _DummyLogger()
    logger.base_log_dir = str(tmp_path)
    monitor = DynamoPrometheusMonitor(
        logger=logger,
        dynamo_cfg={"dgd_name": "jonas-swe"},
        prometheus_cfg={
            "endpoints": {"decode": "http://example/metrics"},
            "export": {"enabled": True},
        },
    )
    monitor.start_time = 100.0
    monkeypatch.setattr(
        "nemo_rl.utils.dynamo_prometheus.time.time",
        lambda: 112.5,
    )
    monkeypatch.setattr(
        monitor,
        "_fetch_endpoint_metrics",
        lambda endpoint_name, endpoint_url: {"decode/some_metric": 7.0},
    )

    monitor._prepare_export()
    monitor._collect_and_log()
    export_dir = monitor._finalize_export()

    samples = [
        json.loads(line)
        for line in (export_dir / "samples.jsonl").read_text().splitlines()
    ]
    assert samples == [
        {
            "dynamo_prometheus/decode/some_metric": 7.0,
            "dynamo_prometheus/elapsed_seconds": 12.5,
            "dynamo_prometheus/wall_time": 112.5,
        }
    ]


def test_export_writes_replay_artifacts(tmp_path, monkeypatch):
    class _FakeResponse:
        status_code = 200
        text = """
# HELP dynamo_component_request_bytes_total request bytes
# TYPE dynamo_component_request_bytes_total counter
dynamo_component_request_bytes_total{dynamo_component="VllmDecodeWorker",dynamo_endpoint="generate",model="Qwen/Qwen2.5"} 10
python_info{version="3.11"} 1
"""

    logger = _DummyLogger()
    logger.base_log_dir = str(tmp_path)
    monitor = DynamoPrometheusMonitor(
        logger=logger,
        dynamo_cfg={"dgd_name": "jonas-swe"},
        prometheus_cfg={
            "endpoints": {"decode": "http://example/metrics"},
            "export": {"enabled": True},
        },
    )
    monitor.start_time = 1710000000.0
    monkeypatch.setattr(
        "nemo_rl.utils.dynamo_prometheus.time.time",
        lambda: 1710000012.5,
    )
    monkeypatch.setattr(
        "nemo_rl.utils.dynamo_prometheus.requests.get",
        lambda url, timeout: _FakeResponse(),
    )

    monitor._prepare_export()
    monitor._collect_and_log()
    export_dir = monitor._finalize_export()

    assert export_dir == tmp_path / "dynamo_prometheus_export"

    samples = [
        json.loads(line)
        for line in (export_dir / "samples.jsonl").read_text().splitlines()
    ]
    assert samples[0]["dynamo_prometheus/elapsed_seconds"] == 12.5
    assert (
        samples[0][
            "dynamo_prometheus/decode/dynamo_component_request_bytes_total/"
            "dynamo_component.VllmDecodeWorker/dynamo_endpoint.generate/"
            "model.Qwen_Qwen2.5"
        ]
        == 10
    )

    raw_scrapes = [
        json.loads(line)
        for line in (export_dir / "raw_scrapes.jsonl").read_text().splitlines()
    ]
    assert raw_scrapes[0]["endpoint_name"] == "decode"
    assert "python_info" in raw_scrapes[0]["text"]

    openmetrics = (export_dir / "data.openmetrics").read_text()
    assert "dynamo_component_request_bytes_total{" in openmetrics
    assert 'nemo_rl_endpoint="decode"' in openmetrics
    assert "python_info" not in openmetrics
    assert "1710000012.500000" in openmetrics
    assert openmetrics.endswith("# EOF\n")

    metadata = json.loads((export_dir / "metadata.json").read_text())
    assert metadata["counts"] == {
        "grafana_dashboard_panels": 1,
        "openmetrics_samples": 1,
        "samples": 1,
        "scrapes": 1,
    }
    assert metadata["files"]["grafana_dashboard"] == "grafana-dashboard.json"
    assert metadata["metric_names"] == ["dynamo_component_request_bytes_total"]

    dashboard = json.loads((export_dir / "grafana-dashboard.json").read_text())
    assert dashboard["title"] == "NeMo RL Dynamo Prometheus Replay"
    assert dashboard["time"]["from"].endswith("+00:00")
    assert dashboard["time"]["to"].endswith("+00:00")
    assert len(dashboard["panels"]) == 1
    target_exprs = [
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel["targets"]
    ]
    assert target_exprs == [
        (
            "sum by (nemo_rl_endpoint, model) "
            "(rate(dynamo_component_request_bytes_total{"
            'nemo_rl_endpoint=~"$endpoint",model=~"$model"'
            "}[$__rate_interval]))"
        )
    ]
