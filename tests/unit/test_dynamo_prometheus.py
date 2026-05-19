import sys

from nemo_rl.utils.dynamo_prometheus import (
    DynamoPrometheusMonitor,
    _resolve_metric_endpoints,
)


class _DummyWandbLogger:
    def define_metric(self, *args, **kwargs):
        pass


class _DummyLogger:
    wandb_logger = _DummyWandbLogger()

    def log_metrics(self, *args, **kwargs):
        pass


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


class _RecordingRun:
    def __init__(self):
        self.logged = []
        self.defined_metrics = []
        self.finished = False
        self.id = "main-run-id"
        self.name = "main-run"
        self.project = "test-project"
        self.entity = "test-entity"
        self.group = "test-group"

    def log(self, metrics, **kwargs):
        self.logged.append((metrics, kwargs))

    def define_metric(self, *args, **kwargs):
        self.defined_metrics.append((args, kwargs))

    def finish(self):
        self.finished = True


class _RecordingWandbLogger:
    def __init__(self):
        self.run = _RecordingRun()
        self.defined_metrics = []

    def define_metric(self, *args, **kwargs):
        self.defined_metrics.append((args, kwargs))


class _RecordingLogger:
    def __init__(self):
        self.wandb_logger = _RecordingWandbLogger()
        self.log_metrics_calls = []

    def log_metrics(self, *args, **kwargs):
        self.log_metrics_calls.append((args, kwargs))


def test_collect_flushes_prometheus_metrics_on_elapsed_time(monkeypatch):
    logger = _RecordingLogger()
    monitor = DynamoPrometheusMonitor(
        logger=logger,
        dynamo_cfg={"dgd_name": "jonas-swe"},
        prometheus_cfg={"endpoints": {"decode": "http://example/metrics"}},
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

    monitor._collect_and_log()

    assert logger.log_metrics_calls == []
    assert logger.wandb_logger.run.logged == []

    assert monitor._flush_samples_to_wandb() == 1
    assert len(logger.wandb_logger.run.logged) == 1
    metrics, kwargs = logger.wandb_logger.run.logged[0]

    assert kwargs == {}
    assert metrics["dynamo_prometheus/elapsed_seconds"] == 12.5
    assert metrics["dynamo_prometheus/wall_time"] == 112.5
    assert metrics["dynamo_prometheus/decode/some_metric"] == 7.0


def test_start_defines_endpoint_metrics_against_elapsed_time():
    logger = _RecordingLogger()
    monitor = DynamoPrometheusMonitor(
        logger=logger,
        dynamo_cfg={"dgd_name": "jonas-swe"},
        prometheus_cfg={"endpoints": {"decode": "http://example/metrics"}},
    )
    monitor._collection_loop = lambda: None

    monitor.start()
    monitor.stop()

    assert (
        ("dynamo_prometheus/elapsed_seconds",),
        {},
    ) in logger.wandb_logger.defined_metrics
    assert (
        ("dynamo_prometheus/wall_time",),
        {},
    ) in logger.wandb_logger.defined_metrics
    assert (
        ("dynamo_prometheus/decode/*",),
        {"step_metric": "dynamo_prometheus/elapsed_seconds"},
    ) in logger.wandb_logger.defined_metrics


def test_separate_wandb_run_groups_with_parent_run(monkeypatch):
    class _FakeWandb:
        def __init__(self):
            self.init_calls = []
            self.run = None

        def init(self, **kwargs):
            self.init_calls.append(kwargs)
            self.run = _RecordingRun()
            self.run.id = "dynamo-run-id"
            self.run.name = kwargs["name"]
            self.run.project = kwargs["project"]
            self.run.entity = kwargs["entity"]
            self.run.group = kwargs["group"]
            return self.run

    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = _RecordingLogger()
    monitor = DynamoPrometheusMonitor(
        logger=logger,
        dynamo_cfg={"dgd_name": "jonas-swe"},
        prometheus_cfg={
            "endpoints": {"decode": "http://example/metrics"},
            "wandb_run": "separate",
        },
        logger_config={
            "wandb": {
                "project": "nemo-gym-swe-dynamo",
                "name": "qwen25-coder-14b-mini-swe-rollout",
                "entity": "nvidia",
                "group": "qwen25-coder-14b-mini-swe-rollout",
            }
        },
    )
    monitor._collection_loop = lambda: None

    monitor.start()
    assert fake_wandb.init_calls == [
        {
            "project": "nemo-gym-swe-dynamo",
            "entity": "nvidia",
            "name": "qwen25-coder-14b-mini-swe-rollout-dynamo-prometheus",
            "group": "qwen25-coder-14b-mini-swe-rollout",
            "job_type": "dynamo-prometheus",
            "reinit": "create_new",
            "resume": "never",
            "config": {
                "source_wandb_run_id": "main-run-id",
                "source_wandb_run_name": "qwen25-coder-14b-mini-swe-rollout",
                "metric_prefix": "dynamo_prometheus",
                "metric_prefixes": ["dynamo_"],
                "endpoints": {"decode": "http://example/metrics"},
            },
        }
    ]
    assert (
        ("dynamo_prometheus/decode/*",),
        {"step_metric": "dynamo_prometheus/elapsed_seconds"},
    ) in fake_wandb.run.defined_metrics

    monitor.stop()
    assert fake_wandb.run.finished is True
