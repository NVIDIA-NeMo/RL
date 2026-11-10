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

"""Which data-plane clients the trainer folds into the cluster view."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.observability import MetricsDataPlaneClient
from nemo_rl.models.policy.tq_policy import TQPolicy


def _client() -> MetricsDataPlaneClient:
    client = MetricsDataPlaneClient(NoOpDataPlaneClient())
    client.register_partition(
        partition_id="p", fields=["x"], num_samples=1, consumer_tasks=["t"]
    )
    return client


def _trainer(own, workers=(), fns=()):
    trainer = SimpleNamespace(
        dp_client=own,
        _extra_snapshot_sources=[],
        _extra_snapshot_fns=list(fns),
        run_all_workers_single_data=lambda *_a, **_k: [{"w": i} for i in workers],
    )
    trainer.collect_data_plane_snapshots = (
        TQPolicy.collect_data_plane_snapshots.__get__(trainer)
    )
    return trainer


def _collect(trainer, driver_client=None):
    with patch("nemo_rl.models.policy.tq_policy.ray.get", return_value=[]):
        return trainer.collect_data_plane_snapshots(driver_client)


def test_supplied_driver_client_comes_first_and_trainer_client_is_kept():
    own, driver = _client(), _client()
    snaps = _collect(_trainer(own), driver)
    assert len(snaps) == 2
    assert snaps[0]["total_ops"] == driver.snapshot()["total_ops"]


def test_trainer_client_is_not_counted_twice_when_it_is_the_driver_client():
    own = _client()
    assert len(_collect(_trainer(own), own)) == 1
    assert len(_collect(_trainer(own))) == 1


def test_worker_ranks_and_callable_sources_join_the_cluster_view():
    own = _client()
    snaps = _collect(
        _trainer(own, workers=[0, 1], fns=[lambda: [{"gen": 0}, {"gen": 1}]])
    )
    assert len(snaps) == 1 + 2 + 2


def test_a_failing_callable_source_is_dropped_not_raised():
    def broken():
        raise RuntimeError("worker gone")

    own = _client()
    snaps = _collect(_trainer(own, fns=[broken, lambda: [{"gen": 0}]]))
    assert len(snaps) == 2


def test_one_snapshot_means_driver_scope():
    own = _client()
    trainer = _trainer(own)
    trainer.get_data_plane_step_metrics = TQPolicy.get_data_plane_step_metrics.__get__(
        trainer
    )
    metrics, scope = trainer.get_data_plane_step_metrics(1.0)
    assert scope == "driver"
    assert metrics
