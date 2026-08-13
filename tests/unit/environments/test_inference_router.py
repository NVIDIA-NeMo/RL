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

import subprocess
import sys
from unittest.mock import MagicMock, call, patch
from urllib.error import URLError

import pytest

from nemo_rl.environments.inference_router import (
    InferenceRouterConfig,
    InferenceRouterProcess,
)


def test_builds_vllm_router_command_and_openai_base_url() -> None:
    router = InferenceRouterProcess(
        worker_base_urls=[
            "http://worker-0:8000/v1",
            "http://worker-1:8001/v1/",
        ],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        config=InferenceRouterConfig(enabled=True),
    )

    assert router.command == [
        sys.executable,
        "-m",
        "vllm_router.launch_router",
        "--worker-urls",
        "http://worker-0:8000",
        "http://worker-1:8001",
        "--policy",
        "consistent_hash",
        "--host",
        "10.0.0.5",
        "--port",
        "6100",
        "--prometheus-port",
        "6600",
    ]
    assert router.openai_base_url == "http://10.0.0.5:6100/v1"
    assert router.readiness_url == "http://10.0.0.5:6100/readiness"
    assert router.session_affinity_header == "X-Session-ID"


def test_builds_smg_command_and_maps_session_affinity() -> None:
    router = InferenceRouterProcess(
        worker_base_urls=["http://worker-0:8000/v1"],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        config=InferenceRouterConfig(
            enabled=True,
            backend="smg",
            policy="consistent_hash",
        ),
    )

    assert router.command == [
        sys.executable,
        "-m",
        "smg.launch_router",
        "--worker-urls",
        "http://worker-0:8000",
        "--policy",
        "consistent_hashing",
        "--host",
        "10.0.0.5",
        "--port",
        "6100",
        "--prometheus-port",
        "6600",
    ]
    assert router.session_affinity_header == "X-SMG-Routing-Key"


@pytest.mark.parametrize(
    ("backend", "policy"),
    [
        ("vllm_router", "least_load"),
        ("vllm_router", "rendezvous_hash"),
        ("smg", "rendezvous_hash"),
    ],
)
def test_rejects_policy_not_supported_by_backend(backend: str, policy: str) -> None:
    with pytest.raises(ValueError, match="does not support policy"):
        InferenceRouterConfig(backend=backend, policy=policy)


def test_starts_and_stops_owned_router_process() -> None:
    router = InferenceRouterProcess(
        worker_base_urls=["http://worker-0:8000/v1"],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        config=InferenceRouterConfig(enabled=True),
    )
    process = MagicMock()
    process.poll.return_value = None

    with patch(
        "nemo_rl.environments.inference_router.subprocess.Popen",
        return_value=process,
    ) as popen:
        router.start()
        popen.assert_called_once_with(router.command)

        router.stop()
        router.stop()

    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=10.0)
    process.kill.assert_not_called()


def test_force_kills_router_process_when_shutdown_times_out() -> None:
    router = InferenceRouterProcess(
        worker_base_urls=["http://worker-0:8000/v1"],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        config=InferenceRouterConfig(enabled=True),
    )
    process = MagicMock()
    process.poll.return_value = None
    process.wait.side_effect = [
        subprocess.TimeoutExpired(router.command, 2.0),
        0,
    ]

    with patch(
        "nemo_rl.environments.inference_router.subprocess.Popen",
        return_value=process,
    ):
        router.start()
        router.stop(timeout=2.0)

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_args_list == [
        call(timeout=2.0),
        call(),
    ]


def test_waits_until_router_is_ready() -> None:
    router = InferenceRouterProcess(
        worker_base_urls=["http://worker-0:8000/v1"],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        config=InferenceRouterConfig(enabled=True),
    )
    process = MagicMock()
    process.poll.return_value = None

    ready_response = MagicMock()
    ready_response.status = 200
    ready_response.__enter__.return_value = ready_response

    with (
        patch(
            "nemo_rl.environments.inference_router.subprocess.Popen",
            return_value=process,
        ),
        patch(
            "nemo_rl.environments.inference_router.urlopen",
            side_effect=[URLError("not ready"), ready_response],
        ) as urlopen,
        patch("nemo_rl.environments.inference_router.time.sleep") as sleep,
    ):
        router.start()
        router.wait_until_ready(timeout=10.0, poll_interval=0.25)

    assert urlopen.call_args_list == [
        call(router.readiness_url, timeout=0.25),
        call(router.readiness_url, timeout=0.25),
    ]
    sleep.assert_called_once_with(0.25)


def test_readiness_fails_when_router_process_exits() -> None:
    router = InferenceRouterProcess(
        worker_base_urls=["http://worker-0:8000/v1"],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        config=InferenceRouterConfig(enabled=True),
    )
    process = MagicMock()
    process.poll.return_value = 17

    with (
        patch(
            "nemo_rl.environments.inference_router.subprocess.Popen",
            return_value=process,
        ),
        patch("nemo_rl.environments.inference_router.urlopen") as urlopen,
    ):
        router.start()
        with pytest.raises(RuntimeError, match="exited with code 17"):
            router.wait_until_ready(timeout=10.0, poll_interval=0.25)

    urlopen.assert_not_called()


def test_readiness_times_out() -> None:
    router = InferenceRouterProcess(
        worker_base_urls=["http://worker-0:8000/v1"],
        host="10.0.0.5",
        port=6100,
        prometheus_port=6600,
        config=InferenceRouterConfig(enabled=True),
    )
    process = MagicMock()
    process.poll.return_value = None

    with (
        patch(
            "nemo_rl.environments.inference_router.subprocess.Popen",
            return_value=process,
        ),
        patch(
            "nemo_rl.environments.inference_router.urlopen",
            side_effect=URLError("not ready"),
        ),
        patch("nemo_rl.environments.inference_router.time.sleep") as sleep,
    ):
        router.start()
        with pytest.raises(TimeoutError, match="did not become ready"):
            router.wait_until_ready(timeout=0.0, poll_interval=0.25)

    sleep.assert_not_called()
