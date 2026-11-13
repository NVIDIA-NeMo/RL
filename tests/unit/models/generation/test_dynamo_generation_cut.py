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

"""NeMo-Gym's checkpoint generation cut through the Dynamo frontend."""

import asyncio
import time

import pytest

pytest.importorskip("nemo_gym")

from nemo_gym._checkpoint.generation_cut import (  # noqa: E402
    GenerationCutInventory,
    GenerationCutPrefix,
    GenerationCutPrefixAck,
)

from nemo_rl.models.generation.dynamo import managed_runtime  # noqa: E402
from nemo_rl.models.generation.dynamo.config import DynamoCfg  # noqa: E402
from nemo_rl.models.generation.dynamo.managed_runtime import (  # noqa: E402
    ManagedDynamoRuntime,
)
from nemo_rl.models.generation.dynamo.vllm.generation_cut import (  # noqa: E402
    CUT_PREFIXES_KEY,
    DEADLINE_KEY,
    GENERATION_CUT_ANNOTATION,
    INVENTORY_KEY,
    EngineCutFanOut,
)

pytestmark = pytest.mark.nemo_gym


class _Item:
    def __init__(self, data) -> None:
        self._data = data

    def data(self):
        return self._data


class _Late(list):
    """An engine's answer that reaches the frontend after its deadline."""


class _Client:
    """Dynamo's request-plane client: each engine answers a direct request."""

    def __init__(self, engines: dict[int, object]) -> None:
        self._engines = engines
        self.requests: dict[int, dict] = {}

    def instance_ids(self) -> list[int]:
        return list(self._engines)

    async def direct(self, request, instance_id):
        self.requests[instance_id] = request
        answer = self._engines[instance_id]
        if isinstance(answer, Exception):
            raise answer

        async def stream():
            if isinstance(answer, _Late):
                await asyncio.sleep(
                    request["extra_args"][DEADLINE_KEY] - time.time() + 5
                )
            yield _Item({"token_ids": [], "engine_data": {CUT_PREFIXES_KEY: answer}})

        return stream()


class _Runtime:
    def __init__(self, client: _Client) -> None:
        self._client = client
        self.paths: list[str] = []

    def endpoint(self, path: str) -> "_Runtime":
        self.paths.append(path)
        return self

    def client(self) -> "asyncio.Future[_Client]":
        # A future, as Dynamo's binding returns: not a coroutine.
        future = asyncio.get_running_loop().create_future()
        future.set_result(self._client)
        return future


def _call(name: str) -> GenerationCutPrefix:
    return GenerationCutPrefix(
        ticket_id=f"ticket-{name}",
        rollout_id="r0",
        attempt=0,
        model_call_id=name,
        admitted_at=1.0,
    )


def test_a_cut_reaches_every_engine_and_each_call_gets_its_engines_answer(
    monkeypatch, tmp_path
) -> None:
    held, lost, late = _call("held"), _call("lost"), _call("late")
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy_model",
        active_prefixes=[held, lost, late],
    )

    def cut(call: GenerationCutPrefix) -> dict:
        return GenerationCutPrefixAck(
            **call.model_dump(),
            disposition="durable_prefix",
            cut_kind="active_prefix",
            frozen_buffer_id="active/checkpoint-1/0",
            staging_keys=(
                f"__generation_cut__/checkpoint-1/r0/{call.model_call_id}/0",
            ),
            prefix_token_count=3,
            prefix_digest="0" * 64,
            effective_output_limit=8,
        ).model_dump(mode="json")

    client = _Client(
        {
            1: [],  # holds none of the calls
            2: [cut(held)],
            3: RuntimeError("engine gone"),
            4: _Late([cut(late)]),
        }
    )
    runtime = _Runtime(client)
    monkeypatch.setenv("DYN_NAMESPACE", "nemo-rl-test")
    monkeypatch.setenv(managed_runtime.CAPTURE_FILE_ENV, str(tmp_path / "capture.pkl"))
    driver = ManagedDynamoRuntime(
        dynamo_cfg=DynamoCfg(),
        model_name="model",
        frontend_python="python",
        ray_address="10.0.0.1:6379",
        serves_chat=True,
    )
    driver._chat_dir = str(tmp_path)

    async def run():
        fan_out = EngineCutFanOut(runtime, asyncio.get_running_loop())
        # Before capture is set up the route refuses every cut, as natively.
        assert fan_out.control_token is None
        driver.publish_chat_capture(
            dp_cfg={"backend": "simple"},
            staging_partition="staging",
            capture_media=False,
            generation_cut_control_token="secret",
            generation_cut_control_timeout_s=0.5,
        )
        assert fan_out.control_token == "secret"
        arrived = time.time()
        return arrived, await fan_out.run_cut(inventory)

    arrived, receipt = asyncio.run(run())

    # Gym accepts it as the answer to its inventory: every call acked once.
    receipt.validate_for(inventory)
    assert {ack.model_call_id: ack.disposition for ack in receipt.prefixes} == {
        "held": "durable_prefix",
        "lost": "durable_failure",
        # Its engine cut it, but answered after the frontend stopped waiting.
        "late": "durable_failure",
    }
    assert runtime.paths == ["nemo-rl-test.backend.generate"]
    # Every engine gets the whole inventory and the frontend's one deadline,
    # counted from the cut's arrival; token_ids is always present.
    deadline = client.requests[1]["extra_args"][DEADLINE_KEY]
    assert arrived + 0.5 <= deadline <= time.time()
    request = {
        "token_ids": [],
        "annotations": [GENERATION_CUT_ANNOTATION],
        "extra_args": {
            INVENTORY_KEY: inventory.model_dump(mode="json"),
            DEADLINE_KEY: deadline,
        },
    }
    assert client.requests == {1: request, 2: request, 3: request, 4: request}
