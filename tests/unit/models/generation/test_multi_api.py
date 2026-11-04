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

import asyncio
import importlib.util
import multiprocessing
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

# Import by path to avoid the vLLM package's GPU/Ray worker imports.
MODULE_PATH = (
    Path(__file__).resolve().parents[4] / "nemo_rl/models/generation/vllm/multi_api.py"
)
spec = importlib.util.spec_from_file_location("multi_api_test_target", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MultiAPITest(unittest.IsolatedAsyncioTestCase):
    def test_count_validation(self):
        self.assertIsNone(module.api_server_count({"vllm_cfg": {}}))
        self.assertIsNone(
            module.api_server_count({"vllm_cfg": {"http_api_server_count": None}})
        )
        self.assertEqual(
            module.api_server_count({"vllm_cfg": {"http_api_server_count": 1}}), 1
        )
        for count in [0, -1, True, 1.5, "8"]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                module.api_server_count({"vllm_cfg": {"http_api_server_count": count}})
        with self.assertRaises(ValueError):
            module.api_server_count({"vllm_cfg": {"http_api_server_count": 8}})
        self.assertEqual(
            module.api_server_count(
                {
                    "vllm_cfg": {
                        "http_api_server_count": 8,
                        "expose_http_server": True,
                        "async_engine": True,
                    }
                }
            ),
            8,
        )

        with self.assertRaises(ValueError):
            module.api_server_count(
                {
                    "vllm_cfg": {
                        "http_api_server_count": 8,
                        "expose_http_server": True,
                        "async_engine": False,
                    }
                }
            )

    async def test_spawned_host_preserves_frontend_settings(self):
        # Exercise child initialization without constructing a GPU worker.
        for dtype in ["int16", "int32"]:
            hosts = []

            class Host:
                def _initialize_http_state(self):
                    self.http_server = None
                    self.setup_token_capture = AsyncMock()
                    self.set_rollout_weight_version = AsyncMock()
                    hosts.append(self)

            engine = Mock()
            connection = Mock()
            connection.recv.side_effect = [
                ("capture", ({"impl": "transfer_queue"}, "stage", True, 4)),
                ("version", (10,)),
                ("stop", ()),
            ]
            listener = Mock()
            listener.getsockname.return_value = ("localhost", 8100)
            http_client = Mock()
            install_yield = Mock()
            modules = {
                "ray": SimpleNamespace(is_initialized=lambda: False, init=Mock()),
                "nemo_rl.models.generation.vllm.patches": SimpleNamespace(
                    install_output_loop_yield=install_yield
                ),
                "vllm.v1.engine.async_llm": SimpleNamespace(
                    AsyncLLM=Mock(return_value=engine)
                ),
                "vllm.v1.executor.abstract": SimpleNamespace(
                    Executor=SimpleNamespace(get_class=lambda cfg: object)
                ),
                "vllm.v1.metrics.prometheus": SimpleNamespace(
                    shutdown_prometheus=Mock()
                ),
                "nemo_rl.models.generation.vllm.vllm_worker_async": SimpleNamespace(
                    VllmAsyncGenerationWorkerImpl=Host,
                    _AsyncLLMHTTPClient=http_client,
                ),
            }
            with patch.dict(sys.modules, modules):
                await asyncio.to_thread(
                    module.run_api_process,
                    connection,
                    {
                        "vllm_cfg": {
                            "flat_logprobs": True,
                            "output_loop_yield_budget_ms": 20,
                        }
                    },
                    SimpleNamespace(disable_log_stats=False, enable_log_requests=False),
                    SimpleNamespace(parallel_config=SimpleNamespace()),
                    {},
                    count=8,
                    index=1,
                    listener=listener,
                    node_ip="localhost",
                    namespace="test",
                    routed_experts_dtype=dtype,
                )
            self.assertEqual(hosts[0].routed_experts_dtype, dtype)
            self.assertTrue(http_client.call_args.kwargs["flat_logprobs"])
            install_yield.assert_called_once_with(engine, reader_budget_s=0.02)
            hosts[0].setup_token_capture.assert_awaited_once_with(
                {"impl": "transfer_queue"},
                "stage",
                capture_media=True,
                top_logprobs=4,
            )
            hosts[0].set_rollout_weight_version.assert_awaited_once_with(10)
            engine.shutdown.assert_called_once()
            listener.close.assert_called_once()

    def test_reply_error_and_timeout(self):
        parent, child = multiprocessing.Pipe()
        try:
            child.send(("error", "capture failed"))
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                module.receive_reply(parent, 0.1)
            with self.assertRaises(TimeoutError):
                module.receive_reply(parent, 0.01)
        finally:
            parent.close()
            child.close()

    async def test_fanout_capture_and_version_acknowledgements(self):
        pool = module.MultiAPIServerPool(SimpleNamespace(), 8)
        peers = []
        for _ in range(7):
            parent, child = multiprocessing.Pipe()
            pool.connections.append(parent)
            pool.children.append(Mock(is_alive=Mock(return_value=True)))
            peers.append(child)
        try:
            for name, args in [
                ("capture", ({"impl": "transfer_queue"}, "stage", False, 4)),
                ("version", (10,)),
            ]:
                task = asyncio.create_task(pool.command(name, *args))
                await asyncio.sleep(0.01)
                self.assertFalse(task.done())
                for peer in peers:
                    self.assertEqual(peer.recv(), (name, args))
                    peer.send(("ok", None))
                await task
        finally:
            for connection in pool.connections + peers:
                connection.close()

    async def test_child_error_cleans_pool_and_engine(self):
        pool = module.MultiAPIServerPool(SimpleNamespace(), 8)
        parent, child = multiprocessing.Pipe()
        process = Mock()
        process.is_alive.side_effect = [True, True, False]
        engine = Mock()
        pool.children = [process]
        pool.connections = [parent]
        pool.engine_manager = engine
        child.send(("error", "initialization failed"))
        try:
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                await pool.command("capture", {}, "stage", False, 4)
            engine.shutdown.assert_called_once()
            process.terminate.assert_called_once()
            self.assertEqual(pool.children, [])
        finally:
            child.close()


if __name__ == "__main__":
    unittest.main()
