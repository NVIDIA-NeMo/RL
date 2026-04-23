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
"""RemoteGeneration — GenerationInterface wrapper for disaggregated vLLM.

Two modes:

  Co-located (``generation`` provided):
    Wraps a VllmGeneration instance in the SAME Ray cluster. All calls delegate
    to the underlying VllmGeneration; a GenerationControlServer runs alongside
    for external HTTP clients (e.g. NemoGym).

  HTTP-only (``generation=None`` + ``server_url``):
    VllmGeneration lives in a SEPARATE Ray cluster (or standalone process).
    Generation requests are sent to per-shard vLLM workers via the OpenAI
    ``/v1/completions`` endpoint (round-robin across shards). Control-plane
    calls (weight sync, lifecycle) go to the GenerationControlServer.
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import AsyncGenerator, Optional

import aiohttp
import ray
import requests
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)

# Timeout for HTTP requests to the control server (seconds).
# Weight sync / NCCL operations can take minutes.
_HTTP_TIMEOUT = 600


@ray.remote(num_cpus=0)
def _http_call_blocking(url: str, json_body: dict | None = None, raw_body: bytes | None = None, timeout: int = _HTTP_TIMEOUT) -> dict:  # pragma: no cover
    """Fire-and-forget HTTP POST wrapped in a Ray task.

    Returns the JSON response dict. Using a Ray remote function lets the
    caller get back a future immediately, which is critical for NCCL
    rendezvous: both training and inference sides must enter simultaneously.
    """
    if raw_body is not None:
        resp = requests.post(url, data=raw_body, timeout=timeout, headers={"Content-Type": "application/octet-stream"})
    elif json_body is not None:
        resp = requests.post(url, json=json_body, timeout=timeout)
    else:
        resp = requests.post(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


class RemoteGeneration(GenerationInterface):
    """GenerationInterface wrapper that supports direct delegation or HTTP-only mode."""

    def __init__(
        self,
        generation: Optional[GenerationInterface],
        server_url: str,
        config: dict,
    ):
        self._generation = generation
        self.server_url = server_url.rstrip("/")
        self._http_mode = generation is None

        if self._http_mode:
            self.cfg = self._fetch_remote_config(config)
        else:
            self.cfg = dict(generation.cfg)

        # Merge caller-provided overrides
        for key in (
            "remote_generation_url",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_token_ids",
            "stop_strings",
        ):
            if key in config:
                self.cfg[key] = config[key]

        # Fetch per-shard vLLM URLs once. JSON completions are sent directly to
        # these (round-robin); the control server is only used for control-plane
        # calls (weight sync, lifecycle, metrics).
        self._shard_urls: list[str] = []
        self._shard_rr_idx = 0
        if self._http_mode:
            self._shard_urls = self._fetch_shard_urls()
            print(
                f"  ✓ Disagg HTTP routing to {len(self._shard_urls)} DP shard(s)",
                flush=True,
            )

        # Expose per-shard URLs so NemoGym / external OpenAI clients can address
        # DP shards directly (they round-robin / sticky-map internally).
        self.dp_openai_server_base_urls = (
            list(self._shard_urls) if self._http_mode else [f"{self.server_url}/v1"]
        )

    def _fetch_shard_urls(self) -> list[str]:
        """Fetch per-shard vLLM URLs from the control server."""
        resp = requests.get(f"{self.server_url}/dp_openai_server_base_urls", timeout=30)
        resp.raise_for_status()
        urls = [u for u in resp.json() if u is not None]
        if not urls:
            raise RuntimeError("No shard URLs returned from generation server")
        return urls

    def _select_shard(self) -> str:
        """Round-robin select a shard URL."""
        url = self._shard_urls[self._shard_rr_idx]
        self._shard_rr_idx = (self._shard_rr_idx + 1) % len(self._shard_urls)
        return url

    def _fetch_remote_config(self, local_config: dict) -> dict:
        """Fetch generation config from the remote control server."""
        for attempt in range(30):
            try:
                resp = requests.get(f"{self.server_url}/config", timeout=10)
                resp.raise_for_status()
                remote_cfg = resp.json()
                print(f"  ✓ Fetched remote generation config from {self.server_url}")
                return remote_cfg
            except Exception as e:
                if attempt < 29:
                    print(f"  Waiting for gen server at {self.server_url} (attempt {attempt + 1}/30): {e}")
                    time.sleep(5)
                else:
                    raise RuntimeError(
                        f"Failed to reach generation server at {self.server_url}/config after 30 attempts"
                    ) from e

    # =====================================================================
    # Data plane — generation
    # =====================================================================

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        if not self._http_mode:
            return self._generation.generate(data, greedy)
        return asyncio.run(self._generate_json_completions(data, greedy))

    async def _generate_json_completions(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Send batch to ONE shard via /v1/completions JSON endpoint (round-robin).

        Uses the OpenAI-compatible completions API with prompt_token_ids.
        This goes through vLLM's full OpenAI serving layer (tokenization, validation, etc).
        """
        gen_timeout = aiohttp.ClientTimeout(total=300)

        shard_url = self._shard_urls[self._shard_rr_idx]
        self._shard_rr_idx = (self._shard_rr_idx + 1) % len(self._shard_urls)
        completions_url = f"{shard_url}/completions"

        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]
        batch_size = input_ids.shape[0]
        max_new_tokens = self.cfg.get("max_new_tokens", 2048)
        max_model_len = self.cfg.get("vllm_cfg", {}).get("max_model_len", 4096)

        temperature = 0.0 if greedy else self.cfg.get("temperature", 1.0)
        top_p = 1.0 if greedy else self.cfg.get("top_p", 1.0)

        # Build per-sample requests
        requests_list = []
        for i in range(batch_size):
            length = input_lengths[i].item()
            prompt_tokens = input_ids[i, :length].tolist()
            max_tokens = min(max_new_tokens, max_model_len - length)
            req = {
                "model": self.cfg.get("model_name", "default"),
                "prompt": prompt_tokens,  # vLLM accepts list[int] as token IDs
                "max_tokens": max(max_tokens, 1),
                "temperature": temperature,
                "top_p": top_p,
                "logprobs": 1,
            }
            if self.cfg.get("stop_token_ids"):
                req["stop_token_ids"] = self.cfg["stop_token_ids"]
            requests_list.append(req)

        # Send all requests concurrently to the same shard
        async with aiohttp.ClientSession(timeout=gen_timeout) as session:
            async def _send_one(req):
                async with session.post(completions_url, json=req) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            responses = await asyncio.gather(*[_send_one(r) for r in requests_list])

        # Parse responses into GenerationOutputSpec
        pad_token_id = self.cfg.get("_pad_token_id", 0)
        all_output_ids = []
        all_gen_lengths = []
        all_unpadded_lengths = []
        all_logprobs = []
        all_truncated = []

        for i, resp_json in enumerate(responses):
            choice = resp_json["choices"][0]
            finish_reason = choice.get("finish_reason", "stop")
            input_length = input_lengths[i].item()

            # Extract generated token IDs from logprobs.tokens ("token_id:NNN" format)
            gen_token_ids = []
            lp_list = []
            logprobs_data = choice.get("logprobs")
            if logprobs_data and "tokens" in logprobs_data:
                for tok_str in logprobs_data["tokens"]:
                    if tok_str.startswith("token_id:"):
                        gen_token_ids.append(int(tok_str.split(":")[1]))
                    else:
                        gen_token_ids.append(0)  # fallback
                lp_list = [lp if lp is not None else 0.0 for lp in logprobs_data.get("token_logprobs", [])]

            gen_length = len(gen_token_ids)
            unpadded_length = input_length + gen_length
            prompt_tokens = input_ids[i, :input_length].tolist()
            full_ids = prompt_tokens + gen_token_ids

            # Pad logprobs: zeros for input tokens, then actual logprobs
            full_logprobs = [0.0] * input_length + lp_list
            # Pad to same length as full_ids
            while len(full_logprobs) < len(full_ids):
                full_logprobs.append(0.0)

            all_output_ids.append(full_ids)
            all_gen_lengths.append(gen_length)
            all_unpadded_lengths.append(unpadded_length)
            all_logprobs.append(full_logprobs)
            all_truncated.append(finish_reason == "length")

        # Pad to uniform sequence length
        max_seq_len = max(len(ids) for ids in all_output_ids)
        for i in range(batch_size):
            pad_len = max_seq_len - len(all_output_ids[i])
            all_output_ids[i].extend([pad_token_id] * pad_len)
            all_logprobs[i].extend([0.0] * pad_len)

        return BatchedDataDict[GenerationOutputSpec]({
            "output_ids": torch.tensor(all_output_ids, dtype=torch.long),
            "generation_lengths": torch.tensor(all_gen_lengths, dtype=torch.long),
            "unpadded_sequence_lengths": torch.tensor(all_unpadded_lengths, dtype=torch.long),
            "logprobs": torch.tensor(all_logprobs, dtype=torch.float32),
            "truncated": torch.tensor(all_truncated, dtype=torch.bool),
        })

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        if not self._http_mode:
            async for result in self._generation.generate_async(data, greedy):
                yield result
            return

        result = await self._generate_json_completions(data, greedy)
        batch_size = result["output_ids"].shape[0]
        for i in range(batch_size):
            single = BatchedDataDict[GenerationOutputSpec]({
                k: v[i:i+1] if isinstance(v, torch.Tensor) else ([v[i]] if isinstance(v, list) else v)
                for k, v in result.items()
            })
            yield (i, single)

    # =====================================================================
    # Weight sync and lifecycle
    # =====================================================================

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        if not self._http_mode:
            return self._generation.init_collective(
                ip, port, world_size, train_world_size=train_world_size
            )

        # HTTP mode: dispatch as a Ray task so it returns a future.
        # The training side calls ray.get(futures_train + futures_inference)
        # and both sides must enter NCCL rendezvous simultaneously.
        return [
            _http_call_blocking.remote(
                f"{self.server_url}/init_collective",
                json_body={
                    "ip": ip,
                    "port": port,
                    "world_size": world_size,
                    "train_world_size": train_world_size,
                },
            )
        ]

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        if not self._http_mode:
            return self._generation.update_weights_from_collective()

        return [
            _http_call_blocking.remote(
                f"{self.server_url}/update_weights_from_collective",
            )
        ]

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        if not self._http_mode:
            return self._generation.prepare_for_generation(*args, **kwargs)

        resp = requests.post(f"{self.server_url}/prepare_for_generation", timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("success", False)

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        if not self._http_mode:
            return self._generation.finish_generation(*args, **kwargs)

        resp = requests.post(f"{self.server_url}/finish_generation", timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("success", False)

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        if not self._http_mode:
            self._generation.prepare_refit_info(state_dict_info)
            return

        buf = io.BytesIO()
        torch.save(state_dict_info, buf)
        resp = requests.post(
            f"{self.server_url}/prepare_refit_info",
            data=buf.getvalue(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        if not self._http_mode:
            return self._generation.update_weights_via_ipc_zmq()
        raise NotImplementedError("update_weights_via_ipc_zmq not supported in HTTP mode")

    def invalidate_kv_cache(self) -> bool:
        if not self._http_mode:
            return self._generation.invalidate_kv_cache()

        resp = requests.post(f"{self.server_url}/invalidate_kv_cache", timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("success", False)

    @property
    def requires_kv_scale_sync(self) -> bool:
        if not self._http_mode:
            return getattr(self._generation, "requires_kv_scale_sync", False)
        return False

    def clear_logger_metrics(self) -> None:
        if not self._http_mode:
            self._generation.clear_logger_metrics()
            return

        try:
            requests.post(f"{self.server_url}/clear_logger_metrics", timeout=30)
        except requests.RequestException:
            pass

    def get_logger_metrics(self) -> dict[str, Any]:
        if not self._http_mode:
            return self._generation.get_logger_metrics()

        try:
            resp = requests.get(f"{self.server_url}/get_logger_metrics", timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return {}

    def snapshot_step_metrics(self) -> None:
        if not self._http_mode:
            if hasattr(self._generation, "snapshot_step_metrics"):
                self._generation.snapshot_step_metrics()
            return

        try:
            requests.post(f"{self.server_url}/snapshot_step_metrics", timeout=30)
        except requests.RequestException:
            pass

    def get_step_metrics(self) -> dict[str, float]:
        if not self._http_mode:
            if hasattr(self._generation, "get_step_metrics"):
                return self._generation.get_step_metrics()
            return {}

        try:
            resp = requests.get(f"{self.server_url}/get_step_metrics", timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return {}
