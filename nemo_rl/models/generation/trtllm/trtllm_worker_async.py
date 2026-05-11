# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Ray actor wrapping ``tensorrt_llm._torch.async_llm.AsyncLLM``.

Sibling of :class:`TrtllmGenerationWorker` that drives the async TRT-LLM
engine. Follows vLLM's ``VllmAsyncGenerationWorker`` pattern: every method
that needs to call into ``AsyncLLM`` is exposed as an ``async def`` with
the ``_async`` suffix. Ray's actor runtime runs those coroutines directly
on the actor's own asyncio event loop, so we don't need an external
``AsyncLoopThread`` (which would deadlock against Ray's loop).

Sync entrypoints inherited from :class:`TrtllmGenerationWorkerImpl` (e.g.
``shutdown``) stay sync. :meth:`TrtllmGeneration` dispatches to the
``_async`` variants when ``trtllm_cfg.async_engine`` is set.
"""

import asyncio
import os
from typing import Any, Optional

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.trtllm.config import TrtllmConfig
from nemo_rl.models.generation.trtllm.trtllm_worker import TrtllmGenerationWorkerImpl


class TrtllmAsyncGenerationWorkerImpl(TrtllmGenerationWorkerImpl):
    """Async variant of :class:`TrtllmGenerationWorkerImpl` (plain class)."""

    def __repr__(self) -> str:
        return "TrtllmAsyncGenerationWorker"

    def __init__(
        self,
        config: TrtllmConfig,
        bundle_indices: Optional[list[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg = config
        self.model_name = self.cfg["model_name"]
        self.is_model_owner = bundle_indices is not None
        self._bundle_indices = bundle_indices
        self._seed = seed
        self.llm = None
        self.TrtSamplingParams = None
        self._http_thread = None
        self._http_base_url: Optional[str] = None
        self._http_server = None

        if not self.is_model_owner:
            return

        from tensorrt_llm import AsyncLLM, SamplingParams as TrtSamplingParams
        from ray.util.placement_group import get_current_placement_group

        self.TrtSamplingParams = TrtSamplingParams

        trtllm_cfg = self.cfg["trtllm_cfg"]
        tp_size = trtllm_cfg["tensor_parallel_size"]

        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        pg = get_current_placement_group()
        assert pg is not None, (
            "TrtllmAsyncGenerationWorker must be scheduled inside a Ray placement "
            "group; got None from get_current_placement_group()."
        )
        # NB: AsyncLLM.__init__ has its OWN placement_groups /
        # placement_bundle_indices / per_worker_gpu_share named parameters,
        # and it unconditionally overwrites any ``ray_placement_config`` we
        # might pass in **kwargs with a fresh one built from those top-level
        # args (see tensorrt_llm/_torch/async_llm.py). So passing
        # ``ray_placement_config`` directly is a no-op (silently dropped).
        # The right way is verl's pattern: top-level keyword args.
        print(
            f"[TrtllmAsyncWorker] bundle_indices={self._bundle_indices}, "
            f"pg={pg}, bundle_specs={pg.bundle_specs}",
            flush=True,
        )

        precision = trtllm_cfg.get("precision", "bfloat16")
        max_batch_size = trtllm_cfg.get("max_batch_size", 64)
        max_num_tokens = trtllm_cfg.get("max_num_tokens", 8192)

        # Verl-style top-level placement kwargs: AsyncLLM.__init__ takes
        # placement_groups / placement_bundle_indices as named parameters and
        # builds the RayPlacementConfig itself with defer_workers_init=True.
        # Passing ray_placement_config in **kwargs is a no-op (silently
        # overwritten inside AsyncLLM.__init__).
        #
        # Use the "one entry per worker" expansion form
        # (placement_groups=[pg, pg, ..., pg], placement_bundle_indices=[[i0],
        # [i1], ...]) — matches the docstring example and trivially
        # generalizes to multi-PG (multi-node TP) once we extend
        # configure_worker to surface per-rank PGs.
        n_workers = len(self._bundle_indices)
        placement_groups_list = [pg] * n_workers
        placement_bundle_indices_list = [[i] for i in self._bundle_indices]

        llm_kwargs: dict[str, Any] = dict(
            model=self.model_name,
            backend="pytorch",
            tensor_parallel_size=tp_size,
            dtype=precision,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            max_seq_len=trtllm_cfg["max_model_len"],
            orchestrator_type="ray",
            ray_worker_extension_cls="nemo_rl.models.generation.trtllm.trtllm_backend.NcclExtension",
            placement_groups=placement_groups_list,
            placement_bundle_indices=placement_bundle_indices_list,
            trust_remote_code=True,
        )

        # Sync construction only — defer __await__ (which fires setup_async)
        # to post_init_async on the Ray actor's asyncio loop.
        self.llm = AsyncLLM(**llm_kwargs)

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    async def post_init_async(self) -> None:
        """Finish async-side engine setup and (optionally) start HTTP server."""
        if not self.is_model_owner or self.llm is None:
            return

        print("[TrtllmAsyncWorker] post_init_async: awaiting setup_async…", flush=True)
        await self.llm.setup_async()
        print("[TrtllmAsyncWorker] AsyncLLM ready", flush=True)

        if self.cfg["trtllm_cfg"].get("expose_http_server"):
            self.start_http_server()

    def shutdown(self) -> bool:
        self.stop_http_server()
        return super().shutdown()

    # ------------------------------------------------------------------ #
    #  HTTP server for NeMo Gym
    # ------------------------------------------------------------------ #

    def start_http_server(self, port: int = 0) -> str:
        """Start an OpenAI-compatible HTTP server backed by ``self.llm``."""
        if self._http_base_url is not None:
            return self._http_base_url

        from transformers import AutoTokenizer

        from nemo_rl.models.generation.trtllm.trtllm_http_server import start_server

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        self._http_thread, self._http_base_url, self._http_server = start_server(
            llm=self.llm,
            tokenizer=tokenizer,
            model_name=self.model_name,
            port=port,
        )
        print(
            f"[TrtllmAsyncWorker] HTTP server started: {self._http_base_url}",
            flush=True,
        )
        return self._http_base_url

    def stop_http_server(self) -> None:
        if self._http_server is not None:
            self._http_server.should_exit = True
            self._http_server = None
            self._http_thread = None
            self._http_base_url = None

    async def report_dp_openai_server_base_url(self) -> Optional[str]:
        return self._http_base_url

    # ------------------------------------------------------------------ #
    #  Collective RPC / refit
    # ------------------------------------------------------------------ #

    async def init_collective_async(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        assert self.llm is not None
        await self.llm.collective_rpc(
            "init_collective",
            args=(rank_prefix, ip, port, world_size, train_world_size),
        )

    async def prepare_refit_info_async(self, state_dict_info: dict[str, Any]) -> None:
        assert self.llm is not None
        await self.llm.collective_rpc("prepare_refit_info", args=(state_dict_info,))

    async def update_weights_from_collective_async(self) -> bool:
        assert self.llm is not None
        try:
            results = await self.llm.collective_rpc("update_weights_from_collective")
            worker_result = results[0] if results else True
            if not worker_result:
                print(
                    f"Error: TRT-LLM worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during TRT-LLM async collective weight update: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def report_device_id_async(self) -> list[str]:
        assert self.llm is not None
        return await self.llm.collective_rpc("report_device_id")

    async def reset_prefix_cache_async(self) -> bool:
        if self.llm is not None:
            try:
                await self.llm.collective_rpc("reset_prefix_cache")
            except Exception:
                pass
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        return True

    # ------------------------------------------------------------------ #
    #  Generation
    # ------------------------------------------------------------------ #

    async def generate_async(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        if len(data["input_ids"]) == 0:
            return BatchedDataDict[GenerationOutputSpec]({
                "output_ids": torch.zeros((0, 0), dtype=torch.long),
                "logprobs": torch.zeros((0, 0), dtype=torch.float),
                "generation_lengths": torch.zeros(0, dtype=torch.long),
                "unpadded_sequence_lengths": torch.zeros(0, dtype=torch.long),
            })

        assert self.llm is not None
        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]

        verify_right_padding(data, pad_value=self.cfg["_pad_token_id"])

        padded_input_length = input_ids.size(1)

        prompts = []
        for i in range(len(input_ids)):
            length = input_lengths[i].item()
            token_ids = input_ids[i, :length].tolist()
            prompts.append({"prompt_token_ids": token_ids})

        sampling_params = self._build_sampling_params(greedy=greedy)

        # Fan all prompts out concurrently; AsyncLLM batches them in-flight.
        outputs = await asyncio.gather(
            *[
                self.llm.generate_async(inputs=p, sampling_params=sampling_params)
                for p in prompts
            ]
        )

        output_ids_list = []
        logprobs_list = []
        generation_lengths = []
        unpadded_sequence_lengths = []

        max_gen_len = max(len(o.outputs[0].token_ids) for o in outputs)

        for i, output in enumerate(outputs):
            seq_len = input_lengths[i].item()
            gen = output.outputs[0]
            gen_tokens = list(gen.token_ids)
            total_length = padded_input_length + max_gen_len

            full_output = torch.full(
                (total_length,), self.cfg["_pad_token_id"], dtype=input_ids.dtype,
            )
            full_output[:seq_len] = input_ids[i][:seq_len]
            full_output[seq_len : seq_len + len(gen_tokens)] = torch.tensor(gen_tokens)
            output_ids_list.append(full_output)

            full_logprobs = torch.zeros(total_length, dtype=torch.float32)
            if gen.logprobs:
                for idx, lp in enumerate(gen.logprobs):
                    pos = seq_len + idx
                    if pos < total_length:
                        if isinstance(lp, (int, float)):
                            full_logprobs[pos] = float(lp)
                        elif isinstance(lp, dict):
                            full_logprobs[pos] = next(iter(lp.values())).logprob
                        else:
                            full_logprobs[pos] = float(lp)
            logprobs_list.append(full_logprobs)

            resp_len = seq_len + len(gen_tokens)
            generation_lengths.append(len(gen_tokens))
            unpadded_sequence_lengths.append(resp_len)

        return BatchedDataDict[GenerationOutputSpec]({
            "output_ids": torch.stack(output_ids_list),
            "logprobs": torch.stack(logprobs_list),
            "generation_lengths": torch.tensor(generation_lengths, dtype=torch.long),
            "unpadded_sequence_lengths": torch.tensor(
                unpadded_sequence_lengths, dtype=torch.long
            ),
        })


@ray.remote(num_cpus=0, max_concurrency=8)
class TrtllmAsyncGenerationWorker(TrtllmAsyncGenerationWorkerImpl):
    """Ray actor wrapper around :class:`TrtllmAsyncGenerationWorkerImpl`."""

    pass
