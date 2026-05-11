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

"""Ray actor wrapping ``tensorrt_llm.LLM`` for inference in the GRPO loop.

The worker creates a TRT-LLM ``LLM`` instance with:
 - ``backend="pytorch"`` — standard nn.Module, hookable
 - ``orchestrator_type="ray"`` — required for ``ray_worker_extension_cls``
 - ``ray_worker_extension_cls`` pointing to ``NcclExtension``

Weight updates flow through ``NcclExtension`` inside TRT-LLM's internal
``RayGPUWorker``, invoked via ``llm._collective_rpc()``.
"""

import gc
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


class TrtllmGenerationWorkerImpl:
    """Plain (non-actor) implementation of the TRT-LLM generation worker.

    Held separately from ``@ray.remote``-wrapped :class:`TrtllmGenerationWorker`
    so the async sibling :class:`TrtllmAsyncGenerationWorkerImpl` can subclass
    it (Ray rejects inheritance between two ``@ray.remote`` actor classes).
    """

    @staticmethod
    def configure_worker(
        num_gpus: int | float,
        bundle_indices: Optional[tuple[int, list[int]]] = None,
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        # TRT-LLM with orchestrator_type="ray" creates its own internal
        # Ray actors that each need a GPU.  The outer actor therefore
        # gives up its GPU reservation.
        resources: dict[str, Any] = {"num_gpus": 0, "num_cpus": 0}
        # TRT-LLM's CudaRunner derives NVRTC -I include paths via
        # `popen("pip show tensorrt_llm")`. With /opt/nemo_rl_venv/bin ahead of
        # /usr/bin on PATH (uv run prepends the venv), pip resolves to the venv
        # pip which has no tensorrt_llm installed, breaking fmha JIT
        # ("cuda.h: no directories in search list"). Force /usr/bin first so
        # `pip` is the system pip that can locate tensorrt_llm in
        # /usr/local/lib/python3.12/dist-packages.
        worker_path = "/usr/bin:" + os.environ.get("PATH", "")
        env_vars: dict[str, str] = {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "NCCL_CUMEM_ENABLE": "1",
            "PATH": worker_path,
        }
        init_kwargs: dict[str, Any] = {}

        if bundle_indices is not None:
            # Pass bundle_indices through __init__ kwargs; the worker resolves the
            # parent placement group via get_current_placement_group() and hands both
            # to TRT-LLM as ray_placement_config (instead of TRTLLM_RAY_BUNDLE_INDICES).
            init_kwargs["bundle_indices"] = bundle_indices[1]
            node_idx = bundle_indices[0]
            if len(bundle_indices[1]) == 1:
                seed = node_idx * 1024 + bundle_indices[1][0]
            else:
                seed = node_idx * 1024 + bundle_indices[1][0] // len(bundle_indices[1])
            init_kwargs["seed"] = seed

        return resources, env_vars, init_kwargs

    def __repr__(self) -> str:
        return "TrtllmGenerationWorker"

    def __init__(
        self,
        config: TrtllmConfig,
        bundle_indices: Optional[list[int]] = None,
        seed: Optional[int] = None,
    ):
        self.cfg = config
        self.model_name = self.cfg["model_name"]
        self.is_model_owner = bundle_indices is not None

        if not self.is_model_owner:
            self.llm = None
            return

        import tensorrt_llm
        from tensorrt_llm import SamplingParams as TrtSamplingParams
        from tensorrt_llm.llmapi.llm_args import RayPlacementConfig
        from ray.util.placement_group import get_current_placement_group

        self.TrtSamplingParams = TrtSamplingParams

        trtllm_cfg = self.cfg["trtllm_cfg"]
        tp_size = trtllm_cfg["tensor_parallel_size"]

        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Hand TRT-LLM the parent placement group + this worker's bundle indices
        # directly via ray_placement_config (path 0 in RayExecutor._get_placement_group),
        # instead of the env-var path (TRTLLM_RAY_BUNDLE_INDICES). This avoids env-var
        # coupling and lets TRT-LLM see the exact PG the outer Ray actor is scheduled on
        # — i.e. the nemo-rl inference cluster's PG.
        pg = get_current_placement_group()
        assert pg is not None, (
            "TrtllmGenerationWorker must be scheduled inside a Ray placement group; "
            "got None from get_current_placement_group()."
        )
        ray_placement_config = RayPlacementConfig(
            placement_groups=[pg],
            placement_bundle_indices=[list(bundle_indices)],
        )
        print(f"[TrtllmWorker] bundle_indices={bundle_indices}, "
              f"pg={pg}, bundle_specs={pg.bundle_specs}", flush=True)

        precision = trtllm_cfg.get("precision", "bfloat16")
        max_batch_size = trtllm_cfg.get("max_batch_size", 64)
        max_num_tokens = trtllm_cfg.get("max_num_tokens", 8192)

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
            ray_placement_config=ray_placement_config,
            trust_remote_code=True,
        )

        self.llm = tensorrt_llm.LLM(**llm_kwargs)

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def is_alive(self) -> bool:
        return True

    def post_init(self) -> None:
        """Hook called once after construction (overridden by async subclass)."""
        if self.is_model_owner and self.cfg["trtllm_cfg"].get("expose_http_server"):
            raise RuntimeError(
                "trtllm_cfg.expose_http_server is only supported with "
                "async_engine=true (TrtllmAsyncGenerationWorker)."
            )

    def shutdown(self) -> bool:
        try:
            if self.llm is not None:
                del self.llm
                self.llm = None
            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"Error during TRT-LLM shutdown: {e}")
            return False

    def report_dp_openai_server_base_url(self) -> Optional[str]:
        """Sync workers don't expose an HTTP server (overridden by async)."""
        return None

    # ------------------------------------------------------------------ #
    #  Collective RPC dispatch (overridden by TrtllmAsyncGenerationWorker)
    # ------------------------------------------------------------------ #

    def _collective_rpc(self, method: str, *, args: tuple = ()):
        """Dispatch a TRT-LLM collective_rpc call.

        The sync ``LLM`` exposes the private ``_collective_rpc``; the async
        subclass overrides this to use the public async ``collective_rpc``
        through a background event loop.
        """
        assert self.llm is not None
        return self.llm._collective_rpc(method, args=args)

    # ------------------------------------------------------------------ #
    #  Collective init / refit
    # ------------------------------------------------------------------ #

    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        self._collective_rpc(
            "init_collective",
            args=(rank_prefix, ip, port, world_size, train_world_size),
        )

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        self._collective_rpc("prepare_refit_info", args=(state_dict_info,))

    def update_weights_from_collective(self) -> bool:
        try:
            results = self._collective_rpc("update_weights_from_collective")
            worker_result = results[0] if results else True
            if not worker_result:
                print(f"Error: TRT-LLM worker failed to update weights. Result: {worker_result}")
                return False
            return True
        except Exception as e:
            print(f"Exception during TRT-LLM collective weight update: {e}")
            import traceback
            traceback.print_exc()
            return False

    def report_device_id(self) -> list[str]:
        return self._collective_rpc("report_device_id")

    # ------------------------------------------------------------------ #
    #  Generation
    # ------------------------------------------------------------------ #

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False,
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

        # Build per-request prompts (token-id lists, trimmed to actual length)
        prompts = []
        for i in range(len(input_ids)):
            length = input_lengths[i].item()
            token_ids = input_ids[i, :length].tolist()
            prompts.append({"prompt_token_ids": token_ids})

        sampling_params = self._build_sampling_params(greedy=greedy)
        outputs = self._generate_impl(prompts, sampling_params)

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

        result: dict = {
            "output_ids": torch.stack(output_ids_list),
            "logprobs": torch.stack(logprobs_list),
            "generation_lengths": torch.tensor(generation_lengths, dtype=torch.long),
            "unpadded_sequence_lengths": torch.tensor(unpadded_sequence_lengths, dtype=torch.long),
        }

        return BatchedDataDict[GenerationOutputSpec](result)

    def _generate_impl(self, prompts, sampling_params):
        """Run a batch of prompts on the sync LLM (overridden by the async subclass)."""
        return self.llm.generate(prompts, sampling_params=sampling_params)

    # ------------------------------------------------------------------ #
    #  Prefix cache
    # ------------------------------------------------------------------ #

    def reset_prefix_cache(self) -> bool:
        if self.llm is not None:
            try:
                self._collective_rpc("reset_prefix_cache")
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        return True

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _build_sampling_params(self, *, greedy: bool):
        top_k_cfg = self.cfg["top_k"]
        top_k_val = 1 if greedy else (top_k_cfg if top_k_cfg is not None else 0)
        temperature = 0.0 if greedy else self.cfg["temperature"]
        stop_ids = self.cfg.get("stop_token_ids") or []

        return self.TrtSamplingParams(
            temperature=temperature,
            top_p=self.cfg["top_p"],
            top_k=top_k_val,
            max_tokens=self.cfg["max_new_tokens"],
            end_id=stop_ids[0] if stop_ids else None,
            logprobs=True,
        )


@ray.remote(num_cpus=0)
class TrtllmGenerationWorker(TrtllmGenerationWorkerImpl):
    """Ray actor wrapper around :class:`TrtllmGenerationWorkerImpl`."""

    pass
