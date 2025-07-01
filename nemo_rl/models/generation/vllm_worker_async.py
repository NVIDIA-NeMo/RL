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

import asyncio
import gc
import uuid
from typing import Any, AsyncGenerator, cast

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm_worker import BaseVllmGenerationWorker


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")}
)
class VllmAsyncGenerationWorker(BaseVllmGenerationWorker):
    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self.llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**llm_kwargs))

    async def init_collective(
        self, data: int, ip: str, port: int, world_size: int
    ) -> None:
        await self.llm.collective_rpc(
            "init_collective",
            args=(
                data,
                ip,
                port,
                world_size,
            ),
        )

    async def generate(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate a batch of data using vLLM's AsyncLLMEngine, yielding results as they are ready.

        Args:
            data: BatchedDataDict with input_ids and input_lengths
            greedy: Whether to use greedy decoding instead of sampling

        Yields:
            Tuple of (original_index, BatchedDataDict conforming to GenerationOutputSpec for the single sequence)
        """
        # Ensure generate_async only receives single samples (batch_size = 1)
        batch_size = data["input_ids"].shape[0]
        assert batch_size == 1, (
            f"generate_async is restricted to handle only single samples, "
            f"but received batch_size={batch_size}. Please handle batching outside this method."
        )

        # Verify inputs have correct padding
        verify_right_padding(data, pad_value=self.cfg["pad_token_id"])

        # Directly get single item from batch (batch_size = 1)
        input_lengths = data["input_lengths"][0].item()
        input_ids = data["input_ids"][0][:input_lengths]  # remove padding if exists
        # stop strings
        specific_stop_strings = data.get("stop_strings", [[]])[0]
        stop_strings = self._merge_stop_strings([specific_stop_strings])
        # max new tokens
        remaining_ctx = self.cfg["vllm_cfg"]["max_model_len"] - input_lengths
        allowed_new_tokens = max(0, min(self.cfg["max_new_tokens"], remaining_ctx))

        # Handle case where no tokens can be generated due to length constraints
        if allowed_new_tokens == 0:
            # Create output tensors with just the input (no generated tokens)
            output_ids = input_ids.unsqueeze(0)

            logprobs = torch.zeros(
                (1, input_lengths),
                dtype=torch.float32,
                device=input_ids.device,
            )

            generation_lengths_tensor = torch.tensor(
                [0], dtype=torch.long, device=input_ids.device
            )

            unpadded_sequence_lengths_tensor = torch.tensor(
                [input_lengths],
                dtype=torch.long,
                device=input_ids.device,
            )

            result_batch = BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids": output_ids,
                    "logprobs": logprobs,
                    "generation_lengths": generation_lengths_tensor,
                    "unpadded_sequence_lengths": unpadded_sequence_lengths_tensor,
                }
            )

            return result_batch

        # Convert inputs to vLLM format
        prompt_token_ids_list = input_ids.tolist()
        prompt = {"prompt_token_ids": prompt_token_ids_list}

        sampling_params_for_request = self._build_sampling_params(
            greedy=greedy,
            stop_strings=stop_strings,
            max_new_tokens=allowed_new_tokens,
        )

        # Generate using vLLM async engine
        request_id = str(uuid.uuid4())
        vllm_request_generator = self.llm.generate(
            prompt=prompt,
            sampling_params=sampling_params_for_request,
            request_id=request_id,
        )

        # Get the final result from the generator
        final_request_output = None
        async for req_output in vllm_request_generator:
            final_request_output = req_output

        if final_request_output is None:
            raise RuntimeError(f"No output received for request {request_id}")

        # Process the output
        generation_details = final_request_output.outputs[0]
        generated_token_ids = list(generation_details.token_ids)
        num_generated_tokens = len(generated_token_ids)
        final_output_tensor_len = input_lengths + num_generated_tokens

        # Concat output_ids and reshape to (1, seq_len) for BatchedDataDict
        generate_token_ids_tensor = torch.tensor(
            generated_token_ids,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        output_ids = torch.cat([input_ids, generate_token_ids_tensor], dim=0)
        output_ids = output_ids.unsqueeze(0)

        # Create logprobs tensor for this single item
        logprobs = torch.zeros(
            (1, final_output_tensor_len),
            dtype=torch.float32,
            device=input_ids.device,
        )
        if hasattr(generation_details, "logprobs") and generation_details.logprobs:
            for idx, logprob_dict_per_token in enumerate(generation_details.logprobs):
                if logprob_dict_per_token and idx < len(generated_token_ids):
                    token_id_at_idx = generated_token_ids[idx]
                    if token_id_at_idx in logprob_dict_per_token:
                        logprob_value = logprob_dict_per_token[token_id_at_idx].logprob
                        position_in_output_tensor = input_lengths + idx
                        if position_in_output_tensor < final_output_tensor_len:
                            logprobs[0, position_in_output_tensor] = logprob_value

        # Generation lengths
        generation_lengths_tensor = torch.tensor(
            [num_generated_tokens],
            dtype=torch.long,
            device=input_ids.device,
        )

        # Unpadded sequence lengths (actual_input + actual_generated)
        unpadded_sequence_lengths_tensor = torch.tensor(
            [final_output_tensor_len],
            dtype=torch.long,
            device=input_ids.device,
        )

        result_batch = BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": output_ids,
                "logprobs": logprobs,
                "generation_lengths": generation_lengths_tensor,
                "unpadded_sequence_lengths": unpadded_sequence_lengths_tensor,
            }
        )

        return result_batch

    async def generate_text(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        raise NotImplementedError(
            "generate_text is not implemented with async_engine=True."
        )

    async def report_device_id(self) -> list[str]:
        """Async version of report_device_id."""
        assert self.llm is not None, (
            "Attempting to report device id with either an uninitialized vLLM or non-model-owner"
        )

        result_or_coro = await self.llm.collective_rpc("report_device_id", args=tuple())

        if asyncio.iscoroutine(result_or_coro):
            list_of_worker_results = await result_or_coro
        else:
            list_of_worker_results = result_or_coro

        return cast(list[str], list_of_worker_results)

    async def update_weights_from_ipc_handles(self, data: dict[str, Any]) -> bool:
        """Async version of update_weights_from_ipc_handles.

        Args:
            data (dict): Dictionary mapping device UUIDs (str) to parameter IPC handles.

        Returns:
            bool: True if weights were successfully updated, False otherwise.
        """
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            result_or_coro = await self.llm.collective_rpc(
                "update_weights_from_ipc_handles", args=(data,)
            )

            if asyncio.iscoroutine(result_or_coro):
                worker_results = await result_or_coro
            else:
                worker_results = result_or_coro

            worker_result = worker_results[0]

            if not worker_result:
                print(
                    f"Error: Worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def update_weights_from_collective(self, data: dict[str, Any]) -> bool:
        """Async version of update_weights_from_collective."""
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            result_or_coro = await self.llm.collective_rpc(
                "update_weights_from_collective", args=(data,)
            )

            if asyncio.iscoroutine(result_or_coro):
                worker_results = await result_or_coro
            else:
                worker_results = result_or_coro

            worker_result = worker_results[0]

            if not worker_result:
                print(
                    f"Error: Worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def reset_prefix_cache(self):
        """Async version of reset_prefix_cache."""
        assert self.llm is not None, (
            "Attempting to reset prefix cache with either an uninitialized vLLM or non-model-owner"
        )

        await self.llm.reset_prefix_cache()
        gc.collect()
        torch.cuda.empty_cache()

    async def sleep(self):
        """Async version of sleep."""
        assert self.llm is not None, (
            "Attempting to sleep with either an uninitialized vLLM or non-model-owner"
        )

        # Reset the prefix cache to ensure that prefix cache is not reused after weights are updated
        await self.llm.reset_prefix_cache()
        await self.llm.sleep(level=1)

        gc.collect()
        torch.cuda.empty_cache()

    async def wake_up(self, **kwargs):
        """Async version of wake_up."""
        assert self.llm is not None, (
            "Attempting to wake up with either an uninitialized vLLM or non-model-owner"
        )

        tags = kwargs.get("tags")

        wake_up_args = {}
        if tags is not None:
            wake_up_args["tags"] = tags

        await self.llm.wake_up(**wake_up_args)

    def shutdown(self) -> bool:
        """Clean up vLLM resources."""
        try:
            if self.llm is not None:
                try:
                    self.llm.shutdown()
                except Exception as e_stop:
                    print(f"Error calling shutdown_background_loop: {e_stop}")
                # Explicitly delete the engine. This may trigger its __del__ method.
                del self.llm

            self.llm = None
            self.tokenizer = None

            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

            return True
        except Exception as e:
            print(f"Error during vLLM shutdown: {e}")
            return False
