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
import threading
import uuid
from typing import Any, AsyncGenerator, Optional, cast

import ray
import torch
import uvicorn
from fastapi import FastAPI
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import _get_free_port_local, _get_node_ip_local
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.utils import format_prompt_for_vllm_generation
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker


def _maybe_correct_merged_tokens(
    tokenizer: PreTrainedTokenizerBase,
    reference_token_ids: list[int],
    actual_token_ids: list[int],
) -> list[int]:
    """This is a subroutine used inside the vLLM Chat Completion server. Some environments (namely Penguin) require an OpenAI compatible server endpoint rather than an inference engine handle. This is fine for the most part, but it may cause issues when the environment is used as a part of training.

    RL training frameworks train models on token IDs, but the OpenAI compatible server communicates in what is basically de-tokenized text. When multiple model calls are made to the OpenAI compatible server in a single trajectory, model generations in previous model calls may be re-tokenized to something that is different than what was generated. This is not too big of an issue (that we know of) at inference time, but the log probs the model produces are different enough for the differently re-tokenized generation result that it causes the training to be off policy. Off policy isn't necessarily a bad thing in isolation, but this source of off-policyness may cause unexpected issues if not properly accounted for. It also mis-aligns the token ID sequences across model calls, which feels very strange during training.

    Thus, in this function we attempt to correct any minor re-tokenization errors in an effort to stay on-policy as possible. We require the tokenizer, the ground truth reference token ids taken directly from previous model calls, and the re-tokenized actual token ids.

    In other words, for the current model call:
    - reference_token_ids = all_prefill_so_far + new_generation
        - all_prefill_so_far: the last model call model engine input token ids. Literally what the model sees during the last generation call.
        - new_generation: the last model call model engine generated token ids. Literally what the model generates during the last generation call.
    - actual_token_ids = all_prefill_so_far_maybe_diff_tokenization + new_generation_maybe_diff_tokenization + tool_response_or_user + assistant_generation_prompt
        - all_prefill_so_far_maybe_diff_tokenization: the re-tokenized version of all_prefill_so_far. Since the token IDs in all_prefill_so_far were de-tokenized and returned as OpenAI schema, they must be re-tokenized for the current model call, which means that it may differ from all_prefill_so_far
        - new_generation_maybe_diff_tokenization: analogous version of all_prefill_so_far_maybe_diff_tokenization for new_generation
        - tool_response_or_user: some returned user or tool message. It doesn't matter that this is tokenized here since it has never been tokenized before. However, at the next model call, this will become part of the all_prefill_so_far.
        - assistant_generation_prompt: a common sequence of tokens to instruct the model to generate an assistant response.

    The goal of this subroutine is to find the prefix in actual_token_ids that corresponds to the de-tokenized text of reference_token_ids.
    The idea of this subroutine implementation is to just de-tokenize subsequences of actual_token_ids (called candidate_token_ids) until the de-tokenized text matches the de-tokenized text of reference_token_ids.

    TODO When NeMo RL supports training image generation models, we want to revisit and possibly update this function. This issue occurs when the model generates tokens that are de-tokenized into text or images, and then re-tokenized into tokens. So if there is a situation like that with images and image tokenization is non-unique, then we will need to uppdate this function.
    """
    if not reference_token_ids:
        return actual_token_ids

    # No re-tokenization errors
    if reference_token_ids == actual_token_ids[: len(reference_token_ids)]:
        return actual_token_ids

    reference_str, actual_str = tokenizer.batch_decode(
        [reference_token_ids, actual_token_ids]
    )

    # For now, if a trajectory is not monotonically increasing, we assert.
    # Eventually when we support non-monotonic training, we need to update this logic
    assert (
        reference_str == actual_str[: len(reference_str)]
    ), f"""Found a non-monotonically increasing trajectory that is not caused by a token merge on re-tokenization!
Reference str: {reference_str}
Actual str: {actual_str}

Reference token ids: {reference_token_ids}
Actual token ids: {actual_token_ids}"""

    # Now we want to try to find the subsequence of actual_token_ids that corresponds to reference_str
    # Our first guess is just the prefix in actual_token_ids of length reference_token_ids. How good of a guess this is depends on the distribution of the number of re-tokenization errors.
    # If there are a lot, this will be a poor guess. If there aren't that many this is a good guess.
    candidate_token_ids = actual_token_ids[: len(reference_token_ids)]
    candidate_str = tokenizer.decode(candidate_token_ids)

    # If it's longer, we remove
    if len(candidate_str) > len(reference_str):
        while (
            candidate_str != reference_str
            and len(candidate_str) > len(reference_str)
            and candidate_token_ids
        ):
            candidate_token_ids.pop()
            candidate_str = tokenizer.decode(candidate_token_ids)
    # If it's shorter we append
    elif len(candidate_str) < len(reference_str):
        while (
            candidate_str != reference_str
            and len(candidate_str) < len(reference_str)
            and len(candidate_token_ids) < len(actual_token_ids) - 1
        ):
            candidate_token_ids.append(actual_token_ids[len(candidate_token_ids)])
            candidate_str = tokenizer.decode(candidate_token_ids)
    # If it's equal we should not need to do any modification. The assert below will directly error out.
    else:
        pass

    # If we break above, it must be that we either found a correct match or that we didn't find a valid match
    # e.g. in cases where there is some token merging that occurs at the very end of the reference_token_ids
    # We scream loudly here.
    assert candidate_str == reference_str

    return reference_token_ids + actual_token_ids[len(candidate_token_ids) :]


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_async_generation_worker")}
)  # pragma: no cover
class VllmAsyncGenerationWorker(BaseVllmGenerationWorker):
    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        from vllm.config import CompilationConfig
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        # (TODO: zhiyul) Remove this workaround after upgrading vLLM where the compilation_config passing issue is resolved.
        if llm_kwargs.get("compilation_config", None):
            llm_kwargs["compilation_config"] = CompilationConfig(
                **llm_kwargs["compilation_config"]
            )

        self.llm_async_engine_args = AsyncEngineArgs(**llm_kwargs)
        self.llm = AsyncLLM.from_engine_args(self.llm_async_engine_args)

        self.server_thread, self.base_url, self.http_server = None, None, None
        if self.cfg["vllm_cfg"].get("expose_http_server"):
            self.server_thread, self.base_url, self.http_server = (
                self._setup_vllm_server()
            )

    async def post_init_async(self):
        self.vllm_device_ids = await self.report_device_id_async()

    async def report_dp_openai_server_base_url(self) -> Optional[str]:
        return self.base_url

    def _setup_vllm_openai_api_server(self, app: FastAPI) -> FastAPI:
        from typing import List, Optional, Union

        from fastapi import Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from vllm.entrypoints.openai.api_server import (
            BaseModelPath,
            OpenAIServingChat,
            OpenAIServingModels,
            OpenAIServingTokenization,
        )
        from vllm.entrypoints.openai.protocol import (
            ChatCompletionRequest,
            ChatCompletionResponse,
            ErrorResponse,
            TokenizeChatRequest,
            TokenizeCompletionRequest,
            TokenizeResponse,
        )

        engine_client = self.llm
        model_config = self.llm_async_engine_args.create_model_config()
        base_model_paths = [
            BaseModelPath(name=model_config.model, model_path=model_config.model)
        ]

        openai_serving_models = OpenAIServingModels(
            engine_client=engine_client,
            model_config=model_config,
            base_model_paths=base_model_paths,
            lora_modules=None,
        )

        class NeMoRLOpenAIChatRequestMixin:
            def model_post_init(self, context):
                # Penguin specific processing. This is just how Penguin returns the extra token information.
                if self.required_prefix_token_ids is None:
                    for message in reversed(self.messages):
                        if "prompt_token_ids" in message:
                            self.required_prefix_token_ids = (
                                message["prompt_token_ids"]
                                + message["generation_token_ids"]
                            )
                            break

                return super().model_post_init(context)

        class NeMoRLOpenAIServingMixin:
            async def _preprocess_chat(
                self,
                request: NeMoRLOpenAIChatRequestMixin,
                tokenizer,
                messages,
                chat_template,
                chat_template_content_format,
                add_generation_prompt=True,
                continue_final_message=False,
                tool_dicts=None,
                documents=None,
                chat_template_kwargs=None,
                tool_parser=None,
                truncate_prompt_tokens=None,
                add_special_tokens=False,
            ):
                # res is conversation, [request_prompt], [engine_prompt]
                res = await super()._preprocess_chat(
                    request,
                    tokenizer,
                    messages,
                    chat_template,
                    chat_template_content_format,
                    add_generation_prompt,
                    continue_final_message,
                    tool_dicts,
                    documents,
                    chat_template_kwargs,
                    tool_parser,
                    truncate_prompt_tokens,
                    add_special_tokens,
                )

                if request.required_prefix_token_ids is None:
                    return res

                engine_prompt = res[2][
                    0
                ]  # We need to modify engine_prompt.prompt_token_ids

                final_prompt_token_ids = _maybe_correct_merged_tokens(
                    tokenizer=tokenizer,
                    reference_token_ids=request.required_prefix_token_ids,
                    actual_token_ids=engine_prompt["prompt_token_ids"],
                )

                engine_prompt["prompt_token_ids"] = final_prompt_token_ids

                return res

        ########################################
        # /v1/chat/completions endpoint
        ########################################

        # This MRO is necessary i.e. NeMoRLOpenAIChatRequestMixin > ChatCompletionRequest
        class NeMoRLChatCompletionRequest(
            NeMoRLOpenAIChatRequestMixin, ChatCompletionRequest
        ):
            required_prefix_token_ids: Optional[List[int]] = None

        # This MRO is necessary i.e. NeMoRLOpenAIServingMixin > OpenAIServingChat
        class NeMoRLOpenAIServingChat(NeMoRLOpenAIServingMixin, OpenAIServingChat):
            pass

        serving_chat_default_kwargs = dict(
            response_role="assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )
        serving_chat_kwargs = serving_chat_default_kwargs | self.cfg["vllm_cfg"].get(
            "http_server_serving_chat_kwargs", dict()
        )
        openai_serving_chat = NeMoRLOpenAIServingChat(
            engine_client,
            model_config,
            openai_serving_models,
            return_tokens_as_token_ids=True,
            **serving_chat_kwargs,
        )

        generation_config = self.cfg

        # The create_chat_completion and tokenize methods are taken from vllm/entrypoints/openai/api_server.py
        @app.post("/v1/chat/completions")
        async def create_chat_completion(
            request: NeMoRLChatCompletionRequest, raw_request: Request
        ):
            # This needs to match the behavior in nemo_rl/models/generation/vllm/vllm_worker.py::BaseVllmGenerationWorker::_build_sampling_params
            # Right now we explicitly assert set this to -1.
            assert request.top_k in (None, -1), (
                f"Top k sampling parameter must be unset, empty, or -1. Got `{request.top_k}`"
            )
            request.top_k = -1

            # The request sampling params need to exactly match those as are set in NeMo RL.
            # If they do not match, the inference will be off policy and destroy training stability.
            assert request.temperature == generation_config["temperature"]
            assert request.top_p == generation_config["top_p"]

            generator = await openai_serving_chat.create_chat_completion(
                request, raw_request
            )

            if isinstance(generator, ErrorResponse):
                return JSONResponse(
                    content=generator.model_dump(), status_code=generator.code
                )

            elif isinstance(generator, ChatCompletionResponse):
                return JSONResponse(content=generator.model_dump())

            return StreamingResponse(content=generator, media_type="text/event-stream")

        ########################################
        # /tokenize endpoint
        ########################################

        # This MRO is necessary i.e. NeMoRLOpenAIChatRequestMixin > TokenizeRequest
        class NeMoRLTokenizeChatRequest(
            NeMoRLOpenAIChatRequestMixin, TokenizeChatRequest
        ):
            required_prefix_token_ids: Optional[List[int]] = None

        NeMoRLTokenizeRequest = Union[
            TokenizeCompletionRequest, NeMoRLTokenizeChatRequest
        ]

        # This MRO is necessary i.e. NeMoRLOpenAIServingMixin > OpenAIServingTokenization
        class NeMoRLOpenAIServingTokenization(
            NeMoRLOpenAIServingMixin, OpenAIServingTokenization
        ):
            pass

        openai_serving_tokenization = NeMoRLOpenAIServingTokenization(
            engine_client,
            model_config,
            openai_serving_models,
            request_logger=serving_chat_kwargs["request_logger"],
            chat_template=serving_chat_kwargs["chat_template"],
            chat_template_content_format=serving_chat_kwargs[
                "chat_template_content_format"
            ],
        )

        @app.post("/tokenize")
        async def tokenize(request: NeMoRLTokenizeRequest, raw_request: Request):
            generator = await openai_serving_tokenization.create_tokenize(
                request, raw_request
            )

            if isinstance(generator, ErrorResponse):
                return JSONResponse(
                    content=generator.model_dump(), status_code=generator.code
                )
            elif isinstance(generator, TokenizeResponse):
                return JSONResponse(content=generator.model_dump())

        return app

    def _setup_vllm_server(self) -> "tuple[threading.Thread, str, uvicorn.Server]":
        import threading

        import uvicorn
        from fastapi import FastAPI

        # We initialize the FastAPI app here in case we want to do some generic configuration before the subsequent server inits
        # e.g. last-run middleware.
        app = FastAPI()

        app = self._setup_vllm_openai_api_server(app)

        ########################################
        # Server spinup
        ########################################

        node_ip = _get_node_ip_local()
        free_port = _get_free_port_local()

        base_url = f"http://{node_ip}:{free_port}/v1"
        print(f"Starting server on {base_url}")

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=free_port,
        )
        server = uvicorn.Server(config=config)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        return thread, base_url, server

    async def init_collective_async(
        self, rank_prefix: int, ip: str, port: int, world_size: int
    ) -> None:
        await self.llm.collective_rpc(
            "init_collective",
            args=(
                rank_prefix,
                ip,
                port,
                world_size,
            ),
        )

    async def generate_async(
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
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_async can only be used when async_engine is enabled in vLLM config."
            )

        # Handle empty input case
        if len(data["input_ids"]) == 0:
            return

        verify_right_padding(data, pad_value=self.cfg["pad_token_id"])

        input_ids_batch = data["input_ids"]
        input_lengths_batch = data["input_lengths"]
        batch_size = input_ids_batch.shape[0]

        # Ensure generate_async only receives single samples (batch_size = 1)
        assert batch_size == 1, (
            f"generate_async is restricted to handle only single samples, "
            f"but received batch_size={batch_size}. Please handle batching outside this method."
        )

        batch_specific_stop_strings_list = data.get(
            "stop_strings", [[] for _ in range(batch_size)]
        )

        # Create tasks for each sample in the batch
        async def process_single_sample(sample_idx):
            """Process a single sample and return the result."""
            current_input_actual_length = input_lengths_batch[sample_idx].item()
            prompt = format_prompt_for_vllm_generation(data, sample_idx)

            per_sample_stop_strings = None
            if batch_specific_stop_strings_list and sample_idx < len(
                batch_specific_stop_strings_list
            ):
                per_sample_stop_strings = batch_specific_stop_strings_list[sample_idx]

            final_stop_strings_for_sample = self._merge_stop_strings(
                [per_sample_stop_strings] if per_sample_stop_strings else None
            )

            remaining_ctx = (
                self.cfg["vllm_cfg"]["max_model_len"] - current_input_actual_length
            )
            allowed_new_tokens = max(0, min(self.cfg["max_new_tokens"], remaining_ctx))

            # Handle case where no tokens can be generated due to length constraints
            if allowed_new_tokens == 0:
                # Access the input data directly from the function parameters
                input_ids_single_row = input_ids_batch[sample_idx]

                # Create output tensors with just the input (no generated tokens)
                output_ids_single_item_batched = input_ids_single_row[
                    :current_input_actual_length
                ].unsqueeze(0)

                logprobs_single_item = torch.zeros(
                    (1, current_input_actual_length),
                    dtype=torch.float32,
                    device=input_ids_single_row.device,
                )

                generation_lengths_tensor = torch.tensor(
                    [0], dtype=torch.long, device=input_ids_single_row.device
                )

                unpadded_sequence_lengths_tensor = torch.tensor(
                    [current_input_actual_length],
                    dtype=torch.long,
                    device=input_ids_single_row.device,
                )

                result_batch = BatchedDataDict[GenerationOutputSpec](
                    {
                        "output_ids": output_ids_single_item_batched,
                        "logprobs": logprobs_single_item,
                        "generation_lengths": generation_lengths_tensor,
                        "unpadded_sequence_lengths": unpadded_sequence_lengths_tensor,
                    }
                )

                return (sample_idx, result_batch)

            sampling_params_for_request = self._build_sampling_params(
                greedy=greedy,
                stop_strings=final_stop_strings_for_sample,
                max_new_tokens=allowed_new_tokens,
            )

            request_id = str(uuid.uuid4())

            # Generate using vLLM async engine
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

            original_input_ids_single_row = input_ids_batch[sample_idx]
            final_output_tensor_len = current_input_actual_length + num_generated_tokens

            # Create output_ids tensor for this single item
            output_ids_single_item = torch.full(
                (final_output_tensor_len,),
                self.cfg["pad_token_id"],
                dtype=original_input_ids_single_row.dtype,
                device=original_input_ids_single_row.device,
            )
            # Copy original input (up to its actual length)
            output_ids_single_item[:current_input_actual_length] = (
                original_input_ids_single_row[:current_input_actual_length]
            )
            # Add generated tokens after the actual input
            output_ids_single_item[
                current_input_actual_length : current_input_actual_length
                + num_generated_tokens
            ] = torch.tensor(
                generated_token_ids,
                dtype=original_input_ids_single_row.dtype,
                device=original_input_ids_single_row.device,
            )

            # Reshape to (1, seq_len) for BatchedDataDict
            output_ids_single_item_batched = output_ids_single_item.unsqueeze(0)

            # Create logprobs tensor for this single item
            logprobs_single_item = torch.zeros(
                (1, final_output_tensor_len),
                dtype=torch.float32,
                device=original_input_ids_single_row.device,
            )
            if hasattr(generation_details, "logprobs") and generation_details.logprobs:
                for idx, logprob_dict_per_token in enumerate(
                    generation_details.logprobs
                ):
                    if logprob_dict_per_token and idx < len(generated_token_ids):
                        token_id_at_idx = generated_token_ids[idx]
                        if token_id_at_idx in logprob_dict_per_token:
                            logprob_value = logprob_dict_per_token[
                                token_id_at_idx
                            ].logprob
                            position_in_output_tensor = (
                                current_input_actual_length + idx
                            )
                            if position_in_output_tensor < final_output_tensor_len:
                                logprobs_single_item[0, position_in_output_tensor] = (
                                    logprob_value
                                )

            # Generation lengths
            generation_lengths_tensor = torch.tensor(
                [num_generated_tokens],
                dtype=torch.long,
                device=original_input_ids_single_row.device,
            )

            # Unpadded sequence lengths (actual_input + actual_generated)
            unpadded_total_length = current_input_actual_length + num_generated_tokens
            unpadded_sequence_lengths_tensor = torch.tensor(
                [unpadded_total_length],
                dtype=torch.long,
                device=original_input_ids_single_row.device,
            )

            result_batch = BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids": output_ids_single_item_batched,
                    "logprobs": logprobs_single_item,
                    "generation_lengths": generation_lengths_tensor,
                    "unpadded_sequence_lengths": unpadded_sequence_lengths_tensor,
                }
            )

            return (sample_idx, result_batch)

        # Create tasks for all samples and yield results as they complete
        sample_tasks = [
            asyncio.create_task(process_single_sample(i)) for i in range(batch_size)
        ]

        # Yield results as they become available
        for completed_task in asyncio.as_completed(sample_tasks):
            try:
                result = await completed_task
                yield result
            except Exception as e:
                # Cancel remaining tasks
                for task in sample_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*sample_tasks, return_exceptions=True)
                raise e

    async def generate_text_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate text responses asynchronously, yielding results as they are ready.

        Args:
            data: BatchedDataDict containing prompts with text strings
            greedy: Whether to use greedy decoding instead of sampling

        Yields:
            Tuple of (original_index, BatchedDataDict containing single text response)
        """
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_text_async can only be used when async_engine is enabled in vLLM config."
            )

        # Handle empty input case
        if len(data["prompts"]) == 0:
            return

        prompts = data["prompts"]
        batch_size = len(prompts)

        # Extract stop_strings if provided, else use default from config
        batch_stop_strings: list[list[str] | None] = data.get(
            "stop_strings", [self.cfg.get("stop_strings")] * batch_size
        )

        # Create tasks for each prompt
        async def process_single_prompt(prompt_idx):
            """Process a single prompt and return the result."""
            prompt = prompts[prompt_idx]

            # Get stop strings for this specific prompt
            per_prompt_stop_strings = None
            if batch_stop_strings and prompt_idx < len(batch_stop_strings):
                per_prompt_stop_strings = batch_stop_strings[prompt_idx]

            # Merge stop strings
            final_stop_strings = self._merge_stop_strings(
                [per_prompt_stop_strings] if per_prompt_stop_strings else None
            )

            # Create sampling parameters
            top_k = self.cfg["top_k"] if self.cfg["top_k"] is not None else -1
            sampling_params = self.SamplingParams(
                temperature=self.cfg["temperature"] if not greedy else 0,
                top_p=self.cfg["top_p"],
                top_k=top_k if not greedy else 1,
                max_tokens=self.cfg["max_new_tokens"],
                stop_token_ids=self.cfg["stop_token_ids"],
                stop=final_stop_strings,
                include_stop_str_in_output=True,  # returning stop strings like hf
            )

            request_id = str(uuid.uuid4())

            # Generate using vLLM async engine
            vllm_request_generator = self.llm.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            # Get the final result from the generator
            final_request_output = None
            async for req_output in vllm_request_generator:
                final_request_output = req_output

            if final_request_output is None:
                raise RuntimeError(f"No output received for request {request_id}")

            # Extract the generated text
            generated_text = final_request_output.outputs[0].text

            # Create result in BatchedDataDict format
            result_batch = BatchedDataDict[GenerationOutputSpec](
                {"texts": [generated_text]}
            )

            return (prompt_idx, result_batch)

        # Create tasks for all prompts and yield results as they complete
        prompt_tasks = [
            asyncio.create_task(process_single_prompt(i)) for i in range(batch_size)
        ]

        # Yield results as they become available
        for completed_task in asyncio.as_completed(prompt_tasks):
            try:
                result = await completed_task
                yield result
            except Exception as e:
                # Cancel remaining tasks
                for task in prompt_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*prompt_tasks, return_exceptions=True)
                raise e

    async def report_device_id_async(self) -> list[str]:
        """Async version of report_device_id."""
        assert self.llm is not None, (
            "Attempting to report device id with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "report_device_id_async can only be used with async_engine=True. Use report_device_id instead."
            )

        result_or_coro = await self.llm.collective_rpc("report_device_id", args=tuple())

        if asyncio.iscoroutine(result_or_coro):
            list_of_worker_results = await result_or_coro
        else:
            list_of_worker_results = result_or_coro

        return cast(list[str], list_of_worker_results)

    async def prepare_refit_info_async(self, state_dict_info: dict[str, Any]) -> None:
        """Async version of prepare_refit_info."""
        await self.llm.collective_rpc("prepare_refit_info", args=(state_dict_info,))

    async def update_weights_from_ipc_handles_async(
        self, ipc_handles: dict[str, Any]
    ) -> bool:
        """Async version of update_weights_from_ipc_handles.

        Args:
            ipc_handles (dict): Dictionary mapping device UUIDs (str) to parameter IPC handles.

        Returns:
            bool: True if weights were successfully updated, False otherwise.
        """
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if not self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_from_ipc_handles_async can only be used with async_engine=True. Use update_weights_from_ipc_handles instead."
                )

            # TODO: switch to update_weights_from_local_ipc_handles for better performance once collectively report_device_id is supported in asyncLLM initialization
            result_or_coro = await self.llm.collective_rpc(
                "update_weights_from_global_ipc_handles", args=(ipc_handles,)
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

    async def update_weights_from_collective_async(self) -> bool:
        """Async version of update_weights_from_collective."""
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if not self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_from_collective_async can only be used with async_engine=True. Use update_weights_from_collective instead."
                )

            result_or_coro = await self.llm.collective_rpc(
                "update_weights_from_collective", args=tuple()
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

    async def reset_prefix_cache_async(self):
        """Async version of reset_prefix_cache."""
        assert self.llm is not None, (
            "Attempting to reset prefix cache with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "reset_prefix_cache_async can only be used with async_engine=True. Use reset_prefix_cache instead."
            )

        await self.llm.reset_prefix_cache()
        gc.collect()
        torch.cuda.empty_cache()

    async def sleep_async(self):
        """Async version of sleep."""
        assert self.llm is not None, (
            "Attempting to sleep with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "sleep_async can only be used with async_engine=True. Use sleep instead."
            )

        # Reset the prefix cache to ensure that prefix cache is not reused after weights are updated
        await self.llm.reset_prefix_cache()
        await self.llm.sleep(level=1)

        gc.collect()
        torch.cuda.empty_cache()

    async def wake_up_async(self, **kwargs):
        """Async version of wake_up."""
        assert self.llm is not None, (
            "Attempting to wake up with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "wake_up_async can only be used with async_engine=True. Use wake_up instead."
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

            if self.server_thread is not None:
                from threading import Thread

                from uvicorn import Server

                self.http_server: Server
                self.server_thread: Thread

                self.http_server.should_exit = True
                self.server_thread.join()

            return True
        except Exception as e:
            print(f"Error during vLLM shutdown: {e}")
            return False
