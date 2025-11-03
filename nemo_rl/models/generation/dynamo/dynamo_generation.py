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
import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Union,
)

import torch
from dynamo._core import DistributedRuntime, Context  # Context for stop_generating support

from nemo_rl.models.generation.dynamo.router import Router, KvRouter, RoundRobinRouter

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.generation.vllm.config import VllmConfig

class DynamoGeneration(GenerationInterface):
    """Dynamo generation interface for RL policies using mock workers for testing."""

    def __init__(
        self,
        config: VllmConfig,
        name_prefix: str = "vllm_policy",
        workers_per_node: Optional[Union[int, list[int]]] = None,
        namespace: str = "dynamo",
        component_name: str = "backend",
        endpoint_name: str = "generate",
        block_size: int = 16,
        partial_rollout: bool = False,
    ):
        self.cfg = config
        self.name_prefix = name_prefix
        self.workers_per_node = workers_per_node
        
        # KvPushRouter configuration
        self.namespace = namespace
        self.component_name = component_name
        self.endpoint_name = endpoint_name
        self.block_size = block_size
        
        # KvPushRouter instance (will be initialized in prepare_for_generation)
        self.dynamo_router: Optional[Router] = None
        self.partial_rollout = partial_rollout
        self.runtime: Optional[DistributedRuntime] = None
        
        # Process handles for dynamo infrastructure and workers
        self.dynamo_processes = []  # List of worker processes
        self.nats_process = None
        self.etcd_process = None
        self.mocker_config_file = None
        
        # Configuration
        self.model_name = self.cfg.get("model_name", "Qwen/Qwen3-0.6B")
        self.served_model_name = self.cfg.get("served_model_name", self.model_name)
        
        # Additional vLLM configuration options
        self.tensor_parallel_size = self.cfg.get("tensor_parallel_size", 1)
        self.data_parallel_size = self.cfg.get("data_parallel_size", 1)
        self.gpu_memory_utilization = self.cfg.get("gpu_memory_utilization", 0.90)
        self.disable_log_requests = self.cfg.get("disable_log_requests", True)
        self.enable_prefix_caching = self.cfg.get("enable_prefix_caching", False)
        
        # Generation control (Dynamo context-based)
        self._generation_context: Optional[List[Context]] = None  # Dynamo context for cancellation
        self._stop_requested = False  # Flag to signal immediate termination

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate responses synchronously, returning a single batch of results.
        
        This method processes all samples in the batch together, returning a single result.
        """
        # Run the async version and collect all results
        async def _async_generate():
            results = {}
            async for sample_idx, result_batch in self.generate_async(data, greedy):
                results[sample_idx] = result_batch
            return results
        
        # Run the async function in the current event loop or create a new one
        try:
            asyncio.get_running_loop()
            # Already in async context, use thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _async_generate())
                results = future.result()
        except RuntimeError:
            results = asyncio.run(_async_generate())
        
        if not results:
            # Return empty batch if no results
            batch_size = data["input_ids"].shape[0] if "input_ids" in data else 0
            device = data["input_ids"].device if "input_ids" in data else torch.device("cpu")
            
            return BatchedDataDict[GenerationOutputSpec]({
                "output_ids": torch.empty((batch_size, 0), dtype=torch.long, device=device),
                "logprobs": torch.empty((batch_size, 0), dtype=torch.float32, device=device),
                "generation_lengths": torch.zeros(batch_size, dtype=torch.long, device=device),
                "unpadded_sequence_lengths": torch.zeros(batch_size, dtype=torch.long, device=device),
            })
        
        # Combine all results into a single batch
        batch_size = len(results)
        sample_indices = sorted(results.keys())
        
        # Get the first result to determine tensor shapes and device
        first_result = results[sample_indices[0]]
        device = first_result["output_ids"].device
        max_seq_len = max(results[i]["output_ids"].shape[1] for i in sample_indices)
        
        # Initialize combined tensors
        combined_output_ids = torch.full(
            (batch_size, max_seq_len),
            self.cfg.get("pad_token_id", 0),
            dtype=first_result["output_ids"].dtype,
            device=device,
        )
        combined_logprobs = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float32,
            device=device,
        )
        combined_generation_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        combined_unpadded_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Fill in the combined tensors
        for batch_idx, sample_idx in enumerate(sample_indices):
            result = results[sample_idx]
            seq_len = result["output_ids"].shape[1]
            
            combined_output_ids[batch_idx, :seq_len] = result["output_ids"][0]
            combined_logprobs[batch_idx, :seq_len] = result["logprobs"][0]
            combined_generation_lengths[batch_idx] = result["generation_lengths"][0]
            combined_unpadded_lengths[batch_idx] = result["unpadded_sequence_lengths"][0]
        
        return BatchedDataDict[GenerationOutputSpec]({
            "output_ids": combined_output_ids,
            "logprobs": combined_logprobs,
            "generation_lengths": combined_generation_lengths,
            "unpadded_sequence_lengths": combined_unpadded_lengths,
        })

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate responses asynchronously, yielding individual samples as they complete.

        This method provides per-sample streaming across all workers, yielding each
        sample result as soon as it's ready, regardless of which worker processed it.
        """
        # Validate input data
        if "input_ids" not in data or "input_lengths" not in data:
            raise AssertionError("input_ids and input_lengths are required in data for Dynamo generation")
        
        if self.dynamo_router is None:
            raise RuntimeError("Dynamo router not initialized. Call prepare_for_generation() first.")

        input_ids_batch = data["input_ids"]
        input_lengths_batch = data["input_lengths"]
        remaining_ctx_batch = data["remaining_ctx_list"]
        batch_size = input_ids_batch.shape[0]

        if batch_size == 0:
            return

        # Get stop strings if provided
        batch_specific_stop_strings_list = data.get(
            "stop_strings", [[] for _ in range(batch_size)]
        )

        # Create tasks for each sample in the batch
        async def process_single_sample(sample_idx):
            """Process a single sample and return the result."""
            try:
                current_input_actual_length = input_lengths_batch[sample_idx].item()
                
                # Get input tokens for this sample
                input_ids_single = input_ids_batch[sample_idx][:current_input_actual_length]
                token_ids = input_ids_single.tolist()
                
                # Get stop strings for this sample
                per_sample_stop_strings = None
                if batch_specific_stop_strings_list and sample_idx < len(batch_specific_stop_strings_list):
                    per_sample_stop_strings = batch_specific_stop_strings_list[sample_idx]

                # Calculate remaining context and allowed tokens
                max_model_len = self.cfg.get("max_model_len", 4096)
                remaining_ctx = remaining_ctx_batch[sample_idx]
                # max_new_tokens = self.cfg.get("max_new_tokens", 256)
                # allowed_new_tokens = max(0, min(max_new_tokens, remaining_ctx))

                # Handle case where no tokens can be generated
                if remaining_ctx == 0:
                    # Create output tensors with just the input (no generated tokens)
                    output_ids_single_item_batched = input_ids_single.unsqueeze(0)
                    
                    logprobs_single_item = torch.zeros(
                        (1, current_input_actual_length),
                        dtype=torch.float32,
                        device=input_ids_single.device,
                    )
                    
                    generation_lengths_tensor = torch.tensor(
                        [0], dtype=torch.long, device=input_ids_single.device
                    )
                    
                    unpadded_sequence_lengths_tensor = torch.tensor(
                        [current_input_actual_length],
                        dtype=torch.long,
                        device=input_ids_single.device,
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

                # Use KvPushRouter to generate
                generated_tokens = []
                logprobs_list = []
                
                # # Get the async iterator from KvPushRouter
                # response_iterator = await self.dynamo_router.generate(
                #     token_ids=token_ids,
                #     model=self.served_model_name,
                #     stop_conditions=stop_conditions,
                #     sampling_options=sampling_options,
                # )
                # TODO: (sechoi) FIX AFTER TESTING
                route_result = await self.dynamo_router.route_request(
                    {"token_ids": token_ids, "remaining_ctx": remaining_ctx}
                )
                response_iterator, stream_context, meta_data = route_result
                selected_instance = meta_data.get("selected_instance", None)
                sampled_max_tokens = meta_data.get("max_tokens", None)

                self._generation_context.append(stream_context)
                
                async for response in response_iterator:
                    if self._stop_requested:
                        break

                    response_data = response.data() if hasattr(response, 'data') else response
                    if response_data is None:
                        continue
                    
                    # Extract token and logprob information from response
                    # The exact structure depends on the LLMEngineOutput format
                    if "token_ids" in response_data and response_data["token_ids"]:
                        generated_tokens.extend(response_data["token_ids"])
                    if "log_probs" in response_data and response_data["log_probs"]:
                        logprobs_list.extend(response_data["log_probs"])
                    
                    # Check if generation is complete
                    if response_data.get("finish_reason") is not None:
                        break

                num_generated_tokens = len(generated_tokens)
                # print(f"generated_tokens len: {num_generated_tokens}")
                
                # Create output tensors
                final_output_tensor_len = current_input_actual_length + num_generated_tokens
                
                output_ids_single_item = torch.full(
                    (final_output_tensor_len,),
                    self.cfg.get("pad_token_id", 0),
                    dtype=input_ids_single.dtype,
                    device=input_ids_single.device,
                )
                
                # Copy original input
                output_ids_single_item[:current_input_actual_length] = input_ids_single
                
                # Add generated tokens
                if num_generated_tokens > 0:
                    output_ids_single_item[
                        current_input_actual_length : current_input_actual_length + num_generated_tokens
                    ] = torch.tensor(
                        generated_tokens,
                        dtype=input_ids_single.dtype,
                        device=input_ids_single.device,
                    )
                
                # Reshape to (1, seq_len) for BatchedDataDict
                output_ids_single_item_batched = output_ids_single_item.unsqueeze(0)
                
                # Create logprobs tensor
                logprobs_single_item = torch.zeros(
                    (1, final_output_tensor_len),
                    dtype=torch.float32,
                    device=input_ids_single.device,
                )
                
                # Fill in logprobs if available
                if logprobs_list and len(logprobs_list) == num_generated_tokens:
                    logprobs_single_item[0, current_input_actual_length:current_input_actual_length + num_generated_tokens] = torch.tensor(
                        logprobs_list, dtype=torch.float32, device=input_ids_single.device
                    )
                
                # Generation lengths
                generation_lengths_tensor = torch.tensor(
                    [num_generated_tokens],
                    dtype=torch.long,
                    device=input_ids_single.device,
                )
                
                # Unpadded sequence lengths
                unpadded_total_length = current_input_actual_length + num_generated_tokens
                unpadded_sequence_lengths_tensor = torch.tensor(
                    [unpadded_total_length],
                    dtype=torch.long,
                    device=input_ids_single.device,
                )
                
                result_batch = BatchedDataDict[GenerationOutputSpec](
                    {
                        "output_ids": output_ids_single_item_batched,
                        "logprobs": logprobs_single_item,
                        "generation_lengths": generation_lengths_tensor,
                        "unpadded_sequence_lengths": unpadded_sequence_lengths_tensor,
                        "stop_reason": "abort" if self._stop_requested else "finished",
                    }
                )
                
                # Add selected_instance and out_remaining_len if available (from routers that return them)
                print(f"DEBUG: Before adding to result_batch, selected_instance={selected_instance}, num_generated_tokens={num_generated_tokens}")
                if selected_instance is not None:
                    result_batch["selected_instance"] = selected_instance
                if sampled_max_tokens is not None:
                    result_batch["out_remaining_len"] = sampled_max_tokens - num_generated_tokens
                
                return (sample_idx, result_batch)
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
                raise e

        # Create a new Dynamo context for this generation batch
        # This context will control all streams generated in this batch
        self._generation_context = []
        self._stop_requested = False  # Reset stop flag for this generation batch
        
        # Create tasks for all samples and yield results as they complete
        sample_tasks = [
            asyncio.create_task(process_single_sample(i)) for i in range(batch_size)
        ]
        
        # Store tasks for cancellation
        self._sample_tasks = sample_tasks

        # Yield results as they become available
        for completed_task in asyncio.as_completed(sample_tasks):
            try:
                result = await completed_task
                yield result
            except asyncio.CancelledError:
                # Task was cancelled, continue to next task
                # Partial results were already handled in process_single_sample
                continue
            except Exception as e:
                # Cancel remaining tasks
                for task in sample_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*sample_tasks, return_exceptions=True)
                raise e

    def get_context(self) -> Optional[Context]:
        return self._generation_context

    def stop_generation(self):
        """Immediately stop all ongoing generation requests and break from iteration loop."""
        # Set flag to break from the async iteration loop immediately
        self._stop_requested = True
        
        if self._generation_context:
            for context in self._generation_context:
                context.stop_generating()

    async def test_dynamo_router(self) -> Dict[str, Any]:
        """Test the Dynamo router to verify it's working correctly."""
        test_results = {"dynamo_router_initialized": False, "generation_works": False, "errors": []}
        
        if not self.dynamo_router:
            test_results["errors"].append("Dynamo router not initialized")
            return test_results
        
        test_results["dynamo_router_initialized"] = True
        
        try:
            route_result = await self.dynamo_router.route_request(
                # token_ids=[1, 2, 3, 4, 5],
                # model=self.served_model_name,
                # stop_conditions={"max_tokens": 1},
                # sampling_options={"temperature": 0.1},
                # output_options={
                #     "include_input_tokens": False,
                #     "return_full_text": False,
                # },
                {"token_ids": [1, 2, 3, 4, 5]},
            )
            
            # Handle routers that return 2 or 3 values (with optional selected_instance)
            if len(route_result) == 3:
                response_iterator, stream_context, selected_instance = route_result
            else:
                response_iterator, stream_context = route_result
                selected_instance = None
            
            response_count = 0
            async for response in response_iterator:
                response_data = response.data() if hasattr(response, 'data') else response
                response_count += 1
                if response_data.get("finish_reason") is not None or response_count >= 5:
                    break
            
            test_results["generation_works"] = response_count > 0
            status = "✅ passed" if response_count > 0 else "❌ failed: No responses"
            print(f"Dynamo Router generation test {status}")
            
            if response_count == 0:
                test_results["errors"].append("No responses received from KvPushRouter")
                
        except Exception as e:
            test_results["errors"].append(f"Generation error: {str(e)}")
            print(f"❌ KvPushRouter generation test error: {e}")
        
        return test_results

    async def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Launch Dynamo infrastructure, Mock worker and initialize KvPushRouter for generation."""
        try:
            print("🚀 Starting Dynamo infrastructure and Mock worker...")
            
            # Check if required binaries are available
            if not self._check_infrastructure_requirements():
                return False
            
            # Clean up old NATS JetStream data
            nats_data_dir = "/tmp/nats"
            if os.path.exists(nats_data_dir):
                print("🧹 Cleaning old NATS JetStream data...")
                shutil.rmtree(nats_data_dir, ignore_errors=True)
            
            # Start NATS server
            print("🔧 Starting NATS server...")
            self.nats_process = subprocess.Popen(
                ["nats-server", "-js"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True
            )
            
            # Clean up old etcd data to avoid stale worker registrations
            etcd_data_dir = "/tmp/etcd-dynamo"
            if os.path.exists(etcd_data_dir):
                print("🧹 Cleaning old etcd data directory...")
                shutil.rmtree(etcd_data_dir, ignore_errors=True)
            
            # Start etcd server
            print("🔧 Starting etcd server...")
            self.etcd_process = subprocess.Popen(
                [
                    "etcd",
                    "--listen-client-urls", "http://0.0.0.0:2379",
                    "--advertise-client-urls", "http://0.0.0.0:2379",
                    "--data-dir", etcd_data_dir
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True
            )
            
            # Wait for infrastructure to be ready
            print("⏳ Waiting for NATS and etcd to be ready...")
            time.sleep(3)  # Give infrastructure time to start
            
            # Launch multiple mock workers based on data_parallel_size
            num_workers = self.data_parallel_size
            print(f"🔧 Starting {num_workers} Dynamo Mock worker(s)...")
            
            # Create single shared mocker configuration file for all workers
            mocker_config = {
                "speedup_ratio": 10.0,  # Makes mock inference faster for testing
                "block_size": self.block_size,
            }
            
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(mocker_config, f)
                self.mocker_config_file = f.name
            
            # Start mock workers with the same config
            for worker_id in range(num_workers):
                dynamo_command = [
                    "python3", "-m", "dynamo.mocker",
                    "--model-path", self.model_name,
                    "--endpoint", f"dyn://{self.namespace}.{self.component_name}.{self.endpoint_name}",
                    "--extra-engine-args", self.mocker_config_file,
                ]
                
                print(f"   Worker {worker_id + 1}/{num_workers}: {' '.join(dynamo_command)}")
                process = subprocess.Popen(
                    dynamo_command,
                    text=True
                )
                self.dynamo_processes.append(process)
            
            # Wait for all workers to be ready
            print(f"⏳ Waiting for {num_workers} Dynamo Mock worker(s) to be ready...")
            if not self._wait_for_workers_ready():
                print("❌ Dynamo Mock workers failed to become ready")
                self._cleanup_processes()
                return False
            
            # Get or create the distributed runtime using the same pattern as tests
            if self.runtime is None:
                self.runtime = self._get_runtime()
            
            self.dynamo_router = RoundRobinRouter(self.runtime, self.namespace, self.component_name, self.endpoint_name, self.served_model_name)
            await self.dynamo_router.initialize(self.block_size, self.cfg.get("max_model_len", 4096))
            
            worker_pids = [p.pid for p in self.dynamo_processes]
            print(f"✅ Dynamo infrastructure initialized!")
            print(f"   Endpoint: {self.namespace}/{self.component_name}/{self.endpoint_name}")
            print(f"   Model: {self.served_model_name} | Block size: {self.block_size}")
            print(f"   Workers: {num_workers} (data_parallel_size)")
            print(f"   PIDs - NATS:{self.nats_process.pid} etcd:{self.etcd_process.pid}")
            print(f"   Worker PIDs: {worker_pids}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting Dynamo worker or initializing KvPushRouter: {e}")
            self._cleanup_processes()
            self.dynamo_router = None
            return False

    def _get_runtime(self) -> DistributedRuntime:
        """Get or create a DistributedRuntime instance safely."""
        try:
            runtime_instance = DistributedRuntime.detached()
            print("📡 Using detached runtime (worker already initialized)")
            return runtime_instance
        except Exception:
            pass
        
        try:
            loop = asyncio.get_running_loop()
            return DistributedRuntime(loop, False)
        except RuntimeError as e:
            raise RuntimeError(
                "Cannot initialize DistributedRuntime: not in async context and no existing worker found. "
                "Ensure this is called from an async context."
            ) from e

    async def _wait_for_workers_registration(self, endpoint, expected_workers: int, max_wait_time: float = 30.0, check_interval: float = 2.0) -> bool:
        """Wait for all workers to register with etcd and be discoverable.
        
        Args:
            endpoint: The endpoint to check for worker registration
            expected_workers: Number of workers expected to register
            max_wait_time: Maximum time to wait in seconds
            check_interval: Time between checks in seconds
            
        Returns:
            True if workers are registered, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                client = await endpoint.client()
                if client is not None:
                    print(f"✅ {expected_workers} worker(s) registered and discoverable!")
                    # Give additional time for all workers to register
                    await asyncio.sleep(5)
                    return True
            except Exception:
                pass
            
            print(f"⏳ Waiting for {expected_workers} worker(s) to register... ({time.time() - start_time:.1f}s)")
            await asyncio.sleep(check_interval)
        
        print("❌ Timeout waiting for workers registration")
        return False

    def _check_infrastructure_requirements(self) -> bool:
        """Check if required infrastructure binaries are available."""
        missing = [b for b in ["nats-server", "etcd"] if not shutil.which(b)]
        
        if missing:
            print(f"❌ Missing required binaries: {', '.join(missing)}")
            print("📋 Install: sudo apt-get install nats-server etcd")
            print("   Or see: https://docs.nats.io & https://etcd.io/docs/v3.5/install/")
            return False
        
        print("✅ All required infrastructure binaries available")
        return True

    def _wait_for_workers_ready(self, max_wait_time: float = 30.0, check_interval: float = 1.0) -> bool:
        """Wait for all Dynamo Mock workers to be ready.
        
        Args:
            max_wait_time: Maximum time to wait in seconds
            check_interval: Time between checks in seconds
            
        Returns:
            True if all workers become ready, False if any terminate or timeout
        """
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            # Check if any process has terminated unexpectedly
            for i, process in enumerate(self.dynamo_processes):
                if process is None or process.poll() is not None:
                    print(f"❌ Dynamo Mock worker {i+1} process terminated unexpectedly")
                    return False
            
            # TODO: sechoi check if there's a function to check this
            # Mock workers start up much faster than vLLM workers
            elapsed = time.time() - start_time
            if elapsed >= 5.0:  # Mock workers ready in ~5 seconds
                print(f"✅ {len(self.dynamo_processes)} Dynamo Mock worker(s) ready!")
                return True
            
            print(f"⏳ Waiting for {len(self.dynamo_processes)} worker(s) to be ready... ({elapsed:.1f}s)")
            time.sleep(check_interval)
        
        print("❌ Timeout waiting for workers")
        return False

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Cleanup KvPushRouter resources and stop Dynamo worker process."""
        return self._cleanup_processes()
    
    def _terminate_process(self, process, name, timeout=10, force_kill=False):
        """Helper to terminate a subprocess with consistent error handling."""
        if process is None:
            return True
        
        try:
            print(f"🛑 Terminating {name}...")
            process.terminate()
            process.wait(timeout=timeout)
            print(f"✅ {name} terminated successfully")
            return True
        except subprocess.TimeoutExpired:
            if force_kill:
                print(f"⚠️  {name} taking longer, forcing kill...")
                process.kill()
                process.wait()
                print(f"✅ {name} forcefully terminated")
                return True
            raise
        except Exception as e:
            print(f"❌ Error terminating {name}: {e}")
            return False

    def _cleanup_processes(self) -> bool:
        """Helper method to clean up all running processes and resources."""
        success = True
        
        # Cleanup KvPushRouter
        if self.dynamo_router is not None:
            try:
                print("🧹 Cleaning up KvPushRouter resources...")
                self.dynamo_router = None
                print("✅ KvPushRouter cleanup completed")
            except Exception as e:
                print(f"❌ Error during KvPushRouter cleanup: {e}")
                success = False
        
        # Terminate worker processes
        for i, process in enumerate(self.dynamo_processes):
            if not self._terminate_process(process, f"Dynamo Mock worker {i+1}"):
                success = False
        self.dynamo_processes = []
        
        if not self._terminate_process(self.nats_process, "NATS server", timeout=5):
            success = False
        self.nats_process = None

        # Wait for etcd to clean up worker registrations via lease expiration
        if success and self.runtime is not None:
            if self._wait_for_worker_cleanup():
                print("✅ All worker registrations cleaned up from etcd")
            else:
                print("⚠️  Worker registrations may still exist in etcd")
        
        if not self._terminate_process(self.etcd_process, "etcd server", timeout=10, force_kill=True):
            success = False
        self.etcd_process = None
        
        # Cleanup runtime
        self.runtime = None
        
        # Cleanup temporary mocker config file
        if self.mocker_config_file is not None:
            try:
                os.unlink(self.mocker_config_file)
                self.mocker_config_file = None
            except Exception as e:
                print(f"⚠️  Error cleaning up mocker config file: {e}")
        
        return success
    
    def _wait_for_worker_cleanup(self, max_wait_time: float = 15.0, check_interval: float = 0.5) -> bool:
        """Wait for worker registrations to be cleaned up from etcd.
        
        Returns True if cleanup is verified, False if timeout or unable to check.
        """
        try:
            etcd_client = self.runtime.etcd_client()
            if not etcd_client:
                return True  # No etcd client, nothing to check
            
            endpoint_path = f"instances/{self.namespace}/{self.component_name}/{self.endpoint_name}"
            start_time = time.time()
            
            # Use asyncio to check etcd
            async def check_registrations():
                while time.time() - start_time < max_wait_time:
                    try:
                        kvs = await etcd_client.kv_get_prefix(endpoint_path)
                        if not kvs or len(kvs) == 0:
                            return True
                        
                        elapsed = time.time() - start_time
                        print(f"⏳ Waiting for {len(kvs)} worker registration(s) to expire... ({elapsed:.1f}s)")
                        await asyncio.sleep(check_interval)
                    except Exception as e:
                        print(f"⚠️  Error checking etcd: {e}")
                        return False
                
                return False
            
            # Run the async check
            try:
                loop = asyncio.get_running_loop()
                # In async context, use thread pool
                import concurrent.futures
                def run_check():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(check_registrations())
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(run_check).result(timeout=max_wait_time + 1)
            except RuntimeError:
                # No event loop, run directly
                return asyncio.run(check_registrations())
                
        except Exception as e:
            print(f"⚠️  Could not verify worker cleanup: {e}")
            return False
    
    def __del__(self):
        """Cleanup resources when object is destroyed."""
        self._cleanup_processes()
    
    # Placeholder methods required for async GRPO integration
    def init_collective(self, ip: str, port: int, world_size: int):
        """Initialize collective communication - not needed for Dynamo."""
        # Return empty list since Dynamo doesn't need collective communication
        return []
    
    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare refit info - not needed for Dynamo."""
        pass
    
    def update_weights_from_ipc_handles(self, ipc_handles: dict[str, Any]) -> bool:
        """Update weights from IPC handles - not needed for Dynamo."""
        return True
    
    def update_weights_from_collective(self):
        """Update weights from collective - not needed for Dynamo."""
        return []