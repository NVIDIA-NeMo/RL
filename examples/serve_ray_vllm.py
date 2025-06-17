import asyncio
import json
import os
import time
from copy import deepcopy
from datetime import datetime

import hydra
import jsonlines
import openai
import ray
from omegaconf import DictConfig, OmegaConf
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app


def start_ray_serve(cfg):
    num_gpus = cfg.cluster.gpus_per_node * cfg.cluster.num_nodes
    num_replicas = num_gpus // cfg.server_cfg.tensor_parallel_size
    print(f"Starting Ray Serve with {num_replicas} replicas")

    llm_config = LLMConfig(
        model_loading_config=dict(
            model_id="dummy_model_id",
            model_source=cfg.model_name,
        ),
        deployment_config=dict(
            autoscaling_config=dict(
                min_replicas=num_replicas,
                max_replicas=num_replicas,
            )
        ),
        engine_kwargs=dict(
            **cfg.server_cfg,
        ),
    )

    app = build_openai_app({"llm_configs": [llm_config]})
    serve.run(app, blocking=False)


async def send_request(
    client, messages, client_cfg, sample_idx, test_idx, extra_env_info
):
    """Send a single async request to the model"""
    try:
        response = await client.chat.completions.create(
            model="dummy_model_id", messages=messages, **client_cfg
        )

        generated_text = response.choices[0].message.content

        result = {
            "sample_idx": sample_idx,
            "test_idx": test_idx,
            "messages": messages,
            "generated_text": generated_text,
            "extra_env_info": extra_env_info,
            "timestamp": datetime.now().isoformat(),
        }

        return result
    except Exception as e:
        print(f"Error in request for sample {sample_idx}, test {test_idx}: {str(e)}")
        return None


def save_generations(generations, output_dir, batch_num):
    """Save generations to a JSONL file atomically, grouped by sample"""
    if not generations:
        return None

    os.makedirs(output_dir, exist_ok=True)
    filename = f"generations_batch_{batch_num:04d}.jsonl"
    filepath = os.path.join(output_dir, filename)
    temp_filepath = filepath + ".tmp"

    # Group generations by sample_idx
    grouped_results = {}
    for result in generations:
        sample_idx = result["sample_idx"]
        if sample_idx not in grouped_results:
            grouped_results[sample_idx] = {
                "sample_idx": sample_idx,
                "messages": result["messages"],
                "extra_env_info": result["extra_env_info"],
                "responses": [],
                "timestamps": [],
            }

        grouped_results[sample_idx]["responses"].append(result["generated_text"])
        grouped_results[sample_idx]["timestamps"].append(result["timestamp"])

    # Write to temporary file first in jsonlines format
    with jsonlines.open(temp_filepath, "w") as writer:
        for sample_idx in sorted(grouped_results.keys()):
            writer.write(grouped_results[sample_idx])

    # Atomic rename - this is the key for crash safety
    os.rename(temp_filepath, filepath)

    print(
        f"Saved {len(grouped_results)} prompts with {len(generations)} total responses to {filepath}"
    )
    return filepath


def save_progress_state_and_generations(
    output_dir, completed_requests, batch_num, start_time, generations=None
):
    """Save the current progress state and any pending generations atomically"""
    state = {
        "completed_requests": completed_requests,
        "batch_num": batch_num,
        "start_time": start_time,
        "last_save_time": time.time(),
        "timestamp": datetime.now().isoformat(),
    }

    state_file = os.path.join(output_dir, "progress_state.json")
    temp_state_file = state_file + ".tmp"

    # Write to temporary file first
    with open(temp_state_file, "w") as f:
        json.dump(state, f, indent=2)

    # Atomic rename - ensures the state file is never corrupted
    os.rename(temp_state_file, state_file)

    print(f"Progress state saved. Completed: {len(completed_requests)} requests")

    # Save generations if provided
    if generations and len(generations) > 0:
        filepath = save_generations(generations, output_dir, batch_num)
        if filepath:
            print("Also saved generations grouped by prompt with progress state")
            return batch_num + 1  # Return next batch number

    return batch_num


def save_progress_state(output_dir, completed_requests, batch_num, start_time):
    """Save the current progress state for resuming atomically"""
    return save_progress_state_and_generations(
        output_dir, completed_requests, batch_num, start_time
    )


def load_progress_state(output_dir):
    """Load previous progress state if it exists"""
    state_file = os.path.join(output_dir, "progress_state.json")

    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            state = json.load(f)

        print(
            f"Resuming from previous run. Already completed: {len(state['completed_requests'])} requests"
        )
        return state

    return None


def should_skip_request(sample_idx, test_idx, completed_requests):
    """Check if this request was already completed"""
    request_id = f"{sample_idx}_{test_idx}"
    return request_id in completed_requests


async def process_requests_concurrently(cfg):
    """Process all requests concurrently with batched saving and timer"""
    client_cfg = OmegaConf.to_container(cfg.client_cfg, resolve=True)
    num_tests_per_prompt = cfg.num_tests_per_prompt
    max_concurrent = cfg.max_concurrent_requests
    output_dir = cfg.output_dir
    max_runtime_hours = cfg.max_runtime_hours

    os.makedirs(output_dir, exist_ok=True)

    # Load previous progress if exists
    previous_state = load_progress_state(output_dir)
    if previous_state:
        completed_requests = set(previous_state["completed_requests"])
        batch_num = previous_state["batch_num"]
        start_time = previous_state["start_time"]
        print(f"Resuming from batch {batch_num}")
    else:
        completed_requests = set()
        batch_num = 0
        start_time = time.time()
        print("Starting fresh run")

    # Use async OpenAI client
    async_client = openai.AsyncOpenAI(base_url="http://localhost:8000/v1")

    with jsonlines.open(cfg.data.jsonl_path, "r") as reader:
        data = [line for line in reader]

    # Prepare all requests (skip already completed ones)
    all_requests = []
    total_requests = 0
    skipped_requests = 0

    for i, sample in enumerate(data):
        assert len(sample["messages"]) == 1
        single_message = sample["messages"][0]

        # Extract extra_env_info
        extra_env_info = {}
        for m in single_message:
            if m["role"] == "user":
                extra_env_info = deepcopy(m.get("metadata", {}))

        # Create multiple requests for this sample
        for test_idx in range(num_tests_per_prompt):
            total_requests += 1

            if should_skip_request(i, test_idx, completed_requests):
                skipped_requests += 1
                continue

            request_coro = send_request(
                async_client, single_message, client_cfg, i, test_idx, extra_env_info
            )
            all_requests.append((request_coro, i, test_idx))

    print(f"Total requests: {total_requests}")
    print(f"Already completed: {skipped_requests}")
    print(f"Remaining to process: {len(all_requests)}")
    print(f"Max runtime: {max_runtime_hours} hours")

    # Process requests in batches
    all_results = []
    processed_count = 0
    last_save_time = time.time()
    save_interval = 300  # Save progress every 5 minutes

    for i in range(0, len(all_requests), max_concurrent):
        # Check timer before starting new batch
        elapsed_time = time.time() - start_time
        elapsed_hours = elapsed_time / 3600

        if elapsed_hours >= max_runtime_hours:
            print(f"\nTime limit reached ({elapsed_hours:.2f} hours).")
            print("Stopping before starting new batch. Saving remaining results...")
            break

        batch = all_requests[i : i + max_concurrent]
        batch_requests = [req[0] for req in batch]
        batch_metadata = [(req[1], req[2]) for req in batch]

        remaining_time = max_runtime_hours - elapsed_hours
        print(
            f"Processing batch {i // max_concurrent + 1}/{(len(all_requests) + max_concurrent - 1) // max_concurrent}"
        )
        print(f"Batch size: {len(batch)}, Remaining time: {remaining_time:.2f} hours")

        # Create tasks from coroutines and map them to their metadata for result processing
        batch_tasks = [asyncio.create_task(req[0]) for req in batch]
        task_to_metadata = {
            task: (batch[i][1], batch[i][2]) for i, task in enumerate(batch_tasks)
        }

        # Process requests as they complete instead of waiting for all to finish
        pending_tasks = set(batch_tasks)

        while pending_tasks:
            try:
                # Wait for at least one task to complete
                done, pending_tasks = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                # Process all completed tasks
                for task in done:
                    try:
                        result = await task
                        if result is not None:
                            all_results.append(result)
                            sample_idx, test_idx = task_to_metadata[task]
                            completed_requests.add(f"{sample_idx}_{test_idx}")
                            processed_count += 1

                            # Check if we should save progress while processing
                            current_time = time.time()
                            if current_time - last_save_time >= save_interval:
                                batch_num = save_progress_state_and_generations(
                                    output_dir,
                                    list(completed_requests),
                                    batch_num,
                                    start_time,
                                    all_results,
                                )
                                all_results = []  # Clear results after saving
                                last_save_time = current_time

                    except Exception as e:
                        print(f"Error in request: {str(e)}")
                        # Find the metadata for this failed request
                        if task in task_to_metadata:
                            metadata = task_to_metadata[task]
                            print(
                                f"Failed request was for sample {metadata[0]}, test {metadata[1]}"
                            )

            except Exception as e:
                print(f"Error in batch processing: {str(e)}")
                break

            # Small delay to prevent overwhelming the server
            await asyncio.sleep(0.1)

    # Final save with any remaining results
    batch_num = save_progress_state_and_generations(
        output_dir, list(completed_requests), batch_num, start_time, all_results
    )

    elapsed_time = time.time() - start_time
    print("\nProcessing completed!")
    print(f"Total runtime: {elapsed_time / 3600:.2f} hours")
    print(f"Processed {processed_count} new requests")
    print(f"Total completed: {len(completed_requests)} requests")
    print(f"Results saved in {output_dir}/")


@hydra.main(version_base=None, config_path="configs", config_name="ray_serve")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    start_ray_serve(cfg)

    try:
        # Run the async processing
        asyncio.run(process_requests_concurrently(cfg))
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Shutting down gracefully...")
    finally:
        print("Shutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    main()
