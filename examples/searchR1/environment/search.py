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

import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import ray
import requests
import torch
from utils import compute_score, extract_solution

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1


class SearchEnvConfig(TypedDict):
    num_workers: int = 1
    topk: int = 3
    search_url: str = "http://localhost:8000/retrieve"
    timeout: float = DEFAULT_TIMEOUT
    max_turns: int = 10


class SearchEnvMetadata(TypedDict):
    ground_truth: str
    num_turns: int = 0


class SearchWorkerResult(TypedDict):
    observation: str
    reward: float
    done: bool
    error: Optional[str]
    metadata: SearchEnvMetadata
    answer: Optional[str]


def _call_search_api(
    retrieval_service_url: str,
    query: str,
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    log_requests: bool = True,
    session: Optional[requests.Session] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Calls the search API with a single query.

    Args:
        retrieval_service_url: The URL of the search API.
        query: The query to search for.
        topk: The number of results to return.
        return_scores: Whether to return scores for the results.
        timeout: The timeout for the request.
        log_requests: Whether to log requests.
        session: The session to use for the request. If none is provided, a new session will be created.

    Returns:
        response: The response from the search API (json if successful, None otherwise)
        error_msg: The error message if the request failed.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    payload = {"query": query, "topk": topk, "return_scores": return_scores}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Use provided session or create a new one for this request
    if session is None:
        session = requests.Session()
        should_close_session = True
    else:
        should_close_session = False

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if log_requests:
                logger.info(
                    f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling search API at {retrieval_service_url}"
                )
            response = session.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt {attempt + 1}/{MAX_RETRIES}"
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            if log_requests:
                logger.info(
                    f"{log_prefix}Search API call successful on attempt {attempt + 1}"
                )

            # Close session if we created it
            if should_close_session:
                session.close()

            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If we reach here, all attempts failed
    logger.error(
        f"{log_prefix}API Request Failed after {MAX_RETRIES} attempts: {last_error}"
    )

    # Close session if we created it
    if should_close_session:
        session.close()

    return None, last_error


def _passages2string(retrieval_result: list[dict]) -> str:
    """Format the retrieval result into a string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"].strip()
        format_reference += f"Doc {idx + 1}: {content}\n"
    return format_reference


class SearchWorker:
    def __init__(self, cfg: SearchEnvConfig):
        self.url = cfg["search_url"]
        self.max_turns = cfg["max_turns"]

    def search(self, query: str) -> str:
        """Call the search API with a single query."""
        api_response, error_msg = _call_search_api(self.url, query)
        result_text = json.dumps(
            {"result": "Search request failed or timed out after retries."}
        )
        if error_msg:
            result_text = json.dumps({"result": f"Search error: {error_msg}"})
        elif api_response:
            try:
                raw_results = api_response.get("result", [])
                if raw_results:
                    pretty_results = []
                    total_results = 0
                    for retrieval in raw_results:
                        formatted = _passages2string(retrieval)
                        pretty_results.append(formatted)
                        total_results += (
                            len(retrieval) if isinstance(retrieval, list) else 1
                        )

                    final_result = "\n---\n".join(pretty_results)
                    result_text = json.dumps({"result": final_result})
                else:
                    result_text = json.dumps({"result": "No search results found."})
            except Exception as e:
                error_msg = f"Error processing search results: {e}"
                result_text = json.dumps({"result": error_msg})
        else:
            result_text = json.dumps(
                {"result": "Unknown API state (no response and no error message)."}
            )
        return result_text

    def _is_done(self, log: LLMMessageLogType, metadata: SearchEnvMetadata) -> bool:
        """Check if the search is done."""
        if metadata["num_turns"] >= self.max_turns:
            return True
        msg = log[-1]["content"]
        return "<answer>" in msg and "</answer>" in msg

    def _get_reward(self, answer: str | None, ground_truth: str, done: bool) -> float:
        """Get the reward for the search. If not done, return 0.0."""
        if done:
            return compute_score(answer, ground_truth)
        else:
            return 0.0

    def _parse_action(self, action: str) -> List[str] | None:
        """Parse the action from the message."""
        match = None
        if "<search>" in action and "</search>" in action:
            match = re.search(r"<search>(.*?)</search>", action, re.DOTALL)
        return match.group(1) if match else None

    def process_turn(
        self, log: LLMMessageLogType, metadata: SearchEnvMetadata
    ) -> SearchWorkerResult:
        """Process a single turn of the search given the generated action."""
        ground_truth = metadata["ground_truth"]

        done = self._is_done(log, metadata)
        answer = extract_solution(log[-1]["content"])
        reward = self._get_reward(answer, ground_truth, done)
        error = None
        if done:
            return SearchWorkerResult(
                observation={
                    "role": "user",
                    "content": "Correct" if reward > 0.0 else "Incorrect",
                },
                reward=reward,
                done=done,
                error=None,
                metadata=metadata,
                answer=answer,
            )
        try:
            query = self._parse_action(log[-1]["content"])
            if query:
                obs = self.search(query)
            else:
                obs = None
                error = "No search query provided."
        except Exception as e:
            error = str(e)
            obs = None

        if obs:
            new_msg = {
                "role": "user",
                "content": "<information>" + obs + "</information>",
            }
        else:
            new_msg = {"role": "user", "content": "<error>" + error + "</error>"}

        new_metadata = metadata.copy()
        new_metadata["num_turns"] += 1

        return SearchWorkerResult(
            observation=new_msg,
            reward=reward,
            done=done,
            error=error,
            metadata=new_metadata,
            answer=answer,
        )


@ray.remote
class SearchEnv(EnvironmentInterface[SearchEnvMetadata]):
    """Search environment that maintains state between steps."""

    def __init__(self, cfg: SearchEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.worker = SearchWorker(cfg)

    def step(  # type: ignore[override]
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[SearchEnvMetadata],
    ) -> EnvironmentReturn:
        """Process a batch of search steps."""
        # chunk the responses and ground truths
        results = [
            self.worker.process_turn(msg_log, metadata)
            for msg_log, metadata in zip(message_log_batch, metadata)
        ]

        observations = [result["observation"] for result in results]
        rewards = [result["reward"] for result in results]
        dones = [result["done"] for result in results]
        metadata = [result["metadata"] for result in results]

        rewards = torch.tensor(rewards).cpu()
        dones = torch.tensor(dones).cpu()
        answers = [result["answer"] for result in results]

        return EnvironmentReturn(
            observations=observations,
            rewards=rewards,
            terminateds=dones,
            metadata=metadata,
            next_stop_strings=[["</search>", "</answer>", "</think>"]]
            * len(observations),
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        """Calculate the success rate based on the final reward."""
        # Calculate success rate based on final reward == 1.0
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        success_rate = (
            (final_rewards == 1.0).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        return batch, {"searchr1_success_rate": success_rate}
