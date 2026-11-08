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
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import ray
import torch
from pydantic import BaseModel

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)

logger = logging.getLogger(__name__)


def _checked_value(value, high: int) -> float:
    """A score (1-5) or ranking (1-6), validated the way Gym's genrm_compare and genrm_eval parsers do."""
    # bool is an int subclass, so float(True) would silently become 1.0
    if isinstance(value, bool):
        raise ValueError(f"boolean {value!r} is not a valid score or ranking")
    value = float(value)
    if not (math.isfinite(value) and value.is_integer() and 1 <= value <= high):
        raise ValueError(f"{value} is not an integer in 1-{high}")
    return value


def _checked_rubric_id(value) -> int:
    """A rubric_id, rejecting bools and fractional numbers that int() would silently truncate (as Gym does)."""
    if isinstance(value, bool) or (isinstance(value, float) and not (math.isfinite(value) and value.is_integer())):
        raise ValueError(f"{value!r} is not a valid rubric_id")
    return int(value)


# The score fields a verdict must give, in `overall` and in every rubric entry, by the sample's num_responses: a
# pointwise judge (1) gives score_1 only, a pairwise judge (2) always gives all three, whatever the ground truth grades.
SCORE_KEYS = {1: ("score_1",), 2: ("score_1", "score_2", "ranking")}
SCORE_MAX = {"score_1": 5, "score_2": 5, "ranking": 6}


def _score_keys(metadata) -> tuple:
    """SCORE_KEYS for the sample, from the num_responses its data processor set (1 = pointwise, 2 = pairwise)."""
    num_responses = metadata.get("num_responses")
    if num_responses not in SCORE_KEYS:
        raise ValueError(
            f"metadata['num_responses'] must be 1 (pointwise) or 2 (pairwise), got {num_responses!r}; the data "
            "processor must set it"
        )
    return SCORE_KEYS[num_responses]


def _nothing_to_grade(metadata) -> bool:
    """A sample that gets 0.0 without being parsed: one its data processor masked out of the loss (loss_multiplier
    0.0, e.g. a prompt over max_seq_length), or a pointwise sample whose ground truth has no score_1 (a ranking-only
    row, e.g. safety), which a judge that sees one response cannot be graded on."""
    score_keys = _score_keys(metadata)  # validates num_responses for every sample, masked or not
    if metadata.get("loss_multiplier", 1.0) == 0.0:
        return True
    return score_keys == SCORE_KEYS[1] and (metadata.get("ground_truth_overall") or {}).get("score_1") is None


def _checked_scores(entry, score_keys: tuple) -> dict:
    """The score fields of one verdict entry (`overall` or a rubric): exactly score_keys, each validated."""
    if not isinstance(entry, dict):
        raise TypeError(f"expected a JSON object, got {type(entry).__name__}")
    present = {key for key in SCORE_MAX if key in entry}
    if present != set(score_keys):
        raise ValueError(f"score fields {sorted(present)} where the prompt asks for exactly {list(score_keys)}")
    return {key: _checked_value(entry[key], SCORE_MAX[key]) for key in score_keys}


class VanillaGenRMConfig(BaseModel, extra="allow"):
    """Configuration for Vanilla GenRM training environment (`env.<name>` in the YAML).

    The environment builds this from the plain dict that MasterConfig.env holds, so a key missing from the YAML takes the
    default below, and the config can be read as attributes (`cfg.score_weight`) like the trainers' own configs."""

    num_workers: int = 1  # Number of worker processes for parallel evaluation
    score_weight: float = 1.0  # weight for individual score L1 (default 1.0)
    ranking_weight: float = 2.0  # weight for ranking L1 (default 2.0)
    rubric_weight: float = 0.5  # multiplier on averaged rubric penalty (2x less than overall) (default 0.5)
    reasoning_split_word: Optional[str] = "</think>"  # Split word to separate reasoning from final output (default: "</think>")
    parse_failure_penalty: float = -20.0


class VanillaGenRMMetadata(TypedDict):
    """Metadata structure for vanilla GenRM environment."""
    id: Union[int, str]
    num_responses: int  # 1 = pointwise verdict, 2 = pairwise verdict (see SCORE_KEYS)
    loss_multiplier: float  # the sample's loss_multiplier from its data processor; 0.0 gets reward 0.0 unparsed
    ground_truth_overall: Optional[Dict] = None  # {"score_1", "score_2", "ranking"}
    ground_truth_rubric_scores: Optional[List[Dict]] = None  # [{"rubric_id", "score_1", "score_2", "ranking"}, ...]

@dataclass
class RubricScores:
    rubric_id: int
    score_1: float
    score_2: Optional[float] = None  # None in a pointwise verdict
    ranking: Optional[float] = None  # None in a pointwise verdict


@dataclass
class ParsedScores:
    score_1: float  # overall score_1
    score_2: Optional[float] = None  # overall score_2; None in a pointwise verdict
    ranking: Optional[float] = None  # overall ranking; None in a pointwise verdict
    rubric_scores: List[RubricScores] = field(default_factory=list)


def _find_balanced_braces(text: str) -> list:
    """Find all top-level balanced { ... } substrings in text.

    Prose outside the blocks (e.g. reasoning before the verdict) must not derail the scan: quotes only delimit strings
    inside a block, and a "}" with no open block is ignored.
    """
    results = []
    depth = 0
    start = -1
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        # At depth 0 a quote is prose (e.g. `a 12" pizza`); toggling on it would invert the string state for the rest
        # of the text and hide the braces of every block after it.
        if ch == '"' and depth > 0:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        # A stray "}" at depth 0 would otherwise drive depth negative, so no later block would ever be recorded.
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                results.append(text[start : i + 1])
                start = -1

    return results


def _fix_invalid_escapes(text: str) -> str:
    r"""Replace invalid JSON escape sequences with escaped backslashes.

    JSON only allows: \" \\ \/ \b \f \n \r \t \uXXXX
    Model outputs often contain LaTeX (e.g. \Delta, \ln) which are invalid.
    """
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _safe_json_loads(text: str) -> Optional[dict]:
    """Parse JSON string, returning None on failure or if not a dict."""
    for candidate in [text, _fix_invalid_escapes(text)]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


@ray.remote
class VanillaGenRMWorker:
    """Worker that evaluates GenRM responses and extracts scoring information."""

    def __init__(self):
        """Initialize the GenRM evaluation worker."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )
        self.logger = logging.getLogger(__name__)

    def _try_parse_json(self, output: str) -> Optional[dict]:
        """Extract the single valid JSON dict from model output text.
    
        Scans all top-level balanced { ... } blocks (fenced blocks are covered,
        since their content is part of the full text) and requires that EXACTLY
        ONE of them parses to a JSON dict. Handles nested JSON (arrays/objects
        inside the top-level object).
    
        Returns None when zero or multiple valid dicts are found. The multiple-
        dict case is deliberate: a repeated verdict block must be a parse
        failure, not a free pass on the last copy.
        """
        valid_dicts = []
        for candidate in _find_balanced_braces(output):
            result = _safe_json_loads(candidate)
            if result is not None:
                valid_dicts.append(result)
                if len(valid_dicts) > 1:
                    logger.warning(
                        "Expected exactly one JSON dict in GenRM output but found more than one; "
                        "treating as parse failure."
                    )
                    return None
    
        return valid_dicts[0] if valid_dicts else None

    def extract_scores_from_response(self, output: str, score_keys: tuple) -> Optional[ParsedScores]:
        """Extract scores and ranking from GenRM model output.

        The output must contain EXACTLY ONE valid JSON dict (fenced or bare).
        Zero valid dicts or more than one — e.g. a degenerate policy repeating
        its verdict block — is a parse failure. Otherwise repeated blocks would
        parse fine and reward-free repetition gets reinforced by RL (this
        collapsed a training run at ~step 600: entropy fell, generations looped
        the verdict JSON until hitting the token cap, and reward never moved).
    
        Expects the unified format with "rubric_evaluations" and "overall" keys.
        `overall` and every rubric entry must give exactly the score fields in
        score_keys (see SCORE_KEYS), whatever the ground truth grades.

        Args:
            output: Text output from the GenRM model (message content only).
            score_keys: The score fields the prompt asks for, SCORE_KEYS[num_responses].

        Returns:
            ParsedScores if exactly one JSON dict was found with valid overall
            scores. None otherwise.
        """
        parsed = self._try_parse_json(output)
    
        if parsed is None:
            logger.warning("No parseable JSON found in GenRM output: %s", output)
            return None
    
        try:
            # Extract overall scores
            overall = parsed.get("overall")
            if not isinstance(overall, dict):
                logger.warning("'overall' is not a dict (got %s) in GenRM output: %s", type(overall).__name__, output)
                return None
            overall_scores = _checked_scores(overall, score_keys)

            # Extract per-rubric scores — must be present and all parseable
            raw_rubrics = parsed.get("rubric_evaluations")
            if not isinstance(raw_rubrics, list) or len(raw_rubrics) == 0:
                logger.warning(
                    "'rubric_evaluations' missing or empty (got %s) in GenRM output: %s",
                    type(raw_rubrics).__name__,
                    output,
                )
                return None
    
            rubric_scores = []
            for rs in raw_rubrics:
                scores = _checked_scores(rs, score_keys)
                rubric_scores.append(RubricScores(rubric_id=_checked_rubric_id(rs["rubric_id"]), **scores))

            return ParsedScores(**overall_scores, rubric_scores=rubric_scores)
        except (TypeError, ValueError, KeyError) as e:
            logger.warning("Failed to extract scores (%s: %s) from GenRM output: %s", type(e).__name__, e, output)
            return None

    def evaluate_responses(
        self,
        assistant_responses: List[str],
        metadata_list: List[VanillaGenRMMetadata],
        config: VanillaGenRMConfig,
    ) -> List[float]:
        """Evaluate assistant responses and compute rewards.

        Args:
            assistant_responses: List of assistant responses to evaluate
            metadata_list: List of metadata containing ground truth scores
            config: Configuration for the environment

        Returns:
            List of rewards for each response
        """
        rewards = []

        for response, metadata in zip(assistant_responses, metadata_list):
            if _nothing_to_grade(metadata):
                rewards.append(0.0)
                continue

            # Extract scores from the model's response
            scores = self.extract_scores_from_response(response, _score_keys(metadata))

            if scores is None:
                rewards.append(config.parse_failure_penalty)
                continue

            # Calculate reward based on configuration
            reward = self._calculate_reward(
                scores, metadata, config
            )
            rewards.append(reward)

        return rewards

    def _calculate_reward(
        self,
        scores: ParsedScores,
        metadata: VanillaGenRMMetadata,
        cfg: VanillaGenRMConfig,
    ) -> float:
        """Calculate reward based on extracted scores vs ground truth using negative L1 distance.

        Always evaluates both individual scores and ranking (combined approach).

        Args:
            scores: Extracted scores from model response
            metadata: Metadata from each sample (contains the ground truth)
            config: Environment configuration

        Returns:
            Calculated reward value (negative L1 distance)
        """
        reward = 0.0
        overall_score_1_l1 = None
        overall_score_2_l1 = None
        overall_ranking_l1 = None
        rubric_score_l1 = None
        rubric_ranking_l1 = None
        overall_ranking_binary_acc = None

        # The parser guarantees the verdict gives exactly these fields; a field is graded when it is one of them and
        # the ground truth has it (a ranking-only pairwise row grades the ranking alone).
        score_keys = _score_keys(metadata)

        # --- Overall reward (only penalize fields present in GT) ---
        gt_overall = metadata.get("ground_truth_overall")
        if gt_overall is not None:
            if gt_overall.get("score_1") is not None:
                overall_score_1_l1 = abs(gt_overall["score_1"] - scores.score_1)
                reward -= cfg.score_weight * overall_score_1_l1
            if "score_2" in score_keys and gt_overall.get("score_2") is not None:
                overall_score_2_l1 = abs(gt_overall["score_2"] - scores.score_2)
                reward -= cfg.score_weight * overall_score_2_l1
            if "ranking" in score_keys and gt_overall.get("ranking") is not None:
                overall_ranking_l1 = abs(gt_overall["ranking"] - scores.ranking)
                reward -= cfg.ranking_weight * overall_ranking_l1
                # Binary accuracy: did the model get the direction right?
                # Rankings 1-3 = one direction, 4-6 = the other; midpoint is 3.5
                gt_dir = gt_overall["ranking"] < 3.5
                pred_dir = scores.ranking < 3.5
                overall_ranking_binary_acc = 1.0 if gt_dir == pred_dir else 0.0

        # --- Per-rubric reward ---
        gt_rubrics = metadata.get("ground_truth_rubric_scores")
        if gt_rubrics:
            gt_by_id = {r["rubric_id"]: r for r in gt_rubrics}
            pred_by_id = {r.rubric_id: r for r in scores.rubric_scores}

            # Predicted rubrics must match GT exactly: same count, same IDs. Count the raw
            # predictions, not pred_by_id -- that dict has already collapsed duplicate IDs, so
            # comparing its length would let a verdict that repeats a rubric_id (scoring only
            # the last copy) pass as if it matched.
            if len(scores.rubric_scores) != len(gt_by_id) or set(pred_by_id.keys()) != set(gt_by_id.keys()):
                return cfg.parse_failure_penalty

            score_errors = []
            ranking_errors = []
            rubric_penalties = []
            for rid, gt in gt_by_id.items():
                pred = pred_by_id[rid]
                penalty = 0.0
                scored = False
                if gt.get("score_1") is not None:
                    s1_err = abs(gt["score_1"] - pred.score_1)
                    score_errors.append(s1_err)
                    penalty += cfg.score_weight * s1_err
                    scored = True
                if "score_2" in score_keys and gt.get("score_2") is not None:
                    s2_err = abs(gt["score_2"] - pred.score_2)
                    score_errors.append(s2_err)
                    penalty += cfg.score_weight * s2_err
                    scored = True
                if "ranking" in score_keys and gt.get("ranking") is not None:
                    r_err = abs(gt["ranking"] - pred.ranking)
                    ranking_errors.append(r_err)
                    penalty += cfg.ranking_weight * r_err
                    scored = True
                # Append on `scored`, not `penalty > 0`: a perfectly predicted rubric scores 0.0
                # and must still count toward the mean. Excluding it shrinks the denominator as
                # the policy improves, so fixing one rubric of several could lower the reward.
                if scored:
                    rubric_penalties.append(penalty)

            if rubric_penalties:
                reward -= cfg.rubric_weight * sum(rubric_penalties) / len(rubric_penalties)
            if score_errors:
                rubric_score_l1 = sum(score_errors) / len(score_errors)
            if ranking_errors:
                rubric_ranking_l1 = sum(ranking_errors) / len(ranking_errors)
        else:
            # No rubric ground truth (v2 / HS5 rows): the rubrics are not graded, but their ids must still number the
            # criteria 1..N like the principle does, each exactly once (in any order).
            pred_ids = sorted(r.rubric_id for r in scores.rubric_scores)
            if pred_ids != list(range(1, len(pred_ids) + 1)):
                return cfg.parse_failure_penalty

        # return payload here
        return reward


@ray.remote
class VanillaGenRMEnvironment(EnvironmentInterface):
    """Environment for training vanilla GenRM models on response evaluation tasks."""

    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM
    # Consecutive responses without reasoning_split_word (counted across step calls) that raise the loud warning. Async
    # rollouts call step() one sample at a time, so a single call cannot tell a truncated sample from a run where no
    # response can contain the split word; a long streak can.
    MISSING_SPLIT_WORD_ALARM_STREAK = 32

    def __init__(self, cfg: dict):
        """Initialize the vanilla GenRM environment.

        Args:
            cfg: Configuration for the environment (`env.<name>` from the YAML, a plain dict); see VanillaGenRMConfig
        """
        self.cfg = VanillaGenRMConfig(**cfg)
        self.num_workers = self.cfg.num_workers

        # Create worker pool
        self.workers = [
            VanillaGenRMWorker.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]

        self.logger = logging.getLogger(__name__)

        # Responses without reasoning_split_word: running totals and the current streak (see step)
        self._num_split_responses = 0
        self._num_missing_split_word = 0
        self._missing_split_word_streak = 0
        self._missing_split_word_alarm_at = self.MISSING_SPLIT_WORD_ALARM_STREAK

    def shutdown(self):
        """Shutdown all workers."""
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[VanillaGenRMMetadata],
    ) -> EnvironmentReturn:
        """Run a step in the vanilla GenRM environment.

        Args:
            message_log_batch: Batch of message logs from the LLM interactions
            metadata: Batch of metadata containing ground truth scores

        Returns:
            EnvironmentReturn with observations, metadata, stop strings, rewards, and done flags
        """
        # Extract assistant responses from message logs
        assistant_responses = []
        for conversation in message_log_batch:
            # Find the last assistant response
            assistant_response = ""
            for message in reversed(conversation):
                if message["role"] == "assistant":
                    assistant_response = message["content"]
                    break
            assistant_responses.append(assistant_response)

        # Split reasoning from responses if reasoning_split_word is configured
        processed_responses = []
        reasoning_split_word = self.cfg.reasoning_split_word
        num_checked = 0
        num_missing = 0
        for response, sample_metadata in zip(assistant_responses, metadata):
            if _nothing_to_grade(sample_metadata):
                # Graded 0.0 without parsing (see evaluate_responses), so it does not count toward the split-word warnings
                processed_response = ""
            elif reasoning_split_word:
                num_checked += 1
                # Check if reasoning split word is in the response
                if reasoning_split_word in response:
                    # Split on reasoning word and take the last part (after reasoning)
                    processed_response = response.split(reasoning_split_word)[-1].lstrip()
                    self._missing_split_word_streak = 0
                    self._missing_split_word_alarm_at = self.MISSING_SPLIT_WORD_ALARM_STREAK
                else:
                    # If split word not found, send empty string to worker
                    processed_response = ""
                    num_missing += 1
                    self._missing_split_word_streak += 1
            else:
                processed_response = response
            processed_responses.append(processed_response)

        self._num_split_responses += num_checked
        self._num_missing_split_word += num_missing
        if num_missing:
            self.logger.warning(
                "%d of %d responses in this step lack reasoning_split_word %r (truncated or unclosed reasoning) and get "
                "parse_failure_penalty; %d of %d (%.1f%%) so far.",
                num_missing,
                num_checked,
                reasoning_split_word,
                self._num_missing_split_word,
                self._num_split_responses,
                100.0 * self._num_missing_split_word / self._num_split_responses,
            )
        if self._missing_split_word_streak >= self._missing_split_word_alarm_at:
            self.logger.error(
                "!!! The last %d responses ALL lack reasoning_split_word %r, so every one of them gets "
                "parse_failure_penalty and training gets no reward signal. Check that the chat template leaves the "
                "reasoning open for the model to close (e.g. chat_template_kwargs enable_thinking is not false) and "
                "that the tokenizer does not decode %r as a special token; with thinking disabled, set "
                "env.<name>.reasoning_split_word=null. !!!",
                self._missing_split_word_streak,
                reasoning_split_word,
                reasoning_split_word,
            )
            self._missing_split_word_alarm_at = self._missing_split_word_streak + self.MISSING_SPLIT_WORD_ALARM_STREAK

        # Distribute work across workers
        batch_size = len(assistant_responses)
        chunk_size = max(1, batch_size // self.num_workers)

        futures = []
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            worker_idx = (i // chunk_size) % self.num_workers

            future = self.workers[worker_idx].evaluate_responses.remote(
                processed_responses[i:end_idx], metadata[i:end_idx], self.cfg
            )
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        rewards_list = [reward for chunk in results for reward in chunk]

        # Create observations (environment feedback)
        observations = []
        for reward in rewards_list:
            content = f"Environment: Score = {reward:.2f}"

            observations.append({"role": "environment", "content": content})

        # Convert to tensors
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32).cpu()
        done_tensor = torch.ones_like(rewards_tensor, dtype=torch.bool).cpu()

        # All episodes terminate after one step for this task
        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=done_tensor,
            answers=None,  # GenRM doesn't extract specific answers like Math environment
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Compute global metrics for the vanilla GenRM environment.

        Args:
            batch: Global batch of rollout data

        Returns:
            Tuple of (processed_batch, metrics_dict)
        """
        metrics = {}
        return batch, metrics
