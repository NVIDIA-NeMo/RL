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

import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import numpy as np


class PairwiseRewardAggregator(ABC):
    """Abstract base class for aggregating pairwise comparison results into scalar rewards."""
    
    def __init__(self, **kwargs):
        """Default initialization that accepts any keyword arguments for compatibility with factory function."""
        # Accept any kwargs to maintain compatibility with the factory function
        # Subclasses can override this if they need specific configuration parameters
        pass
    
    @abstractmethod
    def aggregate_scores(
        self,
        comparison_results: List[Tuple[str, float, float, float]],  # (request_id, score_1, score_2, ranking_score)
        comparison_metadata: List[Tuple[str, int, int]],  # (prompt_key, resp_i, resp_j)
        prompt_groups: Dict[str, Dict[str, Any]]  # {prompt_key: {"conversations": [...], "metadata": {...}, "indices": [...]}}
    ) -> Dict[str, List[float]]:
        """Aggregate pairwise comparison results into final rewards for each response."""
        pass
    
    def get_additional_metrics(
        self,
        comparison_results: List[Tuple[str, float, float, float]],  # (request_id, score_1, score_2, ranking_score)
        comparison_metadata: List[Tuple[str, int, int]],  # (prompt_key, resp_i, resp_j)
        prompt_groups: Dict[str, Dict[str, Any]],
        final_scores: Dict[str, List[float]]  # The aggregated scores from aggregate_scores
    ) -> Dict[str, float]:
        """Compute additional metrics to log alongside the main aggregated scores.
        
        Default implementation returns empty dict. Subclasses can override to provide
        additional metrics like individual scores, win rates, etc.
        
        Args:
            comparison_results: Raw comparison results from the GenRM model
            comparison_metadata: Metadata about which responses were compared
            prompt_groups: Grouped prompt data
            final_scores: The final aggregated scores returned by aggregate_scores
            
        Returns:
            Dictionary of additional metrics to log
        """
        return {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the aggregation method."""
        pass



class IndividualScoreAggregator(PairwiseRewardAggregator):
    """Use GenRM's individual helpfulness scores directly."""
    
    def __init__(self, score_range: Tuple[float, float] = (1.0, 5.0)):
        """
        Args:
            score_range: (min_score, max_score) for normalization
        """
        self.min_score, self.max_score = score_range
    
    def aggregate_scores(
        self,
        comparison_results: List[Tuple[str, float, float, float]],
        comparison_metadata: List[Tuple[str, int, int]],
        prompt_groups: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Aggregate using individual helpfulness scores."""
        
        # Collect individual scores for each response
        individual_scores = {}
        score_counts = {}
        
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            individual_scores[prompt_key] = [0.0 for _ in range(num_responses)]
            score_counts[prompt_key] = [0 for _ in range(num_responses)]
        
        # Process each comparison result
        for (request_id, score_1, score_2, ranking_score), (prompt_key, resp_i, resp_j) in zip(
            comparison_results, comparison_metadata
        ):
            # Accumulate individual scores
            individual_scores[prompt_key][resp_i] += score_1
            individual_scores[prompt_key][resp_j] += score_2
            score_counts[prompt_key][resp_i] += 1
            score_counts[prompt_key][resp_j] += 1
        
        # Calculate average individual scores and normalize
        final_scores = {}
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            final_scores[prompt_key] = []
            for resp_idx in range(num_responses):
                if score_counts[prompt_key][resp_idx] > 0:
                    avg_score = individual_scores[prompt_key][resp_idx] / score_counts[prompt_key][resp_idx]
                else:
                    avg_score = (self.min_score + self.max_score) / 2  # Neutral score if no comparisons
                final_scores[prompt_key].append(avg_score)
            
        return final_scores
    
    @property
    def name(self) -> str:
        return "individual_scores"


class WeightedWinLossAggregator(PairwiseRewardAggregator):
    """Use weighted scores based on the magnitude of GenRM ranking preferences."""
    
    def __init__(self, score_mapping: Dict[int, float] = None):
        """
        Args:
            score_mapping: Mapping from ranking scores (1-6) to weighted points
        """
        self.score_mapping = score_mapping or {
            1: 1.0,    # Much better
            2: 0.75,   # Better  
            3: 0.6,    # Slightly better
            4: 0.4,    # Slightly worse
            5: 0.25,   # Worse
            6: 0.0     # Much worse
        }
    
    def aggregate_scores(
        self,
        comparison_results: List[Tuple[str, float, float, float]],
        comparison_metadata: List[Tuple[str, int, int]],
        prompt_groups: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Aggregate using weighted win-loss scores."""
        
        # Initialize weighted scores
        weighted_scores = {}
        total_comparisons = {}
        
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            weighted_scores[prompt_key] = [0.0 for _ in range(num_responses)]
            total_comparisons[prompt_key] = [0 for _ in range(num_responses)]
        
        # Process each comparison result
        for (request_id, score_1, score_2, ranking_score), (prompt_key, resp_i, resp_j) in zip(
            comparison_results, comparison_metadata
        ):
            # Convert ranking score to weighted points
            ranking_int = int(round(ranking_score))
            
            if ranking_int <= 3:
                # Response i wins with different magnitudes
                weight_i = self.score_mapping.get(ranking_int, 0.5)
                weight_j = 1.0 - weight_i
            else:
                # Response j wins with different magnitudes
                weight_j = self.score_mapping.get(ranking_int, 0.5)
                weight_i = 1.0 - weight_j
            
            # Accumulate weighted scores
            weighted_scores[prompt_key][resp_i] += weight_i
            weighted_scores[prompt_key][resp_j] += weight_j
            total_comparisons[prompt_key][resp_i] += 1
            total_comparisons[prompt_key][resp_j] += 1
        
        # Calculate final scores as weighted averages
        final_scores = {}
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            final_scores[prompt_key] = []
            for resp_idx in range(num_responses):
                if total_comparisons[prompt_key][resp_idx] > 0:
                    avg_weighted_score = weighted_scores[prompt_key][resp_idx] / total_comparisons[prompt_key][resp_idx]
                else:
                    avg_weighted_score = 0.5  # Neutral score if no comparisons
                final_scores[prompt_key].append(avg_weighted_score)
            
        return final_scores
    
    def get_additional_metrics(
        self,
        comparison_results: List[Tuple[str, float, float, float]],
        comparison_metadata: List[Tuple[str, int, int]],
        prompt_groups: Dict[str, Dict[str, Any]],
        final_scores: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Compute individual score metrics alongside the weighted win-loss scores."""
        
        # Collect individual scores for metrics
        all_individual_scores_1 = []
        all_individual_scores_2 = []
        all_individual_scores = []
        
        # Process each comparison result to extract individual scores
        for (request_id, score_1, score_2, ranking_score), (prompt_key, resp_i, resp_j) in zip(
            comparison_results, comparison_metadata
        ):
            all_individual_scores_1.append(score_1)
            all_individual_scores_2.append(score_2)
            all_individual_scores.extend([score_1, score_2])
        
        # Compute statistics for individual scores
        individual_metrics = {}
        if all_individual_scores:
            individual_metrics.update({
                "mean_individual_score": np.mean(all_individual_scores),
                "std_individual_score": np.std(all_individual_scores),
                "min_individual_score": np.min(all_individual_scores),
                "max_individual_score": np.max(all_individual_scores),
                "median_individual_score": np.median(all_individual_scores),
            })
            
            # Also compute individual score metrics per response position
            if all_individual_scores_1:
                individual_metrics.update({
                    "mean_individual_score_first": np.mean(all_individual_scores_1),
                    "mean_individual_score_second": np.mean(all_individual_scores_2),
                })
        
        return individual_metrics
    
    @property
    def name(self) -> str:
        return "weighted_win_loss"



class SimpleTiebreakerAggregator(PairwiseRewardAggregator):
    """Use individual scores primarily, with simple ranking-based tiebreaking when scores are equal.
    
    GenRM Scoring System:
    - Individual scores: 1-5 (where 5 is most helpful)
    - Ranking scores: 1-6 (where 1 = Response 1 much better, 6 = Response 2 much better)
    - Neutral ranking = 3.5 (no preference between responses)
    
    Tiebreaker Logic:
    When individual scores are equal, we use the ranking to break ties:
    - score1 = score1 + (3.5 - ranking_score)
    - score2 = score2 + (ranking_score - 3.5)
    
    Why this works:
    - When ranking < 3.5: Response 1 is preferred, so score1 gets positive adjustment, score2 gets negative
    - When ranking > 3.5: Response 2 is preferred, so score2 gets positive adjustment, score1 gets negative
    - When ranking = 3.5: No preference, so no adjustments (both get 0)
    - The further from 3.5, the larger the adjustment magnitude
    
    Examples:
    - ranking = 1 (Response 1 much better): score1 += 2.5, score2 -= 2.5
    - ranking = 3 (Response 1 slightly better): score1 += 0.5, score2 -= 0.5
    - ranking = 6 (Response 2 much better): score1 -= 2.5, score2 += 2.5
    """
    
    def __init__(self, score_range: Tuple[float, float] = (1.0, 5.0)):
        """
        Args:
            score_range: (min_score, max_score) for normalization
        """
        self.min_score, self.max_score = score_range
    
    def aggregate_scores(
        self,
        comparison_results: List[Tuple[str, float, float, float]],
        comparison_metadata: List[Tuple[str, int, int]],
        prompt_groups: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Aggregate using individual scores with simple ranking tiebreaking."""
        
        # Collect individual scores for each response
        individual_scores = {}
        score_counts = {}
        
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            individual_scores[prompt_key] = [0.0 for _ in range(num_responses)]
            score_counts[prompt_key] = [0 for _ in range(num_responses)]
        
        # Process each comparison result
        for (request_id, score_1, score_2, ranking_score), (prompt_key, resp_i, resp_j) in zip(
            comparison_results, comparison_metadata
        ):
            # Apply simple tiebreaking logic when scores are equal
            if score_1 == score_2:
                # When individual scores are equal, use ranking to break ties
                score_1 = score_1 + 3.5 - ranking_score  # Response 1 adjustment
                score_2 = score_2 + ranking_score - 3.5  # Response 2 adjustment
            
            # Accumulate individual scores (with tiebreaking adjustments if applied)
            individual_scores[prompt_key][resp_i] += score_1
            individual_scores[prompt_key][resp_j] += score_2
            score_counts[prompt_key][resp_i] += 1
            score_counts[prompt_key][resp_j] += 1
        
        # Calculate average individual scores
        final_scores = {}
        for prompt_key, group_data in prompt_groups.items():
            num_responses = len(group_data["conversations"])
            final_scores[prompt_key] = []
            for resp_idx in range(num_responses):
                if score_counts[prompt_key][resp_idx] > 0:
                    avg_score = individual_scores[prompt_key][resp_idx] / score_counts[prompt_key][resp_idx]
                else:
                    avg_score = (self.min_score + self.max_score) / 2  # Neutral score if no comparisons
                final_scores[prompt_key].append(avg_score)
            
        return final_scores
    
    def get_additional_metrics(
        self,
        comparison_results: List[Tuple[str, float, float, float]],
        comparison_metadata: List[Tuple[str, int, int]],
        prompt_groups: Dict[str, Dict[str, Any]],
        final_scores: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Compute individual score metrics alongside the simple tiebreaker scores."""
        
        # Collect individual scores and track tiebreaking usage
        all_individual_scores = []
        all_ranking_scores = []
        tiebreak_used_count = 0
        
        # Process each comparison result to extract individual scores
        for (request_id, score_1, score_2, ranking_score), (prompt_key, resp_i, resp_j) in zip(
            comparison_results, comparison_metadata
        ):
            # Track original individual scores (before any tiebreaking)
            all_individual_scores.extend([score_1, score_2])
            all_ranking_scores.append(ranking_score)
            
            # Count how often tiebreaking is used
            if score_1 == score_2:
                tiebreak_used_count += 1
        
        # Compute statistics for individual scores
        individual_metrics = {}
        if all_individual_scores:
            individual_metrics.update({
                "mean_individual_score": np.mean(all_individual_scores),
                "std_individual_score": np.std(all_individual_scores),
                "min_individual_score": np.min(all_individual_scores),
                "max_individual_score": np.max(all_individual_scores),
            })
        
        if all_ranking_scores:
            individual_metrics.update({
                "mean_ranking_score": np.mean(all_ranking_scores),
                "std_ranking_score": np.std(all_ranking_scores),
            })
            
        # Add tiebreaking statistics
        total_comparisons = len(comparison_results)
        if total_comparisons > 0:
            individual_metrics["tiebreak_usage_rate"] = tiebreak_used_count / total_comparisons
        
        return individual_metrics
    
    @property
    def name(self) -> str:
        return "simple_tiebreaker"


# Factory function to create aggregators
def create_aggregator(method: str, **kwargs) -> PairwiseRewardAggregator:
    """Create a reward aggregator by name."""
    aggregators = {
        "individual_scores": IndividualScoreAggregator,
        "weighted_win_loss": WeightedWinLossAggregator,
        "simple_tiebreaker": SimpleTiebreakerAggregator,
    }
    
    if method not in aggregators:
        raise ValueError(f"Unknown aggregation method: {method}. Available: {list(aggregators.keys())}")
    
    return aggregators[method](**kwargs)
