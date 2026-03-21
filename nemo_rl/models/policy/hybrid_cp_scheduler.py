"""Head-node scheduling for Hybrid Context Parallelism (HCP).

This module integrates Megatron-LM's BalancedCPScheduler to schedule samples
to DP×CP ranks based on sequence lengths, enabling per-sample CP size assignment.
"""

import logging
from typing import Any, List, Tuple

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict, SlicedDataDict
from nemo_rl.models.policy.hybrid_cp_config import HybridCPConfig

# Import from Megatron-LM
try:
    from megatron.core.pipeline_parallel.hybrid_cp_schedule import BalancedCPScheduler
except ImportError:
    raise ImportError(
        "BalancedCPScheduler not found. Please ensure Megatron-LM is updated to include HCP support."
    )

logger = logging.getLogger(__name__)


class MockProcessGroup:
    """Mock process group for head-node scheduling without distributed initialization.

    This allows BalancedCPScheduler to be used on the head node where no actual
    process group exists.
    """

    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        """Return the process group size."""
        return self._size

    def rank(self) -> int:
        """Return rank 0 (head node)."""
        return 0


class HeadNodeHCPScheduler:
    """Head-node scheduler for Hybrid Context Parallelism.

    This scheduler runs on the head node (before data is distributed to workers) to:
    1. Extract sequence lengths from all samples
    2. Use BalancedCPScheduler to assign samples to DP×CP ranks
    3. Attach HCP metadata (local_cp_size, cp_group_id) to each sample
    4. Shard data according to DP×CP rank assignments

    Args:
        hcp_config: HCP configuration
        dp_size: Data parallel size
        cp_size: Global context parallel size
        max_seq_len: Maximum sequence length in the model
    """

    def __init__(
        self,
        hcp_config: HybridCPConfig,
        dp_size: int,
        cp_size: int,
        max_seq_len: int,
    ):
        self.hcp_config = hcp_config
        self.dp_size = dp_size
        self.cp_size = cp_size
        self.max_seq_len = max_seq_len
        self.hdp_size = dp_size * cp_size  # Hybrid DP size (DP × CP)

        # Get max_seqlen_per_dp_cp_rank from config (should be calculated in lm_policy.py)
        # Fallback calculation if not provided (for standalone usage)
        if hcp_config.max_seqlen_per_dp_cp_rank is None:
            self.max_seqlen_per_dp_cp_rank = max_seq_len // cp_size
            print(
                f"HCP: max_seqlen_per_dp_cp_rank not provided in config. "
                f"Calculated as {self.max_seqlen_per_dp_cp_rank} "
                f"(max_seq_len={max_seq_len} // cp_size={cp_size}). "
                f"Consider setting this in the config for better control."
            )
        else:
            self.max_seqlen_per_dp_cp_rank = hcp_config.max_seqlen_per_dp_cp_rank

        # Initialize BalancedCPScheduler with mock process group
        # Use mock since head node doesn't have actual distributed process groups
        mock_dp_cp_group = MockProcessGroup(size=self.hdp_size)
        self.scheduler = BalancedCPScheduler(
            max_seq_len_per_rank=self.max_seqlen_per_dp_cp_rank,
            dp_cp_group=mock_dp_cp_group,
        )

        print(
            f"HCP Scheduler initialized: dp_size={dp_size}, cp_size={cp_size}, "
            f"hdp_size={self.hdp_size}, max_seqlen_per_dp_cp_rank={self.max_seqlen_per_dp_cp_rank}"
        )

    def extract_sequence_lengths(
        self, data: BatchedDataDict, seq_length_key: str = "input_lengths"
    ) -> List[int]:
        """Extract sequence lengths from BatchedDataDict.

        Args:
            data: Input data batch
            seq_length_key: Key containing sequence lengths

        Returns:
            List of sequence lengths for all samples
        """
        if seq_length_key not in data:
            raise ValueError(f"seq_length_key '{seq_length_key}' not found in data")

        seq_lengths = data[seq_length_key]

        if torch.is_tensor(seq_lengths):
            seq_lengths = seq_lengths.cpu().tolist()
        elif not isinstance(seq_lengths, list):
            seq_lengths = list(seq_lengths)

        return seq_lengths

    def schedule_samples(
        self, seq_lengths: List[int]
    ) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
        """Schedule samples to DP×CP ranks using BalancedCPScheduler.

        Args:
            seq_lengths: List of sequence lengths for all samples

        Returns:
            Tuple of (groups, sample_id_groups):
            - groups: List of microbatch groups, each containing CP size assignments per rank
            - sample_id_groups: List of groups, each containing sample IDs assigned to each DP×CP rank
        """
        # Create (sample_id, seq_len) tuples
        sample_id_seqlens = [(i, seq_len) for i, seq_len in enumerate(seq_lengths)]

        # print(
        #     f"HCP: Scheduling {len(seq_lengths)} samples across {self.hdp_size} DP×CP ranks"
        # )

        # Run the scheduler
        groups, sample_id_groups = self.scheduler.get_groups_and_subsamples(
            sample_id_seqlens, self.hcp_config
        )

        # print(f"HCP: Created {len(groups)} microbatch groups")

        # Print sample_id_groups structure for debugging distribution
        # print(f"HCP: sample_id_groups structure (num_rounds={len(sample_id_groups)}, hdp_size={self.hdp_size}):")
        # for round_idx, round_assignments in enumerate(sample_id_groups):
        #     samples_per_rank = [len(sample_ids) for sample_ids in round_assignments]
        #     print(f"  Round {round_idx}: samples per rank = {samples_per_rank}, total = {sum(samples_per_rank)}")

        return groups, sample_id_groups

    def create_hcp_metadata(
        self,
        sample_id_groups: List[List[List[int]]],
        num_samples: int,
    ) -> Tuple[List[int]]:
        """Create HCP metadata for each sample.

        Args:
            sample_id_groups: Sample ID assignments from scheduler
                Shape: [num_rounds][hdp_rank] -> list of sample IDs
                Each round contains assignments for all hdp_size ranks
            num_samples: Total number of samples

        Returns:
            sample_local_cp_size: CP size for each sample (how many ranks share it)

        Note:
            We don't return sample_to_hdp_rank because samples can appear in multiple ranks.
            Instead, we build shards directly from sample_id_groups.
        """
        # Initialize metadata array
        sample_local_cp_size = [0] * num_samples
        samples_assigned = set()

        # Process each round
        for round_idx, round_assignments in enumerate(sample_id_groups):
            # round_assignments[hdp_rank] = list of sample IDs for that rank in this round
            for hdp_rank, sample_ids in enumerate(round_assignments):
                for sample_id in sample_ids:
                    samples_assigned.add(sample_id)

                    # Calculate local_cp_size: how many ranks share this sample in this round?
                    # Only calculate once (first time we see this sample)
                    if sample_local_cp_size[sample_id] == 0:
                        cp_size_for_sample = sum(
                            1 for rank_samples in round_assignments if sample_id in rank_samples
                        )
                        sample_local_cp_size[sample_id] = cp_size_for_sample

        # Verify all samples were assigned
        if len(samples_assigned) != num_samples:
            missing = set(range(num_samples)) - samples_assigned
            raise RuntimeError(
                f"HCP scheduling failed: {len(missing)} samples were not assigned to any rank. "
                f"Missing sample IDs: {list(missing)[:10]}..."
            )

        print(
            f"HCP: Assigned {num_samples} samples to {self.hdp_size} DP×CP ranks. "
            f"CP sizes range: {min(sample_local_cp_size)}-{max(sample_local_cp_size)}"
        )

        return (sample_local_cp_size,)

    def shard_data_by_hdp_rank(
        self,
        data: BatchedDataDict,
        sample_id_groups: List[List[List[int]]],
    ) -> List[SlicedDataDict]:
        """Shard data according to DP×CP rank assignments with HCP metadata.

        Args:
            data: Input data batch
            sample_id_groups: Sample ID assignments from scheduler
                Shape: [num_rounds][hdp_rank] -> list of sample IDs

        Returns:
            List of SlicedDataDict, one per DP×CP rank, with HCP metadata attached

        Note:
            We don't attach local_cp_size per sample here since Megatron-LM's
            hybrid_context_parallel_forward_backward calculates it from sample_id_groups.
        """
        # Log input data info
        # print(f"HCP: Sharding data - total samples: {data.size}")
        if "input_lengths" in data:
            input_lengths = data["input_lengths"]
            # if torch.is_tensor(input_lengths):
            #     print(f"HCP: Input sequence lengths: {input_lengths.tolist()}")
            # else:
            #     print(f"HCP: Input sequence lengths: {list(input_lengths)}")
        # if "input_ids" in data:
        #     print(f"HCP: Input input_ids shape: {data['input_ids'].shape}")

        # Create shards - one per DP×CP rank
        # Each shard will contain all samples assigned to that rank across all rounds
        sharded_data: List[List[BatchedDataDict]] = [[] for _ in range(self.hdp_size)]
        shard_sample_ids: List[List[int]] = [[] for _ in range(self.hdp_size)]

        # Iterate through all rounds and collect samples for each rank
        for round_idx, round_assignments in enumerate(sample_id_groups):
            for hdp_rank, sample_ids in enumerate(round_assignments):
                for sample_id in sample_ids:
                    # Add this sample to this rank's shard
                    sample_data = data.slice(sample_id, sample_id + 1)
                    sharded_data[hdp_rank].append(sample_data)
                    shard_sample_ids[hdp_rank].append(sample_id)

        # Convert to SlicedDataDict and add HCP metadata
        aggregated_shards = []
        for hdp_rank in range(self.hdp_size):
            if not sharded_data[hdp_rank]:
                # This rank has no samples - create empty shard
                shard = SlicedDataDict()
                shard["sample_id_groups"] = sample_id_groups
                shard["shard_sample_ids"] = []  # Empty list for empty shard
                # print(f"HCP: Shard {hdp_rank} (hdp_rank) has 0 samples")
            else:
                shard = SlicedDataDict.from_batches(sharded_data[hdp_rank])

                # Add full sample_id_groups structure for HCP coordination
                # Megatron's hybrid_context_parallel_forward_backward uses this to:
                # 1. Calculate local_cp_size for each sample
                # 2. Coordinate barriers between groups
                shard["sample_id_groups"] = sample_id_groups

                # Add the list of global sample IDs in this shard
                # This is needed by HCPDataIterator to create a dict with sample IDs as keys
                shard["shard_sample_ids"] = shard_sample_ids[hdp_rank]

                # Log shard details
                shard_input_lengths = shard.get("input_lengths", None)
                if shard_input_lengths is not None:
                    if torch.is_tensor(shard_input_lengths):
                        lengths_list = shard_input_lengths.tolist()
                    else:
                        lengths_list = list(shard_input_lengths)
                    # print(
                    #     f"HCP: Shard {hdp_rank} (hdp_rank) has {len(shard_sample_ids[hdp_rank])} samples: "
                    #     f"global_ids={shard_sample_ids[hdp_rank]}, seq_lengths={lengths_list}"
                    # )
                else:
                    print(
                        f"HCP: Shard {hdp_rank} (hdp_rank) has {len(shard_sample_ids[hdp_rank])} samples: "
                        f"global_ids={shard_sample_ids[hdp_rank]}"
                    )

            aggregated_shards.append(shard)

        return aggregated_shards

    def schedule_and_shard(
        self,
        data: BatchedDataDict,
        seq_length_key: str = "input_lengths",
    ) -> List[List[SlicedDataDict]]:
        """Complete HCP scheduling: extract lengths, schedule, and shard data.

        This is the main entry point that combines all steps.

        Args:
            data: Input data batch
            seq_length_key: Key containing sequence lengths

        Returns:
            2D list of SlicedDataDict indexed as [dp_idx][cp_idx], suitable
            for passing to run_all_workers_sharded_data with
            in_sharded_axes=["data_parallel", "context_parallel"].
        """
        # Step 0: Validate batch size consistency across all keys
        batch_sizes = set()
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                batch_sizes.add(v.shape[0])
            elif isinstance(v, list):
                batch_sizes.add(len(v))

        if len(batch_sizes) > 1:
            # Get detailed information about which keys have which sizes
            size_info = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    size = v.shape[0]
                elif isinstance(v, list):
                    size = len(v)
                else:
                    continue
                if size not in size_info:
                    size_info[size] = []
                size_info[size].append(k)

            error_msg = "HCP: Inconsistent batch sizes detected in input data:\n"
            for size, keys in sorted(size_info.items()):
                error_msg += f"  Size {size}: {keys}\n"
            error_msg += (
                "\nThis likely indicates a data preparation mismatch. "
                "Ensure all data tensors have the same batch size before calling HCP scheduler."
            )
            raise ValueError(error_msg)

        # Step 1: Extract sequence lengths
        seq_lengths = self.extract_sequence_lengths(data, seq_length_key)

        # Step 2: Run scheduler
        groups, sample_id_groups = self.schedule_samples(seq_lengths)

        # Step 3: Verify all samples are assigned (for validation)
        (sample_local_cp_size,) = self.create_hcp_metadata(
            sample_id_groups, len(seq_lengths)
        )
        # Note: sample_local_cp_size is calculated here for logging/validation,
        # but not passed to workers (Megatron calculates it from sample_id_groups)

        # Step 4: Shard data with HCP metadata (flat list, one per DP×CP rank)
        flat_shards = self.shard_data_by_hdp_rank(data, sample_id_groups)

        # Step 5: Reshape to 2D [dp_idx][cp_idx] for run_all_workers_sharded_data
        return [
            [flat_shards[dp * self.cp_size + cp] for cp in range(self.cp_size)]
            for dp in range(self.dp_size)
        ]
