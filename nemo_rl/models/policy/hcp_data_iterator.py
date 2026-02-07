"""Data iterator wrapper for Hybrid Context Parallelism in NeMo-RL.

This module provides a wrapper that formats data in the way Megatron-LM's
hybrid_context_parallel_forward_backward expects: (batch, sample_id_groups).
"""

from typing import Any, List, Tuple

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class HCPDataIterator:
    """Iterator wrapper for HCP that yields data in Megatron-LM format.

    Megatron's hybrid_context_parallel_forward_backward expects:
        data = next(data_iterator)
        sample_id_groups = data[1]
        batch = data[0]

    This wrapper:
    1. Takes a BatchedDataDict with sample_id_groups attached
    2. Yields (batch, sample_id_groups) tuples
    3. Each batch contains all samples (Megatron splits them internally)

    Args:
        data: BatchedDataDict with HCP metadata attached
            Must have "sample_id_groups" key
    """

    def __init__(self, data: BatchedDataDict):
        """Initialize HCP data iterator.

        Args:
            data: BatchedDataDict with sample_id_groups attached
        """
        if "sample_id_groups" not in data:
            raise ValueError(
                "HCPDataIterator requires data with 'sample_id_groups' metadata. "
                "Ensure HeadNodeHCPScheduler was used to create this data."
            )

        self.data = data
        self.sample_id_groups = data["sample_id_groups"]
        self._yielded = False

    def __iter__(self):
        """Return self as iterator."""
        return self

    def __next__(self) -> Tuple[List[dict], List[List[List[int]]]]:
        """Yield data in format expected by hybrid_context_parallel_forward_backward.

        Returns:
            Tuple of (batch, sample_id_groups):
            - batch: List of sample dicts, one per sample in the data
            - sample_id_groups: Scheduling structure [num_rounds][hdp_size] -> sample IDs

        Note:
            This iterator yields exactly once per forward/backward pass.
            Megatron's hybrid_context_parallel_forward_backward handles splitting
            the batch into individual samples via _get_new_data_iterator.
        """
        if self._yielded:
            raise StopIteration

        self._yielded = True

        # Convert BatchedDataDict to list of sample dicts
        # Each sample is a dict with keys like 'tokens', 'labels', etc.
        batch = []
        num_samples = self.data.size

        for sample_idx in range(num_samples):
            sample = {}
            for key, value in self.data.items():
                if key == "sample_id_groups":
                    # Skip metadata, don't include in sample
                    continue
                # Extract single sample
                sample[key] = value[sample_idx]
            batch.append(sample)

        return batch, self.sample_id_groups

    def __len__(self) -> int:
        """Return number of yields (always 1 for HCP).

        Note:
            HCP processes all samples in a single forward/backward pass,
            coordinating across groups and ranks internally.
        """
        return 1


def create_hcp_data_iterator(data: BatchedDataDict) -> HCPDataIterator:
    """Create HCP data iterator from BatchedDataDict.

    Args:
        data: BatchedDataDict with sample_id_groups attached

    Returns:
        HCPDataIterator that yields (batch, sample_id_groups)

    Raises:
        ValueError: If data doesn't have sample_id_groups metadata
    """
    return HCPDataIterator(data)
