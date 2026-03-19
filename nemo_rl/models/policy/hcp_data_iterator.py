"""Data iterator wrapper for Hybrid Context Parallelism in NeMo-RL.

This module provides a wrapper that formats data in the way Megatron-LM's
hybrid_context_parallel_forward_backward expects: (new_data_iterator, sample_id_groups).

new_data_iterator yields HCPGroupedSamples objects — one per group/microbatch.
Each HCPGroupedSamples holds the list of BatchedDataDict samples that must be
packed together for that group, along with the local_cp_size for the group.
forward_step_arbitrary_loss is responsible for the actual packing.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from megatron.core import parallel_state
from megatron.core.rerun_state_machine import RerunDataIterator


@dataclass
class HCPGroupedSamples:
    """Container for a group of samples to be packed together in one HCP forward pass.

    Yielded by HCPDataIterator for each group/microbatch.  forward_step_arbitrary_loss
    detects this type and performs the actual packing + CP sharding.

    Attributes:
        samples: List of BatchedDataDict objects, one per sample in the group.
            All samples share the same local_cp_size.
        local_cp_size: Context-parallel size assigned to this group by the HCP scheduler.
    """

    samples: List  # list[BatchedDataDict]
    local_cp_size: int


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
            data: BatchedDataDict with sample_id_groups and shard_sample_ids attached
        """
        if "sample_id_groups" not in data:
            raise ValueError(
                "HCPDataIterator requires data with 'sample_id_groups' metadata. "
                "Ensure HeadNodeHCPScheduler was used to create this data."
            )
        if "shard_sample_ids" not in data:
            raise ValueError(
                "HCPDataIterator requires data with 'shard_sample_ids' metadata. "
                "Ensure HeadNodeHCPScheduler was used to create this data."
            )

        self.data = data
        self.sample_id_groups = data["sample_id_groups"]
        self.shard_sample_ids = data["shard_sample_ids"]
        self._yielded = False
        self.dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)

    def __iter__(self):
        """Return self as iterator."""
        return self

    def _pack_sequences(
        self,
        samples: List[Dict[str, torch.Tensor]],
        local_cp_size: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        def _pack_tensors(tensors):
            return torch.cat([t.reshape(-1) for t in tensors], dim=0)

        # Get the padded lengths of all sub-samples being packed
        sample_padded_lens = torch.tensor([s["tokens"].shape[0] for s in samples], dtype=torch.int32)
        # Create the cu_seqlens_padded tensor
        cu_seqlens_padded = torch.empty(1, sample_padded_lens.numel() + 1, device=torch.cuda.current_device(), dtype=torch.int32)
        cu_seqlens_padded[0, 0] = 0
        cu_seqlens_padded[0, 1:] = torch.cumsum(sample_padded_lens, dim=0)
        max_seqlen = torch.max(sample_padded_lens).to(dtype=torch.int32)

        tokens = _pack_tensors([sample["tokens"] for sample in samples])
        
        #TODO(pmannan): Perform any necessary padding for CP and SP.

        new_sample = {}
        new_sample["tokens"] = tokens
        if local_cp_size is not None:
            # Accept either a Python int or a CUDA scalar tensor; keep as a scalar tensor on GPU.
            # new_sample["local_cp_size"] = local_cp_size.to(device=dev, dtype=torch.int32)
            new_sample["local_cp_size"] = torch.tensor(local_cp_size, device=torch.cuda.current_device(), dtype=torch.int32)

        # new_sample["cu_seqlens_padded"] = cu_seqlens_padded
        # We set cu_seqlens to cu_seqlens_padded here
        # we don't provide cu_seqlens_padded here because `get_batch_on_this_tp_rank` does not consider it
        # at the moment. It is assumed that any necessary padding will be after data is loaded in.
        # TODO(pmannan): We need to be able to differentiate if the original data_iterator is providing
        # padded samples or valid lengths.
        new_sample["cu_seqlens"] = cu_seqlens_padded
        new_sample["max_seqlen"] = max_seqlen

        return new_sample

    def _build_packed_microbatches(
        self,
        grouped_samples: List[List[Dict[str, torch.Tensor]]],
        hdp_rank: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Build packed samples for each microbatch given a pre-built list of `samples` per microbatch.

        Args:
            grouped_samples: List of length `num_microbatches`. Each element is the `samples` list
                (list[sample]) for that microbatch, where `sample` is the dict returned by
                `dataset.__getitem__`.

        Returns:
            new_samples: list of packed samples (dicts) length == num_micro_batches.
        """
        num_micro_batches = len(grouped_samples)
        # seg_starts: List[int] = [0]
        # original_lens_tensors = []
        # padded_lens_tensors = []

        # for i in range(num_micro_batches):
        #     samples = grouped_samples[i]
        #     seg_starts.append(seg_starts[-1] + len(samples))
            # original_lens_tensors.extend([s["tokens"].shape[0] for s in samples])
            # padded_lens_tensors.extend([s["tokens"].shape[0] for s in samples])

        # padded_lens_all_gpu = torch.cat(padded_lens_tensors, dim=0).to(dtype=torch.int32)
        # original_lens_all_gpu = torch.cat(original_lens_tensors, dim=0).to(dtype=torch.int32)

        new_samples: List[Dict[str, torch.Tensor]] = []
        for i in range(num_micro_batches):
            samples = grouped_samples[i]
            local_cp_size = -1
            # sample_id_groups = [[[0, 1, 2], [0, 1, 2]], [[3, 4, 5], [6, 7, 8]]]
            # Indicates the sub-sample ids per microbatch for each DPxCP rank.
            for sub_sample_id in self.sample_id_groups[i][hdp_rank]: # i:0 hdp_rank:0 [[0, 1, 2]]
                # sub_sample_id: 0 / 1 / 2
                partner_cp_size = len(
                    [True for sample_ids in self.sample_id_groups[i] if sub_sample_id in sample_ids]
                )

                if local_cp_size == -1:
                    local_cp_size = partner_cp_size
                else:
                    assert local_cp_size == partner_cp_size, f"\
                        found sample within a packed microbatch with different local_cp_size: \
                        {local_cp_size} != {partner_cp_size}"

            # lp = padded_lens_all_gpu[seg_starts[i] : seg_starts[i + 1]]
            # lo = original_lens_all_gpu[seg_starts[i] : seg_starts[i + 1]]

            new_sample = self._pack_sequences(samples, local_cp_size)
            new_samples.append(new_sample)

        return new_samples

    def _get_local_cp_size_for_group(self, group_idx: int, hdp_rank: int) -> int:
        """Compute the local CP size for all samples in a given group.

        local_cp_size == the number of HDP ranks that share any one sub_sample_id
        within the group (i.e. the CP size used by the HCP scheduler for those samples).
        All sub-samples belonging to the same HDP rank within a group are guaranteed
        to have the same local_cp_size (asserted below).

        Args:
            group_idx: Index into sample_id_groups (i.e. the microbatch/group index).
            hdp_rank: This rank's position in the HDP (data × context parallel) group.

        Returns:
            The local CP size for this group.
        """
        sub_sample_ids = self.sample_id_groups[group_idx][hdp_rank]
        assert len(sub_sample_ids) > 0, f"Group {group_idx} has no samples for hdp_rank {hdp_rank}"

        local_cp_size = -1
        for sub_sample_id in sub_sample_ids:
            partner_cp_size = len(
                [True for sample_ids in self.sample_id_groups[group_idx] if sub_sample_id in sample_ids]
            )
            if local_cp_size == -1:
                local_cp_size = partner_cp_size
            else:
                assert local_cp_size == partner_cp_size, (
                    f"Group {group_idx}: sub_sample_id {sub_sample_id} has local_cp_size "
                    f"{partner_cp_size} but earlier samples had {local_cp_size}. "
                    "All samples within a group must share the same local_cp_size."
                )
        return local_cp_size

    def __next__(self) -> Tuple["RerunDataIterator", List[List[List[int]]]]:
        """Yield data in format expected by hybrid_context_parallel_forward_backward.

        Returns:
            Tuple of (new_data_iterator, sample_id_groups):
            - new_data_iterator: RerunDataIterator that yields one HCPGroupedSamples
              per group/microbatch. forward_step_arbitrary_loss detects HCPGroupedSamples
              and performs the packing + CP sharding itself.
            - sample_id_groups: Scheduling structure [num_rounds][hdp_size] -> sample IDs

        Note:
            Yields exactly once per forward/backward pass.
            hybrid_context_parallel_forward_backward sets num_samples_this_group=1 for
            all groups (after sequence packing), so new_data_iterator is called once
            per group by Megatron's forward loop.
        """
        if self._yielded:
            raise StopIteration

        self._yielded = True

        # Build a per-global-sample-id lookup: {global_id: BatchedDataDict(size=1)}
        batch = {}
        num_samples = self.data.size
        for local_idx in range(num_samples):
            global_sample_id = self.shard_sample_ids[local_idx]
            batch[global_sample_id] = self.data.slice(local_idx, local_idx + 1)

        hdp_rank = self.dp_cp_group.rank()

        # Build one HCPGroupedSamples per group.
        # Each group holds the samples that will be packed into a single forward pass,
        # along with the local_cp_size determined by the HCP scheduler.
        grouped_data = []
        for i in range(len(self.sample_id_groups)):
            samples = [batch[sid] for sid in self.sample_id_groups[i][hdp_rank]]
            local_cp_size = self._get_local_cp_size_for_group(i, hdp_rank)
            grouped_data.append(HCPGroupedSamples(samples=samples, local_cp_size=local_cp_size))

        new_data_iterator = RerunDataIterator(iter(grouped_data))

        return new_data_iterator, self.sample_id_groups

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
