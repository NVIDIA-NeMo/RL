"""Data iterator wrapper for Hybrid Context Parallelism in NeMo-RL.

This module provides a wrapper that formats data in the way Megatron-LM's
hybrid_context_parallel_forward_backward expects: (new_data_iterator, sample_id_groups).

new_data_iterator yields ProcessedMicrobatch objects — one per group/microbatch.
Each ProcessedMicrobatch holds the packed and CP-sharded data for that group,
ready for forward_with_post_processing_fn.
"""

from typing import List, Tuple

import torch

from megatron.core import parallel_state
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.utils import get_batch_on_this_hybrid_cp_rank

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.data import (
    ProcessedMicrobatch,
    _get_pack_sequence_parameters_for_megatron,
    _pack_sequences_for_megatron,
)


class HCPDataIterator:
    """Iterator wrapper for HCP that yields data in Megatron-LM format.

    Megatron's hybrid_context_parallel_forward_backward expects:
        data = next(data_iterator)
        sample_id_groups = data[1]
        new_data_iterator = data[0]

    new_data_iterator yields one ProcessedMicrobatch per group.
    Each ProcessedMicrobatch contains the packed and CP-sharded tokens for
    that group, along with packed_seq_params that includes the per-group
    cp_group set by get_batch_on_this_hybrid_cp_rank.

    Args:
        data: BatchedDataDict with HCP metadata attached
            Must have "sample_id_groups" and "shard_sample_ids" keys
        cfg: Policy configuration dictionary containing megatron_cfg
    """

    def __init__(self, data: BatchedDataDict, cfg: dict):
        """Initialize HCP data iterator.

        Args:
            data: BatchedDataDict with sample_id_groups and shard_sample_ids attached
            cfg: Policy configuration dictionary
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
        self.cfg = cfg
        self.megatron_cfg = cfg["megatron_cfg"]
        self.sample_id_groups = data["sample_id_groups"]
        self.shard_sample_ids = data["shard_sample_ids"]
        self._yielded = False
        self.dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)

    def __iter__(self):
        """Return self as iterator."""
        return self

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

    def _pack_and_shard_group(
        self,
        samples: List[BatchedDataDict],
        local_cp_size: int,
    ) -> ProcessedMicrobatch:
        """Pack and shard a group of samples into a ProcessedMicrobatch.

        Packs the samples' token sequences and shards them for this CP rank
        using get_batch_on_this_hybrid_cp_rank, which sets the correct
        per-group cp_group on the returned packed_seq_params.

        Args:
            samples: List of BatchedDataDict objects (one per sample in the group).
                Each has "input_ids" [1, max_seq_len] and "input_lengths" [1].
            local_cp_size: CP size for this group as assigned by the HCP scheduler.

        Returns:
            ProcessedMicrobatch ready for forward_with_post_processing_fn.
        """
        device = torch.cuda.current_device()

        # Stack samples into a single BatchedDataDict on GPU
        data_dict = BatchedDataDict.from_batches(samples)
        data_dict.to(device)

        input_ids = data_dict["input_ids"]    # [n_samples, max_seq_len]
        seq_lengths = data_dict["input_lengths"]  # [n_samples]

        # Per-sequence padding must be a multiple of (local_cp_size * 2), and
        # if sequence parallelism is enabled, also a multiple of tp_size.
        tp_size = self.megatron_cfg["tensor_model_parallel_size"]
        sp = self.megatron_cfg["sequence_parallel"]
        pad_individual_seqs = local_cp_size * 2
        if tp_size > 1 and sp:
            pad_individual_seqs *= tp_size

        # max_seq_len_in_batch is used only when PP > 1 to determine pad_packed_seq_to.
        # For PP = 1, pad_packed_seq_to is None regardless of this value.
        max_seq_len_in_batch = input_ids.shape[1]

        (
            pad_individual_seqs,
            pad_packed_seq_to_multiple_of,
            pad_packed_seq_to,
        ) = _get_pack_sequence_parameters_for_megatron(
            self.megatron_cfg,
            pad_individual_seqs,
            max_seq_len_in_batch,
            effective_cp_size=local_cp_size,
        )

        # Pack sequences without CP sharding (cp_rank=0, cp_size=1).
        # get_batch_on_this_hybrid_cp_rank will handle the HCP-aware THD sharding.
        (
            all_input_ids,        # [1, T_packed] — full packed sequence
            _,                    # packed_input_ids == all_input_ids when cp_size=1
            _,                    # packed_seq_params — discarded; HCP version set below
            cu_seqlens,
            cu_seqlens_padded,
        ) = _pack_sequences_for_megatron(
            input_ids,
            seq_lengths,
            pad_individual_seqs,
            pad_packed_seq_to_multiple_of,
            pad_packed_seq_to,
            cp_rank=0,
            cp_size=1,
        )

        # max_seqlen per sub-sequence (padded), needed by get_thd_batch_on_this_cp_rank
        max_seqlen = (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]).max().to(dtype=torch.int32)

        # Shard the packed sequence for this HCP rank.
        # all_input_ids is [1, T]; squeeze to [T] since get_batch_on_this_hybrid_cp_rank
        # calls torch.stack([data], 0) internally to restore the batch dimension.
        batch_out, packed_seq_params = get_batch_on_this_hybrid_cp_rank(
            batch={"tokens": all_input_ids.squeeze(0)},
            cu_seqlens=cu_seqlens,
            cu_seqlens_padded=cu_seqlens_padded,
            max_seqlen=max_seqlen,
            local_cp_size=local_cp_size,
        )

        # batch_out["tokens"] has shape [1, T_sharded] after the internal stack + index_select
        input_ids_cp_sharded = batch_out["tokens"]

        return ProcessedMicrobatch(
            data_dict=data_dict,
            input_ids=all_input_ids,
            input_ids_cp_sharded=input_ids_cp_sharded,
            attention_mask=None,
            position_ids=None,
            packed_seq_params=packed_seq_params,
            cu_seqlens_padded=cu_seqlens_padded,
        )

    def __next__(self) -> Tuple["RerunDataIterator", List[List[List[int]]]]:
        """Yield data in format expected by hybrid_context_parallel_forward_backward.

        Returns:
            Tuple of (new_data_iterator, sample_id_groups):
            - new_data_iterator: RerunDataIterator that yields one ProcessedMicrobatch
              per group/microbatch.
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

        # Build one ProcessedMicrobatch per group by packing and sharding.
        processed_mbs = []
        for i in range(len(self.sample_id_groups)):
            samples = [batch[sid] for sid in self.sample_id_groups[i][hdp_rank]]
            local_cp_size = self._get_local_cp_size_for_group(i, hdp_rank)
            processed_mb = self._pack_and_shard_group(samples, local_cp_size)
            processed_mbs.append(processed_mb)

        new_data_iterator = RerunDataIterator(iter(processed_mbs))

        return new_data_iterator, self.sample_id_groups

    def __len__(self) -> int:
        """Return number of yields (always 1 for HCP).

        Note:
            HCP processes all samples in a single forward/backward pass,
            coordinating across groups and ranks internally.
        """
        return 1


def create_hcp_data_iterator(data: BatchedDataDict, cfg: dict) -> HCPDataIterator:
    """Create HCP data iterator from BatchedDataDict.

    Args:
        data: BatchedDataDict with sample_id_groups attached
        cfg: Policy configuration dictionary

    Returns:
        HCPDataIterator that yields (new_data_iterator, sample_id_groups)

    Raises:
        ValueError: If data doesn't have sample_id_groups metadata
    """
    return HCPDataIterator(data, cfg)
