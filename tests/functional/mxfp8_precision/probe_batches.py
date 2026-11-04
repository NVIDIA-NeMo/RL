"""Prepare worker-local probe batches through the production packing implementation."""

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from precision_probe_common import ordered_single_sequence_plan


def prepare_probe_batch(data, cfg, *, logprob=False, canonical_order=True):
    batch = BatchedDataDict(data)
    if not cfg["sequence_packing"]["enabled"]:
        assert not cfg["dynamic_batching"]["enabled"]
        return batch
    packing = cfg["sequence_packing"]
    args = {
        "algorithm": packing["algorithm"],
        "input_key": "input_ids",
        "input_lengths_key": "input_lengths",
        "sequence_length_pad_multiple": cfg["make_sequence_length_divisible_by"],
        "max_tokens_per_microbatch": packing[
            "logprob_mb_tokens" if logprob else "train_mb_tokens"
        ],
    }
    for key in ["microbatch_order", "max_sequences_per_bin", "pair_grouping_key"]:
        if packing.get(key) is not None:
            args[key] = packing[key]
    shards, _ = batch.shard_by_batch_size(
        1, batch_size=len(data["input_ids"]), sequence_packing_args=args
    )
    prepared = shards[0]
    assert prepared.micro_batch_indices is not None
    assert prepared.micro_batch_lengths is not None
    if (
        canonical_order
        and packing.get("max_sequences_per_bin") == 1
        and "probe_row_id" in prepared
    ):
        order, bounds, lengths = ordered_single_sequence_plan(
            prepared["probe_row_id"].cpu().tolist(),
            prepared.micro_batch_indices[0],
            prepared.micro_batch_lengths[0],
        )
        prepared = prepared.select_indices(order)
        prepared.micro_batch_indices = [bounds]
        prepared.micro_batch_lengths = [lengths]
        prepared.elem_counts_per_gb = [len(order)]
    if packing.get("max_sequences_per_bin") == 1:
        assert all(
            len(mb["input_ids"]) == 1
            for mb in prepared.make_microbatch_iterator_for_packable_sequences()
        )
    assert sum(
        len(mb["input_ids"])
        for mb in prepared.make_microbatch_iterator_for_packable_sequences()
    ) == len(data["input_ids"])
    return prepared


def verify_probe_batch_preparation(data, cfg):
    """Exercise full/split and logprob packing before allocating model workers."""
    from collections import Counter

    half = len(data["input_ids"]) // 2
    for start, end, logprob in [
        (0, len(data["input_ids"]), False),
        (0, half, False),
        (half, len(data["input_ids"]), False),
        (half, len(data["input_ids"]), True),
    ]:
        part = {k: v[start:end] for k, v in data.items()}
        prepared = prepare_probe_batch(part, cfg, logprob=logprob)
        expected = Counter(map(tuple, part["input_ids"].tolist()))
        if cfg["sequence_packing"]["enabled"]:
            rows = [
                row
                for mb in prepared.make_microbatch_iterator_for_packable_sequences()
                for row in mb["input_ids"].tolist()
            ]
        else:
            rows = prepared["input_ids"].tolist()
        assert Counter(map(tuple, rows)) == expected, (
            "Packing dropped or duplicated probe inputs"
        )
        if cfg["sequence_packing"].get("max_sequences_per_bin") == 1:
            actual_ids = [
                int(mb["probe_row_id"].item())
                for mb in prepared.make_microbatch_iterator_for_packable_sequences()
            ]
            assert actual_ids == sorted(part["probe_row_id"].tolist()), (
                "Packing changed the controlled accumulation order"
            )
