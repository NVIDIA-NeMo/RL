# `tests/unit/data_plane/` — test inventory

Generated audit of every test function under `tests/unit/data_plane/` with a one-line summary. Use this when deciding what to consolidate or drop.

---

## `test_architecture_invariants.py` (11 tests)

- `test_grpo_sync_engages_tq_policy` — Sync trainer must require a TQ-mediated policy.
- `test_grpo_sync_requires_data_plane_enabled` — Sync trainer hard-fails when invoked without `data_plane.enabled=true`.
- `test_no_feature_gate_pattern_in_either_trainer` — Catch the next "just one if branch" temptation in either trainer.
- `test_factory_does_not_construct_noop` — Production factory must not return a NoOp client.
- `test_factory_rejects_disabled_impl` — Factory must raise — not return None / NoOp — when disabled.
- `test_run_grpo_dispatches_both_trainers` — `examples/run_grpo.py._select_trainer` returns legacy vs sync per config.
- `test_legacy_does_not_import_sync` — Dependency direction: `grpo_sync.py` imports from `grpo.py`, not the reverse.
- `test_pack_per_token_field_is_exported` — `pack_per_token_field` must be importable from `nemo_rl.data_plane.codec`.
- `test_pack_per_token_field_is_wired_into_writeback` — **xfail.** At least one write-back call site must import it (wiring incomplete).
- `test_abc_method_present` — Renaming an ABC method is a wire break — keep the swap surface stable.
- `test_fp8_calib_filter_then_seqlen_check_no_crash` — End-to-end behavioral repro of the job 11920261 calib-vs-seqlen bug.

## `test_codec_jagged.py` (9 tests)

- `test_to_nested_by_length_strips_padding` — Right-pad columns must NOT be in the nested output.
- `test_to_nested_by_length_preserves_dtype` — bf16 in → bf16 out.
- `test_to_nested_by_length_rejects_shape_mismatch` — Shape sanity guard.
- `test_to_nested_by_length_rejects_1d_input` — 1D inputs aren't valid (no seq dim).
- `test_materialize_pads_nested_with_field_specific_pad_value` — Token field padded with pad_token_id; mask padded with 0.
- `test_materialize_passes_through_rectangular_tensors` — Already-padded fields emitted unchanged.
- `test_materialize_jagged_layout_passes_nested_through` — `layout='jagged'` path for nested-consuming callers.
- `test_materialize_default_pad_value_is_zero` — No `pad_value_dict` → pad with 0.
- `test_response_from_nested_extracts_response_slice` — Worker write-back: jagged (prompt+response) → response only.

## `test_codec_mooncake.py` (4 tests)

- `test_promote_1d_leaves_unsqueezes_1d` — `_promote_1d_leaves` turns 1D `(N,)` leaves into `(N, 1)` for mooncake wire.
- `test_promote_1d_roundtrip_via_from_wire` — `_promote_1d_leaves` + `_from_wire` restores original `(N,)` shape and values.
- `test_pack_per_token_field_truncates_sp_padding` — pack_per_token_field slices each row to its own length, dropping SP padding.
- `test_pack_per_token_field_exact_fit_matches_to_nested_by_length` — At exact fit, `pack_per_token_field` matches `to_nested_by_length`.

## `test_codec_wire_stripped.py` (5 tests)

- `test_unwrap_wire_stripped_payload_empty_td_to_none` — Empty TD (batch_dims=0) → None.
- `test_unwrap_wire_stripped_payload_real_nontensor_data_passes_through` — Live NonTensorData payload survives unwrap.
- `test_materialize_handles_wire_stripped_nontensor_stack` — Stack of empty TDs materializes to object array of None.
- `test_materialize_preserves_real_nontensor_data` — NonTensorStack of strings materializes to raw strings.
- `test_materialize_decodes_nontensor_stack_with_tensor_field` — Per-field decode: tensors stay padded, objects ride.

## `test_correctness.py` (16 tests)

- `test_kv_batch_get_after_clear_raises` — v3 driver tried to read input_ids for log_data after clear — must fail loud.
- `test_kv_batch_get_unproduced_field_raises` — Requesting an unproduced field must raise, not return junk.
- `test_get_data_without_select_fields_raises` — P2 invariant — never silently fetch all fields.
- `test_kv_batch_put_rejects_non_tensor_leaves` — P3 — adapters reject non-tensor leaves; no pickle on the bus.
- `test_claim_meta_unregistered_task_raises` — Catches typo'd consumer task names early.
- `test_kv_clear_with_none_drops_partition` — Step-end teardown removes the partition entirely.
- `test_double_register_partition_is_idempotent_overwrite` — Re-registering same partition_id within a step is OK.
- `test_check_consumption_status_only_true_when_all_consumed` — Stage-done signal must not lie.
- `test_shard_meta_for_dp_partitions_keys_disjointly` — Sum of shard sizes == total; pairwise disjoint.
- `test_shard_meta_for_dp_keeps_partition_id` — partition_id propagated to every shard.
- `test_kv_first_write_carries_multimodal_extras_through_tq` — VLM image features round-trip via TQ end-to-end.
- `test_kv_batch_put_preserves_bf16_dtype` — Catches silent fp32 promotion.
- `test_kv_batch_put_preserves_int64_dtype` — input_ids stays int64.
- `test_write_columns_accepts_batched_data_dict_input` — Job 11614968 v2 crash guard: worker write-back accepts BatchedDataDict.
- `test_kv_first_write_rejects_key_count_mismatch` — `len(keys) != n_samples` must fail (silent mis-align otherwise).
- `test_kv_first_write_meta_sequence_lengths_match_input_lengths` — Megatron's balanced packing needs `meta.sequence_lengths` to match.

## `test_factory.py` (5 tests)

- `test_factory_none_cfg_rejected` — None config fails fast, not silently.
- `test_factory_disabled_rejected` — Production factory rejects disabled config.
- `test_factory_noop_impl_rejected` — NoOp impl not selectable from production factory.
- `test_factory_unknown_impl_rejected` — Unknown impl name fails fast with a helpful error.
- `test_factory_disabled_error_message_helpful` — Disabled-config error message names the missing flag.

## `test_interface_contract.py` (7 tests)

- `test_factory_disabled_raises` — Factory has no NoOp fallback — disabled must raise.
- `test_factory_unknown_impl_raises` — Unknown impl raises.
- `test_register_put_get_clear` — End-to-end ABC round-trip.
- `test_claim_meta_advances_consumption` — `claim_meta` advances the per-task consumption cursor.
- `test_get_data_requires_field_selection` — P2 — fetching all fields is forbidden.
- `test_kv_batch_put_rejects_non_tensor_leaves` — P3 — adapter rejects non-tensor leaves.
- `test_close_is_idempotent` — `close()` can be called twice safely.

## `test_kvbatchmeta.py` (10 tests)

- `test_size_matches_keys` — `size` derived from `sample_ids` length.
- `test_default_fields_and_extra_info_optional` — `fields` and `sequence_lengths` default to None.
- `test_pickle_roundtrip_structural_equality` — Cloudpickle round-trip for Ray actor dispatch.
- `test_keys_with_duplicates_allowed_or_warned` — Meta doesn't enforce key uniqueness (caller's contract).
- `test_empty_meta_is_valid` — Empty meta is a valid value (e.g. empty DP shard).
- `test_partition_id_is_required` — `partition_id` is positional + required.
- `test_extra_info_default_is_unique_per_instance` — Mutable default trap — two metas don't share `extra_info`.
- `test_tags_align_with_keys` — `tags` exactly one dict per key, or None.
- `test_tags_travel_with_subset_slice_concat` — Per-key tags follow keys through subset/slice/concat.
- `test_tags_none_when_either_side_missing_in_concat` — concat drops tags if either side has none.

## `test_leader_broadcast.py` (2 tests)

- `test_leader_broadcast_round_trip` — 2-rank gloo broadcast of a BatchedDataDict round-trips.
- `test_get_replica_group_default_is_none` — `TQWorkerMixin._get_replica_group` default is None.

## `test_local_node_ip.py` (5 tests)

- `test_local_node_ip_skips_link_local` — gethostbyname returns 169.254.x.x → helper falls back.
- `test_local_node_ip_skips_loopback` — Returns 127.0.0.1 → helper falls back.
- `test_local_node_ip_returns_routable` — Routable address returned as-is.
- `test_local_node_ip_returns_empty_on_exception` — DNS exception → returns empty string (no crash).
- `test_mc_tcp_bind_address_overwrites_existing` — TQDataPlaneClient `__init__` uses direct assignment (not `setdefault`).

## `test_message_log_decompose.py` (11 tests)

- `test_decompose_message_log_basic_shapes` — Basic shapes of decompose output.
- `test_decompose_message_log_no_assistant_turn` — No-assistant case handled.
- `test_decompose_message_log_picks_first_assistant` — Multiple assistant turns → first wins for `response_token_lengths`.
- `test_decompose_message_log_jagged_turn_count` — Different turn counts pad `turn_lengths` with zeros.
- `test_decompose_message_log_missing_role_raises` — Missing `role` raises KeyError loudly.
- `test_reconstruct_message_log_roundtrip` — decompose → flatten → reconstruct equivalent message_log.
- `test_reconstruct_message_log_returns_views` — Per-turn `token_ids` are views into local storage.
- `test_reconstruct_message_log_attaches_generation_logprobs` — Attached only to assistant turns.
- `test_attach_message_log_view_populates_batch` — `attach_message_log_view` populates batch view.
- `test_attach_message_log_view_noop_when_fields_absent` — Without decomposed fields, attach is a no-op.
- `test_attach_message_log_view_idempotent` — Calling twice produces same shape.

## `test_observability.py` (8 tests)

- `test_put_records_bytes_and_count` — Observability decorator records put bytes + count.
- `test_get_records_after_put` — Records get ops after put.
- `test_register_and_clear_recorded` — register/clear ops are recorded.
- `test_error_status_recorded_and_reraised` — Decorator records error AND re-raises (no swallowing).
- `test_snapshot_accumulates_successful_ops` — Snapshot accumulates over time.
- `test_default_callback_is_noop` — Omitting on_event must not raise.
- `test_close_propagates` — close() is forwarded to wrapped client.
- `test_factory_wraps_when_observability_enabled` — factory.py uses the same MetricsDataPlaneClient.

## `test_preshard_extras.py` (10 tests)

- `test_kv_first_write_writes_seed_fields` — Seed fields written to TQ.
- `test_kv_first_write_carries_multimodal_extras` — VLM extras (pixel_values) ride along, no schema declaration needed.
- `test_kv_first_write_keys_match_uids_x_ngen` — Keys round-trip: `f"{uid}_g{i}"` preserved.
- `test_shard_meta_for_dp_partitions_keys_disjointly` — Sum of shards == total, disjoint.
- `test_shard_meta_for_dp_preserves_partition_id` — partition_id preserved across DP shards.
- `test_shard_meta_for_dp_unsorted_round_trip` — `unsorted_indices` reconstructs input order from concat.
- `test_kvbatchmeta_subset_filters_keys_and_seqlens` — `subset` filters keys + seq_lengths.
- `test_kvbatchmeta_concat_joins_keys_and_seqlens` — `concat` joins.
- `test_kvbatchmeta_slice_takes_range` — `slice` takes a contiguous range.
- `test_kvbatchmeta_concat_rejects_partition_mismatch` — `concat` rejects different `partition_id`s.

## `test_seqpack_equivalence.py` (3 tests, ×2 backends)

- `test_seqpack_legacy_equals_tq[simple|mooncake_cpu]` — Sequence packing byte-equivalence: legacy shards == TQ-roundtripped.
- `test_dynbatch_legacy_equals_tq[simple|mooncake_cpu]` — Same claim for dynamic batching.
- `test_no_packing_legacy_equals_tq[simple|mooncake_cpu]` — Sanity: lossless transport even without packing/dynbatch.

## `test_smoke.py` (5 tests)

- `test_sync_utils_module_imports` — Catches FQN drift after `algorithms.sync_utils` consolidation.
- `test_sync_rollout_actor_registered_under_vllm_tier` — Multinode dep: tensordict must be on the vLLM tier.
- `test_kvbatchmeta_schema_unchanged` — Schema-pin: KVBatchMeta is the cross-process boundary.
- `test_dataplane_client_abc_surface` — Catches accidental ABC method removal/rename.
- `test_async_and_sync_actors_share_env_tier` — Sync mirrors async's env tier (both drive vLLM).

## `test_sync_one_hop.py` (9 tests)

- `test_write_columns_lands_in_tq` — write_columns lands fields in TQ.
- `test_read_columns_returns_only_requested_fields` — read_columns honors `select_fields`.
- `test_write_then_read_roundtrip_after_train_window` — Full lifecycle: rollout puts → driver deltas → read deltas back.
- `test_meta_keys_identity_across_dp_shards` — `shard_meta_for_dp` must NOT mint new keys.
- `test_kv_clear_uses_meta_keys_minted_at_rollout` — Step-end clear targets the SAME keys rollout minted.
- `test_apply_dynamic_sampling_filters_zero_std` — Drops zero-std uids and clears their TQ payload.
- `test_apply_dynamic_sampling_completes_when_train_size_reached` — When cache hits train_prompts_size, is_complete=True.
- `test_apply_dynamic_sampling_overflow_slices_and_clears` — Overflow: slice + clear discards.
- `test_apply_dynamic_sampling_raises_on_max_gen_batches` — Exceeding max_gen_batches raises loudly.

## `test_tq_lifecycle.py` (5 tests, some ×2 backends)

- `test_smoke_round_trip` — Basic register → put → claim_meta → get_data → clear flow.
- `test_smoke_round_trip_backends[simple|mooncake_cpu]` — Same parameterized over both backends.
- `test_smoke_round_trip_1d_fields` — `(N,)` tensors come back as `(N,)`, not `(N,1)`.
- `test_object_round_trip_backends[simple|mooncake_cpu]` — `np.ndarray(dtype=object)` round-trips both backends.
- `test_object_and_tensor_mixed_round_trip_backends[simple|mooncake_cpu]` — Mixed tensor+object in one put.

---

## Potential simplifications (candidates to drop or merge)

| Overlap | Files involved | Suggestion |
|---|---|---|
| `factory disabled/unknown impl rejected` | `test_factory.py` (5 tests) + `test_interface_contract.py::test_factory_*` (2) | Keep `test_factory.py` (more thorough); drop the two duplicates in `test_interface_contract.py` |
| `kv_batch_put_rejects_non_tensor_leaves` | `test_correctness.py` + `test_interface_contract.py` | One is enough — keep `test_correctness.py`'s (P3 framing). |
| `get_data_without_select_fields_raises` / `test_get_data_requires_field_selection` | `test_correctness.py` + `test_interface_contract.py` | Same property; keep `test_correctness.py`. |
| `shard_meta_for_dp_partitions_keys_disjointly` + `_keeps/preserves_partition_id` | `test_correctness.py` (2) + `test_preshard_extras.py` (2) | Pure dup. Drop from `test_correctness.py`. |
| `kv_first_write_carries_multimodal_extras` | `test_correctness.py::test_kv_first_write_carries_multimodal_extras_through_tq` + `test_preshard_extras.py::test_kv_first_write_carries_multimodal_extras` | Pure dup. Keep `test_preshard_extras.py`. |
| ABC surface checks | `test_smoke.py::test_dataplane_client_abc_surface` + `test_architecture_invariants.py::test_abc_method_present` + `test_interface_contract.py` (covers same surface end-to-end) | Three angles on the same invariant. Keep `test_architecture_invariants.py` (most explicit); drop the smoke one. |
| Codec tests across 3 files | `test_codec_jagged.py` (9), `test_codec_mooncake.py` (4), `test_codec_wire_stripped.py` (5) | Distinct paths but small files — could merge into a single `test_codec.py` with `# ── jagged ──` / `# ── mooncake ──` / `# ── wire_stripped ──` sections. Saves 2 file headers. |
| `test_smoke.py` — 5 narrow checks | various | These are best as a single fast "import-this-stuff" smoke test, not 5 separate ones. Consider folding into a parametrized `test_imports_unchanged`. |

### Likely to drop

If you want a one-pass cull, the safest deletes are:
1. `test_interface_contract.py::test_factory_disabled_raises` (dup of `test_factory.py`)
2. `test_interface_contract.py::test_factory_unknown_impl_raises` (dup of `test_factory.py`)
3. `test_interface_contract.py::test_get_data_requires_field_selection` (dup of `test_correctness.py`)
4. `test_interface_contract.py::test_kv_batch_put_rejects_non_tensor_leaves` (dup of `test_correctness.py`)
5. `test_correctness.py::test_shard_meta_for_dp_partitions_keys_disjointly` (dup of `test_preshard_extras.py`)
6. `test_correctness.py::test_shard_meta_for_dp_keeps_partition_id` (dup of `test_preshard_extras.py`)
7. `test_correctness.py::test_kv_first_write_carries_multimodal_extras_through_tq` (dup of `test_preshard_extras.py`)
8. `test_smoke.py::test_dataplane_client_abc_surface` (dup of `test_architecture_invariants.py`)

→ −8 tests, no coverage loss.

### Likely to consolidate (file count, not test count)

- Merge `test_codec_{jagged,mooncake,wire_stripped}.py` → `test_codec.py` (3 files → 1, same 18 tests)
- Merge `test_factory.py` into `test_interface_contract.py` (or vice-versa) since they share scope
- `test_smoke.py` is just 5 import/registration checks — could move into `test_architecture_invariants.py`

### File-count target

| Now | After dedupe + merge |
|---|---|
| 17 files / ~125 tests | 12 files / ~117 tests |
