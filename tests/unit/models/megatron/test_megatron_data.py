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

"""
Unit tests for Megatron data utilities.

This module tests the data processing functions in nemo_rl.models.megatron.data,
focusing on:
- Microbatch processing and iteration
- Sequence packing and unpacking
- Global batch processing
- Sequence dimension validation
"""

from unittest.mock import MagicMock, patch

import pytest
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    PY_EXECUTABLES,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from tests.unit.models.megatron.megatron_data_actors import (
    GetPackSequenceParametersTestActor,
    PackSequencesTestActor,
)


@pytest.mark.mcore
class TestProcessedMicrobatchDataclass:
    """Tests for ProcessedMicrobatch dataclass."""

    def test_processed_microbatch_fields(self):
        """Test that ProcessedMicrobatch has all expected fields."""
        from nemo_rl.models.megatron.data import ProcessedMicrobatch

        mock_data_dict = MagicMock()
        mock_input_ids = torch.tensor([[1, 2, 3]])
        mock_input_ids_cp_sharded = torch.tensor([[1, 2, 3]])
        mock_attention_mask = torch.tensor([[1, 1, 1]])
        mock_position_ids = torch.tensor([[0, 1, 2]])
        mock_packed_seq_params = MagicMock()
        mock_cu_seqlens_padded = torch.tensor([0, 3])

        microbatch = ProcessedMicrobatch(
            data_dict=mock_data_dict,
            input_ids=mock_input_ids,
            input_ids_cp_sharded=mock_input_ids_cp_sharded,
            attention_mask=mock_attention_mask,
            position_ids=mock_position_ids,
            packed_seq_params=mock_packed_seq_params,
            cu_seqlens_padded=mock_cu_seqlens_padded,
        )

        assert microbatch.data_dict == mock_data_dict
        assert torch.equal(microbatch.input_ids, mock_input_ids)
        assert torch.equal(microbatch.input_ids_cp_sharded, mock_input_ids_cp_sharded)
        assert torch.equal(microbatch.attention_mask, mock_attention_mask)
        assert torch.equal(microbatch.position_ids, mock_position_ids)
        assert microbatch.packed_seq_params == mock_packed_seq_params
        assert torch.equal(microbatch.cu_seqlens_padded, mock_cu_seqlens_padded)
        assert microbatch.routed_experts is None
        assert microbatch.routed_experts_cp_sharded is None


@pytest.mark.mcore
class TestGetAndValidateSeqlen:
    """Tests for get_and_validate_seqlen function."""

    def test_get_and_validate_seqlen_valid(self):
        """Test get_and_validate_seqlen with valid data."""
        from nemo_rl.models.megatron.data import get_and_validate_seqlen

        # Create mock data with consistent sequence dimension
        data = MagicMock()
        data.__getitem__ = MagicMock(
            side_effect=lambda k: torch.zeros(2, 10) if k == "input_ids" else None
        )
        data.items = MagicMock(
            return_value=[
                ("input_ids", torch.zeros(2, 10)),
                ("attention_mask", torch.zeros(2, 10)),
            ]
        )

        sequence_dim, seq_dim_size = get_and_validate_seqlen(data)

        assert sequence_dim == 1
        assert seq_dim_size == 10

    def test_get_and_validate_seqlen_mismatch(self):
        """Test get_and_validate_seqlen with mismatched sequence dimensions."""
        from nemo_rl.models.megatron.data import get_and_validate_seqlen

        # Create mock data with mismatched sequence dimension
        data = MagicMock()
        data.__getitem__ = MagicMock(
            side_effect=lambda k: torch.zeros(2, 10) if k == "input_ids" else None
        )
        data.items = MagicMock(
            return_value=[
                ("input_ids", torch.zeros(2, 10)),
                ("other_tensor", torch.zeros(2, 15)),  # Mismatched!
            ]
        )

        with pytest.raises(AssertionError) as exc_info:
            get_and_validate_seqlen(data)

        assert "Dim 1 must be the sequence dim" in str(exc_info.value)

    def test_get_and_validate_seqlen_skips_1d_tensors(self):
        """Test that get_and_validate_seqlen skips 1D tensors."""
        from nemo_rl.models.megatron.data import get_and_validate_seqlen

        # Create mock data with 1D tensor (should be skipped)
        data = MagicMock()
        data.__getitem__ = MagicMock(
            side_effect=lambda k: torch.zeros(2, 10) if k == "input_ids" else None
        )
        data.items = MagicMock(
            return_value=[
                ("input_ids", torch.zeros(2, 10)),
                ("seq_lengths", torch.zeros(2)),  # 1D tensor, should be skipped
            ]
        )

        # Should not raise
        sequence_dim, seq_dim_size = get_and_validate_seqlen(data)
        assert seq_dim_size == 10


@pytest.mark.mcore
class TestProcessMicrobatch:
    """Tests for process_microbatch function."""

    @patch("nemo_rl.models.megatron.data.get_ltor_masks_and_position_ids")
    def test_process_microbatch_no_packing(self, mock_get_masks):
        """Test process_microbatch without sequence packing."""
        from nemo_rl.models.megatron.data import process_microbatch

        # Setup mock
        mock_attention_mask = torch.ones(2, 10)
        mock_position_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
        mock_get_masks.return_value = (mock_attention_mask, None, mock_position_ids)

        # Create test data
        data_dict = MagicMock()
        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 0, 0, 0, 0, 0], [6, 7, 8, 9, 10, 11, 12, 0, 0, 0]]
        )
        data_dict.__getitem__ = MagicMock(return_value=input_ids)

        result = process_microbatch(
            data_dict, pack_sequences=False, straggler_timer=MagicMock()
        )

        # Verify results
        assert torch.equal(result.input_ids, input_ids)
        assert torch.equal(result.input_ids_cp_sharded, input_ids)
        assert result.attention_mask is not None
        assert result.position_ids is not None
        assert result.packed_seq_params is None
        assert result.cu_seqlens_padded is None

        # Verify get_ltor_masks_and_position_ids was called
        mock_get_masks.assert_called_once()

    @patch("nemo_rl.models.megatron.data.get_ltor_masks_and_position_ids")
    def test_process_microbatch_repairs_routed_experts_padding_without_packing(
        self, mock_get_masks
    ):
        """Materialized jagged padding must remain valid router replay data."""
        from nemo_rl.models.megatron.data import process_microbatch

        mock_get_masks.return_value = (
            torch.ones(2, 4),
            None,
            torch.arange(4).unsqueeze(0).expand(2, -1),
        )

        routed_experts = torch.tensor(
            [
                [
                    [[4, 5], [6, 7]],
                    [[8, 9], [10, 11]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                ],
                [
                    [[12, 13], [14, 15]],
                    [[16, 17], [18, 19]],
                    [[20, 21], [22, 23]],
                    [[0, 0], [0, 0]],
                ],
            ],
            dtype=torch.int32,
        )
        data_dict = {
            "input_ids": torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]]),
            "input_lengths": torch.tensor([2, 3]),
            "routed_experts": routed_experts,
        }

        result = process_microbatch(
            data_dict,
            pack_sequences=False,
            straggler_timer=MagicMock(),
        )

        expected = routed_experts.clone()
        default_route = torch.tensor([0, 1], dtype=torch.int32)
        expected[0, 2:] = default_route.view(1, 1, 2).expand(2, 2, 2)
        expected[1, 3:] = default_route.view(1, 1, 2).expand(1, 2, 2)
        assert torch.equal(result.routed_experts, expected)
        assert torch.equal(result.routed_experts_cp_sharded, expected)

    @patch("nemo_rl.models.megatron.data.get_ltor_masks_and_position_ids")
    def test_process_microbatch_requires_lengths_for_dense_routed_experts(
        self, mock_get_masks
    ):
        """Dense routed_experts padding repair needs original sequence lengths."""
        from nemo_rl.models.megatron.data import process_microbatch

        data_dict = {
            "input_ids": torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]]),
            "routed_experts": torch.zeros(2, 4, 2, 2, dtype=torch.int32),
        }

        with pytest.raises(ValueError, match="routed_experts requires input_lengths"):
            process_microbatch(
                data_dict,
                pack_sequences=False,
                straggler_timer=MagicMock(),
            )

        mock_get_masks.assert_not_called()

    @patch("nemo_rl.models.megatron.data.get_context_parallel_rank", return_value=0)
    @patch(
        "nemo_rl.models.megatron.data.get_context_parallel_world_size", return_value=1
    )
    @patch("nemo_rl.models.megatron.data._pack_token_aligned_batch_for_megatron")
    def test_process_microbatch_with_packing(
        self, mock_pack, mock_cp_world, mock_cp_rank
    ):
        """Test process_microbatch with sequence packing."""
        from nemo_rl.models.megatron.data import process_microbatch

        # Setup mocks
        mock_packed_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        mock_packed_seq_params = MagicMock()
        mock_cu_seqlens = torch.tensor([0, 5, 8], dtype=torch.int32)
        mock_cu_seqlens_padded = torch.tensor([0, 5, 8], dtype=torch.int32)
        mock_pack.return_value = (
            {"tokens": mock_packed_input_ids},
            {"tokens": mock_packed_input_ids},
            mock_packed_seq_params,
            mock_cu_seqlens,
            mock_cu_seqlens_padded,
        )

        # Create test data
        data_dict = MagicMock()
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0], [6, 7, 8, 0, 0, 0, 0, 0]])
        seq_lengths = torch.tensor([5, 3])
        data_dict.__getitem__ = MagicMock(
            side_effect=lambda k: input_ids if k == "input_ids" else seq_lengths
        )
        data_dict.__contains__ = MagicMock(
            side_effect=lambda k: k in {"input_ids", "input_lengths"}
        )

        result = process_microbatch(
            data_dict,
            seq_length_key="input_lengths",
            pack_sequences=True,
            straggler_timer=MagicMock(),
        )

        # Verify results
        assert torch.equal(result.input_ids, mock_packed_input_ids)
        assert result.packed_seq_params == mock_packed_seq_params
        # For packed sequences, attention_mask and position_ids are None
        assert result.attention_mask is None
        assert result.position_ids is None
        assert result.cu_seqlens_padded is not None

        # Verify pack was called
        mock_pack.assert_called_once()

    @patch("nemo_rl.models.megatron.data.get_context_parallel_rank", return_value=0)
    @patch(
        "nemo_rl.models.megatron.data.get_context_parallel_world_size", return_value=2
    )
    @patch("nemo_rl.models.megatron.data._pack_token_aligned_batch_for_megatron")
    def test_process_microbatch_packs_routed_experts_with_tokens(
        self, mock_pack, mock_cp_world, mock_cp_rank
    ):
        """Test routed_experts follows the same packed CP slicing path as tokens."""
        from nemo_rl.models.megatron.data import process_microbatch

        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        seq_lengths = torch.tensor([3, 2])
        routed_experts = torch.arange(2 * 4 * 3 * 2, dtype=torch.int32).reshape(
            2, 4, 3, 2
        )
        packed_tokens = torch.tensor([[1, 2, 3, 4, 5, 0]])
        packed_routed_experts = torch.arange(1 * 6 * 3 * 2, dtype=torch.int32).reshape(
            1, 6, 3, 2
        )
        cp_tokens = packed_tokens[:, ::2]
        cp_routed_experts = packed_routed_experts[:, ::2]
        mock_pack.return_value = (
            {"tokens": packed_tokens, "routed_experts": packed_routed_experts},
            {"tokens": cp_tokens, "routed_experts": cp_routed_experts},
            MagicMock(),
            torch.tensor([0, 3, 5], dtype=torch.int32),
            torch.tensor([0, 4, 6], dtype=torch.int32),
        )

        data_dict = {
            "input_ids": input_ids,
            "input_lengths": seq_lengths,
            "routed_experts": routed_experts,
        }

        result = process_microbatch(
            data_dict,
            seq_length_key="input_lengths",
            pack_sequences=True,
            straggler_timer=MagicMock(),
        )

        call_args = mock_pack.call_args.args
        assert torch.equal(call_args[0]["tokens"], input_ids)
        assert torch.equal(call_args[0]["routed_experts"], routed_experts)
        assert torch.equal(result.input_ids, packed_tokens)
        assert torch.equal(result.input_ids_cp_sharded, cp_tokens)
        assert torch.equal(result.routed_experts, packed_routed_experts)
        assert torch.equal(result.routed_experts_cp_sharded, cp_routed_experts)

    def test_process_microbatch_packing_requires_seq_length_key(self):
        """Test that packing requires seq_length_key."""
        from nemo_rl.models.megatron.data import process_microbatch

        data_dict = MagicMock()
        input_ids = torch.tensor([[1, 2, 3]])
        data_dict.__getitem__ = MagicMock(return_value=input_ids)

        with pytest.raises(AssertionError) as exc_info:
            process_microbatch(
                data_dict,
                seq_length_key=None,
                pack_sequences=True,
                straggler_timer=MagicMock(),
            )

        assert "seq_length_key must be provided" in str(exc_info.value)

    def test_process_microbatch_packing_requires_seq_length_in_data(self):
        """Test that packing requires seq_length_key to be in data_dict."""
        from nemo_rl.models.megatron.data import process_microbatch

        data_dict = MagicMock()
        input_ids = torch.tensor([[1, 2, 3]])
        data_dict.__getitem__ = MagicMock(return_value=input_ids)
        data_dict.__contains__ = MagicMock(return_value=False)

        with pytest.raises(AssertionError) as exc_info:
            process_microbatch(
                data_dict,
                seq_length_key="input_lengths",
                pack_sequences=True,
                straggler_timer=MagicMock(),
            )

        assert "input_lengths not found in data_dict" in str(exc_info.value)


@pytest.mark.mcore
class TestPackedContextParallelTokenMapping:
    """Tests for packed token preparation before Megatron-owned CP slicing."""

    def test_pack_token_aligned_tensors_for_megatron(self):
        from nemo_rl.models.megatron.data import (
            _pack_token_aligned_tensors_for_megatron,
        )

        token_metadata = torch.arange(2 * 6 * 2, dtype=torch.long).reshape(2, 6, 2)
        seq_lengths = torch.tensor([5, 4])
        cu_seqlens_padded = torch.tensor([0, 8, 16], dtype=torch.int32)

        packed = _pack_token_aligned_tensors_for_megatron(
            {"metadata": token_metadata},
            seq_lengths,
            cu_seqlens_padded,
        )

        expected_padded = torch.zeros(16, 2, dtype=torch.long)
        expected_padded[0:5] = token_metadata[0, :5]
        expected_padded[8:12] = token_metadata[1, :4]

        assert torch.equal(packed["metadata"], expected_padded.unsqueeze(0))

    def test_pack_token_aligned_tensors_uses_valid_routed_experts_padding(self):
        from nemo_rl.models.megatron.data import (
            _pack_token_aligned_tensors_for_megatron,
        )

        routed_experts = torch.tensor(
            [
                [[[4, 5], [6, 7]], [[8, 9], [10, 11]], [[12, 13], [14, 15]]],
                [[[16, 17], [18, 19]], [[20, 21], [22, 23]], [[24, 25], [26, 27]]],
            ],
            dtype=torch.int32,
        )
        seq_lengths = torch.tensor([2, 1])
        cu_seqlens_padded = torch.tensor([0, 4, 8], dtype=torch.int32)

        packed = _pack_token_aligned_tensors_for_megatron(
            {"routed_experts": routed_experts},
            seq_lengths,
            cu_seqlens_padded,
        )["routed_experts"]

        expected_default_route = torch.tensor([0, 1], dtype=torch.int32).view(
            1, 1, 1, 2
        )
        assert torch.equal(packed[:, 0:2], routed_experts[0:1, :2])
        assert torch.equal(packed[:, 4:5], routed_experts[1:2, :1])
        assert torch.equal(packed[:, 2:4], expected_default_route.expand(1, 2, 2, 2))
        assert torch.equal(packed[:, 5:8], expected_default_route.expand(1, 3, 2, 2))

    def test_slice_batch_delegates_dense_to_megatron(self, monkeypatch):
        from nemo_rl.models.megatron import data as megatron_data

        captured = {}

        def fake_get_batch_on_this_cp_rank(batch):
            captured["batch"] = batch
            return {"tokens": batch["tokens"][:, [0, 3]]}

        monkeypatch.setattr(
            megatron_data,
            "get_batch_on_this_cp_rank",
            fake_get_batch_on_this_cp_rank,
        )

        dense_batch = {"tokens": torch.tensor([[10, 11, 12, 13]])}

        sliced_batch, packed_seq_params = (
            megatron_data._slice_batch_for_megatron_context_parallel(dense_batch)
        )

        assert captured["batch"] is not dense_batch
        assert torch.equal(captured["batch"]["tokens"], dense_batch["tokens"])
        assert torch.equal(sliced_batch["tokens"], torch.tensor([[10, 13]]))
        assert packed_seq_params is None

    def test_slice_batch_delegates_packed_thd_to_megatron(self, monkeypatch):
        from nemo_rl.models.megatron import data as megatron_data

        captured = {}
        expected_params = MagicMock()

        def fake_get_thd_batch_on_this_cp_rank(
            batch,
            cu_seqlens,
            cu_seqlens_padded,
            max_seqlen,
            *,
            cp_size,
            cp_rank,
        ):
            captured["batch"] = batch
            captured["cu_seqlens"] = cu_seqlens
            captured["cu_seqlens_padded"] = cu_seqlens_padded
            captured["max_seqlen"] = max_seqlen
            captured["cp_size"] = cp_size
            captured["cp_rank"] = cp_rank
            return {"tokens": batch["tokens"][:, [0, 3]]}, expected_params

        monkeypatch.setattr(
            megatron_data,
            "get_thd_batch_on_this_cp_rank",
            fake_get_thd_batch_on_this_cp_rank,
        )

        packed_batch = {"tokens": torch.tensor([[10, 11, 12, 13]])}
        cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
        cu_seqlens_padded = torch.tensor([0, 4], dtype=torch.int32)

        sliced_batch, packed_seq_params = (
            megatron_data._slice_batch_for_megatron_context_parallel(
                packed_batch,
                cu_seqlens,
                cu_seqlens_padded,
                4,
                cp_rank=1,
                cp_size=2,
            )
        )

        assert captured["batch"] is not packed_batch
        assert torch.equal(captured["batch"]["tokens"], packed_batch["tokens"])
        assert captured["cu_seqlens"] is cu_seqlens
        assert captured["cu_seqlens_padded"] is cu_seqlens_padded
        assert torch.equal(captured["max_seqlen"], torch.tensor([4], dtype=torch.int32))
        assert captured["cp_rank"] == 1
        assert captured["cp_size"] == 2
        assert torch.equal(sliced_batch["tokens"], torch.tensor([[10, 13]]))
        assert packed_seq_params is expected_params

    def test_pack_token_aligned_batch_slices_all_fields(self, monkeypatch):
        from nemo_rl.models.megatron import data as megatron_data

        expected_params = MagicMock()

        def fake_slice_batch(batch, *args, **kwargs):
            return {
                key: value[:, ::2].contiguous() for key, value in batch.items()
            }, expected_params

        monkeypatch.setattr(
            megatron_data,
            "_slice_batch_for_megatron_context_parallel",
            fake_slice_batch,
        )

        tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        routed_experts = torch.arange(2 * 4 * 2 * 1, dtype=torch.int32).reshape(
            2, 4, 2, 1
        )
        seq_lengths = torch.tensor([3, 2])

        packed, cp_packed, packed_seq_params, cu_seqlens, cu_seqlens_padded = (
            megatron_data._pack_token_aligned_batch_for_megatron(
                {"tokens": tokens, "routed_experts": routed_experts},
                seq_lengths,
                cp_rank=1,
                cp_size=2,
            )
        )

        assert packed_seq_params is expected_params
        assert torch.equal(cu_seqlens, torch.tensor([0, 3, 5], dtype=torch.int32))
        assert torch.equal(cu_seqlens_padded, cu_seqlens)
        assert torch.equal(packed["tokens"], torch.tensor([[1, 2, 3, 4, 5]]))
        assert torch.equal(cp_packed["tokens"], torch.tensor([[1, 3, 5]]))
        assert torch.equal(
            cp_packed["routed_experts"], packed["routed_experts"][:, ::2]
        )

    def test_r3_trace_token_identity_verifies_cp_alignment(self, monkeypatch):
        from nemo_rl.models.megatron import data as megatron_data

        monkeypatch.setenv("NRL_R3_TRACE", "1")
        monkeypatch.setenv("NRL_R3_TRACE_VERIFY_FORWARD", "1")

        input_ids = torch.tensor([[10, 11, 12, 0], [20, 21, 0, 0]])
        input_lengths = torch.tensor([3, 2])
        routed_experts = torch.arange(2 * 4 * 2 * 1, dtype=torch.int32).reshape(
            2, 4, 2, 1
        )
        identity = megatron_data._make_r3_trace_token_identity(
            input_ids,
            input_lengths,
        )
        cp_identity = identity[:, [0, 2, 3]]

        source_rows = cp_identity.reshape(-1, 3)[:, 0].long()
        source_cols = cp_identity.reshape(-1, 3)[:, 1].long()
        valid = cp_identity.reshape(-1, 3)[:, 2].bool()
        cp_tokens = input_ids[source_rows, source_cols].reshape(1, -1)
        cp_routed = routed_experts[source_rows, source_cols].reshape(1, -1, 2, 1)

        assert megatron_data._verify_r3_trace_cp_token_alignment(
            source_input_ids=input_ids,
            source_routed_experts=routed_experts,
            input_ids_cp_sharded=cp_tokens,
            routed_experts_cp_sharded=cp_routed,
            token_identity_cp_sharded=cp_identity,
        ) == int(valid.sum().item())

        broken_routed = cp_routed.clone()
        broken_routed[:, 0] += 1
        with pytest.raises(RuntimeError, match="routed_experts"):
            megatron_data._verify_r3_trace_cp_token_alignment(
                source_input_ids=input_ids,
                source_routed_experts=routed_experts,
                input_ids_cp_sharded=cp_tokens,
                routed_experts_cp_sharded=broken_routed,
                token_identity_cp_sharded=cp_identity,
            )


@pytest.mark.mcore
class TestProcessGlobalBatch:
    """Tests for process_global_batch function."""

    def test_process_global_batch_basic(self):
        """Test basic process_global_batch functionality."""
        from nemo_rl.models.megatron.data import process_global_batch

        # Create mock data
        sample_mask = torch.tensor([1.0, 1.0, 0.0])
        input_ids = torch.zeros(3, 10)
        mock_batch = BatchedDataDict(
            {
                "sample_mask": sample_mask,
                "input_ids": input_ids,
            }
        )

        mock_data = MagicMock()
        mock_data.get_batch.return_value = mock_batch

        mock_dp_group = MagicMock()

        # Mock torch.distributed.all_reduce
        with patch("torch.distributed.all_reduce") as mock_all_reduce:
            result = process_global_batch(
                data=mock_data,
                loss_fn=MagicMock(),
                dp_group=mock_dp_group,
                batch_idx=0,
                batch_size=3,
            )

            batch = result["batch"]
            assert torch.equal(batch["sample_mask"], mock_batch["sample_mask"])
            assert torch.equal(batch["input_ids"], mock_batch["input_ids"])

            # Verify get_batch was called
            mock_data.get_batch.assert_called_once_with(batch_idx=0, batch_size=3)

            # Verify all_reduce was called
            mock_all_reduce.assert_called_once()

    def test_process_global_batch_requires_sample_mask_in_data(self):
        """Test that process_global_batch requires sample_mask."""
        from nemo_rl.models.megatron.data import process_global_batch

        # Create mock data without sample_mask
        mock_batch = MagicMock()
        mock_batch.__contains__ = MagicMock(return_value=False)

        mock_data = MagicMock()
        mock_data.get_batch.return_value = mock_batch

        with pytest.raises(AssertionError) as exc_info:
            process_global_batch(
                data=mock_data,
                loss_fn=MagicMock(),
                dp_group=MagicMock(),
                batch_idx=0,
                batch_size=3,
            )

        assert "sample_mask must be present in the data!" in str(exc_info.value)


@pytest.mark.mcore
class TestGetMicrobatchIterator:
    """Tests for get_microbatch_iterator function."""

    @patch("nemo_rl.models.megatron.data.get_and_validate_seqlen")
    @patch("nemo_rl.models.megatron.data.make_processed_microbatch_iterator")
    def test_get_microbatch_iterator_dynamic_batching(
        self, mock_make_iterator, mock_get_and_validate_seqlen
    ):
        """Test get_microbatch_iterator with dynamic batching."""
        from nemo_rl.models.megatron.data import get_microbatch_iterator

        # Setup mocks
        mock_get_and_validate_seqlen.return_value = (1, 128)
        mock_iterator = iter([MagicMock()])
        mock_make_iterator.return_value = mock_iterator

        mock_data = MagicMock()
        mock_data.make_microbatch_iterator_with_dynamic_shapes.return_value = iter([])
        mock_data.get_microbatch_iterator_dynamic_shapes_len.return_value = 5

        cfg = {
            "dynamic_batching": {"enabled": True},
            "sequence_packing": {"enabled": False},
        }

        (
            iterator,
            data_iterator_len,
            micro_batch_size,
            seq_dim_size,
            padded_seq_length,
        ) = get_microbatch_iterator(
            data=mock_data,
            cfg=cfg,
            mbs=4,
            straggler_timer=MagicMock(),
        )

        # Verify dynamic batching path was taken
        mock_data.make_microbatch_iterator_with_dynamic_shapes.assert_called_once()
        mock_data.get_microbatch_iterator_dynamic_shapes_len.assert_called_once()

        assert data_iterator_len == 5
        assert seq_dim_size == 128

    @patch("nemo_rl.models.megatron.data.get_and_validate_seqlen")
    @patch("nemo_rl.models.megatron.data.make_processed_microbatch_iterator")
    @patch("nemo_rl.models.megatron.data._get_pack_sequence_parameters_for_megatron")
    def test_get_microbatch_iterator_sequence_packing(
        self, mock_get_params, mock_make_iterator, mock_get_and_validate_seqlen
    ):
        """Test get_microbatch_iterator with sequence packing."""
        from nemo_rl.models.megatron.data import get_microbatch_iterator

        # Setup mocks
        mock_get_and_validate_seqlen.return_value = (1, 256)
        mock_get_params.return_value = (8, 16, None)
        mock_iterator = iter([MagicMock()])
        mock_make_iterator.return_value = mock_iterator

        mock_data = MagicMock()
        mock_data.make_microbatch_iterator_for_packable_sequences.return_value = iter(
            []
        )
        mock_data.get_microbatch_iterator_for_packable_sequences_len.return_value = (
            10,
            512,
        )

        cfg = {
            "dynamic_batching": {"enabled": False},
            "sequence_packing": {"enabled": True},
            "megatron_cfg": {
                "tensor_model_parallel_size": 1,
                "sequence_parallel": False,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
            },
            "make_sequence_length_divisible_by": 1,
        }

        (
            iterator,
            data_iterator_len,
            micro_batch_size,
            seq_dim_size,
            padded_seq_length,
        ) = get_microbatch_iterator(
            data=mock_data,
            cfg=cfg,
            mbs=4,
            straggler_timer=MagicMock(),
        )

        # Verify sequence packing path was taken
        mock_data.make_microbatch_iterator_for_packable_sequences.assert_called_once()

        # With sequence packing, micro_batch_size should be 1
        assert micro_batch_size == 1
        assert data_iterator_len == 10

    @patch("nemo_rl.models.megatron.data.get_and_validate_seqlen")
    @patch("nemo_rl.models.megatron.data.make_processed_microbatch_iterator")
    def test_get_microbatch_iterator_regular(
        self, mock_make_iterator, mock_get_and_validate_seqlen
    ):
        """Test get_microbatch_iterator with regular batching."""
        from nemo_rl.models.megatron.data import get_microbatch_iterator

        # Setup mocks
        mock_get_and_validate_seqlen.return_value = (1, 64)
        mock_iterator = iter([MagicMock()])
        mock_make_iterator.return_value = mock_iterator

        mock_data = MagicMock()
        mock_data.size = 16
        mock_data.make_microbatch_iterator.return_value = iter([])

        cfg = {
            "dynamic_batching": {"enabled": False},
            "sequence_packing": {"enabled": False},
        }

        mbs = 4

        (
            iterator,
            data_iterator_len,
            micro_batch_size,
            seq_dim_size,
            padded_seq_length,
        ) = get_microbatch_iterator(
            data=mock_data,
            cfg=cfg,
            mbs=mbs,
            straggler_timer=MagicMock(),
        )

        # Verify regular batching path was taken
        mock_data.make_microbatch_iterator.assert_called_once_with(mbs)

        assert micro_batch_size == mbs
        assert data_iterator_len == 16 // mbs
        assert seq_dim_size == 64

    @patch("nemo_rl.models.megatron.data.get_and_validate_seqlen")
    @patch("nemo_rl.models.megatron.data.make_processed_microbatch_iterator")
    def test_get_microbatch_iterator_auto_detects_seq_length_key(
        self, mock_make_iterator, mock_get_and_validate_seqlen
    ):
        """Test that get_microbatch_iterator auto-detects seq_length_key for packing."""
        from nemo_rl.models.megatron.data import get_microbatch_iterator

        # Setup mocks
        mock_get_and_validate_seqlen.return_value = (1, 128)
        mock_iterator = iter([MagicMock()])
        mock_make_iterator.return_value = mock_iterator

        mock_data = MagicMock()
        mock_data.make_microbatch_iterator_for_packable_sequences.return_value = iter(
            []
        )
        mock_data.get_microbatch_iterator_for_packable_sequences_len.return_value = (
            5,
            256,
        )

        cfg = {
            "dynamic_batching": {"enabled": False},
            "sequence_packing": {"enabled": True},
            "megatron_cfg": {
                "tensor_model_parallel_size": 1,
                "sequence_parallel": False,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
            },
            "make_sequence_length_divisible_by": 1,
        }

        get_microbatch_iterator(
            data=mock_data,
            cfg=cfg,
            mbs=4,
            straggler_timer=MagicMock(),
            seq_length_key=None,  # Should be auto-detected
        )

        # Verify make_processed_microbatch_iterator was called with "input_lengths"
        call_kwargs = mock_make_iterator.call_args[1]
        assert call_kwargs["seq_length_key"] == "input_lengths"


@pytest.mark.mcore
class TestMakeProcessedMicrobatchIterator:
    """Tests for make_processed_microbatch_iterator function."""

    @patch("nemo_rl.models.megatron.data.process_microbatch")
    def test_make_processed_microbatch_iterator_basic(self, mock_process):
        """Test make_processed_microbatch_iterator yields ProcessedMicrobatch."""
        from nemo_rl.models.megatron.data import (
            ProcessedInputs,
            ProcessedMicrobatch,
            make_processed_microbatch_iterator,
        )

        # Setup mocks
        mock_input_ids = MagicMock()
        mock_input_ids_cp_sharded = MagicMock()
        mock_attention_mask = MagicMock()
        mock_position_ids = MagicMock()
        mock_packed_seq_params = None
        mock_cu_seqlens_padded = None

        mock_process.return_value = ProcessedInputs(
            input_ids=mock_input_ids,
            input_ids_cp_sharded=mock_input_ids_cp_sharded,
            attention_mask=mock_attention_mask,
            position_ids=mock_position_ids,
            packed_seq_params=mock_packed_seq_params,
            cu_seqlens_padded=mock_cu_seqlens_padded,
        )

        # Create mock data dict
        mock_data_dict = MagicMock()
        mock_data_dict.to.return_value = mock_data_dict

        raw_iterator = iter([mock_data_dict])

        cfg = {"sequence_packing": {"enabled": False}}

        processed_iterator = make_processed_microbatch_iterator(
            raw_iterator=raw_iterator,
            cfg=cfg,
            seq_length_key=None,
            pad_individual_seqs_to_multiple_of=1,
            pad_packed_seq_to_multiple_of=1,
            straggler_timer=MagicMock(),
            pad_full_seq_to=None,
        )

        # Get first item from iterator
        microbatch = next(processed_iterator)

        # Verify it's a ProcessedMicrobatch
        assert isinstance(microbatch, ProcessedMicrobatch)
        assert microbatch.data_dict == mock_data_dict
        assert microbatch.input_ids == mock_input_ids

        # Verify data was moved to CUDA
        mock_data_dict.to.assert_called_once_with("cuda")

    @patch("nemo_rl.models.megatron.data.process_microbatch")
    def test_make_processed_microbatch_iterator_with_packing(self, mock_process):
        """Test make_processed_microbatch_iterator with sequence packing."""
        from nemo_rl.models.megatron.data import (
            ProcessedInputs,
            make_processed_microbatch_iterator,
        )

        # Setup mocks
        mock_process.return_value = ProcessedInputs(
            input_ids=MagicMock(),
            input_ids_cp_sharded=MagicMock(),
            attention_mask=None,  # None for packed
            position_ids=None,  # None for packed
            packed_seq_params=MagicMock(),
            cu_seqlens_padded=MagicMock(),
        )

        mock_data_dict = MagicMock()
        mock_data_dict.to.return_value = mock_data_dict

        raw_iterator = iter([mock_data_dict])

        cfg = {"sequence_packing": {"enabled": True}}

        processed_iterator = make_processed_microbatch_iterator(
            raw_iterator=raw_iterator,
            cfg=cfg,
            seq_length_key="input_lengths",
            pad_individual_seqs_to_multiple_of=8,
            pad_packed_seq_to_multiple_of=16,
            straggler_timer=MagicMock(),
            pad_full_seq_to=1024,
        )

        microbatch = next(processed_iterator)

        # Verify process_microbatch was called with pack_sequences=True
        mock_process.assert_called_once()
        call_kwargs = mock_process.call_args[1]
        assert call_kwargs["pack_sequences"] is True
        assert call_kwargs["seq_length_key"] == "input_lengths"
        assert call_kwargs["pad_individual_seqs_to_multiple_of"] == 8
        assert call_kwargs["pad_packed_seq_to_multiple_of"] == 16
        assert call_kwargs["pad_full_seq_to"] == 1024


PACK_SEQUENCES_TEST_ACTOR_FQN = (
    f"{PackSequencesTestActor.__module__}.PackSequencesTestActor"
)


@pytest.fixture
def register_pack_sequences_test_actor():
    """Register the PackSequencesTestActor for use in tests."""
    original_registry_value = ACTOR_ENVIRONMENT_REGISTRY.get(
        PACK_SEQUENCES_TEST_ACTOR_FQN
    )
    ACTOR_ENVIRONMENT_REGISTRY[PACK_SEQUENCES_TEST_ACTOR_FQN] = PY_EXECUTABLES.MCORE

    yield PACK_SEQUENCES_TEST_ACTOR_FQN

    # Clean up registry
    if PACK_SEQUENCES_TEST_ACTOR_FQN in ACTOR_ENVIRONMENT_REGISTRY:
        if original_registry_value is None:
            del ACTOR_ENVIRONMENT_REGISTRY[PACK_SEQUENCES_TEST_ACTOR_FQN]
        else:
            ACTOR_ENVIRONMENT_REGISTRY[PACK_SEQUENCES_TEST_ACTOR_FQN] = (
                original_registry_value
            )


@pytest.fixture
def pack_sequences_setup(request):
    """Setup and teardown for pack sequences tests - creates a virtual cluster and reusable actor."""
    # Get parameters from request
    if hasattr(request, "param") and request.param is not None:
        cp_size = request.param
    else:
        cp_size = 1

    cluster = None
    worker_group = None

    try:
        # Skip if not enough GPUs
        if not torch.cuda.is_available() or torch.cuda.device_count() < cp_size:
            pytest.skip(
                f"Not enough GPUs available. Need {cp_size}, got {torch.cuda.device_count()}"
            )

        cluster_name = f"test-pack-sequences-cp{cp_size}"
        print(f"Creating virtual cluster '{cluster_name}' for {cp_size} GPUs...")

        cluster = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[cp_size],
            use_gpus=True,
            max_colocated_worker_groups=1,
        )

        actor_fqn = PACK_SEQUENCES_TEST_ACTOR_FQN

        # Register the actor
        original_registry_value = ACTOR_ENVIRONMENT_REGISTRY.get(actor_fqn)
        ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] = PY_EXECUTABLES.MCORE

        try:
            # For CP tests
            sharding = NamedSharding(layout=list(range(cp_size)), names=["cp"])
            builder = RayWorkerBuilder(actor_fqn, cp_size)

            worker_group = RayWorkerGroup(
                cluster=cluster,
                remote_worker_builder=builder,
                workers_per_node=None,
                sharding_annotations=sharding,
            )

            yield worker_group

        finally:
            # Clean up registry
            if actor_fqn in ACTOR_ENVIRONMENT_REGISTRY:
                if original_registry_value is None:
                    del ACTOR_ENVIRONMENT_REGISTRY[actor_fqn]
                else:
                    ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] = original_registry_value

    finally:
        print("Cleaning up pack sequences test resources...")
        if worker_group:
            worker_group.shutdown(force=True)
        if cluster:
            cluster.shutdown()


@pytest.mark.mcore
@pytest.mark.parametrize("pack_sequences_setup", [1], indirect=True, ids=["cp1"])
def test_pack_sequences_comprehensive(pack_sequences_setup):
    """Comprehensive test of pack sequences functionality without context parallelism."""
    worker_group = pack_sequences_setup

    # Run all tests in a single call to the actor
    futures = worker_group.run_all_workers_single_data("run_all_pack_sequences_tests")
    results = ray.get(futures)

    # Check that all workers succeeded
    for i, result in enumerate(results):
        assert result["success"], f"Worker {i} failed: {result['error']}"

        # Print detailed results for debugging
        if "detailed_results" in result:
            detailed = result["detailed_results"]
            print(f"Worker {i} detailed results:")
            for test_name, test_result in detailed.items():
                status = "PASSED" if test_result["success"] else "FAILED"
                print(f"  {test_name}: {status}")
                if not test_result["success"]:
                    print(f"    Error: {test_result['error']}")


@pytest.mark.mcore
@pytest.mark.parametrize("pack_sequences_setup", [2], indirect=True, ids=["cp2"])
def test_pack_sequences_with_context_parallel(pack_sequences_setup):
    """Test pack sequences functionality with context parallelism."""
    worker_group = pack_sequences_setup

    # Run all tests including CP tests
    futures = worker_group.run_all_workers_single_data("run_all_pack_sequences_tests")
    results = ray.get(futures)

    # Check that all workers succeeded
    for i, result in enumerate(results):
        assert result["success"], f"Worker {i} failed: {result['error']}"

        # Print detailed results for debugging
        if "detailed_results" in result:
            detailed = result["detailed_results"]
            print(f"Worker {i} detailed results:")
            for test_name, test_result in detailed.items():
                if "skipped" in test_result:
                    print(f"  {test_name}: SKIPPED ({test_result['skipped']})")
                else:
                    status = "PASSED" if test_result["success"] else "FAILED"
                    print(f"  {test_name}: {status}")
                    if not test_result["success"]:
                        print(f"    Error: {test_result['error']}")


GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN = f"{GetPackSequenceParametersTestActor.__module__}.GetPackSequenceParametersTestActor"


@pytest.fixture
def register_get_pack_sequence_parameters_test_actor():
    """Register the GetPackSequenceParametersTestActor for use in tests."""
    original_registry_value = ACTOR_ENVIRONMENT_REGISTRY.get(
        GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN
    )
    ACTOR_ENVIRONMENT_REGISTRY[GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN] = (
        PY_EXECUTABLES.MCORE
    )

    yield GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN

    # Clean up registry
    if GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN in ACTOR_ENVIRONMENT_REGISTRY:
        if original_registry_value is None:
            del ACTOR_ENVIRONMENT_REGISTRY[GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN]
        else:
            ACTOR_ENVIRONMENT_REGISTRY[GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN] = (
                original_registry_value
            )


@pytest.fixture
def get_pack_sequence_parameters_setup(request):
    """Setup and teardown for get pack sequence parameters tests - creates a virtual cluster and reusable actor."""
    cluster = None
    worker_group = None

    try:
        # Skip if not enough GPUs
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            pytest.skip(
                f"Not enough GPUs available. Need 1, got {torch.cuda.device_count()}"
            )

        cluster_name = "test-get-pack-sequence-parameters"
        print(f"Creating virtual cluster '{cluster_name}'...")

        cluster = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[1],
            use_gpus=True,
            max_colocated_worker_groups=1,
        )

        actor_fqn = GET_PACK_SEQUENCE_PARAMETERS_TEST_ACTOR_FQN

        # Register the actor
        original_registry_value = ACTOR_ENVIRONMENT_REGISTRY.get(actor_fqn)
        ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] = PY_EXECUTABLES.MCORE

        try:
            # For CP tests
            sharding = NamedSharding(layout=list(range(1)), names=["cp"])
            builder = RayWorkerBuilder(actor_fqn)

            worker_group = RayWorkerGroup(
                cluster=cluster,
                remote_worker_builder=builder,
                workers_per_node=None,
                sharding_annotations=sharding,
            )

            yield worker_group

        finally:
            # Clean up registry
            if actor_fqn in ACTOR_ENVIRONMENT_REGISTRY:
                if original_registry_value is None:
                    del ACTOR_ENVIRONMENT_REGISTRY[actor_fqn]
                else:
                    ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] = original_registry_value
    finally:
        print("Cleaning up get pack sequence parameters test resources...")
        if worker_group:
            worker_group.shutdown(force=True)
        if cluster:
            cluster.shutdown()


@pytest.mark.mcore
@pytest.mark.parametrize(
    "get_pack_sequence_parameters_setup", [1], indirect=True, ids=["cp1"]
)
def test_get_pack_sequence_parameters_for_megatron(get_pack_sequence_parameters_setup):
    """Comprehensive test of pack sequences functionality without context parallelism."""
    worker_group = get_pack_sequence_parameters_setup

    for test_name in [
        "run_all_get_pack_sequence_parameters_for_megatron_tests",
        "run_all_get_pack_sequence_parameters_for_megatron_fp8_tests",
    ]:
        # Run all tests in a single call to the actor
        futures = worker_group.run_all_workers_single_data(test_name)
        results = ray.get(futures)

        # Check that all workers succeeded
        for i, result in enumerate(results):
            assert result["success"], f"Worker {i} failed: {result['error']}"
