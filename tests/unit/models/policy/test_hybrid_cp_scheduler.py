"""Unit tests for Hybrid Context Parallelism scheduler."""

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.hybrid_cp_config import HybridCPConfig
from nemo_rl.models.policy.hybrid_cp_scheduler import HeadNodeHCPScheduler


class TestHybridCPConfig:
    """Test HybridCPConfig validation."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = HybridCPConfig(enabled=True)
        assert config.enabled is True
        assert config.max_seqlen_per_dp_cp_rank is None
        assert config.scheduling_strategy == "dp"
        assert config.balance_slack == 0.05
        assert config.eps_bucket == 0.10

    def test_config_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError, match="scheduling_strategy must be"):
            HybridCPConfig(enabled=True, scheduling_strategy="invalid")

    def test_config_pp_not_supported(self):
        """Test that PP strategy raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Pipeline parallel"):
            HybridCPConfig(enabled=True, scheduling_strategy="pp")

    def test_config_invalid_slack(self):
        """Test that invalid balance_slack raises error."""
        with pytest.raises(ValueError, match="balance_slack must be between"):
            HybridCPConfig(enabled=True, balance_slack=1.5)

    def test_config_invalid_eps(self):
        """Test that invalid eps_bucket raises error."""
        with pytest.raises(ValueError, match="eps_bucket must be between"):
            HybridCPConfig(enabled=True, eps_bucket=-0.1)


class TestHeadNodeHCPScheduler:
    """Test HeadNodeHCPScheduler functionality."""

    @pytest.fixture
    def simple_config(self):
        """Create a simple HCP config for testing."""
        return HybridCPConfig(
            enabled=True,
            max_seqlen_per_dp_cp_rank=10000,
            scheduling_strategy="dp",
        )

    @pytest.fixture
    def scheduler(self, simple_config):
        """Create a scheduler instance for testing."""
        return HeadNodeHCPScheduler(
            hcp_config=simple_config,
            dp_size=2,
            cp_size=4,
            max_seq_len=40960,
        )

    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization."""
        assert scheduler.dp_size == 2
        assert scheduler.cp_size == 4
        assert scheduler.hdp_size == 8
        assert scheduler.max_seqlen_per_dp_cp_rank == 10000
        assert scheduler.scheduler is not None

    def test_scheduler_auto_max_seqlen(self):
        """Test automatic calculation of max_seqlen_per_dp_cp_rank."""
        config = HybridCPConfig(enabled=True)
        scheduler = HeadNodeHCPScheduler(
            hcp_config=config,
            dp_size=2,
            cp_size=4,
            max_seq_len=40960,
        )
        # Should be 40960 / 4 = 10240
        assert scheduler.max_seqlen_per_dp_cp_rank == 10240

    def test_extract_sequence_lengths_from_tensor(self, scheduler):
        """Test extracting sequence lengths from tensor."""
        data = BatchedDataDict()
        data["input_lengths"] = torch.tensor([100, 200, 300, 400])
        data["input_ids"] = torch.zeros((4, 500), dtype=torch.long)

        seq_lengths = scheduler.extract_sequence_lengths(data)
        assert seq_lengths == [100, 200, 300, 400]

    def test_extract_sequence_lengths_from_list(self, scheduler):
        """Test extracting sequence lengths from list."""
        data = BatchedDataDict()
        data["input_lengths"] = [100, 200, 300, 400]
        data["input_ids"] = torch.zeros((4, 500), dtype=torch.long)

        seq_lengths = scheduler.extract_sequence_lengths(data)
        assert seq_lengths == [100, 200, 300, 400]

    def test_extract_sequence_lengths_missing_key(self, scheduler):
        """Test error when sequence length key is missing."""
        data = BatchedDataDict()
        data["input_ids"] = torch.zeros((4, 500), dtype=torch.long)

        with pytest.raises(ValueError, match="not found in data"):
            scheduler.extract_sequence_lengths(data)

    def test_schedule_samples_basic(self, scheduler):
        """Test basic sample scheduling."""
        # Create samples with different lengths
        # max_seqlen_per_dp_cp_rank = 10000
        # So samples needing cp1 (<=10000), cp2 (10001-20000), cp4 (20001-40000)
        seq_lengths = [
            5000,   # cp1
            12000,  # cp2
            8000,   # cp1
            25000,  # cp4
            15000,  # cp2
            30000,  # cp4
        ]

        groups, sample_id_groups = scheduler.schedule_samples(seq_lengths)

        # Verify structure
        assert isinstance(groups, list)
        assert isinstance(sample_id_groups, list)
        assert len(groups) == len(sample_id_groups)

        # Verify all samples are assigned
        all_assigned_samples = set()
        for group in sample_id_groups:
            for rank_samples in group:
                all_assigned_samples.update(rank_samples)
        assert all_assigned_samples == set(range(6))

    def test_create_hcp_metadata(self, scheduler):
        """Test HCP metadata creation."""
        # Manually create a simple sample_id_groups structure
        # Group 0: sample 0 goes to rank 0 (cp1), sample 1 goes to rank 1,2 (cp2)
        sample_id_groups = [
            [
                [0],      # rank 0
                [1],      # rank 1
                [1],      # rank 2 (sample 1 shared across rank 1,2)
                [],       # rank 3
                [],       # rank 4
                [],       # rank 5
                [],       # rank 6
                [],       # rank 7
            ]
        ]

        sample_to_hdp_rank, sample_local_cp_size, sample_cp_group_id = (
            scheduler.create_hcp_metadata(sample_id_groups, num_samples=2)
        )

        # Sample 0 should be on rank 0 with cp1
        assert sample_to_hdp_rank[0] == 0
        assert sample_local_cp_size[0] == 1
        assert sample_cp_group_id[0] == 0

        # Sample 1 should be on rank 1 with cp2 (shared across 2 ranks)
        assert sample_to_hdp_rank[1] == 1
        assert sample_local_cp_size[1] == 2
        assert sample_cp_group_id[1] == 0

    def test_create_hcp_metadata_unassigned_sample(self, scheduler):
        """Test error when sample is not assigned."""
        # Sample 0 is in the group, but sample 1 is missing
        sample_id_groups = [
            [
                [0],      # rank 0
                [],       # rank 1
                [],       # rank 2
                [],       # rank 3
                [],       # rank 4
                [],       # rank 5
                [],       # rank 6
                [],       # rank 7
            ]
        ]

        with pytest.raises(RuntimeError, match="samples were not assigned"):
            scheduler.create_hcp_metadata(sample_id_groups, num_samples=2)

    def test_shard_data_by_hdp_rank(self, scheduler):
        """Test data sharding with HCP metadata."""
        # Create test data
        data = BatchedDataDict()
        data["input_ids"] = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        data["input_lengths"] = torch.tensor([2, 2, 2, 2])

        # Assign samples to ranks: 0->rank0, 1->rank1, 2->rank1, 3->rank2
        sample_to_hdp_rank = [0, 1, 1, 2, -1, -1, -1, -1][:4]
        sample_local_cp_size = [1, 2, 2, 1]
        sample_cp_group_id = [0, 0, 0, 1]

        sharded_data = scheduler.shard_data_by_hdp_rank(
            data, sample_to_hdp_rank, sample_local_cp_size, sample_cp_group_id
        )

        # Should have 8 shards (hdp_size = 8)
        assert len(sharded_data) == 8

        # Rank 0 should have sample 0
        assert sharded_data[0].size == 1
        assert torch.equal(sharded_data[0]["input_ids"], torch.tensor([[1, 2]]))
        assert sharded_data[0]["local_cp_size_per_sample"] == [1]
        assert sharded_data[0]["cp_group_id_per_sample"] == [0]

        # Rank 1 should have samples 1 and 2
        assert sharded_data[1].size == 2
        assert torch.equal(sharded_data[1]["input_ids"], torch.tensor([[3, 4], [5, 6]]))
        assert sharded_data[1]["local_cp_size_per_sample"] == [2, 2]
        assert sharded_data[1]["cp_group_id_per_sample"] == [0, 0]

        # Rank 2 should have sample 3
        assert sharded_data[2].size == 1
        assert torch.equal(sharded_data[2]["input_ids"], torch.tensor([[7, 8]]))
        assert sharded_data[2]["local_cp_size_per_sample"] == [1]
        assert sharded_data[2]["cp_group_id_per_sample"] == [1]

        # Other ranks should be empty
        for rank in range(3, 8):
            assert sharded_data[rank].size == 0

    def test_schedule_and_shard_end_to_end(self, scheduler):
        """Test complete end-to-end scheduling and sharding."""
        # Create test data with varying sequence lengths
        data = BatchedDataDict()
        data["input_ids"] = torch.randint(0, 1000, (8, 500), dtype=torch.long)
        data["input_lengths"] = torch.tensor([5000, 12000, 8000, 25000, 15000, 30000, 6000, 18000])

        sharded_data = scheduler.schedule_and_shard(data, seq_length_key="input_lengths")

        # Should have 8 shards (hdp_size = 8)
        assert len(sharded_data) == 8

        # Verify all samples are accounted for
        total_samples = sum(shard.size for shard in sharded_data)
        assert total_samples == 8

        # Verify all shards have HCP metadata
        for rank, shard in enumerate(sharded_data):
            if shard.size > 0:
                assert "local_cp_size_per_sample" in shard
                assert "cp_group_id_per_sample" in shard
                assert len(shard["local_cp_size_per_sample"]) == shard.size
                assert len(shard["cp_group_id_per_sample"]) == shard.size

                # Verify CP sizes are powers of 2
                for cp_size in shard["local_cp_size_per_sample"]:
                    assert cp_size > 0
                    # Check if power of 2
                    assert (cp_size & (cp_size - 1)) == 0, f"CP size {cp_size} is not a power of 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
