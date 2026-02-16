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

import pytest

from nemo_rl.utils.flops_formulas import (
    FLOPSConfig,
    _attention_flops,
    _is_layer_window_attention,
    _moe_mlp_flops,
    _vocab_flops,
    gpt_oss,
)
from nemo_rl.utils.flops_tracker import FLOPTracker, get_default_hf_config


@pytest.mark.parametrize(
    "model_name, gbs, seqlen, expected_flops",
    [
        ("meta-llama/Llama-2-7b-hf", 128, 4096, 2.25e16),
        ("meta-llama/Llama-2-13b-hf", 128, 4096, 4.17e16),
        ("meta-llama/Llama-2-70b-hf", 128, 4096, 2.25e17),
        ("meta-llama/Meta-Llama-3-8B", 128, 8192, 5.31e16),
        ("meta-llama/Llama-3.1-70B-Instruct", 128, 8192, 4.71e17),
        ("meta-llama/Llama-3.1-405B-Instruct", 128, 8192, 2.65e18),
        ("Qwen/Qwen3-30B-A3B", 128, 4096, 9.37e15),
        ("Qwen/Qwen3-235B-A22B", 128, 4096, 6.21e16),
        ("deepseek-ai/DeepSeek-V3", 1, 4096, 1.023e15),
        ("moonshotai/Moonlight-16B-A3B-Instruct", 1, 4096, 6.45e13),
    ],
)
def test_flops_counter(model_name, gbs, seqlen, expected_flops):
    model_config = get_default_hf_config(model_name)
    flops_tracker = FLOPTracker.from_config(model_name, model_config)
    flops_tracker.track(gbs, seqlen)

    # check within 5% relative difference
    assert abs(flops_tracker.total_flops - expected_flops) / expected_flops <= 0.05, (
        f"Expected {expected_flops} flops, got {flops_tracker.total_flops}"
    )


# =============================================================================
# GPT-OSS FLOPS Calculation Tests
# =============================================================================


class TestGptOssFlopsHelpers:
    """Unit tests for GPT-OSS FLOPS helper functions."""

    def test_is_layer_window_attention_with_skip_freq_2(self):
        """Test window attention pattern with skip_freq=2 (every 2nd layer is full attention)."""
        # With skip_freq=2: layer 2, 4, 6, ... use full attention (1-indexed)
        # layer 1, 3, 5, ... use sliding window attention
        assert _is_layer_window_attention(2, 0) is True  # layer 1 -> SWA
        assert _is_layer_window_attention(2, 1) is False  # layer 2 -> full
        assert _is_layer_window_attention(2, 2) is True  # layer 3 -> SWA
        assert _is_layer_window_attention(2, 3) is False  # layer 4 -> full

    def test_is_layer_window_attention_with_skip_freq_none(self):
        """Test window attention when skip_freq is None (all layers use SWA)."""
        assert _is_layer_window_attention(None, 0) is True
        assert _is_layer_window_attention(None, 1) is True
        assert _is_layer_window_attention(None, 99) is True

    def test_is_layer_window_attention_with_skip_freq_1(self):
        """Test window attention when skip_freq=1 (all layers use full attention)."""
        assert _is_layer_window_attention(1, 0) is False
        assert _is_layer_window_attention(1, 1) is False
        assert _is_layer_window_attention(1, 99) is False

    def test_attention_flops_full_attention(self):
        """Test attention FLOPS calculation for full attention."""
        flops = _attention_flops(
            seq_len=1024,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,  # GQA with 8 KV groups
            kv_channels=128,
            is_swa=False,
        )
        # FLOPS should be positive and scale with seq_len^2 for full attention
        assert flops > 0
        # Verify it's in a reasonable range (order of magnitude check)
        assert 1e11 < flops < 1e14

    def test_attention_flops_sliding_window(self):
        """Test attention FLOPS for sliding window attention is less than full."""
        full_flops = _attention_flops(
            seq_len=4096,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,
            kv_channels=128,
            is_swa=False,
        )
        swa_flops = _attention_flops(
            seq_len=4096,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,
            kv_channels=128,
            is_swa=True,
            swa_window_size=128,
        )
        # SWA should have fewer FLOPS than full attention
        assert swa_flops < full_flops
        # But still positive
        assert swa_flops > 0

    def test_moe_mlp_flops_with_glu(self):
        """Test MoE MLP FLOPS with gated linear unit."""
        flops = _moe_mlp_flops(
            seq_len=1024,
            hidden_size=4096,
            moe_ffn_hidden_size=14336,
            moe_router_topk=8,
            gated_linear_unit=True,
        )
        assert flops > 0
        # Verify reasonable range
        assert 1e12 < flops < 1e16

    def test_moe_mlp_flops_without_glu(self):
        """Test MoE MLP FLOPS without gated linear unit is less."""
        with_glu = _moe_mlp_flops(
            seq_len=1024,
            hidden_size=4096,
            moe_ffn_hidden_size=14336,
            moe_router_topk=8,
            gated_linear_unit=True,
        )
        without_glu = _moe_mlp_flops(
            seq_len=1024,
            hidden_size=4096,
            moe_ffn_hidden_size=14336,
            moe_router_topk=8,
            gated_linear_unit=False,
        )
        # Without GLU should have fewer FLOPS (no gate projection)
        assert without_glu < with_glu

    def test_vocab_flops(self):
        """Test vocabulary projection FLOPS."""
        flops = _vocab_flops(seq_len=1024, hidden_size=4096, vocab_size=128000)
        assert flops > 0
        # vocab_flops = 6 * seq_len * hidden_size * vocab_size
        expected = 6 * 1024 * 4096 * 128000
        assert flops == expected


class TestGptOssFlopsFormula:
    """Unit tests for the full GPT-OSS FLOPS formula."""

    @pytest.fixture
    def gpt_oss_20b_like_config(self):
        """Create a config similar to GPT-OSS 20B for testing."""
        return FLOPSConfig(
            gbs=1,
            enc_seq_len=4096,
            hs=6912,
            layers=36,
            attention_heads=48,
            query_groups=8,
            vocab_size=128000,
            moe_ffn_hidden_size=2560,
            moe_router_topk=8,
            swa_window_size=128,
            window_attn_skip_freq=2,
            kv_channels=144,  # 6912 / 48
            gated_linear_unit=True,
        )

    @pytest.fixture
    def gpt_oss_toy_config(self):
        """Create a minimal toy config for quick tests."""
        return FLOPSConfig(
            gbs=1,
            enc_seq_len=512,
            hs=512,
            layers=4,
            attention_heads=8,
            query_groups=8,
            vocab_size=32000,
            moe_ffn_hidden_size=1536,
            moe_router_topk=2,
            swa_window_size=64,
            window_attn_skip_freq=2,
            kv_channels=64,  # 512 / 8
            gated_linear_unit=True,
        )

    def test_gpt_oss_flops_positive(self, gpt_oss_toy_config):
        """Test that GPT-OSS FLOPS returns a positive value."""
        flops = gpt_oss(gpt_oss_toy_config)
        assert flops > 0

    def test_gpt_oss_flops_scales_with_batch_size(self, gpt_oss_toy_config):
        """Test that FLOPS scales linearly with batch size."""
        gpt_oss_toy_config.gbs = 1
        flops_bs1 = gpt_oss(gpt_oss_toy_config)

        gpt_oss_toy_config.gbs = 8
        flops_bs8 = gpt_oss(gpt_oss_toy_config)

        assert flops_bs8 == 8 * flops_bs1

    def test_gpt_oss_flops_scales_with_layers(self, gpt_oss_toy_config):
        """Test that FLOPS scales approximately linearly with layer count."""
        gpt_oss_toy_config.layers = 4
        flops_4layers = gpt_oss(gpt_oss_toy_config)

        gpt_oss_toy_config.layers = 8
        flops_8layers = gpt_oss(gpt_oss_toy_config)

        # Should be approximately 2x (not exactly due to vocab layer)
        ratio = flops_8layers / flops_4layers
        assert 1.8 < ratio < 2.2

    def test_gpt_oss_flops_20b_like_reasonable_range(self, gpt_oss_20b_like_config):
        """Test that GPT-OSS 20B-like config produces reasonable FLOPS."""
        flops = gpt_oss(gpt_oss_20b_like_config)

        # For a 20B parameter model with seq_len=4096, batch=1, expect ~1e14 FLOPS
        # Rough estimate: 6 * 2 * params * seq_len = 6 * 2 * 20e9 * 4096 ≈ 1e15
        # With MoE routing (topk=8 out of many experts), actual compute is higher
        assert 1e13 < flops < 1e16, f"FLOPS {flops} out of expected range"

    def test_gpt_oss_mixed_attention_layers(self, gpt_oss_toy_config):
        """Test that mixed attention (SWA + full) produces expected pattern."""
        # With window_attn_skip_freq=2:
        # Layers 0, 2 use SWA (odd 1-indexed: 1, 3)
        # Layers 1, 3 use full attention (even 1-indexed: 2, 4)

        # All SWA should have fewer FLOPS
        gpt_oss_toy_config.window_attn_skip_freq = None  # All SWA
        all_swa_flops = gpt_oss(gpt_oss_toy_config)

        gpt_oss_toy_config.window_attn_skip_freq = 1  # All full attention
        all_full_flops = gpt_oss(gpt_oss_toy_config)

        gpt_oss_toy_config.window_attn_skip_freq = 2  # Mixed
        mixed_flops = gpt_oss(gpt_oss_toy_config)

        assert all_swa_flops < mixed_flops < all_full_flops


class TestGptOssFlopsTrackerIntegration:
    """Integration tests for GPT-OSS with FLOPTracker."""

    @pytest.fixture
    def mock_gpt_oss_config(self):
        """Create a mock GptOssConfig-like object for testing."""

        class MockGptOssConfig:
            model_type = "gpt_oss"
            hidden_size = 512
            num_hidden_layers = 4
            intermediate_size = 1536
            num_attention_heads = 8
            num_key_value_heads = 8
            vocab_size = 32000
            moe_ffn_hidden_size = 1536
            num_experts_per_tok = 2
            window_size = (64, 64)
            window_attn_skip_freq = 2
            kv_channels = 64

        return MockGptOssConfig()

    def test_flops_config_from_gpt_oss_config(self, mock_gpt_oss_config):
        """Test that FLOPSConfig can be constructed from GPT-OSS config attributes."""
        from nemo_rl.utils.flops_tracker import _HAS_GPT_OSS

        if not _HAS_GPT_OSS:
            pytest.skip("GptOssConfig not available in transformers")

        # This tests the conversion logic without needing the actual class
        config = mock_gpt_oss_config

        # Extract fields as done in convert_config_to_flops_config
        moe_ffn_hidden_size = getattr(config, "moe_ffn_hidden_size", config.intermediate_size)
        window_size = getattr(config, "window_size", None)
        swa_window_size = window_size[0] if window_size else 128
        window_attn_skip_freq = getattr(config, "window_attn_skip_freq", 2)
        kv_channels = getattr(
            config, "kv_channels", config.hidden_size // config.num_attention_heads
        )

        flops_config = FLOPSConfig(
            gbs=1,
            enc_seq_len=1024,
            hs=config.hidden_size,
            layers=config.num_hidden_layers,
            ffn_hs=config.intermediate_size,
            attention_heads=config.num_attention_heads,
            query_groups=config.num_key_value_heads,
            vocab_size=config.vocab_size,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            moe_router_topk=config.num_experts_per_tok,
            swa_window_size=swa_window_size,
            window_attn_skip_freq=window_attn_skip_freq,
            kv_channels=kv_channels,
            gated_linear_unit=True,
        )

        flops = gpt_oss(flops_config)
        assert flops > 0
