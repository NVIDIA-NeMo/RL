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
"""Unit tests for RolloutEngine.

These tests verify TASK-011 requirements:
- AC1: Create RolloutEngine class separating generation logic from training
- AC2: RolloutEngine handles: prompt batching, response generation, reward collection
- AC3: Configurable generation parameters (temperature, top_p, etc.)
- AC4: Works with multiple generation backends (vLLM, Megatron)
- VERIFY: Instantiate RolloutEngine, call generate() with test prompts
"""

import pytest
import torch

from nemo_rl.algorithms.rollout import (
    RolloutEngine,
    RolloutResult,
    SamplingParams,
    create_rollout_engine,
)


class MockBatchedDataDict(dict):
    """Mock BatchedDataDict for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        return None
    
    def get(self, key, default=None):
        return super().get(key, default)


class MockGenerationBackend:
    """Mock generation backend for testing."""
    
    def __init__(self, response_length: int = 10):
        self.response_length = response_length
        self.prepare_called = False
        self.finish_called = False
        self.generate_called = False
    
    def prepare_for_generation(self):
        self.prepare_called = True
        return True
    
    def finish_generation(self):
        self.finish_called = True
        return True
    
    def generate(self, data, greedy: bool = False):
        self.generate_called = True
        batch_size = data.get('input_ids', torch.zeros(4, 10)).shape[0]
        
        return MockBatchedDataDict({
            'output_ids': torch.randint(0, 1000, (batch_size, 20)),
            'generation_lengths': torch.full((batch_size,), self.response_length),
            'unpadded_sequence_lengths': torch.full((batch_size,), 15),
            'logprobs': torch.randn(batch_size, 20),
        })


class TestSamplingParams:
    """Tests for SamplingParams dataclass."""
    
    def test_default_values(self):
        """Test default parameter values."""
        params = SamplingParams()
        
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k is None
        assert params.max_tokens == 256
        assert params.greedy is False
    
    def test_custom_values(self):
        """Test custom parameter values."""
        params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            max_tokens=512,
            greedy=True,
        )
        
        assert params.temperature == 0.7
        assert params.top_p == 0.95
        assert params.top_k == 50
        assert params.max_tokens == 512
        assert params.greedy is True
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=128,
            stop_strings=["END"],
        )
        
        d = params.to_dict()
        
        assert d["temperature"] == 0.8
        assert d["top_p"] == 0.9
        assert d["max_new_tokens"] == 128
        assert d["stop_strings"] == ["END"]


class TestRolloutResult:
    """Tests for RolloutResult dataclass."""
    
    def test_default_values(self):
        """Test default result values."""
        result = RolloutResult()
        
        assert result.prompts is None
        assert result.responses is None
        assert result.rewards is None
        assert result.metrics == {}
    
    def test_with_values(self):
        """Test result with values."""
        prompts = MockBatchedDataDict({'input_ids': torch.zeros(4, 10)})
        responses = MockBatchedDataDict({'output_ids': torch.zeros(4, 20)})
        rewards = torch.ones(4)
        
        result = RolloutResult(
            prompts=prompts,
            responses=responses,
            rewards=rewards,
            metrics={'mean_reward': 1.0},
        )
        
        assert result.prompts is not None
        assert result.responses is not None
        assert result.rewards is not None
        assert result.metrics['mean_reward'] == 1.0


class TestRolloutEngine:
    """Tests for RolloutEngine class (AC1)."""
    
    def test_init_without_backend(self):
        """Test initialization without backend."""
        engine = RolloutEngine()
        
        assert engine.generation_backend is None
        assert engine.environment is None
    
    def test_init_with_backend(self):
        """Test initialization with mock backend."""
        backend = MockGenerationBackend()
        engine = RolloutEngine(generation_backend=backend)
        
        assert engine.generation_backend is backend
    
    def test_init_with_default_params(self):
        """Test initialization with default params."""
        params = SamplingParams(temperature=0.5)
        engine = RolloutEngine(default_params=params)
        
        assert engine.default_params.temperature == 0.5
    
    def test_repr(self):
        """Test string representation."""
        backend = MockGenerationBackend()
        engine = RolloutEngine(generation_backend=backend)
        
        repr_str = repr(engine)
        
        assert "RolloutEngine" in repr_str
        assert "MockGenerationBackend" in repr_str


class TestRolloutEngineGenerate:
    """Tests for RolloutEngine.generate() (AC2, AC3)."""
    
    def test_generate_requires_backend(self):
        """Test that generate raises without backend."""
        engine = RolloutEngine()
        prompts = MockBatchedDataDict({'input_ids': torch.zeros(4, 10)})
        
        with pytest.raises(RuntimeError, match="Generation backend not set"):
            engine.generate(prompts)
    
    def test_generate_calls_backend(self):
        """Test that generate calls the backend."""
        backend = MockGenerationBackend()
        engine = RolloutEngine(generation_backend=backend)
        
        prompts = MockBatchedDataDict({
            'input_ids': torch.zeros(4, 10),
            'input_lengths': torch.full((4,), 10),
        })
        
        responses = engine.generate(prompts)
        
        assert backend.generate_called
        assert 'output_ids' in responses
    
    def test_generate_uses_default_params(self):
        """Test that generate uses default sampling params."""
        backend = MockGenerationBackend()
        params = SamplingParams(temperature=0.5, greedy=True)
        engine = RolloutEngine(
            generation_backend=backend,
            default_params=params,
        )
        
        prompts = MockBatchedDataDict({'input_ids': torch.zeros(2, 10)})
        
        # Should not raise
        engine.generate(prompts)
        
        assert backend.generate_called
    
    def test_generate_with_custom_params(self):
        """Test generate with custom sampling params (AC3)."""
        backend = MockGenerationBackend()
        engine = RolloutEngine(generation_backend=backend)
        
        prompts = MockBatchedDataDict({'input_ids': torch.zeros(2, 10)})
        params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=512,
        )
        
        responses = engine.generate(prompts, params)
        
        assert responses is not None
    
    def test_generate_prepare_and_finish_called(self):
        """Test that prepare and finish are called."""
        backend = MockGenerationBackend()
        engine = RolloutEngine(generation_backend=backend)
        
        prompts = MockBatchedDataDict({'input_ids': torch.zeros(2, 10)})
        
        engine.generate(prompts)
        
        assert backend.prepare_called
        assert backend.finish_called
    
    def test_generate_updates_stats(self):
        """Test that generate updates statistics."""
        backend = MockGenerationBackend(response_length=10)
        engine = RolloutEngine(generation_backend=backend)
        
        assert engine._total_generations == 0
        
        prompts = MockBatchedDataDict({'input_ids': torch.zeros(4, 10)})
        engine.generate(prompts)
        
        assert engine._total_generations == 1
        assert engine._total_tokens_generated == 40  # 4 * 10


class TestRolloutEngineRewards:
    """Tests for RolloutEngine reward collection (AC2)."""
    
    def test_collect_rewards_with_function(self):
        """Test reward collection with simple function."""
        def simple_reward(prompts, responses):
            batch_size = responses['output_ids'].shape[0]
            return torch.ones(batch_size)
        
        engine = RolloutEngine(reward_fn=simple_reward)
        
        prompts = MockBatchedDataDict({'input_ids': torch.zeros(4, 10)})
        responses = MockBatchedDataDict({
            'output_ids': torch.zeros(4, 20),
            'generation_lengths': torch.full((4,), 10),
        })
        
        result = engine.collect_rewards(prompts, responses)
        
        assert 'rewards' in result
        assert result['rewards'].shape[0] == 4
    
    def test_collect_rewards_requires_env_or_fn(self):
        """Test that collect_rewards raises without env or fn."""
        engine = RolloutEngine()
        
        prompts = MockBatchedDataDict({})
        responses = MockBatchedDataDict({})
        
        with pytest.raises(RuntimeError, match="No reward computation"):
            engine.collect_rewards(prompts, responses)


class TestRolloutEngineRollout:
    """Tests for full rollout pipeline (AC2)."""
    
    def test_rollout_generates_responses(self):
        """Test that rollout generates responses."""
        backend = MockGenerationBackend()
        engine = RolloutEngine(generation_backend=backend)
        
        prompts = MockBatchedDataDict({
            'input_ids': torch.zeros(4, 10),
            'input_lengths': torch.full((4,), 10),
        })
        
        result = engine.rollout(prompts, collect_rewards=False)
        
        assert isinstance(result, RolloutResult)
        assert result.responses is not None
        assert result.generation_lengths is not None
    
    def test_rollout_collects_rewards_when_enabled(self):
        """Test that rollout collects rewards when enabled."""
        backend = MockGenerationBackend()
        
        def reward_fn(prompts, responses):
            return torch.ones(4)
        
        engine = RolloutEngine(
            generation_backend=backend,
            reward_fn=reward_fn,
        )
        
        prompts = MockBatchedDataDict({'input_ids': torch.zeros(4, 10)})
        
        result = engine.rollout(prompts, collect_rewards=True)
        
        assert result.rewards is not None
    
    def test_rollout_includes_metrics(self):
        """Test that rollout includes metrics."""
        backend = MockGenerationBackend()
        engine = RolloutEngine(generation_backend=backend)
        
        prompts = MockBatchedDataDict({'input_ids': torch.zeros(4, 10)})
        
        result = engine.rollout(prompts)
        
        assert 'total_generations' in result.metrics


class TestRolloutEngineStats:
    """Tests for statistics tracking."""
    
    def test_stats_property(self):
        """Test stats property."""
        engine = RolloutEngine()
        stats = engine.stats
        
        assert 'total_generations' in stats
        assert 'total_tokens_generated' in stats
    
    def test_reset_stats(self):
        """Test stats reset."""
        backend = MockGenerationBackend()
        engine = RolloutEngine(generation_backend=backend)
        
        prompts = MockBatchedDataDict({'input_ids': torch.zeros(4, 10)})
        engine.generate(prompts)
        
        assert engine._total_generations > 0
        
        engine.reset_stats()
        
        assert engine._total_generations == 0
        assert engine._total_tokens_generated == 0


class TestCreateRolloutEngine:
    """Tests for create_rollout_engine factory."""
    
    def test_create_rollout_engine_returns_engine(self):
        """Test that factory returns RolloutEngine."""
        engine = create_rollout_engine(backend='mock')
        
        assert isinstance(engine, RolloutEngine)
    
    def test_create_rollout_engine_with_reward_fn(self):
        """Test factory with reward function."""
        def my_reward(p, r):
            return torch.ones(4)
        
        engine = create_rollout_engine(
            backend='mock',
            reward_fn=my_reward,
        )
        
        assert engine.reward_fn is my_reward


# AC4: Works with multiple backends
class TestMultipleBackends:
    """Tests for backend compatibility (AC4)."""
    
    def test_works_with_protocol_backend(self):
        """Test that engine works with any protocol-compatible backend."""
        
        class AnotherBackend:
            """Another backend implementing the protocol."""
            
            def generate(self, data, greedy=False):
                return MockBatchedDataDict({
                    'output_ids': torch.zeros(2, 20),
                    'generation_lengths': torch.full((2,), 10),
                })
            
            def prepare_for_generation(self):
                return True
            
            def finish_generation(self):
                return True
        
        backend = AnotherBackend()
        engine = RolloutEngine(generation_backend=backend)
        
        prompts = MockBatchedDataDict({'input_ids': torch.zeros(2, 10)})
        responses = engine.generate(prompts)
        
        assert responses is not None
        assert 'output_ids' in responses
