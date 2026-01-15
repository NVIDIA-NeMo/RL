#!/usr/bin/env python3
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
"""Verify TASK-014: Policy separation into TrainingPolicy and GenerationPolicy.

This script verifies:
1. TrainingPolicy and GenerationPolicy can be imported separately
2. Each class has distinct responsibilities with no overlap
3. Line count requirements are met (<400 lines per class)
"""

import os
import sys


def test_imports():
    """Test that all policy classes can be imported."""
    print("Testing imports...")
    
    # Test direct imports
    from nemo_rl.models.policy.training_policy import TrainingPolicy, TrainingPolicyProtocol
    from nemo_rl.models.policy.generation_policy import GenerationPolicy, GenerationPolicyProtocol
    from nemo_rl.models.policy.lm_policy import Policy
    
    print("  - TrainingPolicy: OK")
    print("  - TrainingPolicyProtocol: OK")
    print("  - GenerationPolicy: OK")
    print("  - GenerationPolicyProtocol: OK")
    print("  - Policy (unified): OK")
    
    # Test package imports
    from nemo_rl.models.policy import (
        TrainingPolicy as TP,
        GenerationPolicy as GP,
        Policy as P,
    )
    
    assert TP is TrainingPolicy
    assert GP is GenerationPolicy
    assert P is Policy
    print("  - Package imports: OK")
    
    return True


def test_training_policy_interface():
    """Test TrainingPolicy has required methods."""
    print("\nTesting TrainingPolicy interface...")
    from nemo_rl.models.policy.training_policy import TrainingPolicy
    
    required_methods = [
        "train",
        "get_logprobs",
        "get_reference_policy_logprobs",
        "get_topk_logits",
        "save_checkpoint",
        "prepare_for_training",
        "prepare_for_lp_inference",
        "finish_training",
    ]
    
    for method in required_methods:
        assert hasattr(TrainingPolicy, method), f"Missing method: {method}"
        assert callable(getattr(TrainingPolicy, method)), f"Not callable: {method}"
        print(f"  - {method}(): OK")
    
    return True


def test_generation_policy_interface():
    """Test GenerationPolicy has required methods."""
    print("\nTesting GenerationPolicy interface...")
    from nemo_rl.models.policy.generation_policy import GenerationPolicy
    
    required_methods = [
        "generate",
        "score",
        "prepare_for_generation",
        "finish_generation",
        "invalidate_kv_cache",
        "update_weights",
    ]
    
    for method in required_methods:
        assert hasattr(GenerationPolicy, method), f"Missing method: {method}"
        assert callable(getattr(GenerationPolicy, method)), f"Not callable: {method}"
        print(f"  - {method}(): OK")
    
    return True


def test_responsibility_separation():
    """Test that responsibilities are separated correctly."""
    print("\nTesting responsibility separation...")
    from nemo_rl.models.policy.training_policy import TrainingPolicy
    from nemo_rl.models.policy.generation_policy import GenerationPolicy
    
    # TrainingPolicy should NOT have generation methods
    generation_methods = ["generate", "score"]
    for method in generation_methods:
        assert not hasattr(TrainingPolicy, method), f"TrainingPolicy should not have {method}"
    print("  - TrainingPolicy has no generation methods: OK")
    
    # GenerationPolicy should NOT have training methods
    training_methods = ["train", "get_logprobs", "save_checkpoint"]
    for method in training_methods:
        assert not hasattr(GenerationPolicy, method), f"GenerationPolicy should not have {method}"
    print("  - GenerationPolicy has no training methods: OK")
    
    return True


def test_line_counts():
    """Test that files meet line count requirements."""
    print("\nTesting line counts...")
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    policy_dir = os.path.join(base_path, "nemo_rl", "models", "policy")
    
    files_to_check = [
        ("training_policy.py", 400),
        ("generation_policy.py", 400),
    ]
    
    for filename, max_lines in files_to_check:
        filepath = os.path.join(policy_dir, filename)
        with open(filepath, "r") as f:
            line_count = len(f.readlines())
        
        assert line_count < max_lines, f"{filename} has {line_count} lines, should be <{max_lines}"
        print(f"  - {filename}: {line_count} lines (<{max_lines}): OK")
    
    return True


def test_unified_policy_backward_compatibility():
    """Test that unified Policy maintains backward compatibility."""
    print("\nTesting backward compatibility...")
    from nemo_rl.models.policy.lm_policy import Policy
    
    all_methods = [
        # Training methods
        "train",
        "get_logprobs",
        "get_reference_policy_logprobs",
        "save_checkpoint",
        "prepare_for_training",
        # Generation methods
        "generate",
        "score",
        "prepare_for_generation",
        # Infrastructure methods
        "init_collective",
        "shutdown",
        "offload_before_refit",
        "offload_after_refit",
    ]
    
    for method in all_methods:
        assert hasattr(Policy, method), f"Policy missing method: {method}"
        assert callable(getattr(Policy, method)), f"Policy.{method} not callable"
    print(f"  - Policy has all {len(all_methods)} expected methods: OK")
    
    return True


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("TASK-014 Verification: Policy Separation")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("TrainingPolicy Interface", test_training_policy_interface),
        ("GenerationPolicy Interface", test_generation_policy_interface),
        ("Responsibility Separation", test_responsibility_separation),
        ("Line Counts", test_line_counts),
        ("Backward Compatibility", test_unified_policy_backward_compatibility),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except AssertionError as e:
            print(f"\nFAILED: {name}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\nERROR: {name}")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    
    print("\nVERIFY: Import TrainingPolicy and GenerationPolicy separately,")
    print("        verify each has distinct responsibilities with no overlap")
    print("STATUS: PASSED")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
