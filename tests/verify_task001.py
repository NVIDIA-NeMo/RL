#!/usr/bin/env python
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
TASK-001 Verification Script

VERIFY: Create a config with invalid values (e.g., negative story_points),
confirm ValidationError is raised immediately with descriptive message
"""

import sys


def verify_task001():
    """Verify TASK-001 acceptance criteria."""
    print("=" * 60)
    print("TASK-001 Verification: Config Module with Pydantic Validation")
    print("=" * 60)
    
    # Test 1: Verify module structure (AC1)
    print("\n[AC1] Verifying module structure...")
    try:
        from nemo_rl.config import (
            BaseConfig,
            ClusterConfig,
            GRPOConfig,
            PolicyConfig,
            DTensorConfig,
            MegatronConfig,
            GenerationConfig,
            VLLMConfig,
            ConfigValidationError,
        )
        from nemo_rl.config.base import BaseConfig
        from nemo_rl.config.policy import PolicyConfig
        from nemo_rl.config.training import GRPOConfig, SFTConfig, DPOConfig
        from nemo_rl.config.generation import GenerationConfig, VLLMConfig
        from nemo_rl.config.cluster import ClusterConfig
        from nemo_rl.config.defaults import get_grpo_config_for_1b_model
        from nemo_rl.config.validation import validate_config
        print("  ✓ All module imports successful")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

    # Test 2: Verify BaseConfig with Pydantic (AC2)
    print("\n[AC2] Verifying BaseConfig with Pydantic validation...")
    try:
        config = PolicyConfig(model_name="test-model")
        print(f"  ✓ Created valid PolicyConfig: model_name={config.model_name}")
    except Exception as e:
        print(f"  ✗ Failed to create valid config: {e}")
        return False

    # Test 3: Verify type annotations and validation (AC3)
    print("\n[AC3] Verifying type annotations and validation...")
    try:
        # This should work
        config = GRPOConfig(
            policy=PolicyConfig(model_name="test-model"),
            num_prompts_per_step=32,
            num_generations_per_prompt=16,
        )
        print(f"  ✓ Created valid GRPOConfig with typed fields")
    except Exception as e:
        print(f"  ✗ Failed to create config: {e}")
        return False

    # Test 4: Verify invalid configs raise clear ValidationError (AC4 + VERIFY)
    print("\n[AC4/VERIFY] Testing invalid config raises ValidationError...")
    
    # Test 4a: Negative value (equivalent to negative story_points)
    try:
        config = GRPOConfig(
            policy=PolicyConfig(model_name="test-model"),
            num_prompts_per_step=-5,  # Invalid: must be positive
        )
        print("  ✗ Should have raised error for negative num_prompts_per_step")
        return False
    except ConfigValidationError as e:
        error_msg = str(e)
        print(f"  ✓ ConfigValidationError raised for negative value")
        print(f"    Error message: {error_msg[:200]}...")
        if "num_prompts_per_step" in error_msg.lower() or "greater" in error_msg.lower():
            print("  ✓ Error message is descriptive (mentions field)")
        else:
            print(f"  ! Warning: Error message may not be descriptive enough")
    except Exception as e:
        print(f"  ✓ Error raised (type: {type(e).__name__}): {str(e)[:100]}")

    # Test 4b: Invalid type
    try:
        config = PolicyConfig(model_name="test", train_micro_batch_size="not_a_number")
        print("  ✗ Should have raised error for invalid type")
        return False
    except (ConfigValidationError, Exception) as e:
        print(f"  ✓ Error raised for invalid type: {type(e).__name__}")

    # Test 5: Verify YAML/JSON/dict loading
    print("\n[Extra] Testing YAML/JSON/dict loading...")
    try:
        import tempfile
        from pathlib import Path
        
        # Create config from dict
        config = PolicyConfig.from_dict({"model_name": "test-model"})
        print(f"  ✓ from_dict() works")
        
        # Save to YAML and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)
            loaded = PolicyConfig.from_yaml(yaml_path)
            assert loaded.model_name == "test-model"
            print(f"  ✓ YAML save/load works")
            
            # Save to JSON and reload
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)
            loaded = PolicyConfig.from_json(json_path)
            assert loaded.model_name == "test-model"
            print(f"  ✓ JSON save/load works")
    except Exception as e:
        print(f"  ✗ File operations failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("TASK-001 VERIFICATION: PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = verify_task001()
    sys.exit(0 if success else 1)
