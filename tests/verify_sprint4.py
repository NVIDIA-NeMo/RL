#!/usr/bin/env python3
"""Verify Sprint 4 tasks implementation."""

import sys

def test_task_014():
    """Test TASK-014: Policy Separation."""
    print("=" * 60)
    print("TASK-014: Policy Separation")
    print("=" * 60)
    
    try:
        from nemo_rl.models.policy import TrainingPolicy, GenerationPolicy
        print("[PASS] TrainingPolicy and GenerationPolicy imported")
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False
    
    # Check TrainingPolicy methods
    training_methods = ['train', 'get_logprobs', 'save_checkpoint']
    for method in training_methods:
        if hasattr(TrainingPolicy, method):
            print(f"[PASS] TrainingPolicy.{method}() exists")
        else:
            print(f"[FAIL] TrainingPolicy.{method}() missing")
            return False
    
    # Check GenerationPolicy methods
    gen_methods = ['generate', 'update_weights']
    for method in gen_methods:
        if hasattr(GenerationPolicy, method):
            print(f"[PASS] GenerationPolicy.{method}() exists")
        else:
            print(f"[FAIL] GenerationPolicy.{method}() missing")
            return False
    
    # Check separation (no overlap)
    if not hasattr(TrainingPolicy, 'generate'):
        print("[PASS] TrainingPolicy does NOT have generate() - correctly separated")
    else:
        print("[FAIL] TrainingPolicy should NOT have generate()")
        return False
    
    if not hasattr(GenerationPolicy, 'train'):
        print("[PASS] GenerationPolicy does NOT have train() - correctly separated")
    else:
        print("[FAIL] GenerationPolicy should NOT have train()")
        return False
    
    return True

def test_task_040():
    """Test TASK-040: SFT BaseTrainer."""
    print("=" * 60)
    print("TASK-040: SFT Refactor to BaseTrainer")
    print("=" * 60)
    
    try:
        from nemo_rl.algorithms.sft import SFTTrainer
        from nemo_rl.trainers.base import BaseTrainer
        print("[PASS] SFTTrainer imported")
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False
    
    # Check extends BaseTrainer
    if issubclass(SFTTrainer, BaseTrainer):
        print("[PASS] SFTTrainer extends BaseTrainer")
    else:
        print("[FAIL] SFTTrainer does NOT extend BaseTrainer")
        return False
    
    # Check required methods
    required_methods = ['_train_step', '_compute_loss', '_validate_step']
    for method in required_methods:
        if hasattr(SFTTrainer, method):
            print(f"[PASS] SFTTrainer.{method}() exists")
        else:
            print(f"[FAIL] SFTTrainer.{method}() missing")
            return False
    
    return True

def test_task_041():
    """Test TASK-041: DPO BaseTrainer."""
    print("=" * 60)
    print("TASK-041: DPO Refactor to BaseTrainer")
    print("=" * 60)
    
    try:
        from nemo_rl.algorithms.dpo import DPOTrainer
        from nemo_rl.trainers.base import BaseTrainer
        print("[PASS] DPOTrainer imported")
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False
    
    # Check extends BaseTrainer
    if issubclass(DPOTrainer, BaseTrainer):
        print("[PASS] DPOTrainer extends BaseTrainer")
    else:
        print("[FAIL] DPOTrainer does NOT extend BaseTrainer")
        return False
    
    # Check required methods
    required_methods = ['_train_step', '_compute_loss', '_validate_step']
    for method in required_methods:
        if hasattr(DPOTrainer, method):
            print(f"[PASS] DPOTrainer.{method}() exists")
        else:
            print(f"[FAIL] DPOTrainer.{method}() missing")
            return False
    
    return True

def test_task_013():
    """Test TASK-013: Unified ValidationRunner."""
    print("=" * 60)
    print("TASK-013: Unified Validation Logic")
    print("=" * 60)
    
    try:
        from nemo_rl.trainers import ValidationRunner, create_validation_runner
        print("[PASS] ValidationRunner imported")
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False
    
    # Check methods
    required_methods = ['should_validate', 'run', 'register_metric', 'reset']
    for method in required_methods:
        if hasattr(ValidationRunner, method):
            print(f"[PASS] ValidationRunner.{method}() exists")
        else:
            print(f"[FAIL] ValidationRunner.{method}() missing")
            return False
    
    # Test instantiation
    try:
        runner = ValidationRunner(metrics=["loss", "accuracy"], frequency=100, mode="steps")
        print(f"[PASS] ValidationRunner instantiated with config: freq={runner.config.frequency}, mode={runner.config.mode}")
    except Exception as e:
        print(f"[FAIL] ValidationRunner instantiation failed: {e}")
        return False
    
    # Test factory function
    try:
        runner = create_validation_runner(config={"val_period": 50, "val_batches": 5})
        print(f"[PASS] create_validation_runner() works")
    except Exception as e:
        print(f"[FAIL] create_validation_runner() failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    results = {}
    results['TASK-014'] = test_task_014()
    print()
    results['TASK-040'] = test_task_040()
    print()
    results['TASK-041'] = test_task_041()
    print()
    results['TASK-013'] = test_task_013()
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for task, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{task}: {status}")
    
    if all(results.values()):
        print("\nAll tasks verified successfully!")
        sys.exit(0)
    else:
        print("\nSome tasks failed verification!")
        sys.exit(1)
