# Hyperparameter Optimization

This guide covers hyperparameter optimization for NeMo RL using the actual configuration patterns and hyperparameters found in the codebase.

## Overview

NeMo RL uses YAML-based configuration files for hyperparameter management. This guide shows how to optimize the real hyperparameters used in DPO and GRPO training.

## Key NeMo RL Hyperparameters

### Policy Configuration

The main hyperparameters are defined in the policy section:

```yaml
# configs/optimization_example.yaml
policy:
  model_name: "microsoft/DialoGPT-medium"
  max_total_sequence_length: 2048
  precision: "bfloat16"  # or "float16", "float32"
  
  # Optimizer hyperparameters
  optimizer:
    name: "torch.optim.AdamW"
    kwargs:
      lr: 1e-5  # Learning rate - key hyperparameter
      weight_decay: 0.01  # Weight decay
      betas: [0.9, 0.999]  # Adam betas
      eps: 1e-8  # Adam epsilon
      foreach: false  # For distributed training
      fused: false  # For distributed training
  
  # Learning rate scheduler
  scheduler:
    - name: "torch.optim.lr_scheduler.LinearLR"
      kwargs:
        start_factor: 0.1
        end_factor: 1.0
        total_iters: 50
    - name: "torch.optim.lr_scheduler.ConstantLR"
      kwargs:
        factor: 1.0
        total_iters: 10000000000
    - milestones: [50]
  
  # Training hyperparameters
  max_grad_norm: 1.0  # Gradient clipping
  train_micro_batch_size: 1  # Micro batch size
  train_global_batch_size: 16  # Global batch size
```

### DPO-Specific Hyperparameters

For DPO training, these are the key hyperparameters:

```yaml
dpo:
  # Core DPO parameters
  reference_policy_kl_penalty: 0.1  # Beta parameter - critical
  preference_loss_weight: 1.0  # Weight for preference loss
  sft_loss_weight: 0.1  # Weight for SFT loss
  preference_average_log_probs: true  # Average across tokens
  sft_average_log_probs: true  # Average SFT loss
  
  # Training schedule
  max_num_epochs: 3
  max_num_steps: 1000
  val_period: 100
  val_batches: 5
  val_global_batch_size: 32
  val_micro_batch_size: 8
  val_at_start: true
  seed: 42
```

### GRPO-Specific Hyperparameters

For GRPO training, these are the key hyperparameters:

```yaml
grpo:
  # Training parameters
  num_prompts_per_step: 4
  num_generations_per_prompt: 4
  max_num_steps: 1000
  max_rollout_turns: 1
  normalize_rewards: true
  use_leave_one_out_baseline: true
  
  # Validation
  val_period: 100
  val_batch_size: 8
  val_at_start: true
  max_val_samples: 100
```

## Hyperparameter Optimization Strategies

### 1. Learning Rate Optimization

The learning rate is the most critical hyperparameter:

```python
import yaml
from nemo_rl.algorithms.dpo import setup, dpo_train
from nemo_rl.utils.config import load_config

def optimize_learning_rate():
    """Optimize learning rate for DPO training."""
    
    # Learning rate candidates
    lr_candidates = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
    
    best_lr = None
    best_val_loss = float('inf')
    
    for lr in lr_candidates:
        # Load base config
        config = load_config("configs/dpo_base.yaml")
        
        # Update learning rate
        config['policy']['optimizer']['kwargs']['lr'] = lr
        
        # Train and evaluate
        val_loss = train_and_evaluate_dpo(config)
        
        print(f"LR: {lr}, Val Loss: {val_loss}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lr = lr
    
    print(f"Best learning rate: {best_lr}")
    return best_lr

def train_and_evaluate_dpo(config):
    """Train DPO model and return validation loss."""
    # Setup training
    policy, cluster, train_dataloader, val_dataloader, loss_fn, master_config, logger, task_spec, save_state = setup(
        master_config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    # Run training for a few steps
    dpo_train(
        policy=policy,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        master_config=master_config,
        logger=logger,
        checkpointer=checkpointer,
        dpo_save_state=save_state
    )
    
    # Return validation loss
    return get_validation_loss(logger)
```

### 2. DPO Beta Optimization

The KL penalty (beta) is crucial for DPO performance:

```python
def optimize_dpo_beta():
    """Optimize DPO beta parameter."""
    
    # Beta candidates
    beta_candidates = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    best_beta = None
    best_reward_gap = -float('inf')
    
    for beta in beta_candidates:
        # Load base config
        config = load_config("configs/dpo_base.yaml")
        
        # Update beta
        config['dpo']['reference_policy_kl_penalty'] = beta
        
        # Train and evaluate
        reward_gap = train_and_evaluate_dpo_rewards(config)
        
        print(f"Beta: {beta}, Reward Gap: {reward_gap}")
        
        if reward_gap > best_reward_gap:
            best_reward_gap = reward_gap
            best_beta = beta
    
    print(f"Best beta: {best_beta}")
    return best_beta
```

### 3. Batch Size Optimization

Optimize batch sizes for your hardware:

```python
def optimize_batch_sizes():
    """Optimize batch sizes for available memory."""
    
    # Batch size combinations
    batch_configs = [
        {'micro': 1, 'global': 8},
        {'micro': 1, 'global': 16},
        {'micro': 2, 'global': 16},
        {'micro': 2, 'global': 32},
        {'micro': 4, 'global': 32},
    ]
    
    best_config = None
    best_throughput = 0
    
    for config in batch_configs:
        try:
            # Load base config
            yaml_config = load_config("configs/dpo_base.yaml")
            
            # Update batch sizes
            yaml_config['policy']['train_micro_batch_size'] = config['micro']
            yaml_config['policy']['train_global_batch_size'] = config['global']
            
            # Test training
            throughput = test_training_throughput(yaml_config)
            
            print(f"Micro: {config['micro']}, Global: {config['global']}, Throughput: {throughput}")
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_config = config
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM for config: {config}")
                continue
    
    print(f"Best batch config: {best_config}")
    return best_config
```

### 4. Loss Weight Optimization

Optimize the balance between preference and SFT losses:

```python
def optimize_loss_weights():
    """Optimize DPO loss weights."""
    
    # Weight combinations
    weight_configs = [
        {'preference': 1.0, 'sft': 0.0},   # Pure preference
        {'preference': 1.0, 'sft': 0.1},   # Light SFT
        {'preference': 1.0, 'sft': 0.2},   # Balanced
        {'preference': 0.8, 'sft': 0.2},   # More SFT
        {'preference': 0.5, 'sft': 0.5},   # Equal weights
    ]
    
    best_weights = None
    best_accuracy = 0
    
    for weights in weight_configs:
        # Load base config
        config = load_config("configs/dpo_base.yaml")
        
        # Update weights
        config['dpo']['preference_loss_weight'] = weights['preference']
        config['dpo']['sft_loss_weight'] = weights['sft']
        
        # Train and evaluate
        accuracy = train_and_evaluate_accuracy(config)
        
        print(f"Preference: {weights['preference']}, SFT: {weights['sft']}, Accuracy: {accuracy}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = weights
    
    print(f"Best weights: {best_weights}")
    return best_weights
```

## Configuration-Based Optimization

### 1. Grid Search Configuration

Create a grid search over key hyperparameters:

```yaml
# configs/grid_search.yaml
optimization:
  method: "grid_search"
  hyperparameters:
    learning_rate: [1e-6, 5e-6, 1e-5, 5e-5]
    dpo_beta: [0.05, 0.1, 0.2, 0.3]
    batch_size: [8, 16, 32]
    preference_weight: [0.8, 1.0, 1.2]
    sft_weight: [0.0, 0.1, 0.2]
  
  evaluation:
    max_steps: 500
    val_period: 50
    metric: "validation_loss"
```

### 2. Random Search Configuration

Use random search for broader exploration:

```yaml
# configs/random_search.yaml
optimization:
  method: "random_search"
  n_trials: 50
  
  hyperparameters:
    learning_rate:
      type: "float"
      min: 1e-6
      max: 1e-4
      log: true
      
    dpo_beta:
      type: "float"
      min: 0.05
      max: 0.5
      
    batch_size:
      type: "int"
      min: 8
      max: 32
      
    preference_weight:
      type: "float"
      min: 0.5
      max: 1.5
      
    sft_weight:
      type: "float"
      min: 0.0
      max: 0.5
  
  evaluation:
    max_steps: 500
    val_period: 50
    metric: "validation_loss"
```

## Automated Optimization Script

```python
import yaml
import json
import time
from pathlib import Path
from nemo_rl.utils.config import load_config
from nemo_rl.algorithms.dpo import setup, dpo_train

class NeMoRLOptimizer:
    def __init__(self, config_path):
        self.base_config = load_config(config_path)
        self.results = []
        
    def optimize_hyperparameters(self, optimization_config):
        """Run hyperparameter optimization."""
        method = optimization_config['method']
        
        if method == 'grid_search':
            return self.grid_search(optimization_config)
        elif method == 'random_search':
            return self.random_search(optimization_config)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def grid_search(self, config):
        """Perform grid search over hyperparameters."""
        hyperparams = config['hyperparameters']
        param_names = list(hyperparams.keys())
        param_values = list(hyperparams.values())
        
        best_config = None
        best_score = float('inf')
        
        # Generate all combinations
        from itertools import product
        combinations = list(product(*param_values))
        
        for i, combination in enumerate(combinations):
            # Create config with these parameters
            trial_config = self.base_config.copy()
            for name, value in zip(param_names, combination):
                self.set_nested_config(trial_config, name, value)
            
            # Evaluate this configuration
            score = self.evaluate_config(trial_config, config['evaluation'])
            
            print(f"Trial {i+1}/{len(combinations)}: {dict(zip(param_names, combination))} -> {score}")
            
            if score < best_score:
                best_score = score
                best_config = trial_config.copy()
        
        return best_config, best_score
    
    def random_search(self, config):
        """Perform random search over hyperparameters."""
        import random
        import numpy as np
        
        n_trials = config['n_trials']
        hyperparams = config['hyperparameters']
        
        best_config = None
        best_score = float('inf')
        
        for trial in range(n_trials):
            # Generate random parameters
            trial_params = {}
            for name, param_config in hyperparams.items():
                if param_config['type'] == 'float':
                    if param_config.get('log', False):
                        # Log-uniform sampling
                        min_val, max_val = param_config['min'], param_config['max']
                        trial_params[name] = np.exp(
                            np.random.uniform(np.log(min_val), np.log(max_val))
                        )
                    else:
                        trial_params[name] = np.random.uniform(
                            param_config['min'], param_config['max']
                        )
                elif param_config['type'] == 'int':
                    trial_params[name] = np.random.randint(
                        param_config['min'], param_config['max'] + 1
                    )
            
            # Create config with these parameters
            trial_config = self.base_config.copy()
            for name, value in trial_params.items():
                self.set_nested_config(trial_config, name, value)
            
            # Evaluate this configuration
            score = self.evaluate_config(trial_config, config['evaluation'])
            
            print(f"Trial {trial+1}/{n_trials}: {trial_params} -> {score}")
            
            if score < best_score:
                best_score = score
                best_config = trial_config.copy()
        
        return best_config, best_score
    
    def set_nested_config(self, config, param_path, value):
        """Set nested configuration parameter."""
        # Handle nested paths like 'policy.optimizer.kwargs.lr'
        keys = param_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def evaluate_config(self, config, eval_config):
        """Evaluate a configuration."""
        try:
            # Setup training
            policy, cluster, train_dataloader, val_dataloader, loss_fn, master_config, logger, task_spec, save_state = setup(
                master_config=config,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset
            )
            
            # Run training for specified steps
            max_steps = eval_config['max_steps']
            val_period = eval_config['val_period']
            
            # Modified training loop for evaluation
            step = 0
            while step < max_steps:
                # Train for val_period steps
                for _ in range(val_period):
                    if step >= max_steps:
                        break
                    
                    # Single training step
                    self.train_step(policy, train_dataloader, loss_fn)
                    step += 1
                
                # Evaluate
                val_loss = self.evaluate_model(policy, val_dataloader, loss_fn)
                
                # Early stopping if loss is too high
                if val_loss > 10.0:
                    return val_loss
            
            return val_loss
            
        except Exception as e:
            print(f"Error evaluating config: {e}")
            return float('inf')
    
    def save_results(self, best_config, best_score, output_path):
        """Save optimization results."""
        results = {
            'best_config': best_config,
            'best_score': best_score,
            'all_results': self.results,
            'timestamp': time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save best config as YAML
        config_path = output_path.replace('.json', '_best.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)
```

## Best Practices

### 1. Start with Reasonable Ranges

```python
# Good starting ranges for NeMo RL
learning_rate_range = (1e-6, 1e-4)  # Log scale
dpo_beta_range = (0.05, 0.3)        # DPO KL penalty
batch_size_range = (8, 32)           # Memory dependent
weight_decay_range = (0.0, 0.1)     # Regularization
```

### 2. Use Log Scale for Multiplicative Parameters

```python
# Learning rate should use log scale
lr_candidates = np.logspace(-6, -4, 10)  # 10 values from 1e-6 to 1e-4

# Beta can use linear scale
beta_candidates = np.linspace(0.05, 0.3, 6)  # 6 values from 0.05 to 0.3
```

### 3. Consider Hardware Constraints

```python
def check_memory_constraints(config):
    """Check if configuration fits in available memory."""
    batch_size = config['policy']['train_global_batch_size']
    seq_length = config['policy']['max_total_sequence_length']
    
    # Rough memory estimation
    estimated_memory_gb = (batch_size * seq_length * 2) / 1e9
    
    return estimated_memory_gb < available_memory_gb
```

### 4. Monitor Key Metrics

```python
def log_optimization_metrics(logger, config, metrics):
    """Log metrics during optimization."""
    logger.log({
        'learning_rate': config['policy']['optimizer']['kwargs']['lr'],
        'dpo_beta': config['dpo']['reference_policy_kl_penalty'],
        'batch_size': config['policy']['train_global_batch_size'],
        'validation_loss': metrics['val_loss'],
        'reward_gap': metrics.get('reward_gap', 0),
        'training_time': metrics.get('training_time', 0)
    })
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```python
   # Reduce batch size
   config['policy']['train_global_batch_size'] = 8
   config['policy']['train_micro_batch_size'] = 1
   ```

2. **Training Instability**
   ```python
   # Reduce learning rate
   config['policy']['optimizer']['kwargs']['lr'] = 1e-6
   
   # Increase gradient clipping
   config['policy']['max_grad_norm'] = 0.5
   ```

3. **Poor DPO Performance**
   ```python
   # Adjust beta parameter
   config['dpo']['reference_policy_kl_penalty'] = 0.2
   
   # Balance loss weights
   config['dpo']['preference_loss_weight'] = 1.0
   config['dpo']['sft_loss_weight'] = 0.1
   ```

4. **Slow Convergence**
   ```python
   # Increase learning rate
   config['policy']['optimizer']['kwargs']['lr'] = 5e-5
   
   # Adjust scheduler
   config['policy']['scheduler'][0]['kwargs']['total_iters'] = 100
   ```

## Next Steps

- Learn about [Loss Functions](../../core-design/design-principles/loss-functions) for advanced loss customization
- Review [Mathematical Foundations](mathematical-foundations) for theoretical understanding
- Explore [Performance & Scaling](../performance/index) for training optimization
- Study [Troubleshooting](../../guides/troubleshooting) for common issues 