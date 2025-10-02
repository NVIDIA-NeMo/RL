---
description: "Master different learning rate scheduling strategies for optimal training convergence"
categories: ["guides"]
tags: ["learning-rate", "scheduling", "optimization", "training"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "intermediate"
content_type: "guide"
modality: "universal"
---

# Optimize Learning Rate Scheduling

This guide covers different learning rate scheduling strategies for optimal training convergence in NeMo RL.

## Overview

Learning rate scheduling is crucial for training large language models effectively. The right schedule can dramatically improve convergence speed and final model performance.

## Basic Learning Rate Schedules

### Linear Warm-up and Decay

```python
def linear_warmup_decay_schedule(optimizer, total_steps, warmup_steps=1000):
    """Linear warm-up followed by linear decay"""
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warm-up
            return step / warmup_steps
        else:
            # Linear decay
            decay_steps = total_steps - warmup_steps
            current_decay_step = step - warmup_steps
            return max(0.0, 1.0 - current_decay_step / decay_steps)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler
```

### Cosine Annealing

```python
def cosine_annealing_schedule(optimizer, total_steps, warmup_steps=1000):
    """Cosine annealing with warm-up"""
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warm-up
            return step / warmup_steps
        else:
            # Cosine annealing
            decay_steps = total_steps - warmup_steps
            current_decay_step = step - warmup_steps
            progress = current_decay_step / decay_steps
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler
```

### Step Decay

```python
def step_decay_schedule(optimizer, decay_steps, decay_rate=0.1):
    """Step decay learning rate schedule"""
    
    def lr_lambda(step):
        return decay_rate ** (step // decay_steps)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler
```

## Advanced Scheduling Strategies

### Adaptive Learning Rate

```python
class AdaptiveLearningRate:
    def __init__(self, optimizer, initial_lr=1e-4, min_lr=1e-6):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.loss_history = []
        
    def update_lr(self, current_loss):
        """Update learning rate based on loss trend"""
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) < 10:
            return
            
        # Check if loss is improving
        recent_losses = self.loss_history[-10:]
        loss_trend = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        if loss_trend > 0:  # Loss increasing
            # Reduce learning rate
            new_lr = max(current_lr * 0.8, self.min_lr)
        else:  # Loss decreasing
            # Slightly increase learning rate
            new_lr = min(current_lr * 1.05, self.initial_lr)
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
```

### Cyclical Learning Rate

```python
class CyclicalLearningRate:
    def __init__(self, optimizer, base_lr=1e-4, max_lr=1e-3, step_size=2000):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.step_count = 0
        
    def step(self):
        """Update learning rate cyclically"""
        cycle = self.step_count // (2 * self.step_size)
        x = abs(self.step_count / self.step_size - 2 * cycle - 1)
        
        # Triangular cycle
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.step_count += 1
```

## NeMo RL Integration

### Configuration-Based Scheduling

```python
# config.yaml
training:
  optimizer:
    type: "adamw"
    lr: 1e-4
    weight_decay: 0.01
  
  scheduler:
    type: "cosine_annealing"
    warmup_steps: 1000
    total_steps: 100000
    min_lr: 1e-6
```

### Custom Scheduler Implementation

```python
from nemo_rl.utils.trainer_utils import SchedulerInterface

class CustomScheduler(SchedulerInterface):
    def __init__(self, config):
        super().__init__(config)
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.total_steps = config.get('total_steps', 100000)
        self.min_lr = config.get('min_lr', 1e-6)
        
    def get_lr(self, step):
        """Compute learning rate for current step"""
        if step < self.warmup_steps:
            # Linear warm-up
            return self.base_lr * (step / self.warmup_steps)
        else:
            # Cosine annealing
            decay_steps = self.total_steps - self.warmup_steps
            current_decay_step = step - self.warmup_steps
            progress = current_decay_step / decay_steps
            
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(lr, self.min_lr)
```

## Best Practices

### 1. Start with Warm-up
- Always use a warm-up period for large models
- Warm-up should be 5-10% of total training steps
- Helps stabilize early training

### 2. Monitor Learning Rate
- Log learning rate changes
- Plot learning rate vs loss
- Adjust schedule based on observations

### 3. Use Appropriate Decay
- Cosine annealing for long training runs
- Step decay for shorter runs
- Adaptive schedules for dynamic datasets

### 4. Consider Model Size
- Larger models need longer warm-up
- Smaller models can use faster schedules
- Adjust based on model complexity

## Common Patterns

### Learning Rate Finder

```python
class LearningRateFinder:
    def __init__(self, model, optimizer, dataloader):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.lr_history = []
        self.loss_history = []
        
    def find_lr(self, start_lr=1e-7, end_lr=1e-1, num_steps=100):
        """Find optimal learning rate range"""
        
        # Exponential increase
        lr_mult = (end_lr / start_lr) ** (1 / num_steps)
        
        for step in range(num_steps):
            lr = start_lr * (lr_mult ** step)
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            # Run one step
            batch = next(iter(self.dataloader))
            loss = self.model.training_step(batch)
            
            # Record
            self.lr_history.append(lr)
            self.loss_history.append(loss.item())
            
        return self.lr_history, self.loss_history
```

### Learning Rate Monitoring

```python
class LRMonitor:
    def __init__(self):
        self.lr_history = []
        self.step_history = []
        
    def log_lr(self, optimizer, step):
        """Log current learning rate"""
        current_lr = optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        self.step_history.append(step)
        
    def plot_lr_schedule(self):
        """Plot learning rate schedule"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.step_history, self.lr_history)
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True)
        plt.show()
```

## Troubleshooting

### Common Issues

1. **Loss Diverging**: Reduce initial learning rate or increase warm-up
2. **Slow Convergence**: Increase learning rate or adjust decay schedule
3. **Oscillating Loss**: Use smoother decay or reduce learning rate
4. **Plateau**: Try cyclical learning rate or adaptive scheduling

### Debugging Tips

```python
def debug_lr_schedule(scheduler, total_steps):
    """Debug learning rate schedule"""
    lrs = []
    for step in range(total_steps):
        lr = scheduler.get_lr()[0]
        lrs.append(lr)
        
        if step % 1000 == 0:
            print(f"Step {step}: LR = {lr:.2e}")
            
    return lrs
```

## Get Help

- [Troubleshooting](../troubleshooting.md) - Common issues and solutions 