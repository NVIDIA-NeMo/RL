---
description: "Learn techniques to maintain training stability and prevent divergence"
categories: ["guides"]
tags: ["stability", "training", "gradient-clipping", "loss-scaling", "checkpointing"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "intermediate"
content_type: "guide"
modality: "universal"
---

# Stabilize Training

This guide covers techniques to maintain training stability and prevent divergence when training NeMo RL models.

## Overview

Training large language models can be unstable, leading to loss divergence, gradient explosion, or poor convergence. This guide provides practical techniques to maintain stable training.

## Gradient Clipping

### Basic Gradient Clipping

```python
def clip_gradients(model, max_norm=1.0):
    """Clip gradients to prevent explosion"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# Usage in training loop
def training_step(model, batch, optimizer):
    # Forward pass
    loss = model(batch)
    
    # Backward pass
    loss.backward()
    
    # Clip gradients
    clip_gradients(model, max_norm=1.0)
    
    # Update parameters
    optimizer.step()
    optimizer.zero_grad()
    
    return loss
```

### Adaptive Gradient Clipping

```python
class AdaptiveGradientClipper:
    def __init__(self, initial_norm=1.0, adaptation_rate=0.1):
        self.current_norm = initial_norm
        self.adaptation_rate = adaptation_rate
        self.gradient_history = []
        
    def clip_gradients(self, model):
        """Adaptively clip gradients based on history"""
        
        # Compute gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        
        # Record gradient norm
        self.gradient_history.append(total_norm.item())
        
        # Adapt clipping norm based on recent history
        if len(self.gradient_history) > 10:
            recent_norms = self.gradient_history[-10:]
            avg_norm = sum(recent_norms) / len(recent_norms)
            
            # Adjust clipping norm
            if avg_norm > self.current_norm * 1.5:
                self.current_norm *= (1 + self.adaptation_rate)
            elif avg_norm < self.current_norm * 0.5:
                self.current_norm *= (1 - self.adaptation_rate)
                
        # Apply clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.current_norm)
        
        return total_norm
```

## Loss Scaling

### Dynamic Loss Scaling

```python
class DynamicLossScaler:
    def __init__(self, initial_scale=2**16, scale_factor=2, scale_window=2000):
        self.scale = initial_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.consecutive_overflow = 0
        
    def scale_loss(self, loss):
        """Scale loss for mixed precision training"""
        return loss * self.scale
        
    def unscale_gradients(self, optimizer):
        """Unscale gradients and check for overflow"""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.scale)
                    
    def update_scale(self, overflow):
        """Update loss scale based on overflow"""
        if overflow:
            self.consecutive_overflow += 1
            if self.consecutive_overflow >= 2:
                self.scale = max(self.scale / self.scale_factor, 1)
                self.consecutive_overflow = 0
        else:
            self.consecutive_overflow = 0
            if self.consecutive_overflow == 0:
                self.scale = min(self.scale * self.scale_factor, 2**32)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
        
    def training_step(self, batch):
        """Training step with mixed precision"""
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            loss = self.model(batch)
            
        # Scale loss and backward pass
        self.scaler.scale(loss).backward()
        
        # Unscale gradients and clip
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update parameters
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss
```

## Loss Monitoring

### Loss Stability Monitor

```python
class LossStabilityMonitor:
    def __init__(self, window_size=100, threshold=0.1):
        self.loss_history = []
        self.window_size = window_size
        self.threshold = threshold
        self.stability_warnings = []
        
    def update(self, loss):
        """Update loss history and check stability"""
        self.loss_history.append(loss.item())
        
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            
        # Check for instability
        if len(self.loss_history) >= 10:
            recent_losses = self.loss_history[-10:]
            loss_std = torch.std(torch.tensor(recent_losses))
            loss_mean = torch.mean(torch.tensor(recent_losses))
            
            # Coefficient of variation
            cv = loss_std / loss_mean
            
            if cv > self.threshold:
                warning = {
                    'step': len(self.loss_history),
                    'cv': cv.item(),
                    'mean_loss': loss_mean.item(),
                    'std_loss': loss_std.item()
                }
                self.stability_warnings.append(warning)
                
    def get_stability_report(self):
        """Get current stability report"""
        if len(self.loss_history) < 10:
            return {'status': 'insufficient_data'}
            
        recent_losses = self.loss_history[-10:]
        loss_std = torch.std(torch.tensor(recent_losses))
        loss_mean = torch.mean(torch.tensor(recent_losses))
        cv = loss_std / loss_mean
        
        return {
            'status': 'stable' if cv < self.threshold else 'unstable',
            'coefficient_of_variation': cv.item(),
            'mean_loss': loss_mean.item(),
            'std_loss': loss_std.item(),
            'warnings': len(self.stability_warnings)
        }
```

### Gradient Monitoring

```python
class GradientMonitor:
    def __init__(self):
        self.gradient_norms = []
        self.parameter_norms = []
        
    def monitor_gradients(self, model):
        """Monitor gradient and parameter norms"""
        
        # Compute gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)
        
        # Compute parameter norms
        param_norm = 0
        for p in model.parameters():
            param_norm += p.data.norm(2) ** 2
        param_norm = param_norm ** 0.5
        self.parameter_norms.append(param_norm)
        
        return {
            'gradient_norm': total_norm,
            'parameter_norm': param_norm,
            'gradient_parameter_ratio': total_norm / param_norm if param_norm > 0 else 0
        }
```

## Checkpointing and Recovery

### Automatic Checkpointing

```python
class CheckpointManager:
    def __init__(self, save_dir, save_frequency=1000):
        self.save_dir = save_dir
        self.save_frequency = save_frequency
        self.best_loss = float('inf')
        
    def save_checkpoint(self, model, optimizer, step, loss):
        """Save training checkpoint"""
        
        # Regular checkpoint
        if step % self.save_frequency == 0:
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': model.config
            }
            
            checkpoint_path = f"{self.save_dir}/checkpoint_step_{step}.pt"
            torch.save(checkpoint, checkpoint_path)
            
        # Best checkpoint
        if loss < self.best_loss:
            self.best_loss = loss
            best_checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': model.config
            }
            
            best_path = f"{self.save_dir}/best_checkpoint.pt"
            torch.save(best_checkpoint, best_path)
            
    def load_checkpoint(self, model, optimizer, checkpoint_path):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['step'], checkpoint['loss']
```

### Training Recovery

```python
class TrainingRecovery:
    def __init__(self, model, optimizer, checkpoint_manager):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_manager = checkpoint_manager
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
    def detect_instability(self, loss_history, gradient_norms):
        """Detect training instability"""
        
        if len(loss_history) < 10:
            return False
            
        # Check for loss explosion
        recent_losses = loss_history[-10:]
        if max(recent_losses) > 10 * min(recent_losses):
            return True
            
        # Check for gradient explosion
        if len(gradient_norms) > 0:
            recent_grads = gradient_norms[-10:]
            if max(recent_grads) > 10 * min(recent_grads):
                return True
                
        return False
        
    def recover_training(self, step, loss):
        """Attempt to recover from instability"""
        
        if self.recovery_attempts >= self.max_recovery_attempts:
            raise RuntimeError("Maximum recovery attempts exceeded")
            
        self.recovery_attempts += 1
        
        # Load best checkpoint
        best_checkpoint_path = f"{self.checkpoint_manager.save_dir}/best_checkpoint.pt"
        if os.path.exists(best_checkpoint_path):
            step, loss = self.checkpoint_manager.load_checkpoint(
                self.model, self.optimizer, best_checkpoint_path
            )
            
        # Reduce learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.5
            
        return step, loss
```

## Best Practices

### 1. Start Conservative
- Use smaller learning rates initially
- Use gradient clipping from the start
- Track loss and gradients closely

### 2. Use Proper Monitoring
- Track loss stability over time
- Track gradient norms
- Set up automatic checkpointing

### 3. Use Mixed Precision Carefully
- Start with full precision
- Gradually introduce mixed precision
- Track numerical instability

### 4. Have Recovery Strategies
- Save checkpoints regularly
- Use automatic recovery
- Have fallback configurations

## Common Patterns

### Stability Checker

```python
class StabilityChecker:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_monitor = LossStabilityMonitor()
        self.gradient_monitor = GradientMonitor()
        
    def check_stability(self, loss):
        """Comprehensive stability check"""
        
        # Update monitors
        self.loss_monitor.update(loss)
        grad_info = self.gradient_monitor.monitor_gradients(self.model)
        
        # Get stability report
        loss_report = self.loss_monitor.get_stability_report()
        
        # Determine overall stability
        is_stable = (
            loss_report['status'] == 'stable' and
            grad_info['gradient_norm'] < 10.0 and
            grad_info['gradient_parameter_ratio'] < 0.1
        )
        
        return {
            'is_stable': is_stable,
            'loss_report': loss_report,
            'gradient_info': grad_info
        }
```

### Training Stabilizer

```python
class TrainingStabilizer:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        
        # Initialize components
        self.gradient_clipper = AdaptiveGradientClipper()
        self.loss_scaler = DynamicLossScaler()
        self.stability_checker = StabilityChecker(model, optimizer)
        
    def stabilize_training_step(self, batch):
        """Perform stabilized training step"""
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        loss = self.model(batch)
        
        # Scale loss for mixed precision
        scaled_loss = self.loss_scaler.scale_loss(loss)
        
        # Backward pass
        scaled_loss.backward()
        
        # Unscale gradients
        self.loss_scaler.unscale_gradients(self.optimizer)
        
        # Clip gradients
        self.gradient_clipper.clip_gradients(self.model)
        
        # Check stability
        stability_info = self.stability_checker.check_stability(loss)
        
        # Update parameters
        self.optimizer.step()
        
        # Update loss scale
        overflow = torch.isinf(scaled_loss) or torch.isnan(scaled_loss)
        self.loss_scaler.update_scale(overflow)
        
        return loss, stability_info
```

## Troubleshooting

### Common Stability Issues

1. **Loss Explosion**: Reduce learning rate, increase gradient clipping
2. **Gradient Explosion**: Implement adaptive gradient clipping
3. **Numerical Instability**: Use mixed precision carefully, check for NaN/Inf
4. **Oscillating Loss**: Reduce learning rate, increase batch size

### Debugging Tips

```python
def debug_training_stability(model, dataloader, num_steps=100):
    """Debug training stability"""
    
    stability_checker = StabilityChecker(model, model.optimizer)
    
    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break
            
        loss = model.training_step(batch)
        stability_info = stability_checker.check_stability(loss)
        
        if not stability_info['is_stable']:
            print(f"Instability detected at step {i}")
            print(f"Loss report: {stability_info['loss_report']}")
            print(f"Gradient info: {stability_info['gradient_info']}")
            break
```

## Get Help

- [Troubleshooting](../troubleshooting.md) - Common issues and solutions
 