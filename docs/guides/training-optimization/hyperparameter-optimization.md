---
description: "Systematic approaches to finding optimal hyperparameters for your models"
categories: ["guides"]
tags: ["hyperparameters", "optimization", "tuning", "bayesian", "grid-search"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "intermediate"
content_type: "guide"
modality: "universal"
---

# Optimize Hyperparameters

This guide covers systematic approaches to finding optimal hyperparameters for your NeMo RL models.

## Overview

Hyperparameter optimization is crucial for achieving the best performance from your models. This guide covers both simple and advanced techniques for finding optimal hyperparameters.

## Grid Search

### Basic Grid Search

```python
import itertools
from typing import Dict, List, Any

class GridSearch:
    def __init__(self, param_grid: Dict[str, List[Any]]):
        self.param_grid = param_grid
        self.results = []
        
    def generate_combinations(self):
        """Generate all parameter combinations"""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
            
        return combinations
        
    def search(self, model_factory, train_func, eval_func):
        """Perform grid search"""
        combinations = self.generate_combinations()
        
        for i, params in enumerate(combinations):
            print(f"Testing combination {i+1}/{len(combinations)}: {params}")
            
            # Create model with parameters
            model = model_factory(params)
            
            # Train model
            train_result = train_func(model, params)
            
            # Evaluate model
            eval_result = eval_func(model)
            
            # Store results
            result = {
                'params': params,
                'train_result': train_result,
                'eval_result': eval_result,
                'combination_id': i
            }
            self.results.append(result)
            
        return self.results
        
    def get_best_result(self, metric='accuracy'):
        """Get best result based on metric"""
        if not self.results:
            return None
            
        best_result = max(self.results, 
                         key=lambda x: x['eval_result'][metric])
        return best_result
```

### Stratified Grid Search

```python
class StratifiedGridSearch:
    def __init__(self, param_grid, n_trials=100):
        self.param_grid = param_grid
        self.n_trials = n_trials
        self.results = []
        
    def sample_parameters(self):
        """Sample parameters from grid"""
        import random
        
        sampled_params = {}
        for param_name, param_values in self.param_grid.items():
            sampled_params[param_name] = random.choice(param_values)
            
        return sampled_params
        
    def search(self, model_factory, train_func, eval_func):
        """Perform stratified grid search"""
        
        for trial in range(self.n_trials):
            params = self.sample_parameters()
            print(f"Trial {trial+1}/{self.n_trials}: {params}")
            
            # Create and train model
            model = model_factory(params)
            train_result = train_func(model, params)
            eval_result = eval_func(model)
            
            # Store results
            result = {
                'trial': trial,
                'params': params,
                'train_result': train_result,
                'eval_result': eval_result
            }
            self.results.append(result)
            
        return self.results
```

## Bayesian Optimization

### Basic Bayesian Optimization

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np

class BayesianOptimizer:
    def __init__(self, param_bounds, n_initial_points=5, n_iterations=50):
        self.param_bounds = param_bounds
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        
        # Initialize Gaussian Process
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=10,
            random_state=42
        )
        
        self.X = []  # Parameter combinations
        self.y = []  # Objective values
        
    def sample_random_point(self):
        """Sample random point from parameter space"""
        point = {}
        for param_name, (low, high) in self.param_bounds.items():
            if isinstance(low, int) and isinstance(high, int):
                point[param_name] = np.random.randint(low, high + 1)
            else:
                point[param_name] = np.random.uniform(low, high)
        return point
        
    def acquisition_function(self, X_candidates):
        """Expected Improvement acquisition function"""
        if len(self.X) == 0:
            return np.random.random(len(X_candidates))
            
        # Predict mean and std for candidates
        y_pred, std_pred = self.gp.predict(X_candidates, return_std=True)
        
        # Current best
        y_best = max(self.y)
        
        # Expected Improvement
        ei = (y_pred - y_best) / std_pred
        ei = ei * std_pred
        
        return ei
        
    def optimize(self, objective_function):
        """Perform Bayesian optimization"""
        
        # Initial random points
        for i in range(self.n_initial_points):
            params = self.sample_random_point()
            score = objective_function(params)
            
            self.X.append(list(params.values()))
            self.y.append(score)
            
            print(f"Initial point {i+1}: {params} -> {score}")
            
        # Bayesian optimization loop
        for iteration in range(self.n_iterations):
            # Fit GP to current data
            X_array = np.array(self.X)
            y_array = np.array(self.y)
            self.gp.fit(X_array, y_array)
            
            # Generate candidate points
            candidates = []
            for _ in range(100):
                candidate = self.sample_random_point()
                candidates.append(list(candidate.values()))
            candidates = np.array(candidates)
            
            # Select next point using acquisition function
            ei_values = self.acquisition_function(candidates)
            next_idx = np.argmax(ei_values)
            next_params = candidates[next_idx]
            
            # Convert back to dict
            param_names = list(self.param_bounds.keys())
            next_params_dict = dict(zip(param_names, next_params))
            
            # Evaluate objective
            score = objective_function(next_params_dict)
            
            # Update data
            self.X.append(next_params)
            self.y.append(score)
            
            print(f"Iteration {iteration+1}: {next_params_dict} -> {score}")
            
        return self.get_best_result()
        
    def get_best_result(self):
        """Get best result"""
        best_idx = np.argmax(self.y)
        param_names = list(self.param_bounds.keys())
        best_params = dict(zip(param_names, self.X[best_idx]))
        
        return {
            'best_params': best_params,
            'best_score': self.y[best_idx],
            'all_results': list(zip(self.X, self.y))
        }
```

## Hyperparameter Optimization with Optuna

### Optuna Integration

```python
import optuna
from optuna.samplers import TPESampler

class OptunaOptimizer:
    def __init__(self, n_trials=100):
        self.n_trials = n_trials
        self.study = None
        
    def objective(self, trial):
        """Objective function for Optuna"""
        
        # Define hyperparameters
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'warmup_steps': trial.suggest_int('warmup_steps', 100, 2000),
            'gradient_clip_norm': trial.suggest_float('gradient_clip_norm', 0.1, 2.0)
        }
        
        # Create and train model
        model = self.create_model(params)
        train_result = self.train_model(model, params)
        eval_result = self.evaluate_model(model)
        
        return eval_result['accuracy']
        
    def optimize(self):
        """Run optimization"""
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'study': self.study
        }
```

## Multi-Objective Optimization

### Pareto Optimization

```python
class MultiObjectiveOptimizer:
    def __init__(self, objectives):
        self.objectives = objectives
        self.results = []
        
    def evaluate_objectives(self, params):
        """Evaluate all objectives for given parameters"""
        model = self.create_model(params)
        train_result = self.train_model(model, params)
        eval_result = self.evaluate_model(model)
        
        objective_values = {}
        for obj_name, obj_func in self.objectives.items():
            objective_values[obj_name] = obj_func(eval_result)
            
        return objective_values
        
    def is_pareto_dominant(self, result1, result2):
        """Check if result1 dominates result2"""
        at_least_one_better = False
        
        for obj_name in self.objectives.keys():
            val1 = result1['objectives'][obj_name]
            val2 = result2['objectives'][obj_name]
            
            if val1 < val2:  # Assuming minimization
                return False
            elif val1 > val2:
                at_least_one_better = True
                
        return at_least_one_better
        
    def find_pareto_front(self, results):
        """Find Pareto front from results"""
        pareto_front = []
        
        for result in results:
            is_dominated = False
            
            for other_result in results:
                if result != other_result:
                    if self.is_pareto_dominant(other_result, result):
                        is_dominated = True
                        break
                        
            if not is_dominated:
                pareto_front.append(result)
                
        return pareto_front
```

## NeMo RL Integration

### Configuration-Based Optimization

```python
# config.yaml
hyperparameter_optimization:
  method: "optuna"
  n_trials: 100
  objectives:
    - "accuracy"
    - "training_time"
  
  parameters:
    learning_rate:
      type: "float"
      bounds: [1e-5, 1e-3]
      log_scale: true
    batch_size:
      type: "categorical"
      values: [16, 32, 64, 128]
    weight_decay:
      type: "float"
      bounds: [1e-6, 1e-3]
      log_scale: true
```

### Custom Optimizer Implementation

```python
from nemo_rl.utils.optimizer_utils import HyperparameterOptimizer

class CustomHyperparameterOptimizer(HyperparameterOptimizer):
    def __init__(self, config):
        super().__init__(config)
        self.optimization_method = config.get('method', 'grid_search')
        self.n_trials = config.get('n_trials', 100)
        
    def optimize(self, model_factory, train_func, eval_func):
        """Run hyperparameter optimization"""
        
        if self.optimization_method == 'grid_search':
            return self.grid_search(model_factory, train_func, eval_func)
        elif self.optimization_method == 'bayesian':
            return self.bayesian_optimization(model_factory, train_func, eval_func)
        elif self.optimization_method == 'optuna':
            return self.optuna_optimization(model_factory, train_func, eval_func)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
            
    def grid_search(self, model_factory, train_func, eval_func):
        """Grid search optimization"""
        grid_search = GridSearch(self.param_grid)
        results = grid_search.search(model_factory, train_func, eval_func)
        return grid_search.get_best_result()
        
    def bayesian_optimization(self, model_factory, train_func, eval_func):
        """Bayesian optimization"""
        optimizer = BayesianOptimizer(self.param_bounds, n_iterations=self.n_trials)
        
        def objective_function(params):
            model = model_factory(params)
            train_result = train_func(model, params)
            eval_result = eval_func(model)
            return eval_result['accuracy']
            
        return optimizer.optimize(objective_function)
```

## Best Practices

### 1. Start Simple
- Begin with grid search for small parameter spaces
- Use random search for larger spaces
- Graduate to Bayesian optimization for complex spaces

### 2. Define Clear Objectives
- Choose appropriate evaluation metrics
- Consider multiple objectives when needed
- Balance performance vs training time

### 3. Set Appropriate Bounds
- Use log-scale for learning rates
- Set realistic bounds based on domain knowledge
- Avoid overly wide parameter ranges

### 4. Monitor Progress
- Track optimization progress
- Visualize results
- Stop early if no improvement

## Common Patterns

### Parameter Space Definition

```python
def define_parameter_space():
    """Define parameter space for optimization"""
    
    # For grid search
    param_grid = {
        'learning_rate': [1e-5, 1e-4, 1e-3],
        'batch_size': [16, 32, 64],
        'weight_decay': [1e-6, 1e-5, 1e-4]
    }
    
    # For Bayesian optimization
    param_bounds = {
        'learning_rate': (1e-5, 1e-3),
        'batch_size': (16, 128),
        'weight_decay': (1e-6, 1e-4)
    }
    
    return param_grid, param_bounds
```

### Result Analysis

```python
class OptimizationAnalyzer:
    def __init__(self, results):
        self.results = results
        
    def plot_optimization_history(self):
        """Plot optimization history"""
        import matplotlib.pyplot as plt
        
        scores = [result['eval_result']['accuracy'] for result in self.results]
        iterations = range(len(scores))
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, scores)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Optimization History')
        plt.grid(True)
        plt.show()
        
    def analyze_parameter_importance(self):
        """Analyze parameter importance"""
        import pandas as pd
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Analyze correlations
        correlations = df.corr()
        
        return correlations
```

## Troubleshooting

### Common Issues

1. **Slow Optimization**: Reduce parameter space, use fewer trials
2. **Poor Results**: Check parameter bounds, improve objective function
3. **Overfitting**: Use cross-validation, multiple evaluation metrics
4. **Inconsistent Results**: Set random seeds, increase trial count

### Debugging Tips

```python
def debug_optimization(optimizer, objective_function):
    """Debug optimization process"""
    
    # Test objective function
    test_params = optimizer.sample_random_point()
    test_score = objective_function(test_params)
    
    print(f"Test parameters: {test_params}")
    print(f"Test score: {test_score}")
    
    # Check parameter bounds
    for param_name, bounds in optimizer.param_bounds.items():
        print(f"{param_name}: {bounds}")
        
    # Run short optimization
    results = optimizer.optimize(objective_function, n_iterations=5)
    print(f"Short optimization results: {results}")
```

## Get Help

- [Troubleshooting](../troubleshooting.md) - Common issues and solutions 