# LLaDA Generation Registry

This module provides a modular generation system for LLaDA models with a registry-based approach for managing different generation algorithms.

## Overview

The generation registry allows you to:
- Register different generation algorithms with descriptive names
- Use aliases for common algorithm names
- Easily switch between algorithms at runtime
- Maintain backward compatibility with legacy cache-based configuration

## Architecture

### Base Classes

- `GenerationAlgorithm`: Abstract base class for all generation algorithms
- `GenerationRegistry`: Registry system for managing algorithms

### Built-in Algorithms

| Algorithm | Name | Aliases | Description |
|-----------|------|---------|-------------|
| Basic | `basic` | `no_cache`, `simple` | Basic LLaDA generation without caching |
| Prefix Cache | `prefix_cache` | `prefix`, `cache` | Generation with prefix caching for efficiency |
| Dual Cache | `dual_cache` | `dual`, `double_cache` | Generation with dual caching for maximum performance |

## Usage

### Server Usage

#### Per-Request Algorithm Selection (Recommended)
```bash
# Start the server (no algorithm specified - algorithms chosen per request)
python llada_batch_server.py --model-path /path/to/model

# Send requests with different algorithms
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llada-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "generation_algorithm": "basic"
  }'

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llada-8b-instruct", 
    "messages": [{"role": "user", "content": "Hello!"}],
    "generation_algorithm": "prefix_cache"
  }'
```

#### Legacy Cache Parameters (Still Supported)
```bash
# Server startup remains the same
python llada_batch_server.py --model-path /path/to/model

# Requests using legacy cache parameters
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llada-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "use_cache": true,
    "use_dual_cache": true
  }'
```

### Programmatic Usage

```python
from llada_generate import get_algorithm, list_available_algorithms

# Get an algorithm
algorithm = get_algorithm("prefix_cache")
if algorithm and algorithm.is_available():
    output, nfe = algorithm.generate(
        model=model,
        prompt=prompt_tensor,
        steps=16,
        gen_length=128,
        block_length=32,
        temperature=1.0,
        remasking=True,
        threshold=0.95,
        factor=1.0
    )

# List available algorithms
available = list_available_algorithms()
print(f"Available algorithms: {available}")
```

### API Endpoints

The batch server provides new endpoints for algorithm information:

- `GET /generation/algorithms` - List all registered algorithms
- `GET /health` - Includes current algorithm information

## Creating Custom Algorithms

To create a custom generation algorithm:

1. Create a new Python file in the `llada_generate/` directory
2. Subclass `GenerationAlgorithm`
3. Implement the required methods
4. Register your algorithm

```python
# custom_algorithm.py
from .base import GenerationAlgorithm

class CustomGeneration(GenerationAlgorithm):
    def __init__(self):
        super().__init__(
            name="custom",
            description="My custom generation algorithm"
        )
    
    def generate(self, model, prompt, **kwargs):
        # Your generation logic here
        validated_args = self.validate_args(**kwargs)
        # ... implementation
        return output, nfe
    
    def is_available(self):
        # Check if dependencies are available
        return True

# Register the algorithm
from llada_generate import register_algorithm
register_algorithm(CustomGeneration(), aliases=['my_custom'])
```

## Migration Guide

### From Hardcoded Generation Logic

If you have hardcoded generation logic:

```python
# Old way
if use_cache:
    if use_dual_cache:
        output, nfe = generate_with_dual_cache(...)
    else:
        output, nfe = generate_with_prefix_cache(...)
else:
    output, nfe = generate(...)

# New way
algorithm = get_algorithm("prefix_cache")  # or any algorithm
output, nfe = algorithm.generate(...)
```

## Benefits

1. **Per-Request Flexibility**: Each request can use a different generation algorithm
2. **Modularity**: Each algorithm is self-contained and testable
3. **Extensibility**: Easy to add new algorithms without modifying core code
4. **Intelligent Batching**: Server groups requests by algorithm for efficient processing
5. **Maintainability**: Clear separation of concerns
6. **Dynamic Selection**: No need to restart server to change algorithms

## Dependencies

- Fast-dLLM (for generation functions)
- PyTorch
- Transformers

The registry gracefully handles missing dependencies and will mark algorithms as unavailable if their dependencies are not met.
