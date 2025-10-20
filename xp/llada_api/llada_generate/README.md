# LLaDA Generation Registry

This module provides a modular generation system for LLaDA models with a registry-based approach for managing different generation algorithms.

## Overview

The generation registry provides a unified interface for different inference engines and algorithms:
- **Engines**: Inference backends (fast-dllm, dinfer, nemotron) - determines model implementation
- **Algorithms**: Specific generation strategies within each engine
- **Auto-detection**: Automatically selects best engine based on model type
- **Per-request switching**: Change algorithms per request (within same engine)
- **Validation**: Ensures engine/model compatibility

## Architecture

### Module Structure

```
llada_generate/
├── base.py                 # GenerationAlgorithm base class (model loading, tokenization, batching)
├── fast_dllm/              # Fast-dLLM algorithms (inherit from FastDLLMGeneration)
│   ├── _imports.py         # Centralized Fast-dLLM imports from submodule
│   ├── base.py             # FastDLLMGeneration base (LLaDAModelLM loading)
│   ├── basic.py            # Basic generation (no cache)
│   ├── prefix_cache.py     # Prefix cache generation
│   └── dual_cache.py       # Dual cache generation
├── dinfer/                 # dInfer algorithms (inherit from DInferGeneration)
│   ├── _imports.py         # Centralized dInfer imports from submodule
│   ├── base.py             # DInferGeneration base (dInfer LLaDAModelLM + diffusion wrapper)
│   ├── blockwise.py        # BlockWise with threshold decoder
│   ├── hierarchy.py        # BlockWise with hierarchical decoder
│   └── credit.py           # BlockWise with credit decoder
├── nemotron.py             # Nemotron native generation
└── __init__.py             # Registry and public API
```

### Class Hierarchy

```
GenerationAlgorithm (base.py)
├── FastDLLMGeneration (fast_dllm/base.py)
│   ├── BasicGeneration
│   ├── PrefixCacheGeneration
│   └── DualCacheGeneration
├── DInferGeneration (dinfer/base.py)
│   ├── BlockWiseGeneration
│   ├── HierarchyGeneration
│   └── CreditGeneration
└── NemotronGeneration (nemotron.py)
```

### Engines & Algorithms

**Engines** determine the inference backend and model implementation:

| Engine | Model Type | Model Class | Algorithms | Performance |
|--------|------------|-------------|------------|-------------|
| `fast-dllm` | LLaDA | Fast-dLLM LLaDAModelLM | basic, prefix_cache, dual_cache | Optimized |
| `dinfer` | LLaDA | dInfer LLaDAModelLM | dinfer_blockwise, dinfer_hierarchy, dinfer_credit | 10x+ faster (recommended) |
| `nemotron` | Nemotron | AutoModel | nemotron | Native |

**Algorithms** are specific strategies within each engine:

| Algorithm | Engine | Description | Default |
|-----------|--------|-------------|---------|
| `basic` | fast-dllm | Basic without caching | |
| `prefix_cache` | fast-dllm | Prefix caching | |
| `dual_cache` | fast-dllm | Dual caching | ✓ |
| `dinfer_blockwise` | dinfer | Threshold decoder | ✓ |
| `dinfer_hierarchy` | dinfer | Hierarchical decoder | |
| `dinfer_credit` | dinfer | Credit threshold decoder | |
| `nemotron` | nemotron | Native Nemotron | ✓ |

**Engine Selection**:
- LLaDA models: Auto-selects `dinfer` (or use `--engine fast-dllm`)
- Nemotron models: Auto-selects `nemotron`
- Validation ensures engine/model compatibility

## Usage

### Server Usage

#### Simple (Auto-detected Engine)
```bash
# LLaDA - auto-selects dinfer engine
python llada_batch_server.py --model-path GSAI-ML/LLaDA-8B-Instruct

# Nemotron - auto-selects nemotron engine
python llada_batch_server.py --model-path nvidia/Nemotron-Diffusion-Research-4B-v0
```

#### Explicit Engine Selection
```bash
# LLaDA with Fast-dLLM engine
python llada_batch_server.py --model-path GSAI-ML/LLaDA-8B-Instruct --engine fast-dllm

# LLaDA with specific algorithm
python llada_batch_server.py --model-path GSAI-ML/LLaDA-8B-Instruct --engine dinfer --algorithm dinfer_hierarchy
```

#### Per-Request Algorithm Switching (within same engine)
```bash
# Server loaded with dinfer engine
python llada_batch_server.py --model-path GSAI-ML/LLaDA-8B-Instruct --engine dinfer

# Request 1: Use default (dinfer_blockwise)
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'

# Request 2: Switch to hierarchy decoder
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "generation_algorithm": "dinfer_hierarchy"
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

### Fast-dLLM Variant (Recommended if using Fast-dLLM)

Inherit from `FastDLLMGeneration` to reuse model loading:

```python
# fast_dllm/my_algorithm.py
from .base import FastDLLMGeneration
from ._imports import my_generate_function, FAST_DLLM_AVAILABLE

class MyAlgorithm(FastDLLMGeneration):
    def __init__(self):
        super().__init__(name="my_algo", description="My custom Fast-dLLM algorithm")
    
    def generate(self, model, prompt, **kwargs):
        validated_args = self.validate_args(**kwargs)
        return my_generate_function(model, prompt, **validated_args)
    
    def is_available(self):
        return FAST_DLLM_AVAILABLE and my_generate_function is not None
```

### Independent Algorithm

For non-Fast-dLLM algorithms, inherit from `GenerationAlgorithm`:

```python
# my_algorithm.py
from .base import GenerationAlgorithm
from transformers import AutoModel

class MyAlgorithm(GenerationAlgorithm):
    def __init__(self):
        super().__init__(name="my_algo", description="My custom algorithm")
    
    def load_model_class(self, model_path, **kwargs):
        return AutoModel.from_pretrained(model_path, **kwargs)
    
    def generate(self, model, prompt, **kwargs):
        validated_args = self.validate_args(**kwargs)
        # Your generation logic
        return output, nfe
    
    def is_available(self):
        return True  # or check dependencies
```

Then register in `__init__.py`:
```python
from .my_algorithm import MyAlgorithm
register_algorithm(MyAlgorithm(), aliases=['my_custom'])
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

## Key Features

### Algorithm-Based Model Loading
Each algorithm manages its own model loading:
- **Fast-dLLM algorithms**: Use optimized `LLaDAModelLM` class when available
- **Nemotron algorithm**: Uses standard `AutoModel`
- **Graceful fallback**: Missing Fast-dLLM → algorithms use `AutoModel`

### Centralized Tokenization & Batching
The base `GenerationAlgorithm` class provides:
- `tokenize_prompts()`: Chat template support, batching, padding
- `decode_outputs()`: Batch decoding with prompt exclusion
- `generate_batch()`: High-level API combining all steps

### Fast-dLLM Import System
Fast-dLLM algorithms use a centralized import module (`fast_dllm/_imports.py`) that:
- Automatically adds `3rdparty/Fast-dLLM/llada` to `sys.path`
- Imports all Fast-dLLM components in one place
- Gracefully degrades if Fast-dLLM submodule is unavailable

## Dependencies

- **Required**: PyTorch, Transformers
- **Optional**: Fast-dLLM (from `3rdparty/Fast-dLLM` submodule)
- **Optional**: NeMo-RL (for DCP checkpoint loading)

The registry gracefully handles missing dependencies and marks algorithms as unavailable if their dependencies are not met.
