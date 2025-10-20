# Fast-dLLM Generation Algorithms

LLaDA generation algorithms using Fast-dLLM optimizations from the `3rdparty/Fast-dLLM` submodule.

## Structure

```
fast_dllm/
├── _imports.py         # Centralized Fast-dLLM imports (handles sys.path)
├── base.py             # FastDLLMGeneration base class
├── basic.py            # Basic generation (no cache)
├── prefix_cache.py     # Prefix cache generation  
└── dual_cache.py       # Dual cache generation
```

**Hierarchy**: `GenerationAlgorithm` → `FastDLLMGeneration` → `{Basic, PrefixCache, DualCache}Generation`

## Key Features

- **Shared Base Class**: `FastDLLMGeneration` handles model loading with `LLaDAModelLM`
- **Centralized Imports**: `_imports.py` manages Fast-dLLM submodule imports
- **Graceful Fallback**: Falls back to `AutoModel` if Fast-dLLM unavailable
- **No Duplication**: Model loading code in one place

## Adding New Algorithms

```python
# my_algorithm.py
from .base import FastDLLMGeneration
from ._imports import my_generate_function, FAST_DLLM_AVAILABLE

class MyAlgorithm(FastDLLMGeneration):
    def __init__(self):
        super().__init__(name="my_algo", description="My algorithm")
    
    def generate(self, model, prompt, **kwargs):
        validated_args = self.validate_args(**kwargs)
        return my_generate_function(model, prompt, **validated_args)
    
    def is_available(self):
        return FAST_DLLM_AVAILABLE and my_generate_function is not None
```

Then register in `__init__.py`.

See parent README for complete documentation.

