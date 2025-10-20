# dInfer Generation Algorithms

High-performance LLaDA generation algorithms using dInfer (10x+ faster than Fast-dLLM).

## Structure

```
dinfer/
├── _imports.py         # Centralized dInfer imports (handles sys.path)
├── base.py             # DInferGeneration base class
├── blockwise.py        # BlockWise with threshold decoder (recommended)
├── hierarchy.py        # BlockWise with hierarchical decoder
└── credit.py           # BlockWise with credit threshold decoder
```

**Hierarchy**: `GenerationAlgorithm` → `DInferGeneration` → `{BlockWise, Hierarchy, Credit}Generation`

## Available Algorithms

| Algorithm | Decoder | Cache | Best For |
|-----------|---------|-------|----------|
| `dinfer_blockwise` | Threshold | Dual | General purpose (recommended) |
| `dinfer_hierarchy` | Hierarchical | Dual | Enhanced parallel decoding |
| `dinfer_credit` | Credit Threshold | Dual | Credit-based parallel strategy |

## Key Features

- **10x+ Speedup**: Over Fast-dLLM with maintained accuracy
- **Left-Padding**: Optimized batch processing with left-padding
- **Advanced Decoders**: Multiple parallel decoding strategies
- **KV-Cache Management**: Dual cache with vicinity refresh
- **Early Stopping**: Efficient termination

## Usage

```python
from llada_generate import get_algorithm

# Get dInfer algorithm
algorithm = get_algorithm("dinfer_blockwise")
algorithm.load_model_from_hf("GSAI-ML/LLaDA-8B-Instruct")

# Generate (batching handled automatically)
output, nfe = algorithm.generate(
    model=algorithm.model,
    prompt=input_ids,
    gen_length=256,
    block_length=64
)
```

See parent README for complete documentation.

