# Generation Interface

This document explains the token generation interface and various backends for the NeMo RL framework. The generation system is designed with a unified interface that allows different backends (like VLLM, Hugging Face, SGLang, and TRT-LLM) to provide token generation capabilities while adhering to the same API.

## Generation Interface

The core of the generation system is defined in `interfaces.py`, which establishes an abstract interface that all generation backends must implement. This ensures consistency across different implementations and makes it easy to swap backends without changing the calling code.

### Key Components

1. **GenerationConfig**: A TypedDict that defines the configuration for generation:
   ```python
   class GenerationConfig(TypedDict):
       """Configuration for generation."""
       backend: str              # The backend to use (e.g., "vllm", "hf")
       max_new_tokens: int       # Maximum number of tokens to generate
       temperature: float        # Sampling temperature
       top_p: float              # Top-p sampling parameter
       top_k: int                # Top-k sampling parameter
       model_name: str           # Name or path of the model
   ```

2. **GenerationDatumSpec**: A TypedDict that defines the input data format:
   ```python
   class GenerationDatumSpec(TypedDict):
       input_ids: torch.Tensor         # Input token IDs
       attention_mask: torch.Tensor    # Attention mask
       __extra__: Any                  # Additional data specific to the backend
   ```

3. **GenerationOutputSpec**: A TypedDict that defines output data format:
   ```python
   class GenerationOutputSpec(TypedDict):
       output_ids: torch.Tensor
       generation_lengths: torch.Tensor  # Length of just the generated response part
       unpadded_sequence_lengths: torch.Tensor  # Length of full valid sequence (input + generated response)
       logprobs: torch.Tensor
       __extra__: Any                  # Additional output data specific to the backend
   ```

4. **GenerationInterface**: An abstract base class that all generation backends must implement:
   ```python
   class GenerationInterface(ABC):
       """Abstract base class defining the interface for RL policies."""

       @abstractmethod
       def generate(
           self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool
       ) -> BatchedDataDict["GenerationOutputSpec"]:
           pass

       @abstractmethod
       def prepare_for_generation(self, *args, **kwargs):
           pass

       @abstractmethod
       def finish_generation(self, *args, **kwargs):
           pass
   ```

A key design principle for generation backends is that they process tokens directly, without involving the tokenizer. By ensuring that only tokens are exchanged, we eliminate the risk of inconsistencies arising from different tokenizer versions or specifications between the training and generation frameworks.

## VLLM Backend

The VLLM backend (`models/generation/vllm.py`) implements the {py:class}`GenerationInterface <nemo_rl.models.generation.interfaces.GenerationInterface>` to provide efficient text generation using the VLLM library, which is optimized for large language models.

### VllmGeneration Class

The {py:class}`VllmGeneration <nemo_rl.models.generation.vllm.VllmGeneration>` class is the main implementation of the {py:class}`GenerationInterface <nemo_rl.models.generation.interfaces.GenerationInterface>` for VLLM. It performs the following functions:

1. Sets up VLLM workers in a distributed environment using Ray.
2. Manages the lifecycle of these workers (initialization, generation, shutdown).
3. Distributes inputs to workers and collects outputs.
4. Handles weight updates and synchronization.

### VllmGenerationWorker

The {py:class}`VllmGenerationWorker <nemo_rl.models.generation.vllm.VllmGenerationWorker>` is a Ray actor that:

1. Initializes and manages a VLLM model instance.
2. Performs the actual generation on a GPU.
3. Supports dynamic weight updates through IPC handles.
4. Implements sleep/wake mechanisms for efficient resource utilization.

### Custom VLLM Extensions

The {py:class}`UpdatableVllmInternalWorker <nemo_rl.models.generation.vllm_backend.UpdatableVllmInternalWorker>` class in `vllm_backend.py` extends the VLLM worker with additional capabilities:

1. Reporting device IDs to allow mapping of workers to specific GPUs.
2. Updating weights from IPC handles for efficient weight sharing.
3. Checking if weights have been updated correctly.

## Usage Example

To use a generation backend:

```python
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration, VllmConfig

# Set up the configuration
config = VllmConfig(
    model_name="Qwen/Qwen2.5-1.5B",
    max_new_tokens=100,
    temperature=0.7,
    top_p=1,
    top_k=None,
    backend="vllm",
    vllm_cfg={
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 2048,
    }
)

# Configure config with tokenizer
tokenizer = get_tokenizer(config["model_name"])
config = configure_generation_config(config, tokenizer)

# Initialize the cluster and generation backend
cluster = RayVirtualCluster(...)
generator = VllmGeneration(cluster, config)

# Prepare input data
input_data = BatchedDataDict(...)

# Generate text
generator.prepare_for_generation()
output = generator.generate(input_data, greedy=False)
generator.finish_generation()
```

## Extend with New Backends

To add a new generation backend:

1. Create a new class that implements {py:class}`GenerationInterface <nemo_rl.models.generation.interfaces.GenerationInterface>`.
2. Implement the required methods: {py:meth}`generate <nemo_rl.models.generation.interfaces.GenerationInterface.generate>`, {py:meth}`prepare_for_generation <nemo_rl.models.generation.interfaces.GenerationInterface.prepare_for_generation>`, and {py:meth}`finish_generation <nemo_rl.models.generation.interfaces.GenerationInterface.finish_generation>`.
3. Ensure your implementation works with the standard {py:class}`GenerationConfig <nemo_rl.models.generation.interfaces.GenerationConfig>` and {py:class}`GenerationDatumSpec <nemo_rl.models.generation.interfaces.GenerationDatumSpec>` structures.
4. Register your backend with the system (if needed) to make it accessible.

This modular design allows for easy extension with new backends while maintaining a consistent interface for the rest of the system.
