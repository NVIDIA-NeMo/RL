# Generation Interface

This document explains the token generation interface and various backends for the NeMo RL framework. The generation system is designed with a unified interface that allows different backends (like VLLM, Megatron, Hugging Face, SGLang, and TRT-LLM) to provide token generation capabilities while adhering to the same API.

## Generation Interface

The core of the generation system is defined in `interfaces.py`, which establishes an abstract interface that all generation backends must implement. This ensures consistency across different implementations and makes it easy to swap backends without changing the calling code.

### Key Components

1. **GenerationConfig**: A TypedDict that defines the configuration for generation:
   ```python
   class GenerationConfig(TypedDict):
       """Configuration for generation."""
       backend: str              # The backend to use (e.g., "vllm", "megatron", "hf")
       max_new_tokens: int       # Maximum number of tokens to generate
       temperature: float        # Sampling temperature
       top_p: float              # Top-p sampling parameter
       top_k: int | None         # Top-k sampling parameter
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

## Generation Backends

NeMo RL supports multiple generation backends that implement the {py:class}`GenerationInterface <nemo_rl.models.generation.interfaces.GenerationInterface>` to provide efficient text generation for different use cases.

## VLLM Backend

The VLLM backend (`models/generation/vllm/vllm_generation.py`) implements the {py:class}`GenerationInterface <nemo_rl.models.generation.interfaces.GenerationInterface>` to provide efficient text generation using the VLLM library, which is optimized for large language models.

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

## Megatron Backend

The Megatron backend provides native Megatron-Core inference capabilities, eliminating the need for weight conversion between training and generation. This backend is particularly beneficial when using Megatron for training, as it enables seamless integration and optimal performance.

### Key Features

1. **No Weight Conversion**: Uses the same Megatron model format for both training and generation, eliminating conversion overhead and potential inconsistencies.
2. **CUDA Graph Support**: Leverages CUDA graphs for optimized inference performance.
3. **Dynamic Inference Engine**: Utilizes Megatron Core's `DynamicInferenceEngine` for efficient batched generation.
4. **Integrated with Training**: The generation capability is built directly into the `MegatronPolicyWorker`, enabling efficient co-located training and generation.

### MegatronPolicyWorker Generation

The Megatron generation backend is implemented within the {py:class}`MegatronPolicyWorker <nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker>` class. The `generate <nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.generate>` method performs the following:

1. Wraps the Megatron model with `GPTInferenceWrapper` for inference optimization.
2. Creates a `DynamicInferenceContext` to manage inference state and memory.
3. Initializes a `DynamicInferenceEngine` with CUDA graph support enabled.
4. Processes batched requests with proper sampling parameters (temperature, top_k, top_p).
5. Returns outputs conforming to {py:class}`GenerationOutputSpec <nemo_rl.models.generation.interfaces.GenerationOutputSpec>`.

### Configuration

To use the Megatron generation backend, configure your YAML file as follows:

```yaml
policy:
  megatron_cfg:
    enabled: true
  generation:
    backend: megatron
    max_new_tokens: 512
    temperature: 1.0
    top_p: 1.0
    top_k: null
    mcore_generation_config:
      buffer_size_gb: 10               # Memory buffer size for requests, total buffer size is 2x this value (active requests + paused requests)
      num_cuda_graphs: 16              # Number of CUDA graphs to pre-allocate
      max_tokens: 16384                # Maximum number of tokens for inference
      cuda_graph_sizing_distribution: hybrid
      cuda_graph_max_tokens: 512
      prefix_caching_mamba_gb: null
      prefix_caching_eviction_policy: lru
      prefix_caching_coordinator_policy: longest_prefix
      prefix_cache_ttl_seconds: 300.0
      prefix_caching_routing_alpha: 1.0
      http_server_num_replicas: 8
```

### Configuration Parameters

The `mcore_generation_config` section controls Megatron Core inference engine behavior:

- **buffer_size_gb**: Buffer size reserved for active requests that live on the GPU. The total buffer size (stored in unified memory) is 2x this value, with the other half of the buffer reserved for paused requests that live on the CPU.
- **num_cuda_graphs**: Number of CUDA graphs to pre-allocate for different batch sizes. More graphs can improve performance by avoiding runtime graph capture, but consume more memory.
- **max_tokens**: Maximum total number of tokens (across all requests) that can be processed simultaneously. This limits the maximum batch size and sequence length combinations. Increasing this might throw OOM depending on vocab size and buffer size allocated. 

#### CUDA-graph capture

- **cuda_graph_sizing_distribution** (`hybrid`) and **cuda_graph_max_tokens** (`512`) bound graph-capture cost while keeping decode graphs dense and prefill graphs compact. Use `exponential` or `linear` when one layout is a better fit for a known workload.
- **inference_cuda_graph_scope** controls where local inference graphs live: `layer` owns them per Transformer/Mamba layer and `block` owns them per enclosing block. `none` runs eagerly. This setting only applies with `cuda_graph_impl: local`.

#### Prefix-cache routing and HTTP frontends

- **prefix_caching_mamba_gb** (`null`) leaves Mamba-state prefix caching disabled until a hybrid Mamba model has an explicit GPU budget.
- **prefix_caching_eviction_policy** (`lru`), **prefix_caching_coordinator_policy** (`longest_prefix`), **prefix_cache_ttl_seconds** (`300.0`), and **prefix_caching_routing_alpha** (`1.0`) retain useful prefixes while steering requests toward cache hits without ignoring an overloaded inference rank.
- **http_server_num_replicas** (`8`) sets the number of CPU HTTP frontends per model-parallel coordinator. These frontends spread request parsing, prompt processing, and cache-key work across CPU cores.

### Multimodal Megatron Generation

Megatron inference supports image and video inputs in NeMo-RL. Enable multimodal processing with `policy.is_vlm: true`, use the `megatron` generation backend, and provide a `megatron_inference_wrapper`. The wrapper must subclass `megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper.AbstractModelInferenceWrapper` in Megatron-Core and declare `supports_<modality> = True` for each supported modality.

```yaml
policy:
  is_vlm: true
  generation:
    backend: megatron
    mcore_generation_config:
      megatron_inference_wrapper: megatron.core.inference.model_inference_wrappers.multimodal.nemotron_omni_inference_wrapper.NemotronOmniInferenceWrapper
      image_dynamic_resolution: true
      video_num_frames: 16
      video_temporal_patch_size: 2
      video_target_num_patches: 2048
      video_maintain_aspect_ratio: true
      vision_embedding_cache_max_bytes: 0
      allow_stale_multimodal_embeddings: false
data:
  default:
    num_frames: 16
    video_temporal_patch_size: 2
    video_target_num_patches: 2048
    video_maintain_aspect_ratio: true
```

- `image_dynamic_resolution` preserves variable image shapes instead of forcing one fixed resolution; for example, a wide image uses a wider patch grid than a square image.
- `vision_model_type` optionally selects the MCore vision encoder type used by image and video preprocessing. Set it to the encoder expected by the inference wrapper; when omitted, MCore uses its default (`radio`).
- `num_frames` controls uniform video-frame sampling. Use `video_num_frames` for the corresponding MCore key.
- `video_temporal_patch_size` groups sampled frames into temporal tubelets; for example, size `2` turns 16 frames into 8 temporal groups.
- `video_target_num_patches` sets `num_patches_per_frame = patch_height * patch_width <= video_target_num_patches`, which produces `num_patches_per_frame * num_frames / video_temporal_patch_size` total video patches prior to spatial merging (i.e. further grouped / concatenated into MxM patch blocks) that are provided to the vision encoder.
- `video_maintain_aspect_ratio=true` keeps `patch_width / patch_height ~= source_width / source_height`; `false` uses `patch_width = patch_height ~= sqrt(video_target_num_patches)` (for example, `sqrt(256) = 16`).
- `vision_embedding_cache_max_bytes` limits GPU memory used to reuse vision embeddings for repeated media; `0` disables the cache, while `1073741824` permits up to 1 GiB.
- `allow_stale_multimodal_embeddings` controls whether cached embeddings survive model-weight changes. Keep it `false` for RL refits; use `true` only when weights remain fixed.
- `expose_http_server` should be `true` for NeMo Gym.

Keep the video preprocessing values identical in `data.default` and `mcore_generation_config` to avoid disparity between the training policy and inference generation.

#### Multimodal prompt contracts

`multimodal_prompt_config` controls how structured image and video content is
lowered into model prompt tokens. It contains optional `image_spec` and
`video_spec` mappings. Each mapping may override the Megatron inference wrapper's
defaults:

```yaml
policy:
  generation:
    mcore_generation_config:
      megatron_inference_wrapper: path.to.MultimodalInferenceWrapper
      multimodal_prompt_config:
        image_spec:
          model_token: "<image>"
          prefix: ""
          suffix: ""
          input_marker: null
          content_part_separator: ""
          expansion_mode: single
        video_spec:
          model_token: "<video>"
          content_part_separator: "\n"
          expansion_mode: temporal_patch
          include_frame_timestamps_for_nemotron_vl: true
```

The mappings are partial overrides: omitted values come from the selected
wrapper's `multimodal_prompt_config`. Consequently,
`multimodal_prompt_config` requires a `megatron_inference_wrapper`, and that
wrapper must define a default prompt contract.

- `model_token` is the model-specific placeholder that is later replaced by
  multimodal embeddings.
  - Example: `model_token: "<image>"` lowers one structured image block to
    `<image>` before tokenization.
- `prefix` and `suffix` wrap each model placeholder.
  - Example: `prefix: "<img>"`, `model_token: "<image>"`, and
    `suffix: "</img>"` produce `<img><image></img>`.
- `content_part_separator` joins adjacent structured content parts before the
  chat template is applied.
  - Example: Given a text part `Describe this` followed by an image block,
    `content_part_separator: "\n"` produces
    `Describe this\n<img><image></img>` instead of
    `Describe this<img><image></img>`.
- `input_marker` removes a marker already present in incoming text when the
  same message also contains structured media. Current built-in wrappers leave
  this setting unset.
  - Example: For a custom contract with `input_marker: "<video>"`, incoming
    text `Describe this: <video>` plus a structured video block becomes
    `Describe this: ` plus the configured video prompt, rather than retaining
    two video markers.
- `expansion_mode` controls how compact media placeholders are expanded.
  - `single` emits one compact model placeholder per structured media item.
    - Example: A video block initially becomes one `<img><image></img>` span;
      the `<image>` token is expanded internally to the number of projected
      video features.
  - `temporal_patch` rewrites one compact video placeholder into one
    placeholder per temporal tubelet before feature expansion.
    - Example: Four frames with temporal patch size `2` form two tubelets, so
      the compact video span is rewritten as
      `<img><image></img> <img><image></img>`: one span for frames 1–2 and one
      for frames 3–4.
- `include_frame_timestamps_for_nemotron_vl: true` adds the sampled-frame times
  to each temporal-patch span. This option is valid only with
  `expansion_mode: temporal_patch`.
  - Example: For frame indices `[0, 1, 2, 3]`, FPS `1`, and temporal patch size
    `2`, the prompt fragment becomes:

    ```text
    Frame 1 sampled at 0.00 seconds and frame 2 sampled at 1.00 seconds: <img><image></img>
    Frame 3 sampled at 2.00 seconds and frame 4 sampled at 3.00 seconds: <img><image></img>
    ```

Omitting `data.default.video_sampling_style` keeps the default/simple video
path. To opt into Nemotron-VL sampling, set the style explicitly:

```yaml
data:
  default:
    video_sampling_style: nemotron_vl
```

NeMo-RL then automatically materializes the required video prompt contract:

```yaml
multimodal_prompt_config:
  video_spec:
    content_part_separator: "\n"
    expansion_mode: temporal_patch
    include_frame_timestamps_for_nemotron_vl: true
```

The current engine has one prompt contract for all requests. Supporting
different prompt styles in one engine would require a future per-request
endpoint or engine configuration.

## Usage Examples

### Using VLLM Backend

To use the VLLM generation backend:

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

### Using Megatron Backend

To use the Megatron generation backend, configure your YAML file:

```yaml
policy:
  model_name: meta-llama/Llama-3.2-1B-Instruct
  megatron_cfg:
    enabled: true
  generation:
    backend: megatron
    max_new_tokens: 512
    temperature: 1.0
    top_p: 1.0
    top_k: null
    mcore_generation_config:
      buffer_size_gb: 10
      num_cuda_graphs: 16
      max_tokens: 16384
      cuda_graph_sizing_distribution: hybrid
      cuda_graph_max_tokens: 512
      prefix_caching_mamba_gb: null
      prefix_caching_eviction_policy: lru
      prefix_caching_coordinator_policy: longest_prefix
      prefix_cache_ttl_seconds: 300.0
      prefix_caching_routing_alpha: 1.0
      http_server_num_replicas: 8
```

For a complete example, see
`examples/configs/recipes/llm/grpo-llama3.2-1b-instruct-1n8g-megatron_generation.yaml`.

## Extend with New Backends

To add a new generation backend:

1. Create a new class that implements {py:class}`GenerationInterface <nemo_rl.models.generation.interfaces.GenerationInterface>`.
2. Implement the required methods: {py:meth}`generate <nemo_rl.models.generation.interfaces.GenerationInterface.generate>`, {py:meth}`prepare_for_generation <nemo_rl.models.generation.interfaces.GenerationInterface.prepare_for_generation>`, and {py:meth}`finish_generation <nemo_rl.models.generation.interfaces.GenerationInterface.finish_generation>`.
3. Ensure your implementation works with the standard {py:class}`GenerationConfig <nemo_rl.models.generation.interfaces.GenerationConfig>` and {py:class}`GenerationDatumSpec <nemo_rl.models.generation.interfaces.GenerationDatumSpec>` structures.
4. Register your backend with the system (if needed) to make it accessible.

This modular design allows for easy extension with new backends while maintaining a consistent interface for the rest of the system.
