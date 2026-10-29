"""Native vLLM FP32 head configuration and real projection numerics."""

import copy
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from nemo_rl.models.generation.vllm.config import (
    configure_fp32_lm_head,
    vllm_fp32_lm_head_enabled,
)


@pytest.mark.parametrize("enabled", [None, False, True])
def test_fp32_head_preserves_other_overrides(enabled: bool | None) -> None:
    cfg = {} if enabled is None else {"fp32_lm_head": enabled}
    overrides = {
        "rope_theta": 12345,
        "quantization_config": {"ignored_layers": ["lm_head"]},
    }
    kwargs = {"hf_overrides": overrides}
    original = copy.deepcopy(kwargs)
    configure_fp32_lm_head(cfg, kwargs)
    assert overrides == original["hf_overrides"]
    assert kwargs["hf_overrides"] == {
        **overrides,
        **({"head_dtype": "float32"} if enabled else {}),
    }
    configure_fp32_lm_head(cfg, kwargs)  # Idempotent.


@pytest.mark.parametrize("value", ["false", "true", "fp32", 0, 1, [], {}])
def test_fp32_head_rejects_non_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="must be true or false"):
        vllm_fp32_lm_head_enabled({"fp32_lm_head": value})


@pytest.mark.parametrize(
    "enabled,head_dtype", [(True, "bfloat16"), (True, "model"), (False, "float32")]
)
def test_fp32_head_rejects_conflicting_override(enabled: bool, head_dtype: str) -> None:
    with pytest.raises(ValueError, match="conflicts"):
        configure_fp32_lm_head(
            {"fp32_lm_head": enabled}, {"hf_overrides": {"head_dtype": head_dtype}}
        )


@pytest.mark.parametrize(
    "head_dtype", ["float32", "float", "fp32", "FLOAT32", "Float", torch.float32]
)
def test_fp32_head_accepts_matching_dtype_aliases(
    head_dtype: str | torch.dtype,
) -> None:
    kwargs = {"hf_overrides": {"head_dtype": head_dtype}}
    configure_fp32_lm_head({"fp32_lm_head": True}, kwargs)
    assert kwargs["hf_overrides"]["head_dtype"] == "float32"


@pytest.mark.parametrize("head_dtype", ["float32", "FLOAT32", "Float", torch.float32])
@pytest.mark.parametrize("enabled", [None, False])
def test_disabled_fp32_head_rejects_all_fp32_aliases(
    head_dtype: str | torch.dtype, enabled: bool | None
) -> None:
    cfg = {} if enabled is None else {"fp32_lm_head": enabled}
    with pytest.raises(ValueError, match="conflicts"):
        configure_fp32_lm_head(cfg, {"hf_overrides": {"head_dtype": head_dtype}})


@pytest.mark.parametrize("head_dtype", ["model", "bfloat16", "float16"])
def test_disabled_fp32_head_preserves_other_dtypes(head_dtype: str) -> None:
    kwargs = {"hf_overrides": {"head_dtype": head_dtype}}
    configure_fp32_lm_head({"fp32_lm_head": False}, kwargs)
    assert kwargs["hf_overrides"]["head_dtype"] == head_dtype


def test_fp32_head_rejects_legacy_environment_switch() -> None:
    with pytest.raises(ValueError, match="obsolete"):
        configure_fp32_lm_head({"env_vars": {"NRL_VLLM_FP32_LM_HEAD": "1"}}, {})


def test_fp32_head_preserves_native_lora() -> None:
    kwargs = {"enable_lora": True}
    configure_fp32_lm_head({"fp32_lm_head": True}, kwargs)
    assert kwargs == {"enable_lora": True, "hf_overrides": {"head_dtype": "float32"}}


@torch.inference_mode()
def _inspect_loaded_head(worker: Any) -> dict[str, str]:
    """Probe the live model without changing its configured projection dtype."""
    model = worker.model_runner.model
    weight = model.lm_head.weight
    hidden = torch.randn(2, weight.shape[1], device=weight.device, dtype=weight.dtype)
    logits = model.compute_logits(hidden)
    return {
        "head_dtype": str(model.logits_processor.head_dtype),
        "weight_dtype": str(weight.dtype),
        "logits_dtype": str(logits.dtype),
    }


@pytest.mark.vllm
@pytest.mark.timeout(900)
@pytest.mark.parametrize("enabled", [False, True])
def test_fp32_head_reaches_initialized_worker(
    tiny_qwen3_model_path: str, enabled: bool
) -> None:
    # Import the optional inference backend only in the vLLM test lane.
    from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

    worker = VllmGenerationWorkerImpl(
        {
            "model_name": tiny_qwen3_model_path,
            "vllm_cfg": {
                "precision": "bfloat16",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "expert_parallel_size": 1,
                "gpu_memory_utilization": 0.3,
                "max_model_len": 128,
                "async_engine": False,
                "skip_tokenizer_init": False,
                "load_format": "auto",
                "enforce_eager": True,
                "enable_prefix_caching": False,
                "fp32_lm_head": enabled,
            },
            "vllm_kwargs": {
                "max_num_seqs": 2,
                "max_num_batched_tokens": 128,
                "attention_config": {"backend": "FLASH_ATTN"},
            },
        },
        bundle_indices=[],
        seed=42,
    )
    try:
        expected = "torch.float32" if enabled else "torch.bfloat16"
        assert worker.llm.collective_rpc(_inspect_loaded_head) == [
            {
                "head_dtype": expected,
                "weight_dtype": "torch.bfloat16",
                "logits_dtype": expected,
            }
        ]
    finally:
        worker.shutdown()


@pytest.mark.vllm
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 64), (7, 64), (2, 3, 64)])
@torch.inference_mode()
def test_native_fp32_projection_preserves_live_weight(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    # Import the optional vLLM backend only in its own test lane.
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for torch.mm(out_dtype=float32)")
    torch.manual_seed(42)
    hidden = torch.randn(*shape[:-1], shape[-1] * 2, device="cuda", dtype=dtype)[
        ..., ::2
    ]
    weight = torch.nn.Parameter(
        torch.randn(257, 64, device="cuda", dtype=dtype), requires_grad=False
    )
    head = SimpleNamespace(
        weight=weight, quant_method=UnquantizedEmbeddingMethod(), tp_size=1
    )
    with set_current_vllm_config(VllmConfig()):
        processor = LogitsProcessor(257)
    processor.head_dtype = torch.float32
    original_pointer = head.weight.data_ptr()
    for update in [0.0, 0.125]:
        head.weight.add_(update)
        reference = hidden.double() @ head.weight.double().t()
        actual = processor(head, hidden)
        bf16 = (hidden @ head.weight.t()).float()
        assert actual.dtype == torch.float32
        assert head.weight is weight and head.weight.data_ptr() == original_pointer
        assert head.weight.dtype == dtype
        torch.testing.assert_close(actual.double(), reference, atol=2e-4, rtol=1e-4)
        assert (actual.double() - reference).abs().max() < (
            bf16.double() - reference
        ).abs().max() / 20
        assert not torch.equal(actual, actual.to(dtype).float())


@pytest.mark.vllm
def test_native_fp32_projection_rejects_quantized_head() -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    with set_current_vllm_config(VllmConfig()):
        processor = LogitsProcessor(257)
    processor.head_dtype = torch.float32
    with pytest.raises(ValueError, match="unquantized"):
        processor._apply_head(
            SimpleNamespace(quant_method=object()),
            torch.ones(1, 64, dtype=torch.bfloat16),
            None,
        )
