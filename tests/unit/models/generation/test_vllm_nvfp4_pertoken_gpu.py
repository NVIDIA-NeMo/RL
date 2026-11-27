# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blackwell regression for Lightning-sized non-gated NVFP4 expert refits."""

import pytest
import torch

pytestmark = pytest.mark.vllm


def test_lightning_non_gated_native_reload_and_cuda_graph(tmp_path) -> None:
    """Exercise real quantization, native loaders, padding and graph-stable reload.

    This is a single MoE layer, not a full Lightning training validation.
    Run on a scheduled Blackwell allocation with the vLLM extra.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires a Blackwell datacenter GPU")

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    from nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken import (
        NvFp4PerTokenConfig,
        NvFp4PerTokenQuantizer,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{tmp_path / 'dist-init'}",
        local_rank=0,
        backend="nccl",
    )
    try:
        config = VllmConfig()
        config.kernel_config.moe_backend = "flashinfer_trtllm"
        with set_current_vllm_config(config):
            initialize_model_parallel(tensor_model_parallel_size=1)
            with torch.device("cuda"):
                runner = FusedMoEFactory(
                    num_experts=128,
                    top_k=6,
                    hidden_size=2688,
                    intermediate_size=1856,
                    params_dtype=torch.bfloat16,
                    activation="relu2_no_mul",
                    quant_config=NvFp4PerTokenConfig(
                        is_checkpoint_nvfp4_serialized=True,
                    ),
                    tp_size=1,
                    dp_size=1,
                    pcp_size=1,
                    prefix="layers.0.experts",
                    ckpt_names=("up_proj", "down_proj", None),
                )
                experts = runner.routed_experts
                model = torch.nn.Module()
                layer = torch.nn.Module()
                layer.experts = experts
                model.layers = torch.nn.ModuleList([layer])
                quantizer = NvFp4PerTokenQuantizer(model)
                assert not experts.moe_config.is_act_and_mul
                assert experts.w13_weight_scale_2.shape == (128, 1)
                record_metadata_for_reloading(model)

                def load(multiplier: float) -> None:
                    generator = torch.Generator(device="cuda").manual_seed(27)
                    quantizer.reset()
                    for expert in range(128):
                        for projection, shape in (
                            ("up_proj", (1856, 2688)),
                            ("down_proj", (2688, 1856)),
                        ):
                            weight = torch.randn(
                                shape, dtype=torch.bfloat16, generator=generator
                            ) * (0.01 * multiplier)
                            if projection == "up_proj":
                                weight = -weight.abs()
                            emitted = quantizer.process(
                                [
                                    (
                                        f"layers.0.experts.{expert}.{projection}.weight",
                                        weight,
                                    )
                                ]
                            )
                            loaded = list(
                                experts.load_weights(
                                    (name.removeprefix("layers.0.experts."), tensor)
                                    for name, tensor in emitted
                                )
                            )
                            assert len(loaded) == 4
                    quantizer.finish()

                load(1.0)
                experts.quant_method.process_weights_after_loading(experts)
                assert experts.moe_config.hidden_dim == 3072
                assert experts.w13_weight.shape[1] == 1920  # padded up, no gate half
                pointers = {
                    name: p.data_ptr() for name, p in experts.named_parameters()
                }
                x = torch.randn(16, 3072, dtype=torch.bfloat16)
                x[:, 2688:] = 0
                x[0] = 0  # all-zero first GEMM input
                x[1, :2688] = 1  # all-negative up output -> zero ReLU² GEMM2 input
                logits = torch.randn(16, 128, dtype=torch.float32)

                def forward() -> torch.Tensor:
                    return experts.forward_monolithic(x, logits)

                for _ in range(3):
                    forward()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = forward()
                graph.replay()
                before = captured.clone()
                assert torch.isfinite(before).all()
                assert torch.count_nonzero(before[:2]) == 0
                assert torch.count_nonzero(before[2:]) > 0

                initialize_layerwise_reload(model)
                load(2.0)
                finalize_layerwise_reload(model, model_config=None)
                assert pointers == {
                    name: p.data_ptr() for name, p in experts.named_parameters()
                }
                graph.replay()
                after = captured.clone()
                eager = forward()
                torch.cuda.synchronize()
                assert torch.isfinite(after).all()
                assert torch.count_nonzero(after[:2]) == 0
                torch.testing.assert_close(after, eager, rtol=0.01, atol=1e-5)
                # Doubling both projections multiplies down(ReLU(up(x))²) by eight.
                torch.testing.assert_close(after, before * 8, rtol=0.03, atol=1e-4)
    finally:
        cleanup_dist_env_and_memory()
