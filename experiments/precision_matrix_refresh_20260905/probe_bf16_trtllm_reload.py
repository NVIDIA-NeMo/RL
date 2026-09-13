"""Real TP4 TRTLLM expert packing and output parity after two reloads."""

import json
import os
from datetime import timedelta

import torch
from vllm.config import (
    KernelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.v1.worker.workspace import init_workspace_manager

from nemo_rl.models.generation.vllm.local_expert_reload import (
    LocalBf16ExpertReload,
    LocalExpertBinding,
)


@torch.no_grad()
def main() -> None:
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world == 4, "This probe requires four TP ranks"
    torch.cuda.set_device(local_rank)
    cfg = VllmConfig(
        parallel_config=ParallelConfig(tensor_parallel_size=world),
        kernel_config=KernelConfig(moe_backend="flashinfer_trtllm"),
    )
    results: list[dict[str, object]] = []
    with set_current_vllm_config(cfg), torch.device("cuda"):
        init_distributed_environment(
            world, rank, "env://", local_rank, timeout=timedelta(minutes=3)
        )
        initialize_model_parallel(world)
        init_workspace_manager(local_rank)
        experts, hidden = 8, 256
        torch.manual_seed(13)
        inputs = torch.randn(16, hidden, dtype=torch.bfloat16) / 8
        router = torch.randn(16, experts, dtype=torch.float32)

        # Native vLLM pads non-gated experts; gated experts require aligned I.
        for gated, intermediate in ((True, 4096), (False, 2688), (False, 3712)):
            local_width = intermediate // world

            def weights(update: int) -> dict[str, torch.Tensor]:
                torch.manual_seed(31 + update)
                result = {
                    "up": torch.randn(
                        experts, intermediate, hidden, dtype=torch.bfloat16
                    )
                    / 64,
                    "down": torch.randn(
                        experts, hidden, intermediate, dtype=torch.bfloat16
                    )
                    / 64,
                }
                if gated:
                    result["gate"] = (
                        torch.randn(experts, intermediate, hidden, dtype=torch.bfloat16)
                        / 64
                    )
                return result

            def fresh(update: int, prefix: str) -> torch.nn.Module:
                layer = FusedMoE(
                    num_experts=experts,
                    top_k=2,
                    hidden_size=hidden,
                    intermediate_size=intermediate,
                    params_dtype=torch.bfloat16,
                    renormalize=True,
                    tp_size=world,
                    prefix=prefix,
                    activation="silu" if gated else "relu2_no_mul",
                )
                owner = layer.routed_experts
                assert (
                    owner.quant_method.unquantized_backend
                    is UnquantizedMoeBackend.FLASHINFER_TRTLLM
                )
                record_metadata_for_reloading(owner)
                # Define padding in the fresh reference before native loading.
                owner.w13_weight.zero_()
                owner.w2_weight.zero_()
                for name, value in weights(update).items():
                    target_name = "w2_weight" if name == "down" else "w13_weight"
                    target = getattr(owner, target_name)
                    shard_id = {
                        "gate": "w1",
                        "up": "w3" if gated else "w1",
                        "down": "w2",
                    }[name]
                    for expert_id in range(experts):
                        success = owner.weight_loader(
                            target,
                            value[expert_id],
                            target_name,
                            shard_id,
                            expert_id,
                            return_success=True,
                        )
                        assert success
                owner.quant_method.process_weights_after_loading(owner)
                return layer

            layer = fresh(0, f"reload_{gated}_{intermediate}")
            bindings = {
                "up": LocalExpertBinding(
                    "routed_experts.w13_weight",
                    "up_proj",
                    (experts, local_width, hidden),
                ),
                "down": LocalExpertBinding(
                    "routed_experts.w2_weight",
                    "down_proj",
                    (experts, hidden, local_width),
                ),
            }
            if gated:
                bindings["gate"] = LocalExpertBinding(
                    "routed_experts.w13_weight",
                    "gate_proj",
                    (experts, local_width, hidden),
                )
            for update in (1, 2):
                reference = fresh(update, f"reference_{gated}_{intermediate}_{update}")
                reload = LocalBf16ExpertReload(layer, bindings)
                initialize_layerwise_reload(layer.routed_experts)
                for name, value in weights(update).items():
                    axis = 2 if name == "down" else 1
                    reload.load(
                        name, value.narrow(axis, rank * local_width, local_width)
                    )
                reload.require_complete()
                finalize_layerwise_reload(layer.routed_experts, None)
                reload.verify_runtime_storage()
                for name in ("w13_weight", "w2_weight"):
                    torch.testing.assert_close(
                        getattr(layer.routed_experts, name),
                        getattr(reference.routed_experts, name),
                        rtol=0,
                        atol=0,
                    )
                with set_forward_context(None, cfg):
                    actual = layer(hidden_states=inputs, router_logits=router)
                    expected = reference(hidden_states=inputs, router_logits=router)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert torch.isfinite(actual).all()
                results.append(
                    dict(
                        gated=gated,
                        local_width=local_width,
                        update=update,
                        packed_equal=True,
                        output_equal=True,
                    )
                )
        torch.cuda.synchronize()
        torch.distributed.barrier()
        print(json.dumps(dict(rank=rank, cases=results)), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
