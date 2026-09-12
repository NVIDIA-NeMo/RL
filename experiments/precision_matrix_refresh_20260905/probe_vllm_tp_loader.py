"""Check installed expert-loader TP semantics, without loading a model."""

import json
from types import SimpleNamespace

import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


def main() -> None:
    owner = SimpleNamespace(
        moe_config=SimpleNamespace(
            is_act_and_mul=True,
            moe_parallel_config=SimpleNamespace(tp_size=4),
        ),
        _get_hidden_dim=RoutedExperts._get_hidden_dim,
        _narrow_expert_data_for_padding=RoutedExperts._narrow_expert_data_for_padding,
    )
    results: list[dict[str, object]] = []
    for padded in (False, True):
        local_intermediate, hidden = 8, 16
        runtime_intermediate = 12 if padded else local_intermediate
        for rank in range(4):
            for projection in ("w1", "w3", "w2"):
                shard_dim = 1 if projection == "w2" else 0
                global_shape = (hidden, 32) if projection == "w2" else (32, hidden)
                full = torch.arange(512, dtype=torch.float32).reshape(global_shape)
                local = full.narrow(shard_dim, rank * 8, 8).clone()
                shape = (
                    (hidden, runtime_intermediate)
                    if projection == "w2"
                    else (2 * runtime_intermediate, hidden)
                )

                def load(weight: torch.Tensor, *, load_full: bool) -> torch.Tensor:
                    destination = torch.full(shape, -1.0)
                    kwargs = dict(
                        expert_data=destination,
                        shard_dim=shard_dim,
                        loaded_weight=weight,
                        tp_rank=rank,
                        load_full=load_full,
                    )
                    if projection == "w2":
                        RoutedExperts._load_w2(owner, **kwargs)
                    else:
                        RoutedExperts._load_w13(owner, shard_id=projection, **kwargs)
                    return destination

                checkpoint_result = load(full, load_full=False)
                repeated_tp_result = load(local, load_full=False)
                local_result = load(local, load_full=True)
                torch.testing.assert_close(checkpoint_result, local_result, rtol=0, atol=0)
                assert not torch.equal(checkpoint_result, repeated_tp_result)
                offset = runtime_intermediate if projection == "w3" else 0
                logical = local_result.narrow(shard_dim, offset, local_intermediate)
                torch.testing.assert_close(logical, local, rtol=0, atol=0)
                results.append(
                    dict(
                        rank=rank,
                        projection=projection,
                        padded=padded,
                        full_checkpoint_matches_local_no_reshard=True,
                        already_local_input_is_resharded_again=True,
                    )
                )
    print(json.dumps({"cases": results, "passed": len(results)}, indent=2))


if __name__ == "__main__":
    main()
