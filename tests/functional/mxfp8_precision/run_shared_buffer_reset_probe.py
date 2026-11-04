"""Two-rank NCCL/TE ownership-boundary probe, without a training model."""

import argparse
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist


def run_case(*, grad_dtype: torch.dtype, force_sync: bool, mode: str) -> dict:
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.distributed.param_and_grad_buffer import (
        _ParamAndGradBucket,
        _ParamAndGradBucketGroup,
        _ParamAndGradBuffer,
    )
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    rank, world = dist.get_rank(), dist.get_world_size()
    size, padded = 128 * 256, 128 * 256 + 256
    cfg = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        fp8_param_gather=True,
        reuse_grad_buf_for_mxfp8_param_ag=True,
        overlap_param_gather=True,
        overlap_grad_reduce=False,
        grad_reduce_in_fp32=grad_dtype == torch.float32,
    )
    quantizer = MXFP8Quantizer(DType.kFloat8E4M3, rowwise=True, columnwise=True)
    params = [
        torch.nn.Parameter(
            quantizer.quantize(
                torch.zeros(128, 256, device="cuda", dtype=torch.bfloat16)
            )
        )
        for _ in range(2)
    ]
    # Use production bucket/group/optimizer methods with a small explicit layout.
    # Bypass full model construction, optimizer-state allocation and checkpoint IO.
    buffer = object.__new__(_ParamAndGradBuffer)
    buffer.ddp_config = cfg
    buffer.params = params
    buffer.shared_buffer = torch.zeros(2 * padded, device="cuda", dtype=grad_dtype)
    buffer.grad_data = buffer.shared_buffer
    buffer.param_data = buffer.shared_buffer.view(torch.bfloat16)[: 2 * padded]
    buffer.extra_main_grads = []
    index = {
        param: (i * padded, i * padded + size, i) for i, param in enumerate(params)
    }
    buffer.buckets = [
        _ParamAndGradBucket(
            [param],
            buffer.param_data[i * padded : (i + 1) * padded],
            buffer.grad_data[i * padded : (i + 1) * padded],
            i * padded,
            size,
            1.0,
            i,
            index,
            [],
        )
        for i, param in enumerate(params)
    ]
    groups = [
        _ParamAndGradBucketGroup([bucket], cfg, dist.group.WORLD, world)
        for bucket in buffer.buckets
    ]
    optimizer = object.__new__(DistributedOptimizer)
    optimizer.ddp_config = cfg
    optimizer.config = SimpleNamespace(
        reuse_grad_buf_for_mxfp8_param_ag=True, use_megatron_fsdp=False
    )
    optimizer.is_stub_optimizer = False
    optimizer.buffers = [buffer]
    optimizer.model_chunks = []
    optimizer.per_model_bucket_groups = {0: groups}
    optimizer.model_float16_groups = [params]
    optimizer.model_param_gbuf_map = {
        param: (0, None, i) for i, param in enumerate(params)
    }
    start, stop = rank * (padded // world), min((rank + 1) * (padded // world), size)
    optimizer._get_model_param_range_map = MethodType(
        lambda self, param: {"gbuf_world_in_bucket": Range(start, stop)}, optimizer
    )
    ddp = object.__new__(DistributedDataParallel)
    torch.nn.Module.__init__(ddp)
    ddp.config = SimpleNamespace(cuda_graph_impl="none")
    ddp.ddp_config = cfg
    ddp.params_with_grad = params
    ddp.buffers, ddp.expert_parallel_buffers = [buffer], []
    ddp.bucket_groups, ddp.expert_parallel_bucket_groups = groups, []
    ddp.param_to_bucket_group = dict(zip(params, groups))
    for param, bucket in zip(params, buffer.buckets):
        param.main_grad = bucket.grad_data[:size].view(128, 256)

    records = []
    generator = torch.Generator().manual_seed(930)
    original_masters = [
        torch.randn(128, 256, generator=generator).cuda() * 0.02 for _ in params
    ]
    for step in range(2):
        ddp.zero_grad_buffer()
        masters = [master * (1.0 + step * 0.125) for master in original_masters]
        optimizer.shard_fp32_from_float16_groups = [
            [master.flatten()[start:stop] for master in masters]
        ]
        buffer.shared_buffer.fill_(float("nan"))
        optimizer._copy_main_params_to_param_buffer()
        local_padding = buffer.param_data[padded - 128 : padded]
        padding_clean = (
            bool(torch.count_nonzero(local_padding) == 0) if rank == 1 else True
        )
        if mode == "1":
            assert padding_clean, "Poisoned padding survived pre-staging clear"
        for group in groups:
            group.start_param_sync(force_sync=force_sync)
        groups[0].finish_param_sync(skip_next_bucket_dispatch=True)
        # A whole first FP32 grad-bucket clear would erase the second BF16 payload.
        if not force_sync:
            torch.cuda.synchronize()
            sibling = buffer.buckets[1].param_data[:size].reshape(128, 256)
            assert torch.equal(
                sibling, masters[1].bfloat16()
            ), "Sibling payload was erased"
        groups[1].finish_param_sync(skip_next_bucket_dispatch=True)
        for param, master in zip(params, masters):
            reference = quantizer.quantize(master.bfloat16())
            actual, expected = param.get_metadata(), reference.get_metadata()
            for key in [
                "rowwise_data",
                "columnwise_data",
                "rowwise_scale_inv",
                "columnwise_scale_inv",
            ]:
                assert torch.equal(
                    actual[key].view(torch.uint8), expected[key].view(torch.uint8)
                ), key
        all_clean = bool(torch.count_nonzero(buffer.shared_buffer) == 0)
        if mode == "1":
            assert all_clean, "Shared buffer not clean after MXFP8 conversion"
        ddp.zero_grad_buffer()
        expected_grads = [torch.zeros_like(param.main_grad) for param in params]
        for microbatch in range(2):
            for i, param in enumerate(params):
                contribution = torch.full_like(
                    param.main_grad, (i + 1) * (microbatch + 1)
                ).bfloat16()
                param.grad = contribution
                DistributedDataParallel._make_backward_post_hook(ddp, param)()
                expected_grads[i].add_(contribution)
                assert torch.equal(
                    param.main_grad, expected_grads[i]
                ), "Accumulated gradient was erased"
        records.append(
            {
                "step": step,
                "local_staging_padding_clean": padding_clean,
                "shared_allocation_clean_after_conversion": all_clean,
                "rowwise_columnwise_codes_and_scales_equal_fresh_te": True,
                "sibling_payload_preserved": True,
                "two_microbatch_gradients_preserved": True,
            }
        )
    return {"grad_dtype": str(grad_dtype), "force_sync": force_sync, "steps": records}


def main(output: Path) -> None:
    from nemo_rl.models.megatron.mxfp8_buffer_reset import install_mxfp8_buffer_reset

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    mode = os.environ["NRL_MXFP8_RESET_SHARED_BUFFER"]
    assert install_mxfp8_buffer_reset() == (mode == "1")
    cases = [
        run_case(grad_dtype=dtype, force_sync=sync, mode=mode)
        for dtype in [torch.bfloat16, torch.float32]
        for sync in [False, True]
    ]
    result = {
        "completed": True,
        "mode": mode,
        "rank": dist.get_rank(),
        "device": torch.cuda.get_device_name(),
        "cases": cases,
        "scope": "Actual MCore staging, NCCL gathers, TE quantization and DDP gradient accumulation hooks, with synthetic layout and gradients. No full Nano model or optimizer update.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / f"mode-{mode}-rank-{dist.get_rank()}.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args().output_dir)
