"""Native hybrid state checks, including a GPU pinned-cache regression."""

import gc
import copy
import importlib.util
import weakref
import struct
from pathlib import Path

import pytest
import torch
from nemo_rl.models.megatron.optimizer_state_sync import (
    checkpoint_bf16_master_dtypes,
    preserve_hybrid_optimizer_state_dtypes,
    refresh_hybrid_optimizer_after_checkpoint,
    release_unused_pinned_memory_after_checkpoint,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
def test_resume_releases_unused_pinned_cache_without_changing_next_step(hybrid_class):
    parameter = torch.nn.Parameter(
        torch.ones(1024, 8192, dtype=torch.bfloat16, device="cuda")
    )

    def native_optimizer(p):
        return hybrid_class(
            [p],
            offload_fraction=1.0,
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            param_update_in_fp32=True,
            pin_cpu_params=True,
            pin_cpu_grads=True,
            overlap_cpu_optimizer_d2h_h2d=False,
            lr=3e-6,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=0,
        )

    optimizer = native_optimizer(parameter)
    parameter.grad = torch.full_like(parameter, 0.125)
    optimizer.step()
    optimizer.zero_grad()
    saved = copy.deepcopy(optimizer.state_dict())
    for _ in range(3):
        optimizer.load_state_dict(copy.deepcopy(saved))
        refresh_hybrid_optimizer_after_checkpoint(optimizer)
    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    reference = native_optimizer(reference_parameter)
    reference.load_state_dict(copy.deepcopy(saved))
    refresh_hybrid_optimizer_after_checkpoint(reference)
    masters_before = [
        p.detach().clone() for p in optimizer.param_to_fp32_param.values()
    ]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory.host_memory_stats()
    released = release_unused_pinned_memory_after_checkpoint()
    after = torch.cuda.memory.host_memory_stats()
    assert released > 0
    assert (
        released == before["allocated_bytes.current"] - after["allocated_bytes.current"]
    )
    assert before["active_bytes.current"] == after["active_bytes.current"]
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    for master, expected in zip(
        optimizer.param_to_fp32_param.values(), masters_before, strict=True
    ):
        assert torch.equal(master, expected)
    parameter.grad = torch.full_like(parameter, 0.0625)
    reference_parameter.grad = torch.full_like(reference_parameter, 0.0625)
    optimizer.step()
    reference.step()
    torch.cuda.synchronize()
    assert torch.equal(parameter, reference_parameter)
    for actual, expected in zip(
        optimizer.param_to_fp32_param.values(),
        reference.param_to_fp32_param.values(),
        strict=True,
    ):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("use_tracker", [False, True])
def test_reads_actual_dcp_master_formats_per_optimizer(tmp_path, use_tracker):
    import torch.distributed.checkpoint as dcp

    directory = tmp_path / "iter_0000000" if use_tracker else tmp_path
    directory.mkdir(exist_ok=True)
    if use_tracker:
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("0")
    state = {
        "chained_0.optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_(torch.bfloat16, torch.bfloat16).bucket_idx_0.param": torch.ones(
            2, dtype=torch.int16
        ),
        "chained_1.optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_(torch.bfloat16, torch.float32).bucket_idx_0.param": torch.ones(
            2, dtype=torch.float32
        ),
        "chained_0.optimizer.distributed.dp_group_idx_0.gbuf_idx_1.dtype_(torch.float32, torch.float32).bucket_idx_0.param": torch.ones(
            2, dtype=torch.float32
        ),
    }
    dcp.save(state, checkpoint_id=directory)
    assert checkpoint_bf16_master_dtypes(tmp_path) == {0: torch.int16, 1: torch.float32}


def test_rejects_mixed_master_formats_in_same_optimizer(tmp_path):
    import torch.distributed.checkpoint as dcp

    state = {
        "chained_0.optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_(torch.bfloat16, torch.bfloat16).bucket_idx_0.param": torch.ones(
            2, dtype=torch.int16
        ),
        "chained_0.optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_(torch.bfloat16, torch.bfloat16).bucket_idx_1.param": torch.ones(
            2, dtype=torch.float32
        ),
    }
    dcp.save(state, checkpoint_id=tmp_path)
    with pytest.raises(RuntimeError, match="Mixed BF16 master formats"):
        checkpoint_bf16_master_dtypes(tmp_path)


@pytest.fixture(scope="module")
def hybrid_class():
    # This upstream module depends only on Torch and the standard library.
    # Import the actual file, avoiding unrelated CUDA-only package imports.
    source = (
        Path(__file__).resolve().parents[4]
        / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py"
    )
    spec = importlib.util.spec_from_file_location("real_hybrid_state_cpu", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HybridDeviceOptimizer


def make_optimizer(hybrid_class, dtype, *, cpu_class=torch.optim.AdamW):
    """Execute real upstream CPU grouping without constructing CUDA streams."""
    parameter = torch.nn.Parameter(torch.zeros(12, dtype=dtype))
    optimizer = object.__new__(hybrid_class)
    torch.optim.Optimizer.__init__(
        optimizer, [parameter], {"lr": 0.01, "cpu_optimizer_cls": cpu_class}
    )
    optimizer.param_update_in_fp32 = True
    optimizer.pin_cpu_params = False
    (
        optimizer.cpu_param_groups,
        optimizer.gpu_param_groups,
        optimizer.gpu_params_map_cpu_copy,
        optimizer.cpu_copys_map_gpu_param,
        optimizer.param_to_fp32_param,
    ) = optimizer._get_sub_optimizer_param_groups(1.0)
    optimizer.cpu_optimizers = [cpu_class(optimizer.cpu_param_groups, lr=0.01)]
    optimizer.gpu_optimizer = None
    inner = optimizer.param_to_fp32_param.get(parameter, parameter)
    optimizer.inner_param_to_orig_param = {inner: parameter}
    optimizer.param_to_inner_param = {parameter: inner}
    optimizer.fp32_param_to_orig_param = {
        v: k for k, v in optimizer.param_to_fp32_param.items()
    }
    master = torch.tensor(
        [
            3.000123,
            4.000345,
            -3.000123,
            -4.000345,
            0.000123,
            -0.000345,
            0.0,
            -0.0,
            1.0,
            -1.0,
            1.00390625,
            -1.00390625,
        ]
    )
    first = torch.linspace(-0.01, 0.01, master.numel())
    second = torch.full_like(master, 0.001)
    optimizer.state[parameter] = {
        "master_param": master.clone(),
        "step": torch.tensor(3.0),
        "exp_avg": first,
        "exp_avg_sq": second,
    }
    return optimizer, parameter, inner, master, first, second


@pytest.mark.parametrize("source_format", ["fp32", "int16"])
@pytest.mark.parametrize("cpu_class", [torch.optim.Adam, torch.optim.AdamW])
def test_loaded_precision_and_next_real_adam_update(
    hybrid_class, source_format, cpu_class
):
    optimizer, parameter, working, master, first, second = make_optimizer(
        hybrid_class, torch.bfloat16, cpu_class=cpu_class
    )
    assert torch.count_nonzero(working) == 0
    if source_format == "int16":
        # Encode independent 16-bit words using TE's CUDA union/rounding rules.
        # The implementation under test uses signed int32 addition instead.
        high_words, low_words = [], []
        for value in master.tolist():
            low, high = struct.unpack("<hh", struct.pack("<f", value))
            if low < 0:
                high = ((high + 1 + 32768) % 65536) - 32768
            high_words.append(high)
            low_words.append(low)
        parameter.data.copy_(
            torch.tensor(high_words, dtype=torch.int16).view(torch.bfloat16)
        )
        optimizer.state[parameter]["master_param"] = torch.tensor(
            low_words, dtype=torch.int16
        )
    saved_compute = parameter.detach().clone()
    remainder_parameters = refresh_hybrid_optimizer_after_checkpoint(optimizer)
    assert (parameter in remainder_parameters) == (source_format == "int16")
    assert torch.equal(parameter.view(torch.int16), saved_compute.view(torch.int16))
    if source_format == "int16":
        # The explicit halfway cases must retain TE's saved BF16 rounding.
        assert not torch.equal(parameter, master.bfloat16())
    assert torch.equal(working.view(torch.int32), master.view(torch.int32))
    assert not torch.equal(working, master.bfloat16().float())
    state = optimizer.cpu_optimizers[0].state[working]
    assert state["exp_avg"] is first
    assert state["exp_avg_sq"] is second
    gold = torch.nn.Parameter(master.clone())
    reference = cpu_class([gold], lr=0.01)
    reference.state[gold] = {
        "step": torch.tensor(3.0),
        "exp_avg": first.clone(),
        "exp_avg_sq": second.clone(),
    }
    working.grad = torch.linspace(-0.25, 0.5, master.numel())
    gold.grad = working.grad.clone()
    optimizer.cpu_optimizers[0].step()
    reference.step()
    assert torch.equal(working, gold)


@pytest.mark.parametrize("invalid", ["missing", "bf16", "shape"])
def test_invalid_master_rejected_before_mutating(hybrid_class, invalid):
    optimizer, parameter, working, *_ = make_optimizer(hybrid_class, torch.bfloat16)
    if invalid == "missing":
        del optimizer.state[parameter]["master_param"]
    elif invalid == "bf16":
        optimizer.state[parameter]["master_param"] = torch.ones_like(parameter)
    else:
        optimizer.state[parameter]["master_param"] = torch.ones(13)
    with pytest.raises(RuntimeError, match="requires a loaded FP32 master"):
        refresh_hybrid_optimizer_after_checkpoint(optimizer)
    assert torch.count_nonzero(working) == 0


@pytest.mark.parametrize("raise_error", [True, False])
def test_dtype_hooks_removed_on_scope_exit(hybrid_class, raise_error):
    optimizer, *_ = make_optimizer(hybrid_class, torch.bfloat16)
    before = (
        len(optimizer._optimizer_load_state_dict_pre_hooks),
        len(optimizer._optimizer_load_state_dict_post_hooks),
    )
    try:
        with preserve_hybrid_optimizer_state_dtypes(
            optimizer, bf16_master_dtype=torch.int16
        ):
            assert len(optimizer._optimizer_load_state_dict_pre_hooks) == before[0] + 1
            assert len(optimizer._optimizer_load_state_dict_post_hooks) == before[1] + 1
            if raise_error:
                raise ValueError("intentional load failure")
    except ValueError:
        assert raise_error
    assert before == (
        len(optimizer._optimizer_load_state_dict_pre_hooks),
        len(optimizer._optimizer_load_state_dict_post_hooks),
    )


def test_saved_group_step_overrides_dummy_counter_before_adam(hybrid_class):
    optimizer, parameter, working, master, first, second = make_optimizer(
        hybrid_class, torch.bfloat16
    )
    optimizer.param_groups[0]["step"] = 5
    optimizer.state[parameter]["step"].fill_(1)
    refresh_hybrid_optimizer_after_checkpoint(optimizer)
    assert optimizer.cpu_optimizers[0].state[working]["step"].item() == 5
    gold = torch.nn.Parameter(master.clone())
    reference = torch.optim.AdamW([gold], lr=0.01)
    reference.state[gold] = {
        "step": torch.tensor(5.0),
        "exp_avg": first.clone(),
        "exp_avg_sq": second.clone(),
    }
    working.grad = torch.linspace(-0.25, 0.5, master.numel())
    gold.grad = working.grad.clone()
    optimizer.cpu_optimizers[0].step()
    reference.step()
    assert torch.equal(working, gold)


def test_resume_releases_redundant_loaded_fp32_master(hybrid_class):
    optimizer, parameter, working, master, *_ = make_optimizer(
        hybrid_class, torch.bfloat16
    )
    loaded_master = weakref.ref(optimizer.state[parameter]["master_param"])
    assert loaded_master() is not working

    refresh_hybrid_optimizer_after_checkpoint(optimizer)
    gc.collect()

    # A resumed actor must release its extra full FP32 master before its
    # colocated critic is built. Native CPU Adam already owns the exact copy.
    assert loaded_master() is None
    assert optimizer.state[parameter]["master_param"] is working
    assert optimizer.cpu_optimizers[0].state[working]["master_param"] is working
    assert torch.equal(working.view(torch.int32), master.view(torch.int32))
