# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Version-point gradients (grpo.async_grpo.version_point_gradient)."""

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.algorithms.async_utils.replay_buffer import ReplayBufferImpl
from nemo_rl.algorithms.grpo import (
    AsyncGRPOConfig,
    GRPOConfig,
    MasterConfig,
    _build_async_grpo_train_data,
    add_grpo_token_loss_masks_and_generation_logprobs,
    validate_version_point_config,
    version_point_candidate,
    version_point_keep_versions,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.version_snapshots import VersionSnapshots, _bits
from nemo_rl.models.policy import Fp4Config


@pytest.fixture(name="Worker")
def worker_class():
    # Worker integration needs the optional Megatron/Bridge runtime.
    module = pytest.importorskip("nemo_rl.models.policy.workers.megatron_policy_worker")
    return module.MegatronPolicyWorkerImpl


class _FlatModel(torch.nn.Module):
    """Two linears whose params view one flat buffer, like Megatron DDP param buffers."""

    def __init__(self, dtype=torch.float32, device="cpu"):
        super().__init__()
        torch.manual_seed(0)
        self.fc1 = torch.nn.Linear(16, 32, bias=True).to(device, dtype)
        self.fc2 = torch.nn.Linear(32, 4, bias=False).to(device, dtype)
        params = list(self.parameters())
        self.flat = torch.cat([p.detach().reshape(-1) for p in params])
        offset = 0
        for p in params:
            p.data = self.flat[offset : offset + p.numel()].view_as(p)
            offset += p.numel()

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def _snapshots(model, keep_versions=3):
    """Two local "shards" of the flat buffer; the all-gather is a no-op on one process."""
    half = model.flat.numel() // 2
    shards = [model.flat[:half], model.flat[half:]]
    return VersionSnapshots(
        local_shards=lambda: shards,
        sync=lambda: None,
        keep_versions=keep_versions,
        invalidate_weight_caches=lambda: None,
    )


def _fake_worker(Worker, snapshots, rank=0):
    fake = SimpleNamespace(
        rank=rank,
        _version_snapshots=snapshots,
        _version_point=None,
        _drain_param_gather=lambda: None,
    )
    fake.has_version_weights = lambda v: Worker.has_version_weights(fake, v)
    return fake


def _bits_equal(a, b):
    return torch.equal(_bits(a), _bits(b))


def _set_specials(flat):
    # NaN payload, -0.0, subnormal, inf survive the swap bitwise.
    flat[:4] = torch.tensor(
        [float("nan"), -0.0, 1e-40, float("inf")], dtype=torch.float32
    ).to(flat.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.mcore
def test_scope_swaps_in_the_snapshot_and_restores_bitwise(dtype, Worker):
    model = _FlatModel(dtype)
    flat = model.flat
    _set_specials(flat)
    snapshots = _snapshots(model)
    snapshots.snapshot(0)
    theta_g = flat.clone()
    with torch.no_grad():
        flat[4:].add_(0.125)
    snapshots.snapshot(1)
    with torch.no_grad():
        flat[4:].mul_(-1.5)
    theta_t = flat.clone()
    fake = _fake_worker(Worker, snapshots)

    assert Worker.set_version_point(fake, 0) is True
    assert _bits_equal(flat, theta_t)  # nothing swapped before the scope
    with Worker._version_point_scope(fake, enabled=True):
        assert _bits_equal(flat, theta_g)
        # The params view the swapped buffer (fc1.weight is its first 32 x 16 block).
        assert _bits_equal(model.fc1.weight.detach().reshape(-1), theta_g[:512])
    assert _bits_equal(flat, theta_t)

    # An exception inside the scope still restores the current weights exactly.
    with pytest.raises(RuntimeError, match="boom"):
        with Worker._version_point_scope(fake, enabled=True):
            raise RuntimeError("boom")
    assert _bits_equal(flat, theta_t)

    # Disabled (eval) scope: no swap.
    with Worker._version_point_scope(fake, enabled=False):
        assert _bits_equal(flat, theta_t)

    # Unknown version: IS path, no swap.
    assert Worker.set_version_point(fake, 7) is False
    with Worker._version_point_scope(fake, enabled=True):
        assert _bits_equal(flat, theta_t)
    assert Worker.set_version_point(fake, None) is False


@pytest.mark.mcore
def test_gradient_is_taken_at_the_version_point_and_applied_to_current_weights(Worker):
    model = _FlatModel()
    flat = model.flat
    snapshots = _snapshots(model)
    snapshots.snapshot(3)
    reference = _FlatModel()  # a model built at theta_g
    with torch.no_grad():
        flat.add_(torch.randn_like(flat) * 0.1)
    theta_t = flat.clone()
    x = torch.randn(8, 16)

    reference(x).square().sum().backward()
    grad_ref = [p.grad.clone() for p in reference.parameters()]

    fake = _fake_worker(Worker, snapshots)
    assert Worker.set_version_point(fake, 3)
    with Worker._version_point_scope(fake, enabled=True):
        model(x).square().sum().backward()
    Worker.set_version_point(fake, None)
    for p, g in zip(model.parameters(), grad_ref):
        assert torch.equal(p.grad, g)
    assert torch.equal(flat, theta_t)

    # The optimizer then steps the current weights with the version-point gradient:
    # the same step as on a model at theta_t holding the theta_g gradient.
    at_current = _FlatModel()
    with torch.no_grad():
        at_current.flat.copy_(theta_t)
    for p, g in zip(at_current.parameters(), grad_ref):
        p.grad = g.clone()
    torch.optim.SGD(at_current.parameters(), lr=0.1).step()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    assert torch.equal(flat, at_current.flat)
    assert not torch.equal(flat, theta_t)

    # The gradient at the current weights differs (the version point is not a no-op here).
    model.zero_grad()
    with torch.no_grad():
        flat.copy_(theta_t)
    model(x).square().sum().backward()
    assert not all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), grad_ref))


@pytest.mark.mcore
@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE FusedAdam needs CUDA")
def test_precision_aware_master_needs_the_swap_back_before_the_step(Worker):
    """TE FusedAdam with int16 remainders: the master is the bf16 param bits + remainder.

    Stepping after the scope exits equals a reference step from theta_t with the version-point
    gradient; stepping inside the scope (theta_g bits in the param) corrupts the master.
    """
    te_optimizers = pytest.importorskip("transformer_engine.pytorch.optimizers")
    kwargs = dict(
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        adam_w_mode=True,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )

    def master(opt, p):
        # fp32 bits = ((bf16 bits - (rem < 0)) << 16) | (rem & 0xFFFF).
        hi = p.data.view(torch.int16).to(torch.int32)
        lo = opt.get_unscaled_state(p, "master_param").to(torch.int32)
        hi = torch.where(lo < 0, hi - 1, hi)
        return ((hi << 16) | (lo & 0xFFFF)).view(torch.float32)

    x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
    results = {}
    for order in ("after_exit", "inside"):
        model = _FlatModel(torch.bfloat16, "cuda")
        snapshots = _snapshots(model)
        snapshots.snapshot(0)
        with torch.no_grad():
            model.flat.add_(
                (torch.randn_like(model.flat.float()) * 0.05).to(torch.bfloat16)
            )
        theta_t = model.flat.clone()
        params = list(model.parameters())
        opt = te_optimizers.FusedAdam(params, store_param_remainders=True, **kwargs)
        ref_model = _FlatModel(torch.bfloat16, "cuda")
        ref_params = list(ref_model.parameters())
        ref = te_optimizers.FusedAdam(
            ref_params, store_param_remainders=False, **kwargs
        )
        with torch.no_grad():
            ref_model.flat.copy_(theta_t)
        fake = _fake_worker(Worker, snapshots)
        assert Worker.set_version_point(fake, 0)
        with Worker._version_point_scope(fake, enabled=True):
            model(x).float().square().sum().backward()
            if order == "inside":
                opt.step()
        if order == "after_exit":
            opt.step()
        Worker.set_version_point(fake, None)
        for p, q in zip(params, ref_params):
            q.grad = p.grad.clone()
        ref.step()
        results[order] = all(
            torch.equal(
                master(opt, p).view(torch.int32),
                ref.get_unscaled_state(q, "master_param").float().view(torch.int32),
            )
            for p, q in zip(params, ref_params)
        )
    assert results["after_exit"] is True
    assert results["inside"] is False


def test_snapshots_evict_oldest_and_reject_a_conflicting_restore():
    model = _FlatModel(torch.bfloat16)
    snapshots = _snapshots(model, keep_versions=3)
    for version in range(5):
        assert snapshots.snapshot(version)
        with torch.no_grad():
            model.flat.add_(0.5)
    assert snapshots.versions() == [2, 3, 4]
    assert snapshots.stored_bytes() == 3 * model.flat.numel() * 2
    # Same version, same bits: kept. Same version, different bits: dropped (IS path).
    assert snapshots.snapshot(5) and snapshots.snapshot(5)
    with torch.no_grad():
        model.flat[0] += 1.0
    assert snapshots.snapshot(5) is False
    assert not snapshots.has(5)


def test_version_point_candidate_groups_one_version_per_step():
    has = {3, 4, 5}.__contains__
    assert version_point_candidate({4}, has) == (
        4,
        {"version_point_mixed_batch": 0.0, "version_point_no_snapshot": 0.0},
    )
    assert version_point_candidate({4, 5}, has) == (
        None,
        {"version_point_mixed_batch": 1.0, "version_point_no_snapshot": 0.0},
    )
    assert version_point_candidate({1}, has) == (
        None,
        {"version_point_mixed_batch": 0.0, "version_point_no_snapshot": 1.0},
    )


def test_fixed_lead_replay_buffer_serves_one_generation_version_per_step():
    """The buffer serves only groups targeting the current step, each target from one version."""
    lag, groups = 4, 3
    buffer = ReplayBufferImpl(max_size=64, drop_incomplete_targets_on_restore=False)
    # Start-up: version 0 fills targets 0..L; afterwards version v fills target v + L.
    for target in range(lag + 1):
        for g in range(groups):
            buffer.add(
                {"g": (0, target, g)}, weight_version=0, target_weight_version=target
            )
    for step in range(lag + 4):
        if step >= 1:
            for g in range(groups):
                buffer.add(
                    {"g": (step, step + lag, g)},
                    weight_version=step,
                    target_weight_version=step + lag,
                )
        sample = buffer.sample(
            num_prompt_groups=groups, current_weight_version=step, max_age_steps=lag
        )
        versions = set(sample["generation_weight_versions"])
        assert versions == {0 if step <= lag else step - lag}


def _fixed_lead_fallbacks(lag, keep, steps, resume_at=None, restore=True):
    """IS-path steps of a fixed-lead schedule with snapshots after every refit.

    Before the step that trains on weight version t the batch comes from version
    max(0, t - L). A resume at version ``resume_at`` either restores the checkpointed
    snapshots or starts with only a snapshot of the restored weights.
    """
    kept = [0]
    fallbacks = 0
    for t in range(steps):
        if resume_at is not None and t == resume_at:
            kept = kept if restore else [t]
        fallbacks += max(0, t - lag) not in kept
        kept = sorted(set(kept + [t + 1]))[-keep:]  # refit after the step
    return fallbacks


@pytest.mark.parametrize("lag", [1, 2, 8, 16])
def test_keep_l_plus_one_versions_covers_every_fixed_lead_batch(lag):
    keep = version_point_keep_versions(lag)
    assert keep == lag + 1
    assert _fixed_lead_fallbacks(lag, keep, 200) == 0
    assert _fixed_lead_fallbacks(lag, lag, 200) > 0
    # A resume that restores the snapshots has no IS-path step; one that does not has L.
    assert _fixed_lead_fallbacks(lag, keep, 200, resume_at=50) == 0
    assert _fixed_lead_fallbacks(lag, keep, 200, resume_at=50, restore=False) == lag


@pytest.mark.mcore
@pytest.mark.parametrize("storage", ["local", "msc"])
def test_snapshots_survive_a_checkpoint_round_trip(
    tmp_path, monkeypatch, Worker, storage
):
    """Saved and restored snapshots swap in bitwise; a resumed run has 0 IS-path steps."""
    path = str(tmp_path / "snapshots")
    if storage == "msc":
        profile = "vpg-" + uuid.uuid4().hex
        config = tmp_path / "msc.json"
        objects = tmp_path / "objects"
        objects.mkdir()
        config.write_text(
            json.dumps(
                {
                    "profiles": {
                        profile: {
                            "storage_provider": {
                                "type": "file",
                                "options": {"base_path": str(objects)},
                            }
                        }
                    }
                }
            )
        )
        monkeypatch.setenv("MSC_CONFIG", str(config))
        path = f"msc://{profile}/snapshots"
    lag = 3
    keep = version_point_keep_versions(lag)
    model = _FlatModel(torch.bfloat16)
    _set_specials(model.flat)
    fake = _fake_worker(Worker, _snapshots(model, keep), rank=1)
    history = {}
    for version in range(6):
        Worker.snapshot_version_weights(fake, version, keep)
        history[version] = model.flat.clone()
        with torch.no_grad():
            model.flat[4:].add_(0.25)
    with torch.no_grad():
        model.flat.copy_(history[5])  # the checkpoint holds the weights of version 5
    Worker.save_version_snapshots(fake, path)

    # A fresh worker on the restored weights.
    resumed = _fake_worker(Worker, None, rank=1)
    resumed._new_version_snapshots = lambda k: _snapshots(model, k)
    assert Worker.load_version_snapshots(resumed, path, keep) == [2, 3, 4, 5]
    assert Worker.snapshot_version_weights(resumed, 5, keep)["consistent"]
    # Every version the restored buffer can reference (5 - L .. 5) takes the version point.
    for version in range(5 - lag, 6):
        _, vp_metrics = version_point_candidate({version}, resumed.has_version_weights)
        assert vp_metrics["version_point_no_snapshot"] == 0.0
        assert Worker.set_version_point(resumed, version)
        with Worker._version_point_scope(resumed, enabled=True):
            assert _bits_equal(model.flat, history[version])
        assert _bits_equal(model.flat, history[5])
    # No checkpointed snapshots (other rank / older checkpoint): nothing restored.
    other = _fake_worker(Worker, None, rank=0)
    assert Worker.load_version_snapshots(other, path, keep) == []


def _config(**over):
    o = dict(
        enabled=True,
        async_enabled=True,
        in_flight=False,
        gbs=256,
        fp8_param=False,
        colocated=False,
        distributed_optimizer=True,
        batch_invariant=True,
    )
    o.update(over)
    return MasterConfig.model_construct(
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=16,
            num_generations_per_prompt=16,
            async_grpo=AsyncGRPOConfig(
                enabled=o["async_enabled"],
                version_point_gradient=o["enabled"],
                max_trajectory_age_steps=8,
                in_flight_weight_updates=o["in_flight"],
            ),
        ),
        loss_fn=ClippedPGLossConfig.model_construct(
            force_on_policy_ratio=False, use_importance_sampling_correction=True
        ),
        data_plane=None,
        policy={
            "train_global_batch_size": o["gbs"],
            "generation": {"colocated": {"enabled": o["colocated"]}},
            "megatron_cfg": {
                "enabled": True,
                "batch_invariant_mode": o["batch_invariant"],
                "fp8_cfg": {"enabled": o["fp8_param"], "fp8_param": o["fp8_param"]},
                "optimizer": {"use_distributed_optimizer": o["distributed_optimizer"]},
                "distributed_data_parallel_config": {},
            },
        },
    )


def test_validate_version_point_config():
    validate_version_point_config(_config())
    validate_version_point_config(_config(enabled=False, in_flight=True))
    # Works without bitwise parity (IS weight only approximately 1), with a warning.
    with pytest.warns(UserWarning, match="batch_invariant_mode"):
        validate_version_point_config(_config(batch_invariant=False))
    for bad, match in (
        (dict(in_flight=True), "in_flight_weight_updates"),
        (dict(fp8_param=True), "fp8_param"),
        (dict(gbs=128), "one optimizer step per rollout batch"),
        (dict(async_enabled=False), "async GRPO"),
        (dict(colocated=True), "non-colocated"),
        (dict(distributed_optimizer=False), "distributed optimizer"),
    ):
        with pytest.raises(ValueError, match=match):
            validate_version_point_config(_config(**bad))


def test_staging_failure_restores_current_weights_and_clears_caches():
    model = _FlatModel()
    snapshots = _snapshots(model)
    snapshots.snapshot(0)
    model.flat.add_(1)
    current = model.flat.clone()
    calls = []

    def invalidate():
        calls.append(model.flat.clone())
        if len(calls) == 1:
            raise RuntimeError("cache failure")

    snapshots._invalidate_weight_caches = invalidate
    with pytest.raises(RuntimeError, match="cache failure"):
        with snapshots.swapped_in(0):
            pytest.fail("staging must fail before the body")
    assert torch.equal(model.flat, current)
    assert torch.equal(calls[-1], current)
    assert not snapshots._saved_current


@pytest.mark.mcore
def test_pending_gather_finishes_before_current_weights_are_restored(Worker):
    model = _FlatModel()
    snapshots = _snapshots(model)
    snapshots.snapshot(0)
    behavior = model.flat.clone()
    model.flat.add_(1)
    current = model.flat.clone()
    fake = _fake_worker(Worker, snapshots)
    calls = []

    def drain():
        calls.append(1)
        if len(calls) == 2:
            model.flat.copy_(behavior)

    fake._drain_param_gather = drain
    assert Worker.set_version_point(fake, 0)
    with Worker._version_point_scope(fake, enabled=True):
        pass
    assert len(calls) == 2
    assert torch.equal(model.flat, current)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda c: c.policy["megatron_cfg"].update(cuda_graph_impl="local"),
            "cuda_graph_impl",
        ),
        (
            lambda c: c.policy["megatron_cfg"].update(
                fp4_cfg=Fp4Config(enabled=True, fp4_param=True)
            ),
            "fp4_param",
        ),
        (
            lambda c: c.policy["megatron_cfg"].update(
                model_overrides={"moe_router_enable_expert_bias": True}
            ),
            "moe_router_enable_expert_bias",
        ),
        (
            lambda c: setattr(c.grpo.async_grpo, "max_trajectory_age_steps", 0),
            "max_trajectory_age_steps",
        ),
        (
            lambda c: setattr(c.grpo, "seq_logprob_error_threshold", 0.1),
            "seq_logprob_error_in_loss",
        ),
        (
            lambda c: setattr(c.loss_fn, "use_importance_sampling_correction", False),
            "use_importance_sampling_correction",
        ),
    ],
)
def test_version_point_rejects_unsupported_runtime_settings(mutation, message):
    config = _config()
    mutation(config)
    with pytest.raises(ValueError, match=message):
        validate_version_point_config(config)


def test_sync_config_can_omit_the_async_block():
    config = _config()
    config.grpo.async_grpo = None
    validate_version_point_config(config)


def _tokenwise_config():
    config = _config(in_flight=True)
    config.grpo.async_grpo.version_point_per_token = True
    config.policy["generation"]["backend"] = "megatron"
    return config


def test_tokenwise_config_requires_consistent_behavior_context():
    validate_version_point_config(_tokenwise_config())
    mutations = [
        (
            lambda c: setattr(c.grpo.async_grpo, "version_point_gradient", False),
            "version_point_gradient",
        ),
        (
            lambda c: setattr(
                c.grpo.async_grpo, "recompute_kv_cache_after_weight_updates", False
            ),
            "recompute_kv_cache",
        ),
        (
            lambda c: c.policy["generation"].update(backend="vllm"),
            "Megatron generation",
        ),
        (lambda c: setattr(c.loss_fn, "token_level_loss", False), "token-level"),
        (
            lambda c: setattr(c.loss_fn, "sequence_level_importance_ratios", True),
            "token-level",
        ),
        (
            lambda c: setattr(c.grpo, "seq_logprob_error_threshold", 0.1),
            "sequence log-prob",
        ),
        (
            lambda c: c.policy["megatron_cfg"].update(moe_aux_loss_coeff=0.01),
            "moe_aux_loss_coeff",
        ),
    ]
    mutations.extend(
        [
            (lambda c: c.policy.update(tokenizer={"use_processor": True}), "text-only"),
            (
                lambda c: c.policy["megatron_cfg"].update(
                    model_overrides={"mtp_num_layers": 1}
                ),
                "MTP",
            ),
            (
                lambda c: c.policy["megatron_cfg"].update(
                    moe_router_load_balancing_type="sinkhorn"
                ),
                "token-local",
            ),
            (
                lambda c: c.policy["megatron_cfg"].update(
                    model_overrides={
                        "moe_router_load_balancing_type": ["quantile_balancing"]
                    }
                ),
                "token-local",
            ),
        ]
    )
    mutations.extend(
        [
            (
                lambda c: setattr(c.loss_fn, "positive_example_nll_weight", 0.1),
                "positive_example_nll_weight",
            ),
            (
                lambda c: setattr(
                    c.loss_fn, "truncated_importance_sampling_type", "seq-mask-tis"
                ),
                "sequence-level IS",
            ),
        ]
    )
    for mutate, message in mutations:
        config = _tokenwise_config()
        mutate(config)
        with pytest.raises(ValueError, match=message):
            validate_version_point_config(config)


@pytest.mark.mcore
@pytest.mark.parametrize("fail_in_body", [False, True])
def test_split_token_version_pass_restores_current_weights_on_success_and_error(
    Worker, fail_in_body
):
    model = _FlatModel()
    snapshots = _snapshots(model)
    snapshots.snapshot(0)
    behavior = model.flat.clone()
    with torch.no_grad():
        model.flat.add_(1)
    current = model.flat.clone()
    worker = _fake_worker(Worker, snapshots)
    worker._assert_step_open = lambda: {}
    worker.set_version_point = lambda v: Worker.set_version_point(worker, v)
    worker._version_point_scope = lambda enabled: Worker._version_point_scope(
        worker, enabled
    )
    worker._set_mtp_grad_scale_func = lambda _: None
    worker._restore_saved_mcore_hooks = lambda _: None

    def body(state, data):
        assert torch.equal(model.flat, behavior)
        if fail_in_body:
            raise ValueError("intentional body failure")
        return None

    worker._train_microbatch_body = body
    if fail_in_body:
        with pytest.raises(ValueError, match="intentional body failure"):
            Worker.train_microbatch(worker, {}, version_point=0)
    else:
        Worker.train_microbatch(worker, {}, version_point=0)
    assert torch.equal(model.flat, current)
    assert worker._version_point is None


def test_token_stamps_survive_message_flattening_and_training_batch_construction():
    messages = [
        [
            {"role": "user", "content": "prompt", "token_ids": torch.tensor([1, 2, 3])},
            {
                "role": "assistant",
                "content": "response",
                "token_ids": torch.tensor([4, 5]),
                "generation_logprobs": torch.tensor([-0.2, -0.3]),
                "generation_versions": torch.tensor([1, 2]),
            },
            {"role": "user", "content": "tool", "token_ids": torch.tensor([6])},
            {
                "role": "assistant",
                "content": "answer",
                "token_ids": torch.tensor([7]),
                "generation_logprobs": torch.tensor([-0.4]),
                "generation_versions": torch.tensor([3]),
            },
        ]
    ]
    add_grpo_token_loss_masks_and_generation_logprobs(messages)
    flat, lengths = batched_message_log_to_flat_message(
        messages, pad_value_dict={"token_ids": 0}, make_sequence_length_divisible_by=4
    )
    data = _build_async_grpo_train_data(
        flat,
        lengths,
        BatchedDataDict({"loss_multiplier": torch.ones(1)}),
        {"precision": "bfloat16", "router_replay": {"enabled": False}},
    )
    assert data["generation_versions"].tolist() == [[0, 0, 0, 1, 2, 0, 3, 0]]
    assert data["token_mask"].tolist() == [[0, 0, 0, 1, 1, 0, 1, 0]]
    assert data["input_lengths"].tolist() == [7]


@pytest.mark.mcore
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Generation backend imports require a CUDA runtime",
)
@pytest.mark.parametrize("rank", [0, 1])
def test_new_epoch_is_sent_before_post_refit_resume(monkeypatch, rank):
    # The generation backend requires the optional Megatron runtime.
    module = pytest.importorskip("nemo_rl.models.generation.megatron.megatron_worker")
    events = []
    client = SimpleNamespace(
        set_generation_epoch=lambda epoch: events.append(("epoch", epoch)),
        resume_engines=lambda: events.append("resume"),
        unpause_engines=lambda: events.append("unpause"),
    )

    async def wait_until(state):
        events.append(state)

    worker = SimpleNamespace(
        _stamp_token_versions=False,
        _pending_generation_epoch=None,
        inference_client=client,
        dynamic_inference_engine=SimpleNamespace(wait_until=wait_until),
    )
    monkeypatch.setattr(module.torch.distributed, "get_rank", lambda: rank)
    module.MegatronGenerationMixin.set_generation_epoch(worker, 7, at_resume=True)
    assert events == []
    asyncio.run(module.MegatronGenerationMixin._wake_engine(worker))
    if rank == 0:
        assert events == [
            ("epoch", 7),
            "resume",
            module.EngineState.RESUMED,
            "unpause",
            module.EngineState.RUNNING,
        ]
    else:
        assert events == [module.EngineState.RESUMED, module.EngineState.RUNNING]
    assert worker._pending_generation_epoch is None


@pytest.mark.parametrize("tokenwise", [False, True])
def test_tokenwise_config_is_rejected_by_single_controller(tokenwise):
    # SingleController's config is an optional stack for the legacy GRPO tests.
    from nemo_rl.algorithms.single_controller_utils.config import (
        MasterConfig as SingleControllerConfig,
    )
    from nemo_rl.algorithms.single_controller_utils.config import (
        validate_single_controller_config,
    )

    config = SingleControllerConfig.model_construct(
        grpo=GRPOConfig.model_construct(
            async_grpo=AsyncGRPOConfig(
                version_point_gradient=not tokenwise,
                version_point_per_token=tokenwise,
            )
        )
    )
    with pytest.raises(ValueError, match="not supported by SingleController"):
        validate_single_controller_config(config)
