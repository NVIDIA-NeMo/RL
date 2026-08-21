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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import nn


def _load_provider_without_poisoning_packages() -> ModuleType:
    repository_root = Path(__file__).parents[4]
    package_names = (
        "nemo_rl",
        "nemo_rl.algorithms",
        "nemo_rl.algorithms.loss",
        "nemo_rl.models",
        "nemo_rl.models.megatron",
        "nemo_rl.models.megatron.draft",
    )
    module_paths = (
        ("nemo_rl.algorithms.loss.draft", "nemo_rl/algorithms/loss/draft.py"),
        ("nemo_rl.algorithms.loss.dspark", "nemo_rl/algorithms/loss/dspark.py"),
        (
            "nemo_rl.models.megatron.draft.dspark",
            "nemo_rl/models/megatron/draft/dspark.py",
        ),
        (
            "nemo_rl.models.megatron.draft.dspark_provider",
            "nemo_rl/models/megatron/draft/dspark_provider.py",
        ),
    )
    module_names = (*package_names, *(name for name, _ in module_paths))
    previous_modules = {name: sys.modules.get(name) for name in module_names}
    try:
        for package_name in package_names:
            package = ModuleType(package_name)
            package.__path__ = []  # type: ignore[attr-defined]
            sys.modules[package_name] = package
        for module_name, relative_path in module_paths:
            spec = importlib.util.spec_from_file_location(
                module_name, repository_root / relative_path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load {relative_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        loaded = sys.modules["nemo_rl.models.megatron.draft.dspark_provider"]
        assert isinstance(loaded, ModuleType)
        return loaded
    finally:
        for module_name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous


_PROVIDER = _load_provider_without_poisoning_packages()
build_dspark_provider = _PROVIDER.build_dspark_provider


class _CheckpointLinear(nn.Linear):
    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del sharded_offsets, metadata
        return {f"{prefix}{name}": value for name, value in self.state_dict().items()}


class _CheckpointIdentity(nn.Identity):
    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del prefix, sharded_offsets, metadata
        return {}


def _provider_inputs() -> dict[str, object]:
    generator = torch.Generator().manual_seed(271828)
    target_logits = torch.randn(
        2, 3, 9, generator=generator, dtype=torch.float64, requires_grad=True
    )
    return {
        "draft_hidden": torch.randn(
            2, 3, 5, generator=generator, dtype=torch.float64, requires_grad=True
        ),
        "target_output_weight": torch.randn(
            9, 5, generator=generator, dtype=torch.float64, requires_grad=True
        ),
        "target_logits": target_logits,
        "hard_labels": target_logits.detach().argmax(dim=-1),
        "previous_token_ids": torch.tensor([[2, 2, 4], [2, 6, 2]]),
        "valid_mask": torch.tensor([[True, True, False], [True, False, True]]),
        "slot_bins": torch.tensor([[0, 1, 2], [0, 1, 2]]),
        "loss_weights": (1.5, 0.25, 2.0),
        "token_chunk_size": 2,
        "tp_group": None,
    }


def _run_requested_none_group_drift(
    rank: int,
    world_size: int,
    init_method: str,
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        provider = build_dspark_provider(
            body=_CheckpointIdentity(),
            target_vocab_size=9,
            draft_vocab_size=9,
            hidden_size=5,
            markov_rank=3,
            confidence_enabled=False,
            confidence_with_markov=False,
            tensor_parallel_group=(
                None if rank == 0 else torch.distributed.group.WORLD
            ),
            dtype=torch.float64,
        )
        inputs = _provider_inputs()
        inputs["loss_weights"] = (1.0, 1.0, 0.0)

        if rank == 0:
            stats = provider.objective_stats(**inputs)
            assert torch.isfinite(stats.combined.numerators).all()
        else:
            with pytest.raises(ValueError, match="objective TP group must match"):
                provider.objective_stats(**inputs)
    finally:
        torch.distributed.destroy_process_group()


def _assert_requested_none_group_drift_terminates(tmp_path: Path) -> None:
    context = torch.multiprocessing.get_context("spawn")
    init_method = f"file://{tmp_path / 'requested-none-group-drift.init'}"
    processes = [
        context.Process(
            target=_run_requested_none_group_drift,
            args=(rank, 2, init_method),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join()
    assert not alive, "requested=None left a rank blocked in a TP collective"
    assert [process.exitcode for process in processes] == [0, 0]


def test_factory_preserves_public_split_vocab_shapes() -> None:
    """The exact public DeepSeek DSpark artifact uses the full target vocabulary."""
    provider = build_dspark_provider(
        body=_CheckpointIdentity(),
        target_vocab_size=151_936,
        draft_vocab_size=151_936,
        hidden_size=256,
        markov_rank=256,
        confidence_enabled=False,
        confidence_with_markov=False,
        device="meta",
    )

    assert provider.markov_head.target_vocab_size == 151_936
    assert provider.markov_head.draft_vocab_size == 151_936
    assert provider.markov_head.markov_w1.weight.shape == (151_936, 256)
    assert provider.markov_head.markov_w2.weight.shape == (151_936, 256)


def test_factory_returns_private_body_and_head_only_adapter() -> None:
    """Provider state cannot capture a caller-owned target embedding or LM head."""
    body = _CheckpointLinear(5, 5, bias=False).double()
    provider = build_dspark_provider(
        body=body,
        target_vocab_size=9,
        draft_vocab_size=9,
        hidden_size=5,
        markov_rank=3,
        confidence_enabled=True,
        confidence_with_markov=True,
        dtype=torch.float64,
    )

    assert type(provider).__name__.startswith("_")
    assert provider.body is body
    assert set(provider.state_dict()) == {
        "body.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "confidence_head.proj.weight",
        "confidence_head.proj.bias",
    }
    assert not any(
        forbidden in name
        for name, _ in provider.named_parameters()
        for forbidden in ("target", "lm_head", "embed_tokens", "mask_embedding")
    )


def test_provider_uses_live_head_without_owning_or_training_it() -> None:
    """The live target head projects draft hidden state but remains stop-gradient."""
    provider = build_dspark_provider(
        body=_CheckpointIdentity(),
        target_vocab_size=9,
        draft_vocab_size=9,
        hidden_size=5,
        markov_rank=3,
        confidence_enabled=True,
        confidence_with_markov=True,
        dtype=torch.float64,
    )
    inputs = _provider_inputs()

    stats = provider.objective_stats(**inputs)
    stats.combined.normalized(normalization_counts=stats.combined.counts).backward()

    target_output_weight = inputs["target_output_weight"]
    target_logits = inputs["target_logits"]
    draft_hidden = inputs["draft_hidden"]
    assert isinstance(target_output_weight, torch.Tensor)
    assert isinstance(target_logits, torch.Tensor)
    assert isinstance(draft_hidden, torch.Tensor)
    assert target_output_weight.grad is None
    assert target_logits.grad is None
    assert draft_hidden.grad is not None
    assert torch.isfinite(draft_hidden.grad).all()
    for parameter in provider.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_duplicate_markov_tokens_accumulate_into_one_replicated_w1_row() -> None:
    """Repeated previous tokens must accumulate both Markov and confidence gradients."""
    provider = build_dspark_provider(
        body=_CheckpointIdentity(),
        target_vocab_size=9,
        draft_vocab_size=9,
        hidden_size=5,
        markov_rank=3,
        confidence_enabled=True,
        confidence_with_markov=True,
        dtype=torch.float64,
    )
    inputs = _provider_inputs()
    stats = provider.objective_stats(**inputs)
    stats.combined.normalized(normalization_counts=stats.combined.counts).backward()

    gradient = provider.markov_head.markov_w1.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient[2]) > 0
    assert torch.equal(gradient[8], torch.zeros_like(gradient[8]))


def test_confidence_disabled_provider_has_no_confidence_parameters() -> None:
    """A confidence-disabled capability returns explicit zero raw confidence bins."""
    provider = build_dspark_provider(
        body=_CheckpointIdentity(),
        target_vocab_size=9,
        draft_vocab_size=9,
        hidden_size=5,
        markov_rank=3,
        confidence_enabled=False,
        confidence_with_markov=False,
        dtype=torch.float64,
    )
    inputs = _provider_inputs()
    inputs["loss_weights"] = (1.0, 1.0, 0.0)

    stats = provider.objective_stats(**inputs)

    assert provider.confidence_head is None
    assert not any("confidence" in name for name, _ in provider.named_parameters())
    torch.testing.assert_close(stats.confidence.numerators, torch.zeros(3))
    torch.testing.assert_close(stats.confidence.counts, torch.zeros(3))


def test_provider_rejects_mismatched_tp_head_and_objective_group() -> None:
    """The provider cannot silently use different TP groups for heads and loss."""
    provider = build_dspark_provider(
        body=_CheckpointIdentity(),
        target_vocab_size=9,
        draft_vocab_size=9,
        hidden_size=5,
        markov_rank=3,
        confidence_enabled=False,
        confidence_with_markov=False,
        dtype=torch.float64,
    )
    inputs = _provider_inputs()
    inputs["loss_weights"] = (1.0, 1.0, 0.0)
    inputs["tp_group"] = object()

    with pytest.raises(ValueError, match="TP group"):
        provider.objective_stats(**inputs)


def test_requested_none_group_drift_terminates_without_collective(
    tmp_path: Path,
) -> None:
    """A missing requested group must not strand a rank in its configured group."""
    _assert_requested_none_group_drift_terminates(tmp_path)


def test_factory_rejects_body_without_sharded_state_contract() -> None:
    """A body that cannot checkpoint must fail at construction, not at save time."""
    with pytest.raises(TypeError, match="sharded_state_dict"):
        build_dspark_provider(
            body=nn.Identity(),
            target_vocab_size=9,
            draft_vocab_size=9,
            hidden_size=5,
            markov_rank=3,
            confidence_enabled=False,
            confidence_with_markov=False,
            dtype=torch.float64,
        )


def test_provider_resolves_one_dp_cp_group_for_body_and_all_dspark_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CP-replicated body/heads must share one checkpoint replica group."""
    captured_body_metadata: list[dict[str, Any] | None] = []
    captured_head_groups: list[object | None] = []

    class _RecordingBody(nn.Identity):
        def sharded_state_dict(
            self,
            prefix: str = "",
            sharded_offsets: tuple[tuple[int, int, int], ...] = (),
            metadata: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            del prefix, sharded_offsets
            captured_body_metadata.append(metadata)
            return {}

    def _record_sharded_tensors(
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        axis_map: dict[str, int],
        sharded_offsets: tuple[tuple[int, int, int], ...],
        *,
        tp_group: object | None,
        dp_cp_group: object | None,
    ) -> dict[str, torch.Tensor]:
        del axis_map, sharded_offsets, tp_group
        captured_head_groups.append(dp_cp_group)
        return {f"{prefix}{name}": tensor for name, tensor in state_dict.items()}

    monkeypatch.setattr(
        "megatron.core.transformer.utils.make_sharded_tensors_for_checkpoint",
        _record_sharded_tensors,
    )
    dp_cp_group = object()
    resolved_metadata = {"dp_cp_group": dp_cp_group}
    monkeypatch.setattr(
        "nemo_rl.models.megatron.draft.dspark_provider.ensure_metadata_has_dp_cp_group",
        lambda metadata: resolved_metadata if metadata is None else metadata,
    )
    provider = build_dspark_provider(
        body=_RecordingBody(),
        target_vocab_size=9,
        draft_vocab_size=9,
        hidden_size=5,
        markov_rank=3,
        confidence_enabled=True,
        confidence_with_markov=True,
        dtype=torch.float64,
    )
    state = provider.sharded_state_dict(prefix="draft.", metadata=None)

    assert captured_body_metadata == [resolved_metadata]
    assert captured_head_groups == [dp_cp_group, dp_cp_group]
    assert set(state) == {
        "draft.markov_head.markov_w1.weight",
        "draft.markov_head.markov_w2.weight",
        "draft.confidence_head.proj.weight",
        "draft.confidence_head.proj.bias",
    }
