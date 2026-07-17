from types import SimpleNamespace

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard

from nemo_rl.distributed.mx_helpers import (
    MxConfig,
    get_dtensor_local_shard,
    reset_v2_publisher_tensors,
)


class FakeMesh:
    def __init__(self, coordinate: tuple[int, ...], sizes: tuple[int, ...]):
        self._coordinate = coordinate
        self._sizes = sizes

    def get_coordinate(self) -> list[int]:
        return list(self._coordinate)

    def size(self, mesh_dim: int) -> int:
        return self._sizes[mesh_dim]


class FakeDTensor:
    def __init__(
        self,
        local: torch.Tensor,
        *,
        global_shape: tuple[int, ...],
        placements: tuple[object, ...],
        coordinate: tuple[int, ...],
        mesh_sizes: tuple[int, ...],
    ):
        self._local = local
        self.shape = global_shape
        self.placements = placements
        self.device_mesh = FakeMesh(coordinate, mesh_sizes)

    def to_local(self) -> torch.Tensor:
        return self._local


def test_mx_config_preserves_megatron_role_overrides():
    config = MxConfig.from_dict({"megatron_role_overrides": {"linear_fc1": "column"}})

    assert config.megatron_role_overrides == {"linear_fc1": "column"}


def test_get_dtensor_local_shard_describes_uneven_last_shard():
    tensor = FakeDTensor(
        torch.ones(2, 4),
        global_shape=(5, 4),
        placements=(Shard(0),),
        coordinate=(1,),
        mesh_sizes=(2,),
    )

    local, shard_spec = get_dtensor_local_shard(tensor)

    assert local is tensor._local
    assert shard_spec is not None
    assert shard_spec.global_shape == (5, 4)
    assert shard_spec.shard_axis == 0
    assert shard_spec.local_shard_range == (3, 5)


def test_get_dtensor_local_shard_accepts_replicated_tensor():
    tensor = FakeDTensor(
        torch.ones(2, 4),
        global_shape=(2, 4),
        placements=(Replicate(),),
        coordinate=(0,),
        mesh_sizes=(1,),
    )

    local, shard_spec = get_dtensor_local_shard(tensor)

    assert local is tensor._local
    assert shard_spec is None


def test_get_dtensor_local_shard_rejects_multiple_shard_axes():
    tensor = FakeDTensor(
        torch.ones(2, 2),
        global_shape=(4, 4),
        placements=(Shard(0), Shard(1)),
        coordinate=(0, 0),
        mesh_sizes=(2, 2),
    )

    with pytest.raises(NotImplementedError, match="one DTensor shard axis"):
        get_dtensor_local_shard(tensor)


def test_reset_v2_publisher_tensors_prefers_public_api():
    calls: list[str] = []
    publisher = SimpleNamespace(reset_tensors=lambda: calls.append("reset"))

    reset_v2_publisher_tensors(publisher)

    assert calls == ["reset"]


def test_reset_v2_publisher_tensors_supports_released_v2_api():
    publisher = SimpleNamespace(
        _registry=[object()],
        _registered_tensors={"weight": object()},
    )

    reset_v2_publisher_tensors(publisher)

    assert publisher._registry == []
    assert publisher._registered_tensors == {}
