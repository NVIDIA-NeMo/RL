import pytest
import torch
from torch.distributed.tensor import Partial, Replicate, Shard

from nemo_rl.distributed.mx_helpers import (
    ModelExpressPublisherOptions,
    finish_model_express_publication,
    get_dtensor_local_shard,
    start_model_express_publication,
)


class FakeMesh:
    def __init__(self, coordinate: tuple[int, ...] | None, sizes: tuple[int, ...]):
        self._coordinate = coordinate
        self._sizes = sizes

    def get_coordinate(self) -> list[int] | None:
        return list(self._coordinate) if self._coordinate is not None else None

    def size(self, mesh_dim: int) -> int:
        return self._sizes[mesh_dim]


class FakeDTensor:
    def __init__(
        self,
        local: torch.Tensor,
        *,
        global_shape: tuple[int, ...],
        placements: tuple[object, ...],
        coordinate: tuple[int, ...] | None,
        mesh_sizes: tuple[int, ...],
    ):
        self._local = local
        self.shape = global_shape
        self.placements = placements
        self.device_mesh = FakeMesh(coordinate, mesh_sizes)

    def to_local(self) -> torch.Tensor:
        return self._local


class FakePublisher:
    def __init__(self, *, ready: bool = True):
        self.ready = ready
        self.reset_count = 0
        self.published_versions: list[int] = []

    def reset_tensors(self) -> None:
        self.reset_count += 1

    def publish(self, *, version: int) -> str:
        self.published_versions.append(version)
        return f"source-{version}"

    def mark_ready(self) -> bool:
        return self.ready


def test_publisher_options_preserve_megatron_role_overrides():
    options = ModelExpressPublisherOptions(
        mx_server_url="modelexpress-server:8001",
        nic_pin="auto",
        megatron_role_overrides={"linear_fc1": "column"},
    )

    assert options.megatron_role_overrides == {"linear_fc1": "column"}


def test_model_express_publication_lifecycle_repeats_per_version():
    publisher = FakePublisher()

    for version in (7, 8):
        start_model_express_publication(publisher)
        source_id = finish_model_express_publication(
            publisher,
            version=version,
            worker_rank=3,
        )
        assert source_id == f"source-{version}"

    assert publisher.reset_count == 2
    assert publisher.published_versions == [7, 8]


def test_model_express_publication_requires_ready_transition():
    publisher = FakePublisher(ready=False)

    with pytest.raises(RuntimeError, match="trainer rank 3 ready"):
        finish_model_express_publication(
            publisher,
            version=7,
            worker_rank=3,
        )


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


def test_get_dtensor_local_shard_rejects_partial_placement():
    tensor = FakeDTensor(
        torch.ones(2, 4),
        global_shape=(2, 4),
        placements=(Partial(),),
        coordinate=(0,),
        mesh_sizes=(1,),
    )

    with pytest.raises(NotImplementedError, match="partial DTensors"):
        get_dtensor_local_shard(tensor)


def test_get_dtensor_local_shard_rejects_rank_outside_mesh():
    tensor = FakeDTensor(
        torch.ones(2, 4),
        global_shape=(4, 4),
        placements=(Shard(0),),
        coordinate=None,
        mesh_sizes=(2,),
    )

    with pytest.raises(RuntimeError, match="not part of the DTensor device mesh"):
        get_dtensor_local_shard(tensor)
