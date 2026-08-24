import os

import pytest
import ray
import torch

from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    PY_EXECUTABLES,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup


@ray.remote(num_gpus=1)  # pragma: no cover
class GradNormTPTestActor:
    """Builds one TP-sharded and one TP-replicated gradient and checks the norm."""

    def __init__(self, tp_size):
        self.tp_size = tp_size

    def run_grad_norm_check(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Replicate, Shard, distribute_tensor

        from nemo_rl.models.dtensor.parallelize import _is_tp_duplicate, get_grad_norm

        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(0)
        mesh = init_device_mesh("cuda", (1, self.tp_size), mesh_dim_names=("dp", "tp"))
        dp_group = mesh["dp"].get_group()
        tp_group = mesh["tp"].get_group()

        # Same seed on every rank, so the replicated gradient really is identical
        # across the TP group, exactly like a norm/bias gradient in training.
        torch.manual_seed(1234)
        replicated_grad = torch.randn(4096, device="cuda")
        sharded_grad = torch.randn(4096 * self.tp_size, device="cuda")

        replicated_param = torch.nn.Parameter(
            distribute_tensor(
                torch.zeros_like(replicated_grad), mesh, [Replicate(), Replicate()]
            )
        )
        replicated_param.grad = distribute_tensor(
            replicated_grad, mesh, [Replicate(), Replicate()]
        )
        sharded_param = torch.nn.Parameter(
            distribute_tensor(
                torch.zeros_like(sharded_grad), mesh, [Replicate(), Shard(0)]
            )
        )
        sharded_param.grad = distribute_tensor(
            sharded_grad, mesh, [Replicate(), Shard(0)]
        )

        params = [replicated_param, sharded_param]
        expected = torch.linalg.vector_norm(
            torch.cat([replicated_grad, sharded_grad]).to(torch.float64)
        ).item()
        # What the norm would be if each TP rank's copy of the replicated
        # gradient were summed instead of counted once.
        inflated = (
            (
                self.tp_size * replicated_grad.to(torch.float64).pow(2).sum()
                + sharded_grad.to(torch.float64).pow(2).sum()
            )
            .sqrt()
            .item()
        )

        actual = get_grad_norm(params, dp_cp_group=dp_group, tp_group=tp_group)
        inf_norm = get_grad_norm(
            params, dp_cp_group=dp_group, tp_group=tp_group, norm_type=torch.inf
        )
        expected_inf = torch.cat([replicated_grad, sharded_grad]).abs().max().item()

        return {
            "rank": int(os.environ["RANK"]),
            "classified_replicated": _is_tp_duplicate(replicated_param.grad, tp_group),
            "classified_sharded": _is_tp_duplicate(sharded_param.grad, tp_group),
            "expected": expected,
            "inflated": inflated,
            "actual": actual,
            "inf_actual": inf_norm,
            "inf_expected": expected_inf,
        }


GRAD_NORM_TP_ACTOR_FQN = f"{GradNormTPTestActor.__module__}.GradNormTPTestActor"


@pytest.fixture
def register_grad_norm_tp_actor():
    original = ACTOR_ENVIRONMENT_REGISTRY.get(GRAD_NORM_TP_ACTOR_FQN)
    ACTOR_ENVIRONMENT_REGISTRY[GRAD_NORM_TP_ACTOR_FQN] = PY_EXECUTABLES.SYSTEM

    yield GRAD_NORM_TP_ACTOR_FQN

    if GRAD_NORM_TP_ACTOR_FQN in ACTOR_ENVIRONMENT_REGISTRY:
        if original is None:
            del ACTOR_ENVIRONMENT_REGISTRY[GRAD_NORM_TP_ACTOR_FQN]
        else:
            ACTOR_ENVIRONMENT_REGISTRY[GRAD_NORM_TP_ACTOR_FQN] = original


@pytest.mark.parametrize("tp_size", [2])
def test_get_grad_norm_counts_tp_replicated_grads_once(
    register_grad_norm_tp_actor, tp_size
):
    """TP-replicated gradients must contribute to the global norm exactly once.

    Tensor parallelism leaves norm weights, biases and every module outside the
    parallel plan replicated, so all TP ranks hold the same gradient for them.
    Summing the per-rank squared norms over the TP group therefore counts those
    gradients tp_size times and over-reports the norm, which in turn makes
    gradient clipping fire too aggressively.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < tp_size:
        pytest.skip(
            f"Not enough GPUs available. Need {tp_size}, got {torch.cuda.device_count()}"
        )

    cluster = RayVirtualCluster(bundle_ct_per_node_list=[tp_size], use_gpus=True)
    try:
        sharding = NamedSharding(layout=list(range(tp_size)), names=["tp"])
        builder = RayWorkerBuilder(register_grad_norm_tp_actor, tp_size)
        worker_group = RayWorkerGroup(
            cluster=cluster,
            remote_worker_builder=builder,
            workers_per_node=None,
            sharding_annotations=sharding,
        )
        results = ray.get(
            worker_group.run_all_workers_single_data("run_grad_norm_check")
        )
        worker_group.shutdown(force=True)
    finally:
        cluster.shutdown()

    for result in results:
        print(
            f"[rank {result['rank']}] expected={result['expected']:.6f} "
            f"actual={result['actual']:.6f} "
            f"inflated_if_double_counted={result['inflated']:.6f} "
            f"ratio_actual_over_expected={result['actual'] / result['expected']:.6f}"
        )
        assert result["classified_replicated"] is True
        assert result["classified_sharded"] is False
        assert result["actual"] == pytest.approx(result["expected"], rel=1e-5)
        # Guard against the test passing for the wrong reason: the two candidate
        # values must actually differ for tp_size > 1.
        assert result["inflated"] != pytest.approx(result["expected"], rel=1e-5)
        assert result["inf_actual"] == pytest.approx(result["inf_expected"], rel=1e-5)
