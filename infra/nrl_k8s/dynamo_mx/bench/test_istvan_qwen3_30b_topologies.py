from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

MODULE_PATH = Path(__file__).with_name("istvan_qwen3_30b_topologies.py")
SPEC = importlib.util.spec_from_file_location(
    "istvan_qwen3_30b_topologies", MODULE_PATH
)
assert SPEC and SPEC.loader
topologies = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(topologies)

BASE = Path(
    "infra/nrl_k8s/examples/k8s_exemplars/V1/"
    "grpo_moe_qwen3_30b_ep8_tp2_dynamo_mx.gb200.infra.yaml"
)


@pytest.mark.parametrize(
    ("topology", "replicas", "tp", "gpus"),
    [
        ("a", 4, 4, "4"),
        ("b", 8, 2, "2"),
    ],
)
def test_build_config_uses_requested_16_to_16_topology(
    topology: str,
    replicas: int,
    tp: int,
    gpus: str,
) -> None:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))

    config = topologies.build_config(
        base,
        topology=topology,
        rollout_image="registry/image:instrumented",
        trainer_image="registry/nemo-rl:full",
        namespace="bench",
    )

    training = config["kuberay"]["training"]
    trainer = training["spec"]["workerGroupSpecs"][0]
    assert config["launch"]["attach"]["training"] == training["name"]
    # KAI queue must target the real dynamo leaf queue, never "default"/"backfill".
    assert training.get("labels", {}).get("kai.scheduler/queue") == "dynamo"
    # Rollout DGD must target the dynamo KAI queue, not the base's "backfill".
    assert (
        config["dynamo"]["serving"]["annotations"]["nvidia.com/kai-scheduler-queue"]
        == "dynamo"
    )
    # Workspace volume must point at the real bound PVC on both DGD services, and
    # the non-existent RoCE resource claim must be cleared on the worker.
    services = config["dynamo"]["serving"]["overrides"]["services"]
    for svc_name in ("Frontend", "VllmDecodeWorker"):
        vols = services[svc_name]["extraPodSpec"]["volumes"]
        claim = vols[0]["persistentVolumeClaim"]["claimName"]
        assert claim == "shared-model-cache"
        service_env = {
            item["name"]: item["value"]
            for item in services[svc_name]["extraPodSpec"]["mainContainer"]["env"]
        }
        assert (
            service_env["NATS_SERVER"]
            == "nats://dynamo-platform-nats.bench.svc.cluster.local:4222"
        )
    assert services["VllmDecodeWorker"]["extraPodSpec"]["resourceClaims"] == []
    # Worker must run the instrumented image on its actual container, not just
    # the (ignored) service-level image key.
    assert (
        services["VllmDecodeWorker"]["extraPodSpec"]["mainContainer"]["image"]
        == "registry/image:instrumented"
    )
    # Trainer gang runs the Ray-capable NeMo-RL image; rollout runs the
    # instrumented Dynamo image.
    assert config["image"] == "registry/nemo-rl:full"
    assert (
        config["dynamo"]["serving"]["overrides"]["services"]["VllmDecodeWorker"][
            "image"
        ]
        == "registry/image:instrumented"
    )
    assert trainer["replicas"] == 4
    assert trainer["minReplicas"] == 4
    assert trainer["maxReplicas"] == 4
    assert trainer["template"]["spec"]["schedulerName"] == "kai-scheduler"

    worker = config["dynamo"]["serving"]["overrides"]["services"]["VllmDecodeWorker"]
    assert worker["replicas"] == replicas
    assert worker["resources"]["requests"]["gpu"] == gpus
    assert worker["resources"]["limits"]["gpu"] == gpus
    assert replicas * int(gpus) == 16
    args = worker["extraPodSpec"]["mainContainer"]["args"]
    assert args[args.index("--model") + 1] == topologies.MODEL
    assert args[args.index("--tensor-parallel-size") + 1] == str(tp)
    env = {
        item["name"]: item["value"]
        for item in worker["extraPodSpec"]["mainContainer"]["env"]
    }
    assert env["MX_REFIT_TIMING_STDOUT"] == "1"
    assert (
        env["NATS_SERVER"] == "nats://dynamo-platform-nats.bench.svc.cluster.local:4222"
    )
    assert "MX_REFIT_TIMING_STDOUT=1" in config["launch"]["entrypoint"]


def test_build_config_enables_merged_496_on_both_sides() -> None:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))

    config = topologies.build_config(
        base,
        topology="b",
        rollout_image="registry/image:instrumented",
        trainer_image="registry/nemo-rl:full",
        namespace="bench",
        mx_reshard496=True,
    )

    trainer = config["kuberay"]["training"]["spec"]["workerGroupSpecs"][0]
    trainer_env = {
        item["name"]: item["value"]
        for item in trainer["template"]["spec"]["containers"][0]["env"]
    }
    worker = config["dynamo"]["serving"]["overrides"]["services"]["VllmDecodeWorker"]
    rollout_env = {
        item["name"]: item["value"]
        for item in worker["extraPodSpec"]["mainContainer"]["env"]
    }

    assert trainer_env["MX_MEGATRON_RESHARD496"] == "1"
    assert rollout_env["MX_MEGATRON_RESHARD496"] == "1"
    assert rollout_env["MX_NUM_TRAINER_SOURCES"] == "16"
    assert rollout_env["MX_RESHARD_MAX_SEGMENTS_PER_COPY"] == "64"
    assert config["kuberay"]["training"]["name"].endswith("-r496")
