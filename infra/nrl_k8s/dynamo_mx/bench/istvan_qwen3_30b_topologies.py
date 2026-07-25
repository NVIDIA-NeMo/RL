#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate GB200 infra configs for Istvan's Qwen3-30B benchmark topologies."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

TOPOLOGIES = {
    "a": {
        # Legacy slug/file name says dp2. With a 16-rank Megatron world,
        # TP2/PP1 implies DP8; EP4 overlaps those ranks and is not multiplied.
        "slug": "tp2ep4dp2-tp4dp4",
        "recipe": (
            "infra/nrl_k8s/examples/k8s_exemplars/V1/"
            "grpo_moe_qwen3_30b_tp2ep4dp2_tp4dp4_dynamo_mx.yaml"
        ),
        "inference_replicas": 4,
        "inference_tp": 4,
        "gpus_per_replica": 4,
    },
    "b": {
        "slug": "ep8pp2-tp2dp8",
        "recipe": (
            "infra/nrl_k8s/examples/k8s_exemplars/V1/"
            "grpo_moe_qwen3_30b_ep8pp2_tp2dp8_dynamo_mx.yaml"
        ),
        "inference_replicas": 8,
        "inference_tp": 2,
        "gpus_per_replica": 2,
    },
}

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

FABRICS = ("gke-rdma", "aws-efa")

# GKE exposes RoCE as per-NIC extended resources plus multi-network pod
# annotations. AWS exposes EFA as a single extended resource and needs neither.
GKE_RDMA_RESOURCE_PREFIX = "networking.gke.io.networks/"
GKE_RDMA_ANNOTATIONS = (
    "networking.gke.io/default-interface",
    "networking.gke.io/interfaces",
)
EFA_RESOURCE = "vpc.amazonaws.com/efa"

# Every RDMA pod requests the node's FULL complement of EFA adapters, not one per
# GPU. p6e-gb200.36xlarge has 4 GPUs and 4 EFA adapters paired 1:1 by PCI adjacency
# (GPU 29:00.0<->rdmap40s0 28:00.0, 3F<->3e, 9C<->9b, B2<->b1). Kubernetes does not
# coordinate the GPU and EFA device plugins, so a pod asking for fewer than all 4
# can be handed an adapter that is NOT its GPU's partner. NIXL then logs
# "PCI bus ID ... not found in accelerator-EFA mapping, returning all devices" and
# falls back to a non-local adapter: measured 14.3 Gbps vs 383 Gbps on the same
# node pair, a 27x collapse. Requesting all 4 makes the local partner always
# present. Consequence: at most ONE RDMA pod per node (see the topology-B note in
# pensieve/Clusters/AWS/02_efa_networking.md).
EFA_ADAPTERS_PER_NODE = 4

# NIXL's default UCX backend silently falls back to TCP on EFA, so the transport
# must be selected explicitly. FI_HMEM/GPUDirect needs FI_EFA_USE_DEVICE_RDMA;
# SHM is disabled so cross-pod traffic is forced onto the adapter. The GCP
# UCX_TLS / MX_RDMA_NIC_PIN knobs are UCX-only and inert here.
EFA_TRANSPORT_ENV = {
    "MX_NIXL_BACKEND": "LIBFABRIC",
    "FI_PROVIDER": "efa",
    "FI_EFA_USE_DEVICE_RDMA": "1",
    "FI_EFA_ENABLE_SHM_TRANSFER": "0",
}


def _strip_gke_rdma(node: Any) -> None:
    """Drop every GKE multi-network annotation and per-NIC rdma-* resource."""
    if isinstance(node, dict):
        for key in [
            key
            for key in node
            if isinstance(key, str) and key.startswith(GKE_RDMA_RESOURCE_PREFIX)
        ]:
            node.pop(key)
        for key in GKE_RDMA_ANNOTATIONS:
            node.pop(key, None)
        for value in node.values():
            _strip_gke_rdma(value)
    elif isinstance(node, list):
        for item in node:
            _strip_gke_rdma(item)


def _add_efa_to_k8s_resources(node: Any) -> None:
    """Request all EFA adapters in every *native* GPU requests/limits block.

    See EFA_ADAPTERS_PER_NODE for why this is not one-per-GPU.

    Keys on `nvidia.com/gpu`, which only appears in real PodSpec containers. The
    DGD service-level block uses a bare `gpu` key and a restricted schema, so it
    is handled by `_add_efa_to_dgd_resources` instead.
    """
    if isinstance(node, dict):
        for section in ("requests", "limits"):
            block = node.get(section)
            if not isinstance(block, dict) or "nvidia.com/gpu" not in block:
                continue
            if _as_positive_int(block["nvidia.com/gpu"]):
                block[EFA_RESOURCE] = str(EFA_ADAPTERS_PER_NODE)
        for value in node.values():
            _add_efa_to_k8s_resources(value)
    elif isinstance(node, list):
        for item in node:
            _add_efa_to_k8s_resources(item)


def _as_positive_int(value: Any) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _add_efa_to_dgd_resources(service: dict[str, Any]) -> None:
    """Add EFA to a DGD service, respecting the CRD's restricted resource schema.

    The DGD `resources.{requests,limits}` schema allows only
    {cpu, custom, gpu, gpuType, memory}; a bare `vpc.amazonaws.com/efa` key there
    is **silently pruned** by structural-schema pruning and the pod comes up with
    no EFA device. Extended resources must go under `custom`. Also mirrored onto
    extraPodSpec.mainContainer (a raw container spec, so arbitrary keys are legal)
    so the request survives regardless of how the operator merges the two.
    """
    resources = service.get("resources")
    if not isinstance(resources, dict):
        return
    gpus = None
    for section in ("requests", "limits"):
        block = resources.get(section)
        if isinstance(block, dict):
            gpus = gpus or _as_positive_int(block.get("gpu"))
    if not gpus:
        return
    adapters = str(EFA_ADAPTERS_PER_NODE)
    for section in ("requests", "limits"):
        block = resources.setdefault(section, {})
        block.setdefault("custom", {})[EFA_RESOURCE] = adapters

    main_container = service.setdefault("extraPodSpec", {}).setdefault(
        "mainContainer", {}
    )
    container_resources = main_container.setdefault("resources", {})
    for section in ("requests", "limits"):
        container_resources.setdefault(section, {})[EFA_RESOURCE] = adapters


def _apply_efa_pod_spec(pod_spec: dict[str, Any], container_key: str) -> None:
    """Add the EFA toleration and the caps RDMA registration needs."""
    tolerations = pod_spec.setdefault("tolerations", [])
    if not any(
        isinstance(item, dict) and item.get("key") == EFA_RESOURCE
        for item in tolerations
    ):
        tolerations.append({"key": EFA_RESOURCE, "operator": "Exists"})

    containers: list[dict[str, Any]] = []
    if container_key == "containers":
        containers = [c for c in pod_spec.get("containers", []) if isinstance(c, dict)]
    else:
        main = pod_spec.get(container_key)
        if isinstance(main, dict):
            containers = [main]
    for container in containers:
        security = container.setdefault("securityContext", {})
        capabilities = security.setdefault("capabilities", {})
        add = capabilities.setdefault("add", [])
        for capability in ("IPC_LOCK", "SYS_PTRACE"):
            if capability not in add:
                add.append(capability)


def _set_arg(args: list[str], flag: str, value: str) -> None:
    try:
        index = args.index(flag)
    except ValueError:
        args.extend([flag, value])
    else:
        if index + 1 >= len(args):
            args.append(value)
        else:
            args[index + 1] = value


def build_config(
    base: dict[str, Any],
    *,
    topology: str,
    rollout_image: str,
    trainer_image: str,
    namespace: str,
    mx_load_mode: str = "stock",
    fp8: bool = False,
    mx_reshard496: bool = False,
    fabric: str = "gke-rdma",
) -> dict[str, Any]:
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}; expected {tuple(TOPOLOGIES)}")
    if fabric not in FABRICS:
        raise ValueError(f"unknown fabric {fabric!r}; expected {FABRICS}")
    spec = TOPOLOGIES[topology]
    config = copy.deepcopy(base)
    slug = spec["slug"]
    if fp8:
        # Distinct DGD/Ray names so the FP8 arm never collides with the BF16 run.
        slug = f"{slug}-fp8"
    if mx_reshard496:
        slug = f"{slug}-r496"

    config["namespace"] = namespace
    # Allow either GB200 customer GPU pool (o7v or w0e). Pinning o7v left nine
    # otherwise-free RDMA-capable GB200 nodes unusable during gang scheduling.
    shared = config.get("_shared", {})
    for selector_key in ("headNodeSelector", "workerNodeSelector"):
        selector = shared.get(selector_key)
        if isinstance(selector, dict):
            selector.pop("cloud.google.com/gke-nodepool", None)
    # The Ray trainer (KubeRay head/workers) must run the NeMo-RL image that ships
    # Ray + Megatron; the instrumented benchmark image is a Dynamo/vLLM image with
    # no `ray` binary, so it belongs only on the rollout workers below.
    config["image"] = trainer_image
    if fabric == "aws-efa":
        transport_exports = "".join(
            f"export {name}={value}\n" for name, value in EFA_TRANSPORT_ENV.items()
        )
    else:
        transport_exports = "export UCX_TLS='^tcp'\nexport NIXL_UCX_TLS='^tcp'\n"
    config["launch"]["entrypoint"] = (
        "set -eu\n"
        "cd /opt/nemo-rl\n"
        f"{transport_exports}"
        "export MX_REFIT_TIMING_STDOUT=1\n"
        f"python -u examples/run_grpo.py --config {spec['recipe']} "
        f"+policy.generation.dynamo_cfg.dgd_name={namespace}-dyn-{slug}\n"
    )

    training = config["kuberay"]["training"]
    training["name"] = f"{namespace}-rc-{slug}"
    config["launch"].setdefault("attach", {})["training"] = training["name"]
    # Pin the real dynamo KAI queue. On gcp-dev-02 the KAI leaf queues are
    # {default, default-queue, dynamo, test}; the dynamo queue has unlimited GPU
    # quota. When the label is absent the cluster's Kyverno policy defaults GPU
    # pods to "backfill", which is not a queue on this cluster
    # (kai-scheduler -> "QueueDoesNotExist: Queue 'backfill'"), so we must set it
    # explicitly. Kyverno only injects a default when the label is missing, so an
    # explicit value takes precedence.
    training.setdefault("labels", {})["kai.scheduler/queue"] = "dynamo"
    ray_spec = training["spec"]
    ray_spec["headGroupSpec"]["template"]["spec"]["schedulerName"] = "kai-scheduler"
    worker_groups = ray_spec["workerGroupSpecs"]
    if len(worker_groups) != 1:
        raise ValueError("expected exactly one trainer worker group")
    trainer_workers = worker_groups[0]
    trainer_workers["replicas"] = 4
    trainer_workers["minReplicas"] = 4
    trainer_workers["maxReplicas"] = 4
    trainer_workers["numOfHosts"] = 1
    trainer_workers["template"]["spec"]["schedulerName"] = "kai-scheduler"
    # The cluster admission policy defaults GPU pods to 800Gi ephemeral storage,
    # which prevents eight otherwise-free GB200 nodes from fitting this
    # benchmark. Models live on the shared PVC; keep only bounded container
    # scratch and reduce over-sized CPU/host-memory reservations.
    trainer_container = trainer_workers["template"]["spec"]["containers"][0]
    trainer_resources = trainer_container["resources"]
    trainer_resources["requests"].update(
        {"cpu": "16", "memory": "192Gi", "ephemeral-storage": "100Gi"}
    )
    trainer_resources["limits"].update(
        {"cpu": "32", "memory": "320Gi", "ephemeral-storage": "200Gi"}
    )
    # Some GB200 nodes in this pool expose CUDA P2P peer access as disabled
    # ("P2P is disabled between NVLINK connected GPUs"), and NVLS/MNNVL multicast
    # is unsupported in-container, so the Megatron TP/EP/PP NCCL init dies with
    # "unhandled cuda error" on those nodes. Recipe-level megatron_cfg.env_vars
    # are applied too late to influence the communicator, so pin the SHM path in
    # the container env (inherited by every Ray/Megatron actor from process
    # start). Training step time is slower on the SHM path, but the refit metric
    # (MX wire + load_weights) is independent of the trainer's TP collectives.
    trainer_env = trainer_container.setdefault("env", [])
    _trainer_env_by_name = {
        str(item.get("name")): item
        for item in trainer_env
        if isinstance(item, dict) and item.get("name")
    }
    for _nccl_name, _nccl_val in (
        ("NCCL_P2P_DISABLE", "1"),
        ("NCCL_NVLS_ENABLE", "0"),
        ("NCCL_MNNVL_ENABLE", "0"),
    ):
        _trainer_env_by_name[_nccl_name] = {"name": _nccl_name, "value": _nccl_val}
    if mx_reshard496:
        _trainer_env_by_name["MX_MEGATRON_RESHARD496"] = {
            "name": "MX_MEGATRON_RESHARD496",
            "value": "1",
        }
    if fabric == "aws-efa":
        # Set on the container (not just the launch entrypoint) so every Ray
        # actor inherits it from process start, like the NCCL vars above.
        for _name, _value in EFA_TRANSPORT_ENV.items():
            _trainer_env_by_name[_name] = {"name": _name, "value": _value}
    trainer_container["env"] = list(_trainer_env_by_name.values())

    serving = config["dynamo"]["serving"]
    serving["name"] = f"{namespace}-dyn-{slug}"
    # The rollout DGD's Grove GangSet reads the KAI queue from this annotation.
    # The base manifest pins "backfill", which does not exist on gcp-dev-02
    # (DGD reconcile fails: "queue 'backfill' not found"). dgd.annotations are
    # merged last by build_dgd_manifest, so this overrides the manifest value.
    serving.setdefault("annotations", {})["nvidia.com/kai-scheduler-queue"] = "dynamo"
    worker = serving["overrides"]["services"]["VllmDecodeWorker"]
    worker["replicas"] = int(spec["inference_replicas"])
    worker["image"] = rollout_image
    worker["resources"]["requests"]["gpu"] = str(spec["gpus_per_replica"])
    worker["resources"]["limits"]["gpu"] = str(spec["gpus_per_replica"])
    # DGD ExtraPodSpec is a Kubernetes PodSpec and cannot contain metadata;
    # unknown `extraPodSpec.metadata` is pruned by the CRD. Move the GKE
    # multi-network annotations to the operator-supported service field.
    legacy_metadata = worker["extraPodSpec"].pop("metadata", None)
    if isinstance(legacy_metadata, dict):
        worker["extraPodMetadata"] = legacy_metadata
    container = worker["extraPodSpec"]["mainContainer"]
    container_resources = container.setdefault("resources", {})
    container_resources.setdefault("requests", {}).update(
        {"cpu": "16", "memory": "128Gi", "ephemeral-storage": "100Gi"}
    )
    container_resources.setdefault("limits", {}).update(
        {"cpu": "32", "memory": "800Gi", "ephemeral-storage": "200Gi"}
    )
    # nrl_k8s only fills mainContainer.image when it is unset, and the base DGD
    # manifest already pins a (non-instrumented) image there, so a service-level
    # `image:` override is silently ignored. Set the container image directly so
    # the rollout actually runs the instrumented refit-stage-v2 build.
    container["image"] = rollout_image
    args = container["args"]
    _set_arg(args, "--model", MODEL)
    _set_arg(args, "--served-model-name", MODEL)
    _set_arg(args, "--tensor-parallel-size", str(spec["inference_tp"]))
    if fp8:
        # BF16 trainer -> FP8 inference. vLLM dynamic fp8 quantizes the BF16
        # checkpoint at cold load and re-quantizes the BF16 tensors the MX
        # EP-gather streamed install hands to model.load_weights each refit.
        _set_arg(args, "--quantization", "fp8")
    # TP2 needs at least ~30 GiB/rank for BF16 model weights before vLLM can
    # reserve any KV blocks. 0.1 is below that floor on GB200 and fails engine
    # startup; 0.3 leaves ample headroom for streamed MX refit staging.
    _set_arg(args, "--gpu-memory-utilization", "0.3")
    mx_server_url = f"modelexpress-server.{namespace}.svc.cluster.local:8001"
    _set_arg(args, "--model-express-url", mx_server_url)
    env = container.setdefault("env", [])
    env_by_name = {
        str(item.get("name")): item
        for item in env
        if isinstance(item, dict) and item.get("name")
    }
    env_by_name.setdefault(
        "MX_REFIT_TIMING_STDOUT",
        {"name": "MX_REFIT_TIMING_STDOUT", "value": "1"},
    )
    env_by_name.setdefault(
        "MX_REFIT_TIMING_SCHEMA",
        {"name": "MX_REFIT_TIMING_SCHEMA", "value": "refit-stage-v2"},
    )
    env_by_name["MX_REFIT_TIMING_DIR"] = {
        "name": "MX_REFIT_TIMING_DIR",
        "value": "/tmp/mx-refit-timing",
    }
    env_by_name["MX_LOAD_MODE"] = {
        "name": "MX_LOAD_MODE",
        "value": mx_load_mode,
    }
    env_by_name["MODEL_EXPRESS_URL"] = {
        "name": "MODEL_EXPRESS_URL",
        "value": mx_server_url,
    }
    # This override replaces (rather than merges with) the operator-generated
    # container environment. Keep the request-plane endpoint explicit; without
    # it DistributedRuntime falls back to localhost and fails before vLLM/MX
    # initialization even though the platform NATS service is healthy.
    nats_server_url = f"nats://dynamo-platform-nats.{namespace}.svc.cluster.local:4222"
    env_by_name["NATS_SERVER"] = {
        "name": "NATS_SERVER",
        "value": nats_server_url,
    }
    env_by_name["MX_EP_GATHER_STAGING"] = {
        "name": "MX_EP_GATHER_STAGING",
        "value": "host",
    }
    env_by_name["MX_EP_GATHER_STREAM_INSTALL"] = {
        "name": "MX_EP_GATHER_STREAM_INSTALL",
        "value": "1",
    }
    env_by_name["MX_SCRATCH_ARENA_GB"] = {
        "name": "MX_SCRATCH_ARENA_GB",
        "value": "24",
    }
    if mx_reshard496:
        env_by_name["MX_MEGATRON_RESHARD496"] = {
            "name": "MX_MEGATRON_RESHARD496",
            "value": "1",
        }
        env_by_name["MX_NUM_TRAINER_SOURCES"] = {
            "name": "MX_NUM_TRAINER_SOURCES",
            "value": "16",
        }
        # #528 strided-descriptor bound. Any captured copy whose exact plan
        # exceeds this many RDMA segments is rerouted to a bounded full-source
        # contiguous staging pull + local view replay (FullPullSource), which
        # collapses the millions of 1-2 KB strided descriptors that capped the
        # pre-#528 baseline at 2.5-5.3 Gbps. 64 is the library default; set it
        # explicitly so the reshard policy is reproducible from the manifest.
        env_by_name["MX_RESHARD_MAX_SEGMENTS_PER_COPY"] = {
            "name": "MX_RESHARD_MAX_SEGMENTS_PER_COPY",
            "value": "64",
        }
    # Some GB200 nodes in this pool expose CUDA P2P peer access as disabled
    # ("P2P is disabled between NVLINK connected GPUs ... probably a hardware
    # issue"), and NVLS/MNNVL multicast is unsupported inside these containers,
    # so vLLM's intra-replica TP NCCL init fails with "unhandled cuda error" on
    # those nodes. Route the small TP decode group over the SHM path so every
    # replica comes up regardless of node placement. The refit measurement is
    # MX-wire + load_weights bound, not TP-collective bound, so decode-path
    # transport choice does not materially affect the reported refit timings.
    for _nccl_name, _nccl_val in (
        ("NCCL_P2P_DISABLE", "1"),
        ("NCCL_NVLS_ENABLE", "0"),
        ("NCCL_MNNVL_ENABLE", "0"),
    ):
        env_by_name[_nccl_name] = {"name": _nccl_name, "value": _nccl_val}
    # The base override's env list replaces the DGD manifest env and omits
    # HF_HOME, so the worker used the default cache and tried to pull the full
    # 30B from HuggingFace (429 Too Many Requests). Point it at the shared model
    # cache the trainer already populated (mounted at /mnt/rl-workspace) so the
    # cold load reads from disk instead of the network. Use the ${user:} template
    # (resolved by nrl_k8s) rather than the namespace: the cache lives under the
    # username (e.g. "kavink"), which differs from the namespace ("kavin").
    env_by_name["HF_HOME"] = {
        "name": "HF_HOME",
        "value": "/mnt/rl-workspace/${user:}/hf-cache",
    }
    env_by_name.setdefault("HF_HUB_OFFLINE", {"name": "HF_HUB_OFFLINE", "value": "1"})
    if fabric == "aws-efa":
        for _name, _value in EFA_TRANSPORT_ENV.items():
            env_by_name[_name] = {"name": _name, "value": _value}
    container["env"] = list(env_by_name.values())

    # The base DGD manifest hard-codes a "rl-workspace" PVC and a
    # "roce-mx-qwen3-4b" ResourceClaimTemplate from another user's environment;
    # neither exists in this namespace, so the gang cannot schedule
    # (PVC-not-found on the frontend, FailedResourceClaimCreation on the worker).
    # Point the workspace volume at the real bound PVC (shared-model-cache, the
    # same claim the trainer uses) and drop the RoCE claim -- RDMA is provided by
    # the networking.gke.io/networks requests already set above.
    workspace_volumes = [
        {
            "name": "rl-workspace",
            "persistentVolumeClaim": {"claimName": "shared-model-cache"},
        }
    ]
    services = serving["overrides"]["services"]
    for svc_name in ("Frontend", "VllmDecodeWorker"):
        svc = services.setdefault(svc_name, {})
        extra_pod_spec = svc.setdefault("extraPodSpec", {})
        extra_pod_spec["volumes"] = copy.deepcopy(workspace_volumes)
        main_container = extra_pod_spec.setdefault("mainContainer", {})
        service_env = {
            str(item.get("name")): item
            for item in main_container.setdefault("env", [])
            if isinstance(item, dict) and item.get("name")
        }
        service_env["NATS_SERVER"] = {
            "name": "NATS_SERVER",
            "value": nats_server_url,
        }
        main_container["env"] = list(service_env.values())
    worker["extraPodSpec"]["resourceClaims"] = []

    if fabric == "aws-efa":
        # Done last so it also covers the resource blocks rewritten above. The
        # GKE annotations/rdma-* resources are meaningless on EKS and would make
        # every pod unschedulable, so they are removed rather than left inert.
        _strip_gke_rdma(config)
        _add_efa_to_k8s_resources(config)
        _add_efa_to_dgd_resources(worker)
        _apply_efa_pod_spec(ray_spec["headGroupSpec"]["template"]["spec"], "containers")
        for worker_group in worker_groups:
            _apply_efa_pod_spec(worker_group["template"]["spec"], "containers")
        for svc_name in ("Frontend", "VllmDecodeWorker"):
            _apply_efa_pod_spec(
                services[svc_name].setdefault("extraPodSpec", {}), "mainContainer"
            )

    config.setdefault("labels", {})["mx.nvidia.com/benchmark"] = "qwen3-30b-istvan"
    config["labels"]["mx.nvidia.com/topology"] = slug
    config["labels"]["mx.nvidia.com/fabric"] = fabric
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", choices=sorted(TOPOLOGIES), required=True)
    # Instrumented Dynamo/vLLM image for the rollout workers (refit target side).
    parser.add_argument(
        "--image", "--rollout-image", dest="rollout_image", required=True
    )
    # NeMo-RL image with Ray + Megatron for the trainer gang. Defaults to whatever
    # the base infra already pins, which is a known-good full image.
    parser.add_argument("--trainer-image", dest="trainer_image", default=None)
    parser.add_argument("--namespace", default="kavin")
    parser.add_argument(
        "--mx-load-mode",
        choices=("stock", "direct"),
        default="stock",
        help="vLLM install path: stock model.load_weights or direct MDL",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="BF16 trainer -> FP8 inference (vLLM --quantization fp8)",
    )
    parser.add_argument(
        "--mx-reshard496",
        action="store_true",
        help="publish HF aliases and receive through merged #496 planning",
    )
    parser.add_argument(
        "--fabric",
        choices=FABRICS,
        default="gke-rdma",
        help=(
            "RDMA fabric to target. gke-rdma keeps the GKE multi-network/RoCE "
            "wiring; aws-efa strips it and requests vpc.amazonaws.com/efa with "
            "the libfabric NIXL backend (requires -efa images)"
        ),
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(
            "infra/nrl_k8s/examples/k8s_exemplars/V1/"
            "grpo_moe_qwen3_30b_ep8_tp2_dynamo_mx.gb200.infra.yaml"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    trainer_image = args.trainer_image or base.get("image")
    if not trainer_image:
        raise SystemExit(
            "no trainer image: base infra has no top-level 'image' and "
            "--trainer-image was not provided"
        )
    generated = build_config(
        base,
        topology=args.topology,
        rollout_image=args.rollout_image,
        trainer_image=trainer_image,
        namespace=args.namespace,
        mx_load_mode=args.mx_load_mode,
        fp8=args.fp8,
        mx_reshard496=args.mx_reshard496,
        fabric=args.fabric,
    )
    defaults = generated.get("defaults")
    if isinstance(defaults, str) and not Path(defaults).is_absolute():
        generated["defaults"] = str((args.base.parent / defaults).resolve())
    manifest = generated["dynamo"]["serving"].get("manifest")
    if isinstance(manifest, str) and not Path(manifest).is_absolute():
        generated["dynamo"]["serving"]["manifest"] = str(
            (args.base.parent / manifest).resolve()
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(generated, sort_keys=False),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
