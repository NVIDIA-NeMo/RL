# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import os
import socket
import sys
import time
from typing import NamedTuple, NotRequired, Optional, TypedDict

import ray
from ray.util.placement_group import (
    PlacementGroup,
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusterConfig(TypedDict):
    gpus_per_node: int
    num_nodes: int
    # Port range for the distributed master address (TCPStore / NCCL rendezvous)
    # and per-worker available ports used by RayVirtualCluster.  These ports are
    # kept below the OS ephemeral range (32768-60999 on stock Linux) to avoid
    # TOCTOU collisions with kernel-assigned source ports.  When absent,
    # RayVirtualCluster falls back to DEFAULT_MASTER_PORT_RANGE_LOW/HIGH
    # (1400-1999).  See ray.sub for the full port layout.
    master_port_range_low: NotRequired[int]
    master_port_range_high: NotRequired[int]
    segment_size: NotRequired[
        int | None
    ]  # Nodes per NVLink domain segment for topology-aware alignment; None to disable


# Get the directory path of the current module and the root of the package
dir_path = os.path.dirname(os.path.abspath(__file__))
git_root = os.path.abspath(os.path.join(dir_path, "../.."))


class PY_EXECUTABLES:
    SYSTEM = sys.executable

    # Use NeMo-RL direct dependencies.
    BASE = f"uv run --locked --directory {git_root}"

    # Use NeMo-RL direct dependencies and vllm.
    VLLM = f"uv run --locked --extra vllm --directory {git_root}"

    # Use NeMo-RL direct dependencies and fsdp.
    FSDP = f"uv run --locked --extra fsdp --directory {git_root}"

    # Use NeMo-RL direct dependencies and nemo-automodel.
    AUTOMODEL = f"uv run --locked --extra automodel --directory {git_root}"

    # Use NeMo-RL direct dependencies and Megatron.
    MCORE = f"uv run --locked --extra mcore --directory {git_root}"

    # Use NeMo-Gym dependencies
    NEMO_GYM = f"uv run --locked --extra nemo_gym --directory {git_root}"

    # Use NeMo-RL direct dependencies and SGLang.
    SGLANG = f"uv run --locked --extra sglang --directory {git_root}"

    # Use NeMo-RL direct dependencies and TRT-LLM.
    TRTLLM = f"uv run --locked --extra trtllm --directory {git_root}"


# Default port ranges — kept below the OS ephemeral range.  On some DGX/GB200
# nodes the ephemeral floor is as low as 9000 (32768 on stock Linux), so every
# service port is pinned below 9000 to avoid TOCTOU collisions.  See ray.sub for
# the full layout including Ray's own GCS / worker gRPC ports.
#
#   1400-1999    Master address / TCPStore       (cluster.master_port_range_low/high)
#   3000-4999    NeMo RL generation HTTP servers + SGLang engine NCCL/dist_init
#                                                 (policy.generation.port_range_low/high)
#   5000-5999    NeMo Gym HTTP servers           (env.nemo_gym.port_range_low/high)
#   7000-8999    vLLM engine rendezvous          (VLLM_PORT env var, 100-port spacing)
#   8600-8799    SGLang router                   (DEFAULT_SGLANG_ROUTER_PORT_RANGE_*, hard-coded;
#                                                 carved out of the vLLM band — only one rollout
#                                                 backend runs at a time)
#   8800-8999    SGLang Prometheus metrics       (DEFAULT_SGLANG_PROMETHEUS_PORT_RANGE_*, hard-coded)
DEFAULT_GENERATION_PORT_RANGE_LOW = 3000
DEFAULT_GENERATION_PORT_RANGE_HIGH = 4999
DEFAULT_GYM_PORT_RANGE_LOW = 5000
DEFAULT_GYM_PORT_RANGE_HIGH = 5999
# vLLM TP/DP rendezvous ports.  Each engine gets PORTS_PER_ENGINE ports starting
# at LOW + engine_index * PORTS_PER_ENGINE.  With 8 GPUs and TP=1 (8 engines):
# 7000 + 8*100 = 7800, still below the 9000 ephemeral floor.
DEFAULT_VLLM_PORT_RANGE_LOW = 7000
DEFAULT_VLLM_PORTS_PER_ENGINE = 100
# SGLang control-plane ports, carved out of the top of the vLLM rendezvous band —
# safe because only one rollout backend runs at a time, and a vLLM run only
# climbs past 8600 with >=16 engines on a single node.  Both bands also steer
# clear of the Ray dashboard carve-out at 8265 (see ray.sub).
DEFAULT_SGLANG_ROUTER_PORT_RANGE_LOW = 8600
DEFAULT_SGLANG_ROUTER_PORT_RANGE_HIGH = 8799
DEFAULT_SGLANG_PROMETHEUS_PORT_RANGE_LOW = 8800
DEFAULT_SGLANG_PROMETHEUS_PORT_RANGE_HIGH = 8999
# Master address / TCPStore range, tucked below the Ray worker-gRPC band (2000+).
DEFAULT_MASTER_PORT_RANGE_LOW = 1400
DEFAULT_MASTER_PORT_RANGE_HIGH = 1999

# ---------------------------------------------------------------------------
# Topology resource keys
# ---------------------------------------------------------------------------
# These constants define the Ray custom-resource keys that ray.sub injects
# into each worker node at cluster start-up. The probe pipeline is:
#
#   ray.sub  (topology_probe.sh)    -- parses nvidia-smi -q for ClusterUUID
#                                   -- parses SLURM_TOPOLOGY_ADDR for topo_rank
#                                   -- prefixes ClusterUUID with NVLINK_DOMAIN_PREFIX
#                                   -- registers both as Ray custom resources
#   virtual_cluster.py              -- reads these resources to sort ranks
#
# If you rename any of the below keys, you must also update the corresponding strings in ray.sub

NVLINK_DOMAIN_PREFIX = "nvlink_domain_"
"""Ray resource key prefix for the NVLink domain.
Each node registers one resource ``nvlink_domain_<ClusterUUID>: 1``
where ClusterUUID is parsed directly from ``nvidia-smi -q`` output by ray.sub.
Nodes sharing the same key belong to the same NVLink switch fabric (e.g. one GB200 NVL72 rack)."""

TOPO_RANK_KEY = "topo_rank"
"""Ray resource key for the SLURM topological rank.
Derived from ``SLURM_TOPOLOGY_ADDR`` (when ``SLURM_TOPOLOGY_ADDR_PATTERN=block.node``),
falling back to ``SLURM_PROCID + 2`` on worker nodes (head node is pinned to ``1``),
then to hostname digits when SLURM is unavailable. Values are always ``>= 1`` so that
Ray does not drop the custom resource (Ray drops value-0 custom resources).
Used to sort nodes within and across NVLink domains so rank assignment follows physical topology."""

NVLINK_DOMAIN_UNKNOWN = "unknown"
"""Sentinel returned when no NVLink domain info is available for a node."""

TOPO_RANK_UNKNOWN: int = -1
"""Sentinel returned when no topological rank is available for a node."""


@ray.remote  # pragma: no cover
def _get_node_ip_and_free_port(
    port_range_low: int = DEFAULT_MASTER_PORT_RANGE_LOW,
    port_range_high: int = DEFAULT_MASTER_PORT_RANGE_HIGH,
) -> tuple[str, int]:
    return _get_node_ip_local(), _get_free_port_local(port_range_low, port_range_high)


def _get_node_ip_local() -> str:
    # Get the IP address of the current node
    node_ip = ray._private.services.get_node_ip_address()

    return node_ip


def _bind_socket_in_range(
    sock: socket.socket,
    port_range_low: int,
    port_range_high: int,
    max_retries: int = 50,
) -> int:
    """Try to bind *sock* to a random port in [port_range_low, port_range_high).

    Raises ``RuntimeError`` after *max_retries* failed attempts.
    """
    import random

    for _ in range(max_retries):
        port = random.randint(port_range_low, port_range_high - 1)
        try:
            sock.bind(("", port))
            return port
        except OSError:
            continue
    raise RuntimeError(
        f"Could not find a free port in range [{port_range_low}, {port_range_high}) "
        f"after {max_retries} attempts."
    )


def _get_free_port_local(
    port_range_low: int = DEFAULT_MASTER_PORT_RANGE_LOW,
    port_range_high: int = DEFAULT_MASTER_PORT_RANGE_HIGH,
) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        port = _bind_socket_in_range(s, port_range_low, port_range_high)
        s.listen(1)

    return port


def _get_free_consecutive_ports_local(
    port_range_low: int,
    port_range_high: int,
    consecutive: int = 1,
    start_port: Optional[int] = None,
) -> int:
    """Find ``consecutive`` contiguous bindable ports and return the base.

    Scans upward from *start_port* within [port_range_low, port_range_high).
    *start_port* lets a caller thread a per-node cursor so successive blocks do
    not overlap. Raises ``RuntimeError`` if no such block exists in the range.
    """
    assert consecutive >= 1, f"consecutive must be >= 1, got {consecutive}"
    base = port_range_low if start_port is None else max(start_port, port_range_low)
    while base + consecutive - 1 < port_range_high:
        socks: list[socket.socket] = []
        try:
            for offset in range(consecutive):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("", base + offset))
                s.listen(1)
                socks.append(s)
            return base
        except OSError:
            base += 1
        finally:
            for s in socks:
                s.close()
    raise RuntimeError(
        f"Could not find {consecutive} consecutive free ports in "
        f"[{port_range_low}, {port_range_high})."
    )


def init_ray(log_dir: Optional[str] = None) -> None:
    """Initialise Ray.

    Try to attach to an existing local cluster.
    If that cluster uses the same CUDA_VISIBLE_DEVICES or Slurm managed tag we will reuse it.
    Otherwise, we will detach and start a fresh local cluster.

    Args:
        log_dir: Optional directory to store Ray logs and temp files.
    """
    # Strip MPI/PMIx/SLURM launcher vars from the driver env before they get
    # captured into runtime_env (both by `dict(os.environ)` below and by
    # RayWorkerGroup, which re-reads os.environ). Otherwise they are forwarded
    # into every actor, where they cause two failures in the TRT-LLM generation
    # worker: (1) PMIX_/SLURM_STEP_ID make OMPI's MPI_Init abort with "OMPI was
    # not built with SLURM's PMI support"; (2) any residual OMPI_/MPI_/SLURM_
    # var makes TRT-LLM think it runs under an MPI launcher and pick the MPI
    # orchestrator (MPI_Comm_Spawn -> MPI_ERR_SPAWN) instead of the Ray
    # orchestrator. init_ray() runs before any worker group is built, so
    # popping here cleans the env for all downstream captures.
    for _k in list(os.environ):
        if _k.startswith(("PMIX_", "PMI_", "MPI_", "OMPI_", "SLURM_")):
            os.environ.pop(_k, None)

    # Set up runtime environment
    env_vars = dict(os.environ)
    env_vars.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None)
    runtime_env = {
        "env_vars": env_vars,  # Pass thru all user environment variables
    }

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
    # sort cvd to ensure consistent tag
    cvd = ",".join(sorted(cvd.split(",")))
    cvd_tag_prefix = "nrl_tag_"
    cvd_tag = f"{cvd_tag_prefix}{cvd.replace(',', '_')}"

    # Try to attach to an existing cluster
    try:
        ray.init(
            address="auto",
            log_to_driver=True,
            include_dashboard=False,
            runtime_env=runtime_env,
            _temp_dir=os.path.abspath(log_dir) if log_dir else None,
        )

        cluster_res = ray.cluster_resources()

        # Check reusability for NeMo-RL managed local clusters
        if any(k.startswith(cvd_tag_prefix) for k in cluster_res):
            # Reuse if the driver's cvd_tag matches a tag in the cluster.
            # This is for reusing a previously self-started local cluster.
            if cvd_tag in cluster_res:
                logger.info(
                    f"Connected to existing Ray cluster (driver CVD_TAG '{cvd_tag}' matched): {cluster_res}"
                )
                return

            # If neither reuse condition is met, but we connected to *something*
            logger.info(
                f"Existing Ray cluster found ({cluster_res}) but it does not meet reuse criteria. "
                f"Driver's cvd_tag: '{[k for k in cluster_res if k.startswith(cvd_tag_prefix)][0]}'. Expected cvd_tag: '{cvd_tag}'. "
                "Starting a new local cluster..."
            )
            ray.shutdown()

            # Clear driver-side package cache so working_dir is re-uploaded
            import importlib

            import ray._private.runtime_env.packaging as _pkg

            importlib.reload(_pkg)

        # Always reuse if it's an externally managed cluster.
        else:
            logger.info(f"Connected to existing Ray cluster: {cluster_res}")
            return

    except ConnectionError:
        logger.debug("No existing Ray cluster found, will start a new one.")
        # If ConnectionError, proceed to start a new local cluster without further action here.
        # Clear driver-side package cache so working_dir is re-uploaded
        ray.shutdown()
        pass

    # Start a brand-new local cluster
    # Reuse `runtime_env` but drop `working_dir` to avoid packaging the whole repo (prevents ray OSError: Failed to download runtime_env file package issue)
    local_runtime_env = dict(runtime_env)
    local_runtime_env.pop("working_dir", None)

    ray.init(
        log_to_driver=True,
        include_dashboard=True,
        runtime_env=local_runtime_env,
        _temp_dir=os.path.abspath(log_dir) if log_dir else None,
        resources={cvd_tag: 1},
    )
    logger.info(
        f"Started local cluster with tag '{cvd_tag}': {ray.cluster_resources()}"
    )


@ray.remote(num_gpus=1)
def _get_gpu_id_info() -> tuple[int, str, int]:  # pragma: no cover
    """Return (gpu_id, nvlink_domain, topo_rank) for the current worker's bundle.

    Reads custom resources set by ray.sub (see NVLINK_DOMAIN_PREFIX / TOPO_RANK_KEY).
    """
    gpu_id = ray.get_gpu_ids()[0]
    nvlink_domain = NVLINK_DOMAIN_UNKNOWN
    topo_rank = TOPO_RANK_UNKNOWN
    runtime_ctx = ray.get_runtime_context()
    node_id = runtime_ctx.get_node_id()
    all_node_resources: dict = {}
    for node in ray.nodes():
        if node.get("NodeID") == node_id:
            all_node_resources = node.get("Resources", {})
            break
    for key, val in all_node_resources.items():
        if key.startswith(NVLINK_DOMAIN_PREFIX):
            nvlink_domain = key
        if key == TOPO_RANK_KEY:
            topo_rank = int(val)
    return gpu_id, nvlink_domain, topo_rank


def get_reordered_bundle(
    pg: PlacementGroup,
    segment_size: int | None = None,
    gpus_per_node: int | None = None,
) -> tuple[list[int], list[int], tuple[str, ...]]:
    """Return bundle indices and GPU IDs ordered by physical topology.

    Spins up one short-lived task per bundle (``_get_gpu_id_info``) to discover
    each bundle's physical GPU ID, NVLink domain, and ``topo_rank``, then orders
    bundles via ``_sort_bundle_indices_by_topology``:

      * No topology info available -> sort by ``(node_id, gpu_id)``.
      * Topology info available    -> sort by ``(domain_min_topo_rank, topo_rank, gpu_id)``.
      * ``segment_size`` set        -> additionally drop bundles from NVLink
        domains that cannot form a complete ``segment_size``-node segment.

    Args:
        pg: Placement group whose bundles to reorder.
        segment_size: Nodes per NVLink domain segment for topology-aware
            alignment; ``None`` disables segment filtering.
        gpus_per_node: Required when ``segment_size`` is set.

    Returns:
        ``(reordered_bundle_indices, reordered_gpu_ids, nvlink_domain_per_bundle_index)``
        where ``nvlink_domain_per_bundle_index`` is indexed by *original* bundle index.
    """
    pg_data = placement_group_table(pg)
    num_bundles = len(pg_data["bundles"])
    bundle_to_node_ids = pg_data["bundles_to_node_id"]

    # Fire-and-forget tasks to get GPU id + topology info per bundle.
    # Tasks reuse the raylet's worker pool and avoid GCS actor registrations.
    info_refs = [
        _get_gpu_id_info.options(
            num_cpus=0.01,  # both small to enable assignment in colocated case
            num_gpus=0.01,
            resources=None,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            ),
        ).remote()
        for i in range(num_bundles)
    ]
    infos = ray.get(info_refs)
    gpu_ids = [info[0] for info in infos]
    nvlink_domains = [info[1] for info in infos]
    topo_ranks = [info[2] for info in infos]

    bundle_data = [
        (gpu_ids[i], nvlink_domains[i], topo_ranks[i], bundle_to_node_ids[i])
        for i in range(num_bundles)
    ]
    reordered_bundle_indices = _sort_bundle_indices_by_topology(
        bundle_data,
        segment_size=segment_size,
        gpus_per_node=gpus_per_node,
    )
    reordered_gpu_ids = [gpu_ids[i] for i in reordered_bundle_indices]
    return reordered_bundle_indices, reordered_gpu_ids, tuple(nvlink_domains)


class ResourceInsufficientError(Exception):
    """Exception raised when the cluster does not have enough resources to satisfy the requested configuration."""


def get_ray_cluster_topology() -> dict[str, tuple[str, int]]:
    """Query all alive Ray nodes for their NVLink domain and topo_rank.

    Returns:
        Dict mapping node_id -> (nvlink_domain, topo_rank).
        nvlink_domain is NVLINK_DOMAIN_UNKNOWN and topo_rank is TOPO_RANK_UNKNOWN
        if topology info is unavailable.
    """
    topology: dict[str, tuple[str, int]] = {}
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        node_id = node.get("NodeID", "")
        resources = node.get("Resources", {})
        nvlink_domain = NVLINK_DOMAIN_UNKNOWN
        topo_rank = TOPO_RANK_UNKNOWN
        for key, val in resources.items():
            if key.startswith(NVLINK_DOMAIN_PREFIX):
                nvlink_domain = key
            if key == TOPO_RANK_KEY:
                topo_rank = int(val)
        topology[node_id] = (nvlink_domain, topo_rank)
    return topology


def select_segment_nodes(
    topology: dict[str, tuple[str, int]],
    segment_size: int,
    num_nodes: int,
) -> tuple[list[str], list[str]]:
    """Partition Ray node IDs into segment-aligned selected nodes and remainder.

    Greedily selects complete segments (segment_size nodes) from each NVLink domain,
    sorted by topological order, until num_nodes is reached.

    Args:
        topology: Dict mapping node_id -> (nvlink_domain, topo_rank) from get_ray_cluster_topology().
        segment_size: Number of nodes per NVLink domain segment.
        num_nodes: Total number of nodes to select.

    Returns:
        (selected_node_ids, remaining_node_ids): Selected nodes are in topological order.

    Raises:
        ValueError: If segment_size does not evenly divide num_nodes.
        ResourceInsufficientError: If not enough complete segments can be formed.
    """
    if num_nodes % segment_size != 0:
        raise ValueError(
            f"num_nodes ({num_nodes}) must be divisible by "
            f"segment_size ({segment_size})."
        )

    domain_nodes: dict[str, list[tuple[str, int]]] = {}
    for nid, (domain, topo_rank) in topology.items():
        # Skip nodes with no NVLink-domain info. They all collapse into a single
        # NVLINK_DOMAIN_UNKNOWN pseudo-domain with TOPO_RANK_UNKNOWN (-1), so they
        # would sort first and be selected — but the resulting placement-group
        # constraint {NVLINK_DOMAIN_UNKNOWN: 0.001} names a Ray resource that
        # ray.sub never registers, so the bundle can never schedule. Excluding
        # them here means we only ever pin to real, registered NVLink domains
        # (and these nodes fall through to remaining_node_ids).
        if domain == NVLINK_DOMAIN_UNKNOWN:
            continue
        domain_nodes.setdefault(domain, []).append((nid, topo_rank))
    for domain in domain_nodes:
        domain_nodes[domain].sort(key=lambda x: x[1])

    # Sort domains by the minimum topo_rank of their nodes.
    sorted_domains = sorted(
        domain_nodes.items(),
        key=lambda item: item[1][0][1],
    )

    num_segments_needed = num_nodes // segment_size
    selected_node_ids: list[str] = []
    segments_taken = 0

    for domain, nodes in sorted_domains:
        if segments_taken >= num_segments_needed:
            break
        segments_available = len(nodes) // segment_size
        segments_to_take = min(segments_available, num_segments_needed - segments_taken)
        nodes_to_take = segments_to_take * segment_size
        for nid, _ in nodes[:nodes_to_take]:
            selected_node_ids.append(nid)
        segments_taken += segments_to_take

    if segments_taken < num_segments_needed:
        domain_summary = {d: len(ns) for d, ns in sorted_domains}
        raise ResourceInsufficientError(
            f"Cannot form {num_segments_needed} complete segments of {segment_size} nodes. "
            f"Nodes per domain: {domain_summary}. "
            f"Need {num_nodes} nodes total."
        )

    remaining_node_ids = [nid for nid in topology if nid not in set(selected_node_ids)]

    domains_used = set()
    for nid in selected_node_ids:
        domains_used.add(topology[nid][0])
    logger.info(
        f"[TOPOLOGY] Segment selection: {segments_taken} segments of {segment_size} nodes "
        f"from {len(domains_used)} NVLink domains -> {len(selected_node_ids)} selected nodes, "
        f"{len(remaining_node_ids)} remaining nodes"
    )

    return selected_node_ids, remaining_node_ids


def prepare_segment_topology(
    segment_size: int | None,
    num_nodes: int,
    *,
    topology: dict[str, tuple[str, int]] | None = None,
    role: str = "training",
) -> tuple[list[dict[str, float]] | None, list[str], dict[str, tuple[str, int]]]:
    """Compute node resource constraints for topology-aware cluster placement.

    Fetches cluster topology if not provided, selects segment-aligned nodes,
    and returns per-node domain constraints ready for ``RayVirtualCluster``.

    Args:
        segment_size: Nodes per NVLink domain segment. ``None`` disables topology logic.
        num_nodes: Number of nodes to select.
        topology: Pre-fetched topology dict; fetched automatically when ``None``.
        role: Label used in progress messages (e.g. ``"training"``, ``"inference"``).

    Returns:
        ``(node_resource_constraints, remaining_node_ids, topology)``

        - *node_resource_constraints*: per-node domain-pinning dicts for
          ``RayVirtualCluster``, or ``None`` when ``segment_size`` is ``None`` or
          no NVLink domain info is available.
        - *remaining_node_ids*: node IDs not selected; pass the corresponding
          sub-topology to a follow-up call to allocate an inference cluster.
        - *topology*: the topology dict used (empty dict when ``segment_size`` is
          ``None``, for safe sub-topology slicing by callers).
    """
    if segment_size is None:
        return None, [], {}

    if topology is None:
        topology = get_ray_cluster_topology()

    has_topology = any(
        domain != NVLINK_DOMAIN_UNKNOWN for domain, _ in topology.values()
    )
    if not has_topology:
        print(
            f"  ⚠ segment_size={segment_size} is set but no NVLink domain info "
            "found, falling back to unordered allocation",
            flush=True,
        )
        return None, list(topology.keys()), topology

    selected_node_ids, remaining_node_ids = select_segment_nodes(
        topology, segment_size, num_nodes
    )
    node_resource_constraints = [{topology[nid][0]: 0.001} for nid in selected_node_ids]
    print(
        f"  ✓ Topology-aware allocation: {num_nodes} {role} nodes in "
        f"{len(set(topology[nid][0] for nid in selected_node_ids))} NVLink domains "
        f"(segment_size={segment_size})",
        flush=True,
    )
    return node_resource_constraints, remaining_node_ids, topology


def _sort_bundle_indices_by_topology(
    bundle_data: list[tuple[int, str, int, str]],
    segment_size: int | None = None,
    gpus_per_node: int | None = None,
) -> list[int]:
    """Compute topology-aware sort order for bundle indices.

    When topology information is available: sort by (domain_min_topo_rank, topo_rank, gpu_id).
    When segment_size is set: additionally validate that each NVLink domain contributes
    complete segments (segment_size nodes), discarding bundles from incomplete domains.
    Else: sort by (node_id, gpu_id).

    Args:
        bundle_data: For each bundle i, (gpu_id, nvlink_domain, topo_rank, node_id).
        segment_size: If set, number of nodes per NVLink domain segment. Bundles from
            domains with fewer than segment_size nodes are excluded.
        gpus_per_node: Required when segment_size is set. Number of GPUs per node.

    Returns:
        List of bundle indices in sorted order.

    Raises:
        ValueError: If segment_size is set but gpus_per_node is not.
    """
    if segment_size is not None and gpus_per_node is None:
        raise ValueError("gpus_per_node is required when segment_size is set")

    if not bundle_data:
        return []

    has_topology = any(
        b[1] != NVLINK_DOMAIN_UNKNOWN or b[2] != TOPO_RANK_UNKNOWN for b in bundle_data
    )

    # Without topology info, fall back to deterministic (node_id, gpu_id) ordering.
    if not has_topology:
        basic = [
            (i, node_id, gpu_id)
            for i, (gpu_id, _, _, node_id) in enumerate(bundle_data)
        ]
        return [idx for idx, _, _ in sorted(basic, key=lambda x: (x[1], x[2]))]

    class BundleInfo(NamedTuple):
        idx: int
        node_id: str
        gpu_id: int
        domain: str
        topo_rank: int

    bundle_infos = [
        BundleInfo(
            idx=i,
            node_id=node_id,
            gpu_id=gpu_id,
            domain=nvlink_domain,
            topo_rank=topo_rank,
        )
        for i, (gpu_id, nvlink_domain, topo_rank, node_id) in enumerate(bundle_data)
    ]

    if segment_size is not None:
        assert gpus_per_node is not None
        domain_bundles: dict[str, list[BundleInfo]] = {}
        for info in bundle_infos:
            domain_bundles.setdefault(info.domain, []).append(info)

        filtered: list[BundleInfo] = []
        for domain, bundles in domain_bundles.items():
            domain_node_count = len(set(b.node_id for b in bundles))
            usable_nodes = (domain_node_count // segment_size) * segment_size
            usable_gpus = usable_nodes * gpus_per_node
            bundles.sort(key=lambda x: (x.topo_rank, x.gpu_id))
            kept = bundles[:usable_gpus]
            discarded = bundles[usable_gpus:]
            if discarded:
                logger.info(
                    f"[TOPOLOGY] Domain {domain}: keeping {len(kept)} bundles "
                    f"({usable_nodes} nodes), discarding {len(discarded)} bundles "
                    f"({domain_node_count - usable_nodes} incomplete segment nodes)"
                )
            filtered.extend(kept)
        bundle_infos = filtered

    domain_to_min_topo_rank: dict[str, int] = {}
    for info in bundle_infos:
        if (
            info.domain not in domain_to_min_topo_rank
            or info.topo_rank < domain_to_min_topo_rank[info.domain]
        ):
            domain_to_min_topo_rank[info.domain] = info.topo_rank

    indices = [
        info.idx
        for info in sorted(
            bundle_infos,
            key=lambda x: (domain_to_min_topo_rank[x.domain], x.topo_rank, x.gpu_id),
        )
    ]
    for rank, idx in enumerate(indices):
        gpu_id, nvlink_domain, topo_rank, node_id = bundle_data[idx]
        logger.info(
            f"[TOPOLOGY] Rank {rank} -> GPU {gpu_id} on node {node_id} "
            f"(nvlink_domain: {nvlink_domain}, topo_rank: {topo_rank})"
        )
    return indices


class RayVirtualCluster:
    """Creates a virtual distributed cluster using Ray placement groups.

    This class simplifies distributed training setup by:
    - Creating placement groups that represent logical compute nodes
    - Allocating GPU and CPU resources for distributed workers
    - Managing communication between distributed processes

    - Bundle: A resource allocation unit (ex: 4 GPUs on a single node)
    - Worker: A process that performs computation (model training/inference)
    - Node: A physical or virtual machine containing multiple bundles
    """

    def __init__(
        self,
        bundle_ct_per_node_list: list[int],
        use_gpus: bool = True,
        max_colocated_worker_groups: int = 1,
        num_gpus_per_node: int = 8,
        name: str = "",
        placement_group_strategy: str = "SPREAD",
        port_range_low: Optional[int] = None,
        port_range_high: Optional[int] = None,
        segment_size: int | None = None,
        node_resource_constraints: list[dict[str, float]] | None = None,
    ):
        """Initialize a virtual cluster using Ray placement groups.

        Args:
            bundle_ct_per_node_list: List specifying GPU bundles per node
                                    (e.g., [2,2] creates 2 nodes with 2 GPU bundles each)
            use_gpus: Whether to allocate GPU resources
            max_colocated_worker_groups: Maximum number of worker groups that can be colocated
            num_gpus_per_node: Number of GPUs per node
            name: Name prefix for placement groups
            placement_group_strategy: Ray placement group strategy ("STRICT_PACK", "PACK", or "SPREAD")
            port_range_low: Lower bound (inclusive) of the port range for master address allocation.
                Falls back to DEFAULT_MASTER_PORT_RANGE_LOW if None.
            port_range_high: Upper bound (exclusive) of the port range for master address allocation.
                Falls back to DEFAULT_MASTER_PORT_RANGE_HIGH if None.
            segment_size: Nodes per NVLink domain segment for topology-aware alignment.
                         When set, _sort_bundle_indices_by_topology trims incomplete domain segments.
            node_resource_constraints: Per-logical-node extra Ray resource requirements.
                         Length must match bundle_ct_per_node_list. Each dict is merged into
                         every bundle spec for that node, pinning it to a physical domain.
                         Built from NVLink domain resources injected by ray.sub.
                         Example: [{"nvlink_domain_<uuid>": 0.001}] * 16 pins 16 nodes to a single NVLink domain.
        """
        if node_resource_constraints is not None:
            assert len(node_resource_constraints) == len(bundle_ct_per_node_list), (
                f"node_resource_constraints length ({len(node_resource_constraints)}) must match "
                f"bundle_ct_per_node_list length ({len(bundle_ct_per_node_list)})"
            )

        self._bundle_ct_per_node_list = bundle_ct_per_node_list
        self._world_size = sum(self._bundle_ct_per_node_list)
        self._node_placement_groups: Optional[list[PlacementGroup]] = None
        self._sorted_bundle_indices: Optional[list[int]] = None
        self._nvlink_domain_per_bundle_index: Optional[tuple[str, ...]] = None

        self.num_gpus_per_node = num_gpus_per_node
        self.use_gpus = use_gpus
        if use_gpus:
            assert num_gpus_per_node > 0, (
                "num_gpus_per_node must be greater than 0 if using GPUs"
            )
        self.max_colocated_worker_groups = max_colocated_worker_groups
        self.name = name
        self.placement_group_strategy = placement_group_strategy
        self.port_range_low = (
            port_range_low
            if port_range_low is not None
            else DEFAULT_MASTER_PORT_RANGE_LOW
        )
        self.port_range_high = (
            port_range_high
            if port_range_high is not None
            else DEFAULT_MASTER_PORT_RANGE_HIGH
        )
        self._allocated_master_ports: set[int] = set()
        self.segment_size = segment_size
        self.node_resource_constraints = node_resource_constraints

    def _init_placement_groups(
        self, strategy: str | None = None, use_unified_pg: bool = False
    ) -> list[PlacementGroup]:
        """Creates placement groups based on whether cross-node model parallelism is needed.

        Args:
            strategy: Ray placement group strategy (defaults to self.placement_group_strategy)
            use_unified_pg: If True, create a single unified placement group.
                          If False, create per-node placement groups.

        Returns:
            List of placement groups
        """
        if self._node_placement_groups is not None:
            return self._node_placement_groups

        if strategy is None:
            strategy = self.placement_group_strategy

        # Add retry logic that was previously in __init__
        max_retries = int(os.environ.get("NRL_VIRTUAL_CLUSTER_MAX_RETRIES", 6))
        assert max_retries > 0, (
            f"NRL_VIRTUAL_CLUSTER_MAX_RETRIES={max_retries} must be an integer greater than 0"
        )

        for i in range(max_retries):
            try:
                self._node_placement_groups = self._create_placement_groups_internal(
                    strategy, use_unified_pg
                )
                if use_unified_pg and self.use_gpus:
                    self._sorted_bundle_indices = self._get_sorted_bundle_indices()
                return self._node_placement_groups
            except ResourceInsufficientError as e:
                print(e)
                print(
                    f"Retrying placement group creation... {i + 1}/{max_retries}. Next retry in {2**i} seconds."
                )
                time.sleep(2**i)
                continue
        raise ResourceInsufficientError(
            f"Maximum number of retries reached ({max_retries}). Cluster resources may be insufficient or cluster itself is highly unstable. Please check your cluster configuration and your cluster logs."
        )

    def _create_placement_groups_internal(
        self, strategy: str, use_unified_pg: bool = False
    ) -> list[PlacementGroup]:
        """Internal method to create placement groups without retry logic."""
        # Check available resources in the Ray cluster
        cluster_resources = ray.cluster_resources()
        total_available_gpus = int(cluster_resources.get("GPU", 0))
        total_available_cpus = int(cluster_resources.get("CPU", 0))

        # Calculate required resources
        total_requested_gpus = (
            sum(self._bundle_ct_per_node_list) if self.use_gpus else 0
        )
        total_requested_cpus = (
            sum(self._bundle_ct_per_node_list) * self.max_colocated_worker_groups
        )

        # Validate resources
        if self.use_gpus and total_requested_gpus > total_available_gpus:
            raise ResourceInsufficientError(
                f"Not enough GPUs available. Requested {total_requested_gpus} GPUs, but only {total_available_gpus} are available in the cluster."
            )

        if total_requested_cpus > total_available_cpus:
            raise ResourceInsufficientError(
                f"Not enough CPUs available. Requested {total_requested_cpus} CPUs, but only {total_available_cpus} are available in the cluster."
            )

        num_cpus_per_bundle = self.max_colocated_worker_groups
        # num_gpus_per_bundle == 1 indicates that there is 1 GPU per process
        num_gpus_per_bundle = 1 if self.use_gpus else 0

        def _make_bundle(node_idx: int) -> dict:
            bundle: dict = {"CPU": num_cpus_per_bundle, "GPU": num_gpus_per_bundle}
            if (
                self.node_resource_constraints
                and self.node_resource_constraints[node_idx]
            ):
                bundle.update(self.node_resource_constraints[node_idx])
            return bundle

        placement_groups = []
        if use_unified_pg:
            # Create a single unified placement group for cross-node model parallelism
            all_bundles = []
            for node_idx, bundle_count in enumerate(self._bundle_ct_per_node_list):
                for _ in range(bundle_count):
                    all_bundles.append(_make_bundle(node_idx))

            placement_groups = [
                placement_group(
                    bundles=all_bundles, strategy=strategy, name=f"{self.name}-unified"
                )
            ]
        else:
            # Create per-node placement groups to respect bundle_ct_per_node_list
            for node_idx, bundle_count in enumerate(self._bundle_ct_per_node_list):
                if bundle_count > 0:
                    node_bundles = [_make_bundle(node_idx) for _ in range(bundle_count)]
                    pg = placement_group(
                        bundles=node_bundles,
                        strategy="PACK",  # Use PACK to keep bundles together
                        name=f"{self.name}-node{node_idx}",
                    )
                    placement_groups.append(pg)

        # Add timeout to prevent hanging indefinitely
        try:
            ray.get(
                [pg.ready() for pg in placement_groups], timeout=180
            )  # 3-minute timeout
        except (TimeoutError, ray.exceptions.GetTimeoutError):
            # Clean up any created placement groups
            for pg in placement_groups:
                try:
                    remove_placement_group(pg)
                except Exception:
                    pass
            raise TimeoutError(
                "Timed out waiting for placement groups to be ready. The cluster may not have enough resources "
                "to satisfy the requested configuration, or the resources may be busy with other tasks."
            )

        return placement_groups

    def get_placement_groups(self) -> list[PlacementGroup]:
        # Initialize placement groups if not already created
        if self._node_placement_groups is None:
            self._init_placement_groups()

        assert self._node_placement_groups is not None, (
            "Placement groups must be initialized before calling get_placement_groups"
        )
        return [pg for pg in self._node_placement_groups if pg.bundle_specs]

    def world_size(self) -> int:
        return self._world_size

    def node_count(self) -> int:
        return sum(1 for count in self._bundle_ct_per_node_list if count > 0)

    def get_available_address_and_port(
        self, pg_idx: int, bundle_idx: int
    ) -> tuple[str, int]:
        """Gets an available address and port for the given placement group index and bundle index.

        Returns:
            Tuple of (address, port)
        """
        results = self.get_available_addresses_and_ports_batch([(pg_idx, bundle_idx)])
        return results[0]

    def get_available_addresses_and_ports_batch(
        self,
        pg_bundle_pairs: list[tuple[int, int]],
        batch_size: int = 256,
    ) -> list[tuple[str, int]]:
        """Discovers available addresses and ports for multiple bundles in parallel.

        Fires all remote tasks up front, then collects results in batches via ray.wait()
        to avoid putting too many objects into the Ray object
        store at once.
        See https://docs.ray.io/en/latest/ray-core/patterns/ray-get-too-many-objects.html

        Args:
            pg_bundle_pairs: List of ``(pg_idx, bundle_idx)`` pairs.
            batch_size: Maximum number of ready futures to fetch at once.

        Returns:
            List of ``(address, port)`` tuples in the same order as ``pg_bundle_pairs``.
        """
        placement_groups = self.get_placement_groups()
        refs: list[ray.ObjectRef] = []
        for pg_idx, bundle_idx in pg_bundle_pairs:
            pg = (
                placement_groups[0]
                if len(placement_groups) == 1
                else placement_groups[pg_idx]
            )
            if not pg.bundle_specs:
                raise RuntimeError(
                    "No valid placement groups found to get available address and port"
                )

            refs.append(
                _get_node_ip_and_free_port.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=bundle_idx
                    ),
                    # Need to explicitly set to 0 since it's possible for this to be unschedulable if all CPUs are already in use.
                    num_cpus=0,
                ).remote(self.port_range_low, self.port_range_high)
            )

        if len(refs) <= batch_size:
            return ray.get(refs)

        # ray.wait returns refs in completion order, so map each ref back to
        # its input index to preserve worker-to-port ordering.
        ref_to_idx = {ref: idx for idx, ref in enumerate(refs)}
        results: list[Optional[tuple[str, int]]] = []
        for _ in refs:
            results.append(None)
        remaining = list(refs)
        while remaining:
            ready, remaining = ray.wait(
                remaining, num_returns=min(batch_size, len(remaining))
            )
            for ref, value in zip(ready, ray.get(ready)):
                results[ref_to_idx[ref]] = value

        ordered_results: list[tuple[str, int]] = []
        for result in results:
            assert result is not None
            ordered_results.append(result)
        return ordered_results

    def get_master_address_and_port(self) -> tuple[str, int]:
        """Gets the master address and port for the distributed training setup.

        Each call returns a unique port that has not been returned by previous
        calls on this cluster instance.  This prevents NCCL process-group
        collisions when multiple worker groups (e.g. policy and value) share the
        same cluster and node.

        Returns:
            Tuple of (address, port)
        """
        # Get placement groups if not already created
        if not self._node_placement_groups:
            self.get_placement_groups()

        if self._sorted_bundle_indices is not None:
            pg_idx, bundle_idx = 0, self._sorted_bundle_indices[0]
        else:
            pg_idx, bundle_idx = 0, 0

        max_retries = 10
        for _ in range(max_retries):
            addr, port = self.get_available_address_and_port(pg_idx, bundle_idx)
            if port not in self._allocated_master_ports:
                self._allocated_master_ports.add(port)
                return addr, port

        raise RuntimeError(
            f"Failed to find a unique master port after {max_retries} retries. "
            f"Already allocated ports: {self._allocated_master_ports}"
        )

    def _get_sorted_bundle_indices(self) -> Optional[list[int]]:
        """Gets the sorted bundle indices for the placement groups.

        Returns:
            List of bundle indices in sorted order.
        """
        if self._node_placement_groups is None:
            raise ValueError(
                "Placement groups must be initialized before calling _get_sorted_bundle_indices"
            )

        if not self.use_gpus:
            self._nvlink_domain_per_bundle_index = None
            return None

        if len(self._node_placement_groups) != 1:
            self._nvlink_domain_per_bundle_index = None
            return None

        reordered_bundle_indices, _, nvlink_domain_per_bundle_index = (
            get_reordered_bundle(
                self._node_placement_groups[0],
                segment_size=self.segment_size,
                gpus_per_node=self.num_gpus_per_node if self.segment_size else None,
            )
        )
        self._nvlink_domain_per_bundle_index = nvlink_domain_per_bundle_index
        num_bundles = len(nvlink_domain_per_bundle_index)
        assert len(reordered_bundle_indices) == num_bundles, (
            f"Topology sort returned {len(reordered_bundle_indices)} bundle indices "
            f"but the cluster has {num_bundles}. Some NVLink domains had incomplete "
            f"segments and were trimmed. Ensure cluster.segment_size divides evenly "
            f"into each domain's node count and that node_resource_constraints are set "
            f"to pin nodes to complete segments before creating this cluster."
        )
        return reordered_bundle_indices

    def shutdown(self) -> bool:
        """Cleans up and releases all resources associated with this virtual cluster.

        This includes removing all placement groups and resetting the internal state.

        This method is idempotent and can be safely called multiple times.
        """
        # Skip if Ray is already gone (typically from __del__ during _Py_Finalize
        # after Ray's atexit teardown). Any Ray API call would trigger a fatal
        # core_worker_process.cc:88 CHECK. Placement groups die with Ray.
        if not ray.is_initialized():
            return True
        if self._node_placement_groups is not None:
            # Remove all placement groups
            for pg in self._node_placement_groups:
                try:
                    remove_placement_group(pg)
                except Exception as e:
                    # Log but continue if a placement group can't be removed
                    print(f"Error removing placement group {pg.id}: {e}")

            # Reset internal state
            self._node_placement_groups = None
            self._sorted_bundle_indices = None
            self._nvlink_domain_per_bundle_index = None

        return True

    def __del__(self) -> None:
        """Shutsdown the virtual cluster when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call shutdown and the pointer to
        the cluster is lost due to leaving a function scope. It's always recommended that the
        user calls shutdown().
        """
        self.shutdown()
