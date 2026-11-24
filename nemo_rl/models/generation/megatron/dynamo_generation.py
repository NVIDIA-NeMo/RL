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
"""Generation over NeMo-RL-owned Megatron prefill/decode engines served through Dynamo.

One engine is one ``MegatronGeneration`` over its own worker group, the same
dedicated inference policy the Megatron backend already builds, with its role
in its config. ``MegatronDynamoGeneration`` makes the engines exist, in refit
rank order, and is their refit side.
"""

import copy
import dataclasses
import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, cast

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, _get_node_ip_local
from nemo_rl.models.generation.disagg_layout import ROLES, Role
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
    reject_unenforceable_refit_deadline,
)
from nemo_rl.models.generation.megatron.config import MCoreGenerationConfig
from nemo_rl.models.generation.megatron.dynamo_config import (
    MegatronDynamoCfg,
    megatron_dynamo_cfg,
    plan_megatron_engines,
    role_overrides,
    validate_megatron_dynamo_generation,
)
from nemo_rl.models.generation.megatron.megatron_generation import MegatronGeneration
from nemo_rl.models.megatron.router_replay import router_replay_enabled
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.weight_sync.interfaces import WeightSynchronizer

if TYPE_CHECKING:
    from ray.actor import ActorHandle

    from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
    from nemo_rl.distributed.worker_groups import RayWorkerGroup
    from nemo_rl.models.policy.lm_policy import Policy
    from nemo_rl.weight_sync.membership import RefitMembership

LOGGER = logging.getLogger(__name__)


def role_policy_config(
    policy_config: Mapping[str, Any],
    role: Role,
    overrides: Mapping[str, int],
) -> dict[str, Any]:
    """The policy config a worker of one role builds its engine from.

    A deep copy of ``policy_config`` with the role in
    ``generation._disaggregation_role`` (what makes the worker build the handoff
    engine for that role) and ``overrides`` (the role's
    ``mcore_generation_config.prefill_engine`` / ``decode_engine`` keys) merged into
    ``generation.mcore_generation_config``, which is where the dedicated
    inference policy takes its layout from. ``generation.dynamo`` is turned
    off: each engine is a plain Megatron engine sized by its own layout, and
    only ``MegatronDynamoGeneration`` serves through Dynamo. ``policy_config``
    is left as is.

    Args:
        policy_config: The full ``policy`` config.
        role: ``prefill`` or ``decode``.
        overrides: Keys from ``dynamo_config.role_overrides``; empty for
            a role that keeps the shared ``mcore_generation_config``.

    Returns:
        A fresh config for one engine of ``role``.
    """
    config = copy.deepcopy(dict(policy_config))
    generation: dict[str, Any] = dict(config["generation"])
    # Off, or the engine's NVLink span would be a whole prefill/decode replica
    # instead of its own width.
    generation["dynamo"] = False
    generation["_disaggregation_role"] = role
    if overrides:
        generation["mcore_generation_config"] = {
            **(generation.get("mcore_generation_config") or {}),
            **overrides,
        }
    config["generation"] = generation
    return config


def engine_node_slots(
    widths: Sequence[int], gpus_per_node: int, nodes_per_segment: int | None
) -> list[list[int]]:
    """Inference-node indices each engine takes, in launch order.

    An engine no wider than a node shares nodes with its neighbours: it starts
    the next node when it does not fit in the GPUs the earlier engines left. A
    wider engine takes ``width / gpus_per_node`` whole nodes, starting on a
    fresh one. With ``nodes_per_segment`` set (the inference cluster's node
    pins come in NVLink-domain segments of that many nodes), no engine crosses
    a segment boundary, so every engine stays inside one NVLink domain.

    Args:
        widths: GPUs each engine spans, in launch order.
        gpus_per_node: GPUs on every inference node.
        nodes_per_segment: Nodes per pinned segment, or None without pins.

    Returns:
        One list of node indices per engine.

    Raises:
        ValueError: An engine wider than a node is not a whole number of
            nodes, or is wider than one segment.
    """
    slots: list[list[int]] = []
    node, used = 0, 0
    for width in widths:
        if width <= gpus_per_node:
            if used + width > gpus_per_node:
                node, used = node + 1, 0
            slots.append([node])
            used += width
            continue
        if width % gpus_per_node != 0:
            raise ValueError(
                f"A Megatron engine behind Dynamo spans {width} GPUs, which is wider "
                f"than a node and not a multiple of {gpus_per_node} GPUs per node; "
                "an engine wider than a node takes whole nodes."
            )
        span = width // gpus_per_node
        start = node + 1 if used else node
        if nodes_per_segment is not None:
            if span > nodes_per_segment:
                raise ValueError(
                    f"A Megatron engine behind Dynamo spans {span} nodes, more than "
                    f"one NVLink segment of {nodes_per_segment} nodes; engines "
                    "wider than an NVLink domain are not supported."
                )
            segment_end = (start // nodes_per_segment + 1) * nodes_per_segment
            if start + span > segment_end:
                start = segment_end
        slots.append(list(range(start, start + span)))
        node, used = start + span, 0
    return slots


@ray.remote(num_cpus=0)
def cluster_node_ip() -> str:  # pragma: no cover
    """IP of the node a placement-group bundle lives on."""
    return _get_node_ip_local()


def _deadline(timeout_s: float | None) -> float | None:
    """Monotonic deadline ``timeout_s`` from now; ``None`` means no deadline."""
    return None if timeout_s is None else time.monotonic() + timeout_s


def _seconds_left(deadline: float | None) -> float | None:
    """Time left before ``deadline``, never negative; ``None`` waits indefinitely."""
    return None if deadline is None else max(0.0, deadline - time.monotonic())


def _wait_on_engine(
    refs: list[ray.ObjectRef], *, deadline: float | None, what: str, instance_id: str
) -> list[Any]:
    """``ray.get`` under ``deadline``; a timeout becomes a ``RuntimeError`` naming the engine.

    Not a ``TimeoutError``: the SingleController treats any ``TimeoutError``
    raised inside a refit as its own refit budget running out and reports a
    generic abort without the engine's name. Ray's ``GetTimeoutError`` is
    one, so it is chained as the cause instead.

    Raises:
        RuntimeError: When the deadline passes before every ref is ready.
    """
    try:
        return ray.get(refs, timeout=_seconds_left(deadline))
    except ray.exceptions.GetTimeoutError as error:
        raise RuntimeError(
            f"{what} on Megatron engine {instance_id} did not finish before its "
            "deadline"
        ) from error


@dataclasses.dataclass(frozen=True)
class MegatronDynamoEngineHandle:
    """A handle to one running Megatron engine: what the refit side drives it through.

    Holds plain data and actor handles only. ``MegatronGeneration`` (through
    its ``Policy``) and the ``RayVirtualCluster`` behind its worker group both
    shut the engine down when garbage collected, so any process that unpickled
    either and then dropped it would kill the engine. This handle never does;
    the ``MegatronDynamoGeneration`` that built the engine owns its lifetime,
    and the SingleController actor's copy drives the engine through the handles.

    Attributes:
        role: ``prefill`` or ``decode``.
        world_size: Ranks the engine spans.
        instance_id: ``<role>-<ordinal>``, the engine's name in logs and errors.
        node_ip: The node rank zero sits on: where the engine's coordinator
            listens.
        workers: Every worker of the engine, in rank order.
        leader: The worker leading the engine's single DP shard.
    """

    role: Role
    world_size: int
    instance_id: str
    node_ip: str
    workers: tuple["ActorHandle", ...]
    leader: "ActorHandle"

    @classmethod
    def from_generation(
        cls,
        generation: MegatronGeneration,
        *,
        role: Role,
        world_size: int,
        instance_id: str,
        node_ip: str,
    ) -> "MegatronDynamoEngineHandle":
        """The handle of a running engine, read off the ``MegatronGeneration`` that owns it.

        Args:
            generation: The engine's owning object; its worker group supplies
                the handles.
            role: ``prefill`` or ``decode``.
            world_size: Ranks the engine spans.
            instance_id: ``<role>-<ordinal>``, the engine's name in logs and errors.
            node_ip: The node rank zero sits on: where the engine's
                coordinator listens.

        Returns:
            A handle holding the workers in rank order and the single DP shard's leader.
        """
        group = generation.worker_group
        workers = tuple(group.workers)
        return cls(
            role=role,
            world_size=world_size,
            instance_id=instance_id,
            node_ip=node_ip,
            workers=workers,
            leader=workers[group.get_dp_leader_worker_idx(0)],
        )

    def _all(self, method: str, **kwargs: Any) -> list[ray.ObjectRef]:
        return [getattr(worker, method).remote(**kwargs) for worker in self.workers]

    def prepare_refit_info(self, state_dict_info: Any) -> list[ray.ObjectRef]:
        # MegatronGeneration.prepare_refit_info fans the same call out and
        # returns nothing; the futures are awaited here instead.
        return self._all("prepare_refit_info", state_dict_info=state_dict_info)

    def init_collective(
        self,
        ip: str,
        port: int,
        world_size: int,
        *,
        train_world_size: int,
        rank_offset: int,
    ) -> list[ray.ObjectRef]:
        # Every worker gets the engine's offset; each adds its own rank to it.
        return [
            worker.init_collective.remote(
                ip=ip,
                port=port,
                world_size=world_size,
                train_world_size=train_world_size,
                rank_offset=rank_offset,
            )
            for worker in self.workers
        ]

    def init_collective_mcore_generation(
        self,
        ip: str,
        port: int,
        world_size: int,
        *,
        rank_offset: int,
        refit_execution_batch_bytes: int | None,
        refit_backend: str,
    ) -> list[ray.ObjectRef]:
        # As init_collective: every worker gets the engine's offset and adds
        # its own rank inside the engine to it.
        return self._all(
            "init_collective_mcore_generation",
            ip=ip,
            port=port,
            world_size=world_size,
            rank_offset=rank_offset,
            refit_execution_batch_bytes=refit_execution_batch_bytes,
            refit_backend=refit_backend,
        )

    def update_weights_from_collective(
        self, refit_timeout_s: float | None
    ) -> list[ray.ObjectRef]:
        return self._all(
            "update_weights_from_collective", refit_timeout_s=refit_timeout_s
        )

    def swap_weights_via_reshard(self) -> list[ray.ObjectRef]:
        # The native refit's receive side; the trainers send with is_source=True.
        return self._all("swap_weights_via_reshard", is_source=False)

    def suspend_for_refit(self) -> list[ray.ObjectRef]:
        return self._all("suspend_for_refit")

    def resume_after_refit(self) -> list[ray.ObjectRef]:
        return self._all("resume_after_refit")

    def prepare_for_generation(self, *, tags: list[str]) -> list[ray.ObjectRef]:
        return self._all("prepare_for_generation", tags=tags)

    def set_rollout_weight_version(self, version: int) -> list[ray.ObjectRef]:
        return self._all("set_rollout_weight_version", version=version)

    def liveness_ref(self) -> ray.ObjectRef:
        return self.leader.is_alive.remote()


class MegatronDynamoGeneration(GenerationInterface):
    """The generation object for ``policy.generation.dynamo=true`` on the Megatron backend.

    A sibling of ``MegatronGeneration``, not a subclass: ``resolve_generation_class``
    picks it over ``MegatronGeneration`` when the switch is set. Instead of one
    dedicated inference policy it owns one ``MegatronGeneration`` per prefill or
    decode engine, each over its own virtual cluster on the nodes it takes and
    built with ``skip_weight_load=True`` so its inference engine comes up at the
    first weight refit. An engine wider than a node takes whole nodes inside one
    pinned NVLink segment. Every engine has
    the layout the policy config gives plus its role's
    ``mcore_generation_config.prefill_engine`` / ``decode_engine`` overrides, so prefill and
    decode engines may differ in TP and PP (the KV handoff reshards between
    them). An engine's role, carried in its config, also picks the engine
    variant and KV-transfer agents its ranks build.

    This object is also the engines' refit side: every refit call the trainer
    makes here (collective init, prepare the weights, transfer, suspend,
    prepare the KV cache, resume) is fanned out over the engines' Ray workers
    in engine order. Engine ``i`` joins the refit group at the trainers plus
    the widths of the engines before it, so a prefill engine at TP 4 may sit
    next to decode engines at TP 2. After a failed suspend the fleet resumes
    every engine, including one that stopped partway (each worker resumes from
    the state its engine is actually in); the original error still
    propagates, with any resume failure attached. The object owns no policy
    and no worker group of its own.

    Pickling keeps the engine handles (plain data plus the workers' actor
    handles) and drops the owning objects, so the SingleController actor's copy
    drives refits without being able to shut an engine down.

    Rollouts are not served yet: the engines register with no frontend, so
    :meth:`validate_settings` refuses the switch and ``generate_async`` raises.
    """

    @classmethod
    def validate_settings(cls, master_config: "MasterConfig") -> None:
        """Refuse ``policy.generation.dynamo=true`` until rollouts are served.

        The Megatron backend's checks run first, then the engines' config is checked
        so a shape the engines could never serve is named as such. Router replay
        is refused next: the KV handoff moves a prompt's cache from a prefill to
        a decode engine but not its routing-replay buffers. Any config is then
        refused, because the prefill/decode engines are built and refit but no
        frontend serves rollouts from them yet, and the run would otherwise fail
        at its first generation.
        """
        MegatronGeneration.validate_settings(master_config)
        validate_megatron_dynamo_generation(master_config.policy["generation"])
        if router_replay_enabled(master_config.policy):
            raise ValueError(
                "policy.router_replay.enabled must be false when "
                "policy.generation.dynamo=true: the KV handoff does not transfer "
                "the prompt's routing-replay buffers."
            )
        raise NotImplementedError(
            "policy.generation.dynamo=true is not served by MegatronGeneration yet: "
            "the Megatron prefill/decode engines are built and refit, but no "
            "frontend serves rollouts from them. Remove policy.generation.dynamo."
        )

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: PreTrainedTokenizerBase,
        cluster: Optional[RayVirtualCluster] = None,
        policy: Optional["Policy"] = None,
        name_prefix: str = "megatron_generation",
        processor: Optional[AutoProcessor] = None,
        skip_weight_load: bool = False,
        reserved_http_server_ports: Optional[dict[int, int]] = None,
    ):
        """Build one engine per role; the engines come up at the first refit.

        Same signature as ``MegatronGeneration`` so setup builds either class the
        same way. ``cluster`` contributes the node shape, port range, node pins
        and segment size only; each engine gets its own cluster on the nodes it
        takes. Its placement groups stay unmaterialized, or the engines' GPUs
        would be booked twice.
        ``policy`` (colocation) and ``processor`` (VLM prompts) are refused;
        ``name_prefix``, ``skip_weight_load`` and ``reserved_http_server_ports``
        do not apply, since every engine is built with ``skip_weight_load=True``
        and serves no HTTP server of its own.

        Args:
            config: PolicyConfig for the Megatron model.
            tokenizer: The tokenizer handed to every engine.
            cluster: The inference cluster the engines take the place of.
            policy: Must be None; the engines cannot share the trainer's workers.
            name_prefix: Unused; each engine names its own worker group.
            processor: Must be None; the frontend serves text prompts only.
            skip_weight_load: Unused; every engine defers its weights to the refit.
            reserved_http_server_ports: Unused; the engines expose no HTTP server.

        Raises:
            ValueError: On an engine wider than a node that is not a whole
                number of nodes or wider than a pinned segment, or more engines
                than the node pins hold.
            RuntimeError: When an engine's ranks do not land node by node as
                planned.
        """
        if policy is not None:
            raise ValueError(
                "policy.generation.dynamo=true needs a dedicated inference "
                "cluster; set policy.generation.colocated.enabled=false."
            )
        if processor is not None:
            raise NotImplementedError(
                "Megatron engines behind Dynamo serve text prompts only."
            )
        assert cluster is not None, "Megatron engines behind Dynamo need a cluster."
        # The state the weight synchronizer and setup read off a generation object.
        self._policy_config = config
        # validate_megatron_dynamo_generation below checks it is a Megatron one.
        self.cfg = cast(MCoreGenerationConfig, config["generation"])
        # The frontend URL once the engines serve; the engines expose no HTTP server.
        self.dp_openai_server_base_urls: list[Optional[str]] = []
        # Installed by setup via create_weight_synchronizer.
        self.weight_synchronizer: Optional[WeightSynchronizer] = None
        self._tokenizer: Optional[PreTrainedTokenizerBase] = tokenizer
        # The engines take the policy config as-is; MegatronGeneration types it.
        self._dynamo_cfg: MegatronDynamoCfg = megatron_dynamo_cfg(self.cfg)
        # Names these engines' placement groups apart from any other run's.
        self._fleet_id = uuid.uuid4().hex[:8]
        self._num_gpus_per_node = cluster.num_gpus_per_node
        self._port_range_low = cluster.port_range_low
        self._port_range_high = cluster.port_range_high
        self._node_resource_constraints = cluster.node_resource_constraints
        # Nodes per pinned NVLink segment; engines never cross one.
        self._segment_size = cluster.segment_size
        # One handle per engine, in launch (refit rank) order; what refit calls use.
        self._engine_handles: list[MegatronDynamoEngineHandle] = []
        # The owning objects, in the same order; empty on a copy.
        self._engines: list[MegatronGeneration] = []
        self._clusters: list[RayVirtualCluster] = []
        # True on a copy that was unpickled (the SingleController actor's).
        self._is_pickled_copy = False

        validate_megatron_dynamo_generation(self.cfg)
        roles, widths = plan_megatron_engines(self._policy_config, cluster.world_size())
        try:
            self._start_engines(roles, widths)
        except Exception:
            self.shutdown()
            raise

    # ------------------------------------------------------------------
    # Pickling: the SingleController actor gets a handles-only copy
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        """Leave the owning objects behind when shipped to an actor.

        A copy that dropped a ``MegatronGeneration`` or a ``RayVirtualCluster``
        would shut the engine down, so the copy keeps the engine handles (plain
        data plus worker handles) and drives refits through them.
        """
        state = self.__dict__.copy()
        state["_engines"] = []
        state["_clusters"] = []
        state["_tokenizer"] = None
        state["_is_pickled_copy"] = True
        return state

    @property
    def is_pickled_copy(self) -> bool:
        """True on an unpickled copy, which drives the engines but owns none."""
        return self._is_pickled_copy

    # ------------------------------------------------------------------
    # Geometry, workers and liveness
    # ------------------------------------------------------------------

    @property
    def engine_handles(self) -> list[MegatronDynamoEngineHandle]:
        """One handle per engine, in launch (refit shard) order."""
        return list(self._engine_handles)

    @property
    def shard_widths(self) -> tuple[int, ...]:
        """Ranks each engine spans, in handle (refit shard) order."""
        return tuple(handle.world_size for handle in self._engine_handles)

    @property
    def inference_world_size(self) -> int:
        """Ranks across all engines: their widths summed."""
        return sum(self.shard_widths)

    @property
    def num_engines(self) -> int:
        """How many engines run, each a refit shard of its own."""
        return len(self._engine_handles)

    def get_inference_world_size(self) -> int | None:
        """Ranks across the engines: their widths summed."""
        return self.inference_world_size

    @property
    def uses_native_refit(self) -> bool:
        """Whether the engines refit through Megatron-Core's native path (``refit_transport: mcore``).

        Read by the Megatron weight synchronizer to pick its transport. False
        is the packed collective.
        """
        return self.cfg.get("refit_transport") == "mcore"

    @property
    def worker_group(self) -> "RayWorkerGroup":
        """There is no single worker group; each engine has its own.

        AttributeError on purpose: the collective weight synchronizer probes
        for a worker group with getattr and tracks no membership without one.
        """
        raise AttributeError(
            "Megatron engines behind Dynamo have no single worker group; each "
            "engine is its own MegatronGeneration."
        )

    def shard_liveness_ref(self, shard_idx: int) -> ray.ObjectRef:
        """Liveness of the worker leading engine ``shard_idx`` (its rank zero)."""
        return self._engine_handles[shard_idx].liveness_ref()

    # ------------------------------------------------------------------
    # Fan-out over the engines
    # ------------------------------------------------------------------

    def _call_engines(
        self,
        indices: Sequence[int],
        call: Callable[[MegatronDynamoEngineHandle], list[ray.ObjectRef]],
        *,
        what: str,
        deadline: float | None,
    ) -> None:
        """Run ``call`` on every listed engine at once; raise the first failure.

        A missed deadline is a ``RuntimeError`` naming the engine (see
        :func:`_wait_on_engine`). Any other failure keeps its type (callers
        recover from ``RayActorError``) and gains a note naming the engine and
        the step.
        """
        pending = [(index, call(self._engine_handles[index])) for index in indices]
        for index, futures in pending:
            instance_id = self._engine_handles[index].instance_id
            try:
                _wait_on_engine(
                    futures, deadline=deadline, what=what, instance_id=instance_id
                )
            except Exception as error:
                error.add_note(f"{what} failed on engine {instance_id}")
                raise

    def _attempt(
        self,
        indices: Sequence[int],
        call: Callable[[MegatronDynamoEngineHandle], list[ray.ObjectRef]],
        *,
        what: str,
        deadline: float | None,
        errors: list[str],
    ) -> list[int]:
        """Run ``call`` on every listed engine at once; return the ones that succeeded."""
        pending = [(index, call(self._engine_handles[index])) for index in indices]
        succeeded: list[int] = []
        for index, futures in pending:
            instance_id = self._engine_handles[index].instance_id
            try:
                _wait_on_engine(
                    futures, deadline=deadline, what=what, instance_id=instance_id
                )
            except Exception as error:
                errors.append(f"{instance_id}: {what}: {error!r}")
            else:
                succeeded.append(index)
        return succeeded

    # ------------------------------------------------------------------
    # Refit collective
    # ------------------------------------------------------------------

    def set_refit_membership(self, membership: "RefitMembership") -> None:
        """Accept only every engine: the engine set is fixed.

        ``RefitMembership`` describes equal-width shards, so its
        ``workers_per_shard`` cannot describe engines of different widths and
        is not read. Each refit call computes the engines' ranks from their
        handles instead (:meth:`engine_rank_offsets`); nothing is stored.

        Raises:
            NotImplementedError: For a membership that leaves an engine out.
        """
        every_engine = list(range(self.num_engines))
        if membership.surviving_shards != every_engine:
            raise NotImplementedError(
                "Megatron engines behind Dynamo are a fixed fleet; refit membership "
                f"must be every engine {every_engine}, got "
                f"{membership.surviving_shards}."
            )

    def init_collective(
        self,
        ip: str,
        port: int,
        world_size: int,
        *,
        train_world_size: int,
    ) -> list[ray.ObjectRef]:
        """Join every engine to the refit group at its rank prefix.

        Engine ``i``'s workers are offset to the absolute rank where its shard
        starts: the trainers plus its prefix within the generation block. The
        packed collective joins NeMo RL's ``model_update_group``; the native
        refit (``refit_transport: mcore``) builds Megatron-Core's refit group
        and its transfer plan, with every training rank joining at the same
        time.

        The native plan needs no per-engine pools. Each engine is its own
        torch.distributed world, and Megatron-Core shifts every group rank an
        engine reports by that engine's offset, so one plan sees each engine at
        its true ranks with its own tensor-parallel group. The plan is worked
        out per receiving rank from that rank's own layout, so a TP 2 prefill
        engine and TP 1 decode engines are refit in the same transfer.
        """
        expected = train_world_size + self.inference_world_size
        if world_size != expected:
            raise ValueError(
                f"NCCL world_size={world_size} does not match expected {expected} "
                f"(train={train_world_size} + engines={self.inference_world_size})"
            )
        offsets = self.engine_rank_offsets(train_world_size)
        futures: list[ray.ObjectRef] = []
        if self.uses_native_refit:
            mcore_cfg = self.cfg["mcore_generation_config"]
            refit_backend = mcore_cfg["refit_backend"]
            # validate_megatron_dynamo_generation admits only these two.
            assert refit_backend == "nccl" or refit_backend == "gloo", refit_backend
            for handle, offset in zip(self._engine_handles, offsets, strict=True):
                futures.extend(
                    handle.init_collective_mcore_generation(
                        ip,
                        port,
                        world_size,
                        rank_offset=offset,
                        refit_execution_batch_bytes=mcore_cfg[
                            "refit_execution_batch_bytes"
                        ],
                        refit_backend=refit_backend,
                    )
                )
            return futures
        for handle, offset in zip(self._engine_handles, offsets, strict=True):
            futures.extend(
                handle.init_collective(
                    ip,
                    port,
                    world_size,
                    train_world_size=train_world_size,
                    rank_offset=offset,
                )
            )
        return futures

    def engine_rank_offsets(self, train_world_size: int) -> list[int]:
        """Refit-group rank of each engine's rank zero, in handle order.

        The trainers take ranks ``0 .. train_world_size - 1``; engine ``i``
        starts right after the trainers and the engines before it.
        """
        offsets: list[int] = []
        rank = train_world_size
        for width in self.shard_widths:
            offsets.append(rank)
            rank += width
        return offsets

    def prepare_refit_info(
        self, state_dict_info: Optional[dict[str, Any]]
    ) -> Optional[list[str]]:
        """Prepare Bridge conversion tasks on every engine.

        A no-op with the native refit, which moves Megatron shards and never
        imports HF tensors (as on ``MegatronGeneration``).
        """
        if self.uses_native_refit:
            return None
        if state_dict_info is None:
            raise ValueError("state_dict_info must not be None for Megatron refit")
        # Stored once; every worker resolves the same object.
        info_ref = ray.put(state_dict_info)
        self._call_engines(
            range(self.num_engines),
            lambda handle: handle.prepare_refit_info(info_ref),
            what="prepare_refit_info",
            # No deadline of its own, as on the other generation backends.
            deadline=None,
        )
        return None

    def update_weights_from_collective(
        self, refit_timeout_s: Optional[float] = None
    ) -> list[ray.ObjectRef]:
        """Receive one refit on every engine, into the live parameters.

        The packed collective is a broadcast of HF tensors from train rank
        zero. The native refit is Megatron-Core's ``swap_model_weights``, with
        every engine rank receiving its own shards; it cannot be aborted, so a
        deadline is refused before any engine enters it.
        """
        futures: list[ray.ObjectRef] = []
        if self.uses_native_refit:
            reject_unenforceable_refit_deadline("Megatron", refit_timeout_s)
            for handle in self._engine_handles:
                futures.extend(handle.swap_weights_via_reshard())
            return futures
        for handle in self._engine_handles:
            futures.extend(handle.update_weights_from_collective(refit_timeout_s))
        return futures

    def set_rollout_weight_version(self, version: int) -> None:
        """Rotate the policy epoch every engine stamps on its requests."""
        self._call_engines(
            range(self.num_engines),
            lambda handle: handle.set_rollout_weight_version(version),
            what="set_rollout_weight_version",
            deadline=None,
        )

    # ------------------------------------------------------------------
    # Fence around the transfer
    # ------------------------------------------------------------------

    def _restore_engines(
        self, indices: Sequence[int], *, deadline: float | None
    ) -> list[str]:
        """Resume ``indices``; return the failures."""
        errors: list[str] = []
        self._attempt(
            indices,
            lambda handle: handle.resume_after_refit(),
            what="resume_after_refit",
            deadline=deadline,
            errors=errors,
        )
        return errors

    def suspend_for_refit(self, *, timeout_s: float | None = None) -> None:
        """Suspend every engine before a weight update.

        The caller prepares the weights afterwards, as the Megatron weight
        synchronizer does for one dedicated engine.

        If any step fails or the deadline passes, the fleet resumes every
        engine and the original error still propagates, with any resume
        failures added to it as notes. Each worker resumes from the state its
        engine is actually in, so an engine the failed suspend left paused or
        suspended goes back into service too. The resume shares the suspend's
        deadline: if the suspend used it up, the resumes
        are still sent but not waited for, and the note says they were not
        confirmed in time.

        Args:
            timeout_s: One deadline for the suspend and any resume after it;
                ``None`` waits indefinitely, as on ``MegatronGeneration``.

        Raises:
            RuntimeError: When an engine misses the deadline; the message
                names the engine.
        """
        deadline = _deadline(timeout_s)
        indices = list(range(self.num_engines))
        try:
            self._call_engines(
                indices,
                lambda handle: handle.suspend_for_refit(),
                what="suspend_for_refit",
                deadline=deadline,
            )
        except Exception as error:
            # The weights are unchanged, so a resume puts each engine back in
            # service. Same deadline: the caller's budget covers the restore too.
            restore_errors = self._restore_engines(indices, deadline=deadline)
            if restore_errors:
                error.add_note(
                    "resuming engines after the failed suspend also failed: "
                    + "; ".join(restore_errors)
                )
            raise

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Run one tagged refit phase (``weights``, ``kv_cache``) on every engine.

        ``weights`` readies the refit destination buffers; ``kv_cache`` rebuilds
        the cache after the transfer. Order across engines does not matter. An
        untagged call is the build-time wake; the engines come up at their
        first refit instead, so it does nothing.
        """
        tags = kwargs.get("tags")
        if tags is None:
            return True
        self._call_engines(
            range(self.num_engines),
            lambda handle: handle.prepare_for_generation(tags=list(tags)),
            what=f"prepare_for_generation({list(tags)})",
            # The caller's timeout_s; None waits indefinitely.
            deadline=_deadline(kwargs.get("timeout_s")),
        )
        return True

    def resume_after_refit(self, *, timeout_s: float | None = None) -> None:
        """Resume serving on every engine; the first resume is when the engines start serving.

        Every engine is attempted even if another fails, so one bad engine
        does not keep the rest down.

        Args:
            timeout_s: One deadline for every engine's resume; ``None`` waits
                indefinitely.

        Raises:
            RuntimeError: Any engine failed to resume or missed the deadline.
        """
        errors = self._restore_engines(
            range(self.num_engines), deadline=_deadline(timeout_s)
        )
        if errors:
            raise RuntimeError(
                "Megatron resume after weight update failed for " + "; ".join(errors)
            )

    def finish_generation(self, *, release_gpu: bool = True) -> bool:
        """Dedicated engines keep serving between steps, as without Dynamo."""
        return True

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Not served yet: the engines register with no frontend."""
        raise NotImplementedError(
            "Megatron engines behind Dynamo do not serve rollouts yet; the "
            "engines register with no frontend."
        )

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Not served yet: the engines register with no frontend."""
        raise NotImplementedError(
            "Megatron engines behind Dynamo do not serve rollouts yet; the "
            "engines register with no frontend."
        )
        yield  # pragma: no cover  (makes this an async generator)

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def _start_engines(self, roles: Sequence[Role], widths: Sequence[int]) -> None:
        """Bring up one engine per planned entry, in plan order.

        Args:
            roles: ``prefill`` / ``decode`` per engine, from
                ``dynamo_config.plan_megatron_engines``; the order is the refit
                rank order.
            widths: GPU ranks each engine spans, parallel to ``roles``.

        Raises:
            ValueError: On an engine wider than a node that is not a whole
                number of nodes or wider than a pinned segment, or more engines
                than the node pins hold.
            RuntimeError: When an engine's ranks do not land node by node as
                planned.
        """
        # Which inference nodes each engine takes: sub-node engines share
        # nodes in launch order, wider ones take whole nodes inside one pinned
        # segment (engine_node_slots).
        constraints = self._node_resource_constraints
        nodes_per_segment = self._segment_size if constraints is not None else None
        # One engine at a time first, so a width that cannot be placed names
        # its role and the config key to change.
        for role, width in zip(roles, widths, strict=True):
            try:
                engine_node_slots([width], self._num_gpus_per_node, nodes_per_segment)
            except ValueError as err:
                raise ValueError(
                    f"{err} Change policy.generation.mcore_generation_config."
                    f"{role}_engine (or the shared mcore_generation_config) "
                    "tensor_model_parallel_size x pipeline_model_parallel_size."
                ) from err
        node_slots_by_engine = engine_node_slots(
            widths, self._num_gpus_per_node, nodes_per_segment
        )
        nodes_needed = node_slots_by_engine[-1][-1] + 1
        if constraints is not None and nodes_needed > len(constraints):
            widths_by_role = dict(zip(roles, widths, strict=True))
            raise ValueError(
                f"{len(roles)} Megatron engines of widths {widths_by_role} GPUs tile "
                f"{nodes_needed} nodes of {self._num_gpus_per_node} GPUs, more "
                f"than the {len(constraints)} pinned inference nodes."
            )
        # Each role's engines are built from the policy config with that
        # role and its overrides.
        policy_config_by_role: dict[str, dict[str, Any]] = {
            role: role_policy_config(
                self._policy_config, role, role_overrides(self.cfg, role)
            )
            for role in ROLES
        }
        role_counts: dict[str, int] = {}
        for engine_index, (role, width) in enumerate(zip(roles, widths, strict=True)):
            ordinal = role_counts.get(role, 0)
            role_counts[role] = ordinal + 1
            instance_id = f"{role}-{ordinal}"
            engine_policy_config = cast("PolicyConfig", policy_config_by_role[role])
            node_slots = node_slots_by_engine[engine_index]
            cluster = RayVirtualCluster(
                name=f"dynamo_{self._fleet_id}_{instance_id}",
                bundle_ct_per_node_list=(
                    [width]
                    if len(node_slots) == 1
                    else [self._num_gpus_per_node] * len(node_slots)
                ),
                use_gpus=True,
                num_gpus_per_node=self._num_gpus_per_node,
                max_colocated_worker_groups=1,
                port_range_low=self._port_range_low,
                port_range_high=self._port_range_high,
                node_resource_constraints=(
                    None
                    if constraints is None
                    else [constraints[node] for node in node_slots]
                ),
            )
            self._clusters.append(cluster)
            # Reserved the way the engine will reserve them (always PACK,
            # since Dynamo engines are never colocated; idempotent), so a
            # split engine fails before any worker is scheduled.
            MegatronGeneration.init_cluster_placement_groups(
                cluster, engine_policy_config
            )
            node_ip = self._engine_node_ip(cluster, instance_id, len(node_slots))
            engine = MegatronGeneration(
                config=engine_policy_config,
                tokenizer=self._tokenizer,
                cluster=cluster,
                name_prefix=f"dynamo_{instance_id}",
                skip_weight_load=True,
            )
            self._engines.append(engine)
            self._engine_handles.append(
                MegatronDynamoEngineHandle.from_generation(
                    engine,
                    role=role,
                    world_size=width,
                    instance_id=instance_id,
                    node_ip=node_ip,
                )
            )
            print(
                f"  [Dynamo:{instance_id}] engine ready for refit on "
                f"{node_ip} ({width} ranks)",
                flush=True,
            )

    def _engine_node_ip(
        self, cluster: RayVirtualCluster, instance_id: str, num_nodes: int
    ) -> str:
        """Rank zero's node, after checking the engine landed node by node as planned.

        Ranks are read in rank order (``MegatronGeneration._rank_placement``).
        Each block of ``num_gpus_per_node`` ranks (all ranks, for an engine
        within one node) must share one node, so its tensor-parallel group stays
        on NVLink, and the blocks must sit on ``num_nodes`` different nodes.
        ``PACK`` is best effort: on a fragmented or partly busy cluster Ray may
        spread bundles, and the engine would then silently run its collectives
        across nodes.
        """
        placement_groups = cluster.get_placement_groups()
        refs = [
            cluster_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_groups[pg_index],
                    placement_group_bundle_index=bundle_index,
                )
            ).remote()
            for pg_index, bundle_index in MegatronGeneration._rank_placement(cluster)
        ]
        rank_ips = ray.get(refs)
        block = len(rank_ips) if num_nodes == 1 else self._num_gpus_per_node
        block_ips = []
        for start in range(0, len(rank_ips), block):
            ips = set(rank_ips[start : start + block])
            if len(ips) != 1:
                raise RuntimeError(
                    f"Megatron engine {instance_id} ranks {start}-{start + block - 1} "
                    f"span multiple nodes {sorted(ips)}; each node's worth of ranks "
                    "must share one node."
                )
            block_ips.append(ips.pop())
        if len(set(block_ips)) != num_nodes:
            raise RuntimeError(
                f"Megatron engine {instance_id} landed on {len(set(block_ips))} "
                f"nodes, planned {num_nodes}. Ray placed the engine's per-node "
                "bundles on fewer nodes than planned; make sure the pinned NVLink "
                "domain has that many free whole nodes."
            )
        return rank_ips[0]

    def shutdown(self) -> bool:
        """Stop every engine, then release their placement groups.

        Best effort and idempotent: every failure is logged and the teardown
        continues. A copy that owns no engine does nothing.
        """
        if self._is_pickled_copy:
            return True
        # Includes engines that failed before their handle existed.
        for index in range(len(self._engines)):
            self._stop_engine(index)
        for cluster in self._clusters:
            try:
                cluster.shutdown()
            except Exception:
                LOGGER.exception("Could not remove an engine placement group")
        self._engine_handles.clear()
        self._engines.clear()
        self._clusters.clear()
        return True

    def _stop_engine(self, index: int) -> bool:
        """Stop one engine's policy; whether it stopped cleanly."""
        try:
            engine_stopped = self._engines[index].shutdown()
        except Exception:
            LOGGER.exception("Could not stop a Megatron engine")
            return False
        if not engine_stopped:
            LOGGER.error("A Megatron engine did not shut down cleanly")
        return engine_stopped

    def __del__(self) -> None:
        """Safety net so the engines do not outlive the object that built them."""
        if getattr(self, "_engines", None):
            self.shutdown()
