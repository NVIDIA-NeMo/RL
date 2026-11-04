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
"""Generation over NeMo-RL-owned Megatron prefill/decode engines behind a Dynamo frontend.

One engine is one ``MegatronGeneration`` over its own worker group, the same
dedicated inference policy the Megatron backend already builds, with its role
in its config. Once the first refit has built the engine, its rank zero hosts
Megatron-LM's Dynamo adapter (``MegatronLLMEngine``, driven by a Dynamo
``Worker`` task on the inference loop), which registers the engine with the
frontend, forwards requests to the engine's coordinator, and carries the KV
handoff between prefill and decode. ``MegatronDynamoGeneration`` makes the
engines exist, in refit rank order, starts their endpoints, and is their refit
side.
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
from nemo_rl.models.generation.dynamo.managed_runtime import (
    ManagedDynamoRuntime,
    wait_for_engine_registrations,
)
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.generation.megatron.config import MCoreGenerationConfig
from nemo_rl.models.generation.megatron.dynamo_config import (
    DYNAMO_COMPONENT_BY_ROLE,
    MegatronDynamoCfg,
    MegatronDynamoRole,
    disaggregation_cfg,
    engine_roles,
    engine_widths,
    megatron_dynamo_cfg,
    validate_megatron_dynamo_generation,
)
from nemo_rl.models.generation.megatron.megatron_generation import MegatronGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.weight_sync.interfaces import WeightSynchronizer
from nemo_rl.weight_sync.membership import RefitMembership, desired_membership

if TYPE_CHECKING:
    from ray.actor import ActorHandle

    from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
    from nemo_rl.data.captured_media import MediaColumnSpec
    from nemo_rl.data_plane.interfaces import DataPlaneConfig
    from nemo_rl.distributed.worker_groups import RayWorkerGroup
    from nemo_rl.models.policy.lm_policy import Policy

LOGGER = logging.getLogger(__name__)


def role_policy_config(
    policy_config: Mapping[str, Any],
    role: MegatronDynamoRole,
    overrides: Mapping[str, int],
) -> dict[str, Any]:
    """The policy config a worker of one role builds its engine from.

    A deep copy of ``policy_config`` with the role in
    ``generation._dynamo_role`` (what makes the worker build the handoff
    engine for that role) and ``overrides`` (the role's
    ``dynamo_cfg.prefill_engine`` / ``decode_engine`` layout keys) merged into
    ``generation.mcore_generation_config``, which is where the dedicated
    inference policy takes its layout from. ``generation.dynamo`` is turned
    off: each engine is a plain Megatron engine sized by its own layout, and
    only ``MegatronDynamoGeneration`` serves through Dynamo. ``policy_config``
    is left as is.

    Args:
        policy_config: The full ``policy`` config.
        role: ``prefill`` or ``decode``.
        overrides: Layout keys from ``MegatronDynamoCfg.role_overrides``; empty for
            a role that keeps the shared layout.

    Returns:
        A fresh config for one engine of ``role``.
    """
    config = copy.deepcopy(dict(policy_config))
    generation: dict[str, Any] = dict(config["generation"])
    # Off, or the engine's NVLink span would be a whole prefill/decode replica
    # instead of its own width.
    generation["dynamo"] = False
    generation["_dynamo_role"] = role
    if overrides:
        generation["mcore_generation_config"] = {
            **(generation.get("mcore_generation_config") or {}),
            **overrides,
        }
    config["generation"] = generation
    return config


@ray.remote(num_cpus=0)
def cluster_node_ip() -> str:  # pragma: no cover
    """IP of the node a placement-group bundle lives on."""
    return _get_node_ip_local()


def _seconds_left(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


class PinnedHandoffTimeout(RuntimeError):
    """Blocks were still pinned for decode handoffs when the suspend deadline passed.

    Deliberately not a ``TimeoutError``: the controller's refit await treats any
    ``TimeoutError`` escaping the refit as its own deadline and replaces it with
    a "rank not answering" abort, dropping this message and its notes.
    """


@dataclasses.dataclass(frozen=True)
class MegatronDynamoEngine:
    """One running Megatron engine, as the refit side sees it: its record and worker handles.

    Holds plain data and actor handles only. ``MegatronGeneration`` (through
    its ``Policy``) and the ``RayVirtualCluster`` behind its worker group both
    shut the engine down when garbage collected, so any process that unpickled
    either and then dropped it would kill the engine. This record never does;
    the ``MegatronDynamoGeneration`` that built the engine owns its lifetime,
    and the SingleController actor's copy drives the engine through the handles.

    Attributes:
        role: ``prefill`` or ``decode``.
        world_size: Ranks the engine spans.
        instance_id: ``<role>-<ordinal>``, the engine's name in logs and errors.
        node_ip: The node every rank of the engine sits on.
        workers: Every worker of the engine, in rank order.
        leader: The worker leading the engine's single DP shard.
    """

    role: MegatronDynamoRole
    world_size: int
    instance_id: str
    node_ip: str
    workers: tuple["ActorHandle", ...]
    leader: "ActorHandle"

    @classmethod
    def of(
        cls,
        generation: MegatronGeneration,
        *,
        role: MegatronDynamoRole,
        world_size: int,
        instance_id: str,
        node_ip: str,
    ) -> "MegatronDynamoEngine":
        """The record of a running engine, read off the ``MegatronGeneration`` that owns it.

        Args:
            generation: The engine's owning object; its worker group supplies
                the handles.
            role: ``prefill`` or ``decode``.
            world_size: Ranks the engine spans.
            instance_id: ``<role>-<ordinal>``, the engine's name in logs and errors.
            node_ip: The node every rank of the engine sits on.

        Returns:
            A record holding the workers in rank order and the single DP shard's leader.
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

    def update_weights_from_collective(
        self, refit_timeout_s: float | None
    ) -> list[ray.ObjectRef]:
        return self._all(
            "update_weights_from_collective", refit_timeout_s=refit_timeout_s
        )

    def pause_for_refit(self) -> list[ray.ObjectRef]:
        return self._all("pause_for_refit")

    def suspend_for_refit(self) -> list[ray.ObjectRef]:
        return self._all("suspend_for_refit")

    def resume_after_refit(self) -> list[ray.ObjectRef]:
        return self._all("resume_after_refit")

    def pinned_handoff_block_count(self) -> ray.ObjectRef:
        return self.leader.pinned_handoff_block_count.remote()

    def prepare_for_generation(self, *, tags: list[str]) -> list[ray.ObjectRef]:
        return self._all("prepare_for_generation", tags=tags)

    def attach_dynamo_events(self, *, parent_event_address: str) -> list[ray.ObjectRef]:
        return self._all(
            "attach_dynamo_events", parent_event_address=parent_event_address
        )

    def set_rollout_weight_version(self, version: int) -> list[ray.ObjectRef]:
        return self._all("set_rollout_weight_version", version=version)

    def liveness_ref(self) -> ray.ObjectRef:
        return self.leader.is_alive.remote()


class MegatronDynamoGeneration(GenerationInterface):
    """The generation object for ``policy.generation.dynamo=true`` on the Megatron backend.

    A sibling of ``MegatronGeneration``, not a subclass: ``resolve_generation_class``
    picks it over ``MegatronGeneration`` when the switch is set. Instead of one
    dedicated inference policy it owns one ``MegatronGeneration`` per prefill or
    decode engine, each over
    its own one-node virtual cluster and built with ``skip_weight_load=True`` so
    its inference engine comes up at the first weight refit. Every engine has
    the layout the policy config gives plus its role's
    ``dynamo_cfg.prefill_engine`` / ``decode_engine`` overrides, so prefill and
    decode engines may differ in TP and PP (the KV handoff reshards between
    them). An engine's role, carried in its config, also picks the engine
    variant and KV-transfer agents its ranks build, the Dynamo component it
    registers under, and its place in the pause and teardown order.

    It also owns the ``ManagedDynamoRuntime`` (etcd, NATS and the frontend) the
    engines register with, started right after the engines in the order the
    vLLM backend uses, so the frontend runs from the engines' own environment.
    The first resume after a refit is also the engines' startup: that refit
    built them, so only now do their Dynamo endpoints start on each rank zero
    and register with the frontend, and the resume waits until the frontend
    lists every engine.

    This object is also the engines' refit side: every refit call the trainer
    makes here (collective init, prepare the weights, transfer, suspend,
    prepare the KV cache, resume) is fanned out over the engines' Ray workers
    in engine order. Rank placement comes from the shared ``RefitMembership``
    arithmetic: engine ``i`` starts at the trainers plus the widths of the
    engines before it, so a prefill engine at TP 4 may sit next to decode
    engines at TP 2. Prefill engines are paused first and suspended only once
    decode has pulled every block they pin; every engine resumes in one wave.
    After a failed suspend the fleet attempts a best-effort resume of every
    engine the suspend reached and the original error still propagates. The
    object owns no policy and no worker group of its own.

    Pickling keeps the engine records (plain data plus the workers' actor
    handles), the frontend URL, the namespace and engine environment, and
    drops the owning objects and the runtime, so the SingleController actor's
    copy drives refits and starts the engines' endpoints without being able to
    shut an engine or a service down.

    ``generate_async`` posts native rollouts to the frontend as token-ID
    completions, one request per prompt.
    """

    # How often the refit pause re-reads the prefill engines' pinned handoff
    # blocks while waiting for them to drain.
    _idle_poll_interval_s: float = 0.2
    # How often that wait prints the counts it is still waiting on.
    _pin_report_interval_s: float = 10.0

    @classmethod
    def validate_settings(cls, master_config: "MasterConfig") -> None:
        """Reject what prefill/decode engines behind the Dynamo frontend cannot serve."""
        # Deferred: both modules pull in the rollout and data-plane stacks.
        from nemo_rl.data_plane.interfaces import storage_unit_placement
        from nemo_rl.models.megatron.router_replay import router_replay_enabled

        MegatronGeneration.validate_settings(master_config)
        policy_config = master_config.policy
        validate_megatron_dynamo_generation(policy_config["generation"])
        # The raw flag: should_use_nemo_gym() asserts the HTTP server is exposed,
        # which this mode never does, so it cannot answer the question.
        if master_config.env.get("should_use_nemo_gym"):
            # Gym needs a server URL before generation is built; the frontend's
            # is known only once the runtime is up.
            raise NotImplementedError(
                "NeMo-Gym rollouts are not yet served through the Dynamo frontend "
                "with policy.generation.dynamo=true; set env.should_use_nemo_gym="
                "false or leave policy.generation.dynamo false."
            )
        if policy_config.get("is_vlm"):
            raise ValueError(
                "policy.is_vlm=true is not supported with policy.generation.dynamo="
                "true; the frontend serves text prompts only."
            )
        if router_replay_enabled(policy_config):
            raise ValueError(
                "policy.router_replay.enabled must be false when "
                "policy.generation.dynamo=true: the KV handoff does not transfer "
                "the prompt's routing-replay buffers."
            )
        dp_config = master_config.data_plane
        placement = storage_unit_placement(dp_config)
        if placement is not None and (placement == "all" or "inference" in placement):
            # Planning storage units materializes the named clusters' placement
            # groups. The engines run on their own per-engine clusters, so a
            # materialized inference cluster would book their GPUs a second
            # time and the engines could never start.
            raise ValueError(
                f"data_plane.{dp_config['backend']}.storage_unit_placement="
                f"{placement!r} must not include the 'inference' cluster with "
                "policy.generation.dynamo=true; the Megatron engines reserve the "
                "inference GPUs through their own clusters. Place storage units "
                "on 'train' instead."
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
        """Build one engine per role, then start etcd, NATS and the frontend.

        Same signature as ``MegatronGeneration`` so setup builds either class the
        same way. The order the vLLM backend uses: the engines' workers first,
        so the frontend runs from their environment, then the runtime. The
        engines have no weights yet; they start serving and register at the
        first refit (:meth:`resume_after_refit`).

        ``cluster`` contributes the node shape, port range and node pins only;
        each engine gets its own one-node cluster. Its placement groups stay
        unmaterialized, or the engines' GPUs would be booked twice. ``policy``
        (colocation) and ``processor`` (VLM prompts) are refused;
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
            ValueError: On an engine wider than a node, or more engines than
                the node pins hold.
            RuntimeError: When an engine's GPUs land on more than one node.
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
        self.cfg: MCoreGenerationConfig = config["generation"]
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
        # Bounds the refit calls (pause and resume by their timeout_s, which
        # defaults to this, one deadline per call) and every endpoint and
        # shutdown call to an engine's workers.
        self._control_timeout_s = float(self._dynamo_cfg.control_timeout_s)
        # One record per engine, in launch (refit rank) order; what refit calls use.
        self._engine_records: list[MegatronDynamoEngine] = []
        # The owning objects, in the same order; empty on a copy.
        self._engines: list[MegatronGeneration] = []
        self._clusters: list[RayVirtualCluster] = []
        # Set once every engine's endpoint is started; a repeat call is a no-op.
        self._endpoints_started = False
        # True on a copy that was unpickled (the SingleController actor's).
        self._is_client_copy = False
        # The runtime stays with the driver; a pickled copy keeps the plain
        # values below, which is all the first resume needs to serve the engines.
        self._dynamo_runtime: Optional[ManagedDynamoRuntime] = None
        self._dynamo_namespace = ""
        self._dynamo_engine_env: dict[str, str] = {}
        self._dynamo_frontend_url = ""
        self._dynamo_fleet_ready = False

        validate_megatron_dynamo_generation(self.cfg)
        widths = engine_widths(self._policy_config)
        roles = engine_roles(disaggregation_cfg(self.cfg), cluster.world_size(), widths)
        try:
            self._start_engines(roles, widths)
            runtime = ManagedDynamoRuntime(
                dynamo_cfg=self._dynamo_cfg,
                model_name=str(self._policy_config["model_name"]),
                # The engines' venv carries ai-dynamo, so the frontend runs the
                # same release as every engine.
                frontend_python=self.engine_python(),
            )
            self._dynamo_runtime = runtime
            runtime.start()
        except Exception:
            self.shutdown()
            raise
        self._dynamo_namespace = runtime.namespace
        self._dynamo_engine_env = runtime.engine_env
        self._dynamo_frontend_url = runtime.frontend_url
        self.dp_openai_server_base_urls = [runtime.frontend_url]
        print(f"  [Dynamo] Forwarding rollouts to {runtime.frontend_url}", flush=True)

    # ------------------------------------------------------------------
    # Pickling: the SingleController actor gets a handles-only copy
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        """Leave the owning objects and the Dynamo runtime behind when shipped to an actor.

        A copy that dropped a ``MegatronGeneration`` or a ``RayVirtualCluster``
        would shut the engine down, so the copy keeps the engine records (plain
        data plus worker handles), the frontend URL, the namespace and engine
        environment, and can drive refits and start the engines' endpoints but
        owns no process.
        """
        state = self.__dict__.copy()
        state["_engines"] = []
        state["_clusters"] = []
        state["_tokenizer"] = None
        state["_dynamo_runtime"] = None
        state["_is_client_copy"] = True
        return state

    @property
    def is_client_copy(self) -> bool:
        """True on an unpickled copy, which drives the engines but owns none."""
        return self._is_client_copy

    # ------------------------------------------------------------------
    # Geometry, workers and liveness
    # ------------------------------------------------------------------

    @property
    def engine_records(self) -> list[MegatronDynamoEngine]:
        """One record per engine, in launch (refit shard) order."""
        return list(self._engine_records)

    @property
    def shard_widths(self) -> tuple[int, ...]:
        """Ranks each engine spans, in record (refit shard) order."""
        return tuple(record.world_size for record in self._engine_records)

    @property
    def inference_world_size(self) -> int:
        """Ranks across all engines: their widths summed."""
        return sum(self.shard_widths)

    @property
    def num_engines(self) -> int:
        """How many engines run, each a refit shard of its own."""
        return len(self._engine_records)

    def _prefill_indices(self) -> list[int]:
        return [
            i
            for i, record in enumerate(self._engine_records)
            if record.role == "prefill"
        ]

    def _decode_indices(self) -> list[int]:
        return [
            i
            for i, record in enumerate(self._engine_records)
            if record.role == "decode"
        ]

    def expected_components(self) -> dict[str, int]:
        """Dynamo component name to the number of engines registering under it."""
        counts: dict[str, int] = {}
        for record in self._engine_records:
            component = DYNAMO_COMPONENT_BY_ROLE[record.role]
            counts[component] = counts.get(component, 0) + 1
        return counts

    def engine_python(self) -> str:
        """The engines' interpreter; its environment carries ai-dynamo."""
        if not self._engines:
            raise RuntimeError("A copy of MegatronDynamoGeneration owns no engines")
        return str(self._engines[0].worker_group.py_executable)

    def check_alive(self, *, timeout_s: float | None = None) -> None:
        """Raise if any engine's leading worker has died or its endpoint stopped serving.

        Used while waiting for the engines to register with the frontend, and
        while the refit pause waits for decode to release prefill's pins, so a
        dead engine, or an endpoint task that failed or exited on rank zero,
        fails the wait at once instead of at its timeout. Endpoints are
        checked only once ``start_endpoints`` started all of them. A worker
        whose actor died raises ``RayActorError`` on the probe; that counts as
        dead, and the first such error is chained.

        Args:
            timeout_s: One deadline for every probe; ``None`` uses
                ``dynamo_cfg.control_timeout_s``.

        Raises:
            RuntimeError: An engine died or its endpoint stopped serving.
            ray.exceptions.GetTimeoutError: A probe ran past the deadline.
        """
        timeout_s = self._control_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        pending = [(record, record.liveness_ref()) for record in self._engine_records]
        dead: list[str] = []
        first_error: Exception | None = None
        for record, future in pending:
            try:
                alive = ray.get(future, timeout=_seconds_left(deadline))
            except ray.exceptions.RayActorError as error:
                first_error = first_error or error
                alive = False
            if not alive:
                dead.append(record.instance_id)
        if dead:
            raise RuntimeError(
                f"Megatron engines died: {', '.join(dead)}"
            ) from first_error
        if not self._endpoints_started:
            return
        errors = ray.get(
            [
                record.leader.dynamo_endpoint_error.remote()
                for record in self._engine_records
            ],
            timeout=_seconds_left(deadline),
        )
        stopped = [
            f"{record.instance_id}: {error}"
            for record, error in zip(self._engine_records, errors)
            if error is not None
        ]
        if stopped:
            raise RuntimeError(
                "The Dynamo endpoint of Megatron engines stopped serving: "
                + "; ".join(stopped)
            )

    def get_inference_world_size(self) -> int | None:
        """Ranks across the engines: their widths summed."""
        return self.inference_world_size

    @property
    def uses_native_refit(self) -> bool:
        """Never: the engines take the packed collective, as ``refit_transport`` must be null.

        Read by the Megatron weight synchronizer to pick its transport.
        """
        return False

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
        return self._engine_records[shard_idx].liveness_ref()

    # ------------------------------------------------------------------
    # Fan-out over the engines
    # ------------------------------------------------------------------

    def _call_engines(
        self,
        indices: Sequence[int],
        call: Callable[[MegatronDynamoEngine], list[ray.ObjectRef]],
        *,
        what: str,
        deadline: float,
    ) -> None:
        """Run ``call`` on every listed engine at once; raise the first failure.

        The failure keeps its type (callers recover from ``RayActorError``) and
        gains a note naming the engine and the step.
        """
        pending = [(index, call(self._engine_records[index])) for index in indices]
        for index, futures in pending:
            try:
                ray.get(futures, timeout=_seconds_left(deadline))
            except Exception as error:
                error.add_note(
                    f"{what} failed on engine {self._engine_records[index].instance_id}"
                )
                raise

    def _attempt(
        self,
        indices: Sequence[int],
        call: Callable[[MegatronDynamoEngine], list[ray.ObjectRef]],
        *,
        what: str,
        deadline: float,
        errors: list[str],
    ) -> list[int]:
        """Run ``call`` on every listed engine at once; return the ones that succeeded."""
        pending = [(index, call(self._engine_records[index])) for index in indices]
        succeeded: list[int] = []
        for index, futures in pending:
            try:
                ray.get(futures, timeout=_seconds_left(deadline))
            except Exception as error:
                errors.append(
                    f"{self._engine_records[index].instance_id}: {what}: {error!r}"
                )
            else:
                succeeded.append(index)
        return succeeded

    # ------------------------------------------------------------------
    # Refit collective
    # ------------------------------------------------------------------

    def check_full_fleet(self, membership: RefitMembership) -> None:
        """Reject any membership but every engine, in order: the engine set is fixed.

        Engine ``i``'s prefix is the sum of the widths before it, and its width
        must be the one its record reports.
        """
        widths = self.shard_widths
        full_fleet: dict[int, int] = {}
        prefix = 0
        for index, width in enumerate(widths):
            full_fleet[index] = prefix
            prefix += width
        if (
            membership.shard_prefixes != full_fleet
            or membership.workers_per_shard != widths
        ):
            raise NotImplementedError(
                "Megatron engines behind Dynamo are a fixed fleet; refit membership "
                f"must be every engine {full_fleet} with ranks {list(widths)}, got "
                f"{membership.shard_prefixes} with {list(membership.workers_per_shard)}."
            )

    def full_fleet_membership(self, train_world_size: int) -> RefitMembership:
        """Every engine, each a DP shard of its own TP x PP ranks after the trainers."""
        return desired_membership(
            absent_shards=[],
            dp_size=self.num_engines,
            total_gen_workers=self.inference_world_size,
            train_world_size=train_world_size,
            shard_widths=self.shard_widths,
        )

    def set_refit_membership(self, membership: RefitMembership) -> None:
        """Check the membership; the engine set is fixed, so it must be every engine.

        Nothing is stored: every refit call recomputes the full-fleet membership
        from the engine records, so there is no per-call membership to keep.
        """
        self.check_full_fleet(membership)

    def init_collective(
        self,
        ip: str,
        port: int,
        world_size: int,
        *,
        train_world_size: int,
    ) -> list[ray.ObjectRef]:
        """Join every engine to the NeMo RL ``model_update_group`` at its rank prefix.

        Engine ``i``'s workers are offset to the absolute rank where its shard
        starts: the trainers plus its prefix within the generation block.
        """
        expected = train_world_size + self.inference_world_size
        if world_size != expected:
            raise ValueError(
                f"NCCL world_size={world_size} does not match expected {expected} "
                f"(train={train_world_size} + engines={self.inference_world_size})"
            )
        membership = self.full_fleet_membership(train_world_size)
        futures: list[ray.ObjectRef] = []
        for index, record in enumerate(self._engine_records):
            futures.extend(
                record.init_collective(
                    ip,
                    port,
                    membership.world_size,
                    train_world_size=membership.train_world_size,
                    rank_offset=membership.train_world_size
                    + membership.shard_prefixes[index],
                )
            )
        return futures

    def prepare_refit_info(
        self, state_dict_info: Optional[dict[str, Any]]
    ) -> Optional[list[str]]:
        """Prepare Bridge conversion tasks on every engine."""
        if state_dict_info is None:
            raise ValueError("state_dict_info must not be None for Megatron refit")
        # Stored once; every worker resolves the same object.
        info_ref = ray.put(state_dict_info)
        self._call_engines(
            range(self.num_engines),
            lambda record: record.prepare_refit_info(info_ref),
            what="prepare_refit_info",
            deadline=time.monotonic() + self._control_timeout_s,
        )
        return None

    def update_weights_from_collective(
        self, refit_timeout_s: Optional[float] = None
    ) -> list[ray.ObjectRef]:
        """Receive one collective refit on every engine: a packed broadcast into the live parameters."""
        futures: list[ray.ObjectRef] = []
        for record in self._engine_records:
            futures.extend(record.update_weights_from_collective(refit_timeout_s))
        return futures

    def set_rollout_weight_version(self, version: int) -> None:
        """Rotate the policy epoch every engine stamps on its requests."""
        self._call_engines(
            range(self.num_engines),
            lambda record: record.set_rollout_weight_version(version),
            what="set_rollout_weight_version",
            deadline=time.monotonic() + self._control_timeout_s,
        )

    # ------------------------------------------------------------------
    # Fence around the transfer
    # ------------------------------------------------------------------

    def _wait_for_prefill_handoffs(
        self, prefill: Sequence[int], deadline: float
    ) -> None:
        """Wait until no prefill engine holds a pinned handoff.

        A pinned block is prompt KV a decode engine may still be pulling over
        NIXL; suspending the prefill engine while any are pinned could corrupt
        that transfer. Requests may still be in flight (nothing holds new ones
        back), but the prefill engines are paused, so no new handoff starts.
        The count decreases as a still-running decode engine finds room for a
        handoff it claimed (free KV blocks, then a batch slot), pulls it and
        releases it after its first streamed token, or as Megatron-LM's
        prefill coordinator releases an offer no decode engine claimed once
        its offer timeout (300 s by default) passes.

        Every engine is probed on each poll, so a dead engine or endpoint
        fails the wait as itself. The counts are printed on the first nonzero
        read and then every ``_pin_report_interval_s``.

        Raises:
            PinnedHandoffTimeout: Blocks were still pinned when the deadline
                passed, including a pin read or probe the deadline cut off.
            ray.exceptions.GetTimeoutError: A pin read ran past the deadline
                while every engine's latest count was zero (always so for the
                first read).
            ray.exceptions.RayActorError: A prefill worker died during a pin
                read. Other errors from a pin read also propagate unchanged;
                each carries a note naming the engine.
            RuntimeError: The liveness probe found an engine dead or its
                endpoint no longer serving.
        """
        # The last count read from each engine, kept across polls so a spent
        # deadline reports what is still pinned.
        pinned: dict[str, int] = {}
        stuck = (
            "prefill engines still hold KV blocks pinned for decode handoffs: "
            "{pinned}. A decode engine releases a pin it claimed after it "
            "admits the handoff and streams its first token; the prefill "
            "coordinator releases an offer no decode engine claimed once "
            "Megatron-LM's offer timeout (300 s by default) passes. So either a "
            "decode engine that claimed a handoff has stalled or died (see its "
            "rank-zero log), or this wait ended before an unclaimed offer "
            "expired; raise policy.generation.dynamo_cfg.control_timeout_s "
            "above that offer timeout."
        )
        next_report = float("-inf")
        while True:
            # Checked before a read: one issued with no time left cannot be
            # answered (Ray's get at timeout 0 raises unless the value is ready).
            if any(pinned.values()) and _seconds_left(deadline) <= 0:
                raise PinnedHandoffTimeout(stuck.format(pinned=pinned))
            pending = [
                (index, self._engine_records[index].pinned_handoff_block_count())
                for index in prefill
            ]
            for index, future in pending:
                instance_id = self._engine_records[index].instance_id
                try:
                    pinned[instance_id] = int(
                        ray.get(future, timeout=_seconds_left(deadline))
                    )
                except ray.exceptions.GetTimeoutError as error:
                    if any(pinned.values()):
                        raise PinnedHandoffTimeout(
                            stuck.format(pinned=pinned)
                        ) from error
                    error.add_note(
                        f"reading pinned handoff blocks failed on engine {instance_id}"
                    )
                    raise
                except Exception as error:
                    error.add_note(
                        f"reading pinned handoff blocks failed on engine {instance_id}"
                    )
                    raise
            if not any(pinned.values()):
                return
            if time.monotonic() >= next_report:
                print(
                    "  [Dynamo] Refit waiting for the KV blocks prefill pins "
                    f"for decode handoffs to drain: {pinned}",
                    flush=True,
                )
                next_report = time.monotonic() + self._pin_report_interval_s
            # Only a running decode engine releases a pin it claimed, so a dead
            # engine or endpoint must surface as itself, not as this wait
            # running out.
            # The probe shares the wait's deadline; one it cuts off is the
            # wait running out.
            try:
                self.check_alive(timeout_s=_seconds_left(deadline))
            except ray.exceptions.GetTimeoutError as error:
                raise PinnedHandoffTimeout(stuck.format(pinned=pinned)) from error
            time.sleep(min(self._idle_poll_interval_s, _seconds_left(deadline)))

    def _restore_engines(self, indices: Sequence[int], *, deadline: float) -> list[str]:
        """Resume ``indices`` in one wave, in record order; return the failures."""
        errors: list[str] = []
        self._attempt(
            list(indices),
            lambda record: record.resume_after_refit(),
            what="resume_after_refit",
            deadline=deadline,
            errors=errors,
        )
        return errors

    def suspend_for_refit(self, *, timeout_s: float | None = None) -> None:
        """Suspend every engine in the order that keeps prefill handoffs whole.

        Prefill engines stop scheduling first, so no new handoff can start,
        then this waits until decode has pulled every block prefill still
        pins. Only then are the prefill engines suspended, then the decode
        engines. The caller prepares the weights afterwards, as the Megatron
        weight synchronizer does for one dedicated engine.

        If any step fails or the deadline passes, the fleet attempts a
        best-effort resume of every engine the suspend reached (prefill after
        a failed pause, pin wait or prefill suspend; every engine after a
        failed decode suspend) and the original error still propagates, with
        any resume failures added to it as notes. A worker that failed
        mid-transition may need the next refit to recover.

        Args:
            timeout_s: One deadline for the whole suspend; ``None`` uses
                ``dynamo_cfg.control_timeout_s``.

        Raises:
            PinnedHandoffTimeout: Blocks were still pinned when the deadline
                passed, including a pin read or liveness probe the deadline
                cut off.
            ray.exceptions.GetTimeoutError: A pause or suspend call ran past
                the deadline, or a pin read did while every engine's latest
                count was zero.
            RuntimeError: An engine died or its endpoint stopped serving
                while prefill still pinned blocks.
            ray.exceptions.RayActorError: A worker died during a pause,
                suspend or pin read. Other worker errors propagate unchanged;
                every error from those calls carries a note naming the engine
                and the step.
        """
        timeout_s = self._control_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        prefill = self._prefill_indices()
        decode = self._decode_indices()
        # Every engine a suspend step was sent to, even one whose call failed:
        # the call may have taken effect anyway.
        reached = list(prefill)
        try:
            self._call_engines(
                prefill,
                lambda record: record.pause_for_refit(),
                what="pause_for_refit",
                deadline=deadline,
            )
            self._wait_for_prefill_handoffs(prefill, deadline)
            self._call_engines(
                prefill,
                lambda record: record.suspend_for_refit(),
                what="suspend_for_refit",
                deadline=deadline,
            )
            reached.extend(decode)
            self._call_engines(
                decode,
                lambda record: record.suspend_for_refit(),
                what="suspend_for_refit",
                deadline=deadline,
            )
        except Exception as error:
            # The weights are unchanged, so a plain resume puts each engine back
            # in service. A fresh deadline: the suspend may have used up its own.
            restore_errors = self._restore_engines(
                reached, deadline=time.monotonic() + self._control_timeout_s
            )
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
        timeout_s = kwargs.get("timeout_s")
        timeout_s = self._control_timeout_s if timeout_s is None else timeout_s
        self._call_engines(
            range(self.num_engines),
            lambda record: record.prepare_for_generation(tags=list(tags)),
            what=f"prepare_for_generation({list(tags)})",
            deadline=time.monotonic() + timeout_s,
        )
        return True

    def resume_after_refit(self, *, timeout_s: float | None = None) -> None:
        """Resume every engine in one wave; the first resume is also the startup.

        No role order is needed on the way back up. The engines keep their KV on
        the GPU across suspend and resume (the persist cache mode the handoff
        requires), and a handoff that reaches a decode engine still waking is
        queued and read into buffers the resume does not touch, so prefill may
        serve again before decode does. In the weight synchronizer's sequence the
        ``kv_cache`` phase has already woken every engine at once, so this call
        usually finds them serving and resumes only what that phase did not
        reach. Every engine is attempted even if another fails, so one bad
        engine does not keep the rest down. The first refit built the engines, so only
        now do their endpoints start and register with the frontend (as vLLM
        engines do once they serve), and this waits until the frontend routes
        the model before returning.

        Raises:
            RuntimeError: Any engine failed to resume.
        """
        timeout_s = self._control_timeout_s if timeout_s is None else timeout_s
        errors = self._restore_engines(
            range(self.num_engines), deadline=time.monotonic() + timeout_s
        )
        if errors:
            raise RuntimeError(
                "Megatron resume after weight update failed for " + "; ".join(errors)
            )
        self._await_dynamo_fleet_ready()

    def _await_dynamo_fleet_ready(self) -> None:
        """Start the engines' endpoints and wait for the frontend to list them; once."""
        if self._dynamo_fleet_ready:
            return
        model_name = str(self._policy_config["model_name"])
        self.start_endpoints(
            namespace=self._dynamo_namespace,
            engine_env=self._dynamo_engine_env,
            model_name=model_name,
            timeout_s=self._dynamo_cfg.startup_timeout_s,
        )
        if self._dynamo_runtime is not None:
            self._dynamo_runtime.wait_for_engines(
                self.expected_components(), check_engines=self.check_alive
            )
        else:
            # A copy without the service processes (the controller's) waits on
            # the frontend over HTTP alone.
            wait_for_engine_registrations(
                frontend_url=self._dynamo_frontend_url,
                namespace=self._dynamo_namespace,
                expected_components=self.expected_components(),
                model_name=model_name,
                timeout_s=self._dynamo_cfg.startup_timeout_s,
                check_alive=self.check_alive,
            )
        self._dynamo_fleet_ready = True

    def start_endpoints(
        self,
        *,
        namespace: str,
        engine_env: Mapping[str, str],
        model_name: str,
        timeout_s: float,
    ) -> None:
        """Serve every engine through Dynamo; the engines must already be built.

        Engine by engine, in record order: rank zero starts Megatron-LM's
        adapter (``start_dynamo_endpoint``), then every rank points its events
        at it (``attach_dynamo_events``) and rank zero's "ready" lets the adapter
        register the engine. The vLLM backend does the same once its engines
        serve; Megatron engines get their weights, and so their engine, at the
        first refit, so this runs right after it. Goes through the workers'
        handles, so the SingleController actor's copy can run it too; the
        workers return the first call's endpoint on a repeat.

        Args:
            namespace: Dynamo namespace shared with the frontend.
            engine_env: etcd / NATS variables from the managed runtime.
            model_name: The served model; also the name it is served under.
            timeout_s: Bound on the whole call.
        """
        if self._endpoints_started:
            return
        deadline = time.monotonic() + timeout_s
        # Ports are node-local: a later engine on a node avoids the ports the
        # earlier engines on that node took.
        ports_by_node: dict[str, set[int]] = {}
        for record in self._engine_records:
            used_ports = ports_by_node.setdefault(record.node_ip, set())
            try:
                metadata = ray.get(
                    record.leader.start_dynamo_endpoint.remote(
                        namespace=namespace,
                        manager_env=dict(engine_env),
                        model_name=model_name,
                        served_model_name=model_name,
                        # Native rollouts post token-ID completions; chat stays off.
                        endpoint_types=["completions"],
                        engine_start_timeout_s=self._dynamo_cfg.startup_timeout_s,
                        excluded_ports=sorted(used_ports),
                    ),
                    timeout=_seconds_left(deadline),
                )
                used_ports.update(
                    {int(metadata["event_port"]), int(metadata["system_port"])}
                )
                ray.get(
                    record.attach_dynamo_events(
                        parent_event_address=metadata["parent_event_address"]
                    ),
                    timeout=_seconds_left(deadline),
                )
            except Exception as error:
                error.add_note(
                    f"while starting the Dynamo endpoint of {record.instance_id}"
                )
                raise
            print(
                f"  [Dynamo:{record.instance_id}] endpoint started on "
                f"{metadata['node_ip']}, handoff owner={metadata['handoff_owner']}",
                flush=True,
            )
        self._endpoints_started = True

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
            "Megatron engines behind Dynamo serve generate_async() only; the "
            "frontend takes one request at a time."
        )

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Post the batch to the frontend, one request per prompt; yield the assembled batch."""
        # Deferred: the rollout client pulls in aiohttp.
        from nemo_rl.models.generation.megatron.dynamo_rollouts import (
            generate_through_frontend,
        )

        if len(data["input_ids"]) == 0:
            return
        output = await generate_through_frontend(
            self.cfg,
            frontend_url=self.dp_openai_server_base_urls[0] or "",
            data=data,
            greedy=greedy,
            request_timeout_s=self._dynamo_cfg.request_timeout_s,
        )
        output["gen_leader_worker_idx"] = [0]
        yield 0, output

    def setup_token_capture(
        self,
        dp_cfg: "DataPlaneConfig",
        staging_partition: str,
        *,
        capture_media: bool = False,
        top_logprobs: int = 0,
    ) -> "Optional[MediaColumnSpec]":
        """Not served yet: token capture rides on NeMo-Gym rollouts through the frontend."""
        raise NotImplementedError(
            "Token capture needs NeMo-Gym rollouts, which are not yet served "
            "through the Dynamo frontend."
        )

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def _start_engines(
        self,
        roles: Sequence[MegatronDynamoRole],
        widths: Mapping[MegatronDynamoRole, int],
    ) -> None:
        """Bring up one engine per role, in role order.

        Args:
            roles: ``prefill`` / ``decode`` per engine, from
                ``dynamo_config.engine_roles``; the order is the refit rank order.
            widths: GPU ranks an engine of each role spans, from
                ``dynamo_config.engine_widths``.

        Raises:
            ValueError: On an engine wider than a node, or more engines than
                the node pins hold.
            RuntimeError: When an engine's GPUs land on more than one node.
        """
        roles = list(roles)
        # Distinct roles in first-seen order; typed so the per-role lookups
        # below keep the role literal.
        distinct_roles: list[MegatronDynamoRole] = []
        for role in roles:
            if role not in distinct_roles:
                distinct_roles.append(role)
        for role in distinct_roles:
            if widths[role] > self._num_gpus_per_node:
                raise ValueError(
                    f"A Megatron {role} engine spans {widths[role]} GPUs "
                    f"but nodes have {self._num_gpus_per_node}; multi-node "
                    "engines are not supported behind Dynamo. Lower "
                    f"policy.generation.dynamo_cfg.{role}_engine or "
                    "mcore_generation_config tensor_model_parallel_size x "
                    "pipeline_model_parallel_size."
                )
        # Which pinned inference node each engine takes, when the shared
        # inference cluster has node pins. Engines tile the pinned nodes in
        # launch order, each engine wholly on one node: an engine that does not
        # fit in the GPUs the earlier engines left on the current node starts
        # the next one. With one width this is engine_index // engines_per_node.
        constraints = self._node_resource_constraints
        pin_index_by_engine: list[int] = []
        node_index = 0
        used_on_node = 0
        for role in roles:
            width = widths[role]
            if used_on_node + width > self._num_gpus_per_node:
                node_index += 1
                used_on_node = 0
            pin_index_by_engine.append(node_index)
            used_on_node += width
        nodes_needed = node_index + 1
        if constraints is not None and nodes_needed > len(constraints):
            widths_by_role = {role: widths[role] for role in distinct_roles}
            raise ValueError(
                f"{len(roles)} Megatron engines of widths {widths_by_role} GPUs tile "
                f"{nodes_needed} nodes of {self._num_gpus_per_node} GPUs, more "
                f"than the {len(constraints)} pinned inference nodes."
            )
        # Each role's engines are built from the policy config with that
        # role and its layout overrides.
        policy_config_by_role: dict[str, dict[str, Any]] = {}
        for role in distinct_roles:
            policy_config_by_role[role] = role_policy_config(
                self._policy_config, role, self._dynamo_cfg.role_overrides(role)
            )
        role_counts: dict[str, int] = {}
        for engine_index, role in enumerate[MegatronDynamoRole](roles):
            ordinal = role_counts.get(role, 0)
            role_counts[role] = ordinal + 1
            instance_id = f"{role}-{ordinal}"
            width = widths[role]
            engine_policy_config = cast("PolicyConfig", policy_config_by_role[role])
            cluster = RayVirtualCluster(
                name=f"dynamo_{self._fleet_id}_{instance_id}",
                bundle_ct_per_node_list=[width],
                use_gpus=True,
                num_gpus_per_node=self._num_gpus_per_node,
                max_colocated_worker_groups=1,
                port_range_low=self._port_range_low,
                port_range_high=self._port_range_high,
                node_resource_constraints=(
                    None
                    if constraints is None
                    else [constraints[pin_index_by_engine[engine_index]]]
                ),
            )
            self._clusters.append(cluster)
            # Reserved the way the engine will reserve them (always PACK,
            # since Dynamo engines are never colocated; idempotent), so a
            # split engine fails before any worker is scheduled.
            MegatronGeneration.init_cluster_placement_groups(
                cluster, engine_policy_config
            )
            node_ip = self._engine_node_ip(cluster, instance_id)
            engine = MegatronGeneration(
                config=engine_policy_config,
                tokenizer=self._tokenizer,
                cluster=cluster,
                name_prefix=f"dynamo_{instance_id}",
                skip_weight_load=True,
            )
            self._engines.append(engine)
            self._engine_records.append(
                MegatronDynamoEngine.of(
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

    def _engine_node_ip(self, cluster: RayVirtualCluster, instance_id: str) -> str:
        """Node every bundle of an engine's cluster sits on.

        ``PACK`` is best effort: on a fragmented or partly busy cluster Ray may
        spread an engine's bundles over several nodes, and its tensor-parallel
        group would then silently run across them.
        """
        refs = [
            cluster_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_bundle_index=bundle_index,
                )
            ).remote()
            for placement_group in cluster.get_placement_groups()
            for bundle_index in range(placement_group.bundle_count)
        ]
        node_ips = set(ray.get(refs))
        if len(node_ips) != 1:
            raise RuntimeError(
                f"Megatron engine {instance_id} spans multiple nodes "
                f"{sorted(node_ips)}. Multi-node engines are not supported "
                "behind Dynamo."
            )
        return next(iter(node_ips))

    def _prefill_coordinator_addresses(self) -> dict[int, str]:
        """Each listening prefill engine's coordinator address, by engine index.

        Read from every prefill engine's rank zero at once, under one
        control-timeout deadline. A read that fails or runs past it is logged
        and that engine left out, so the others still count; so is an engine
        whose coordinator is not up yet (before the first refit builds it).
        Only issuing the reads raises (when Ray itself is gone).
        """
        deadline = time.monotonic() + self._control_timeout_s
        pending = [
            (
                index,
                self._engine_records[index].leader.dynamo_coordinator_address.remote(),
            )
            for index in self._prefill_indices()
        ]
        addresses: dict[int, str] = {}
        for index, future in pending:
            try:
                address = ray.get(future, timeout=_seconds_left(deadline))
            except Exception:
                LOGGER.exception(
                    "Could not read the coordinator address of %s",
                    self._engine_records[index].instance_id,
                )
                continue
            if address:
                addresses[index] = address
        return addresses

    def shutdown(self) -> bool:
        """Decode first, attest their termination to prefill, then prefill, then the services.

        Prefill engines keep KV blocks pinned for decode attempts they have not
        heard back from. Once a decode engine's endpoint task is cancelled and
        its workers' kill has been issued (not awaited, which is short of the
        hook's "every rank exited" precondition and safe only because the
        prefill engines stop next), its owner ID is reported to each prefill
        coordinator so the pins are released before the prefill engines
        themselves go down. A decode engine that did not stop cleanly is not
        attested, and with no owner to attest (before the first refit started
        the endpoints) no prefill engine is asked. The frontend, NATS and etcd
        stop last. Best effort and idempotent: every failure is logged and the
        teardown continues. A copy that owns neither engines nor services does
        nothing.
        """
        if self._is_client_copy:
            return True
        decode = self._decode_indices()
        prefill = self._prefill_indices()
        # Read before the decode engines stop: only their rank zero knows the
        # owner prefill pins KV blocks for.
        try:
            owners = self._handoff_owners(decode)
        except Exception:
            LOGGER.exception("Could not read the decode engines' handoff owners")
            owners = {index: None for index in decode}
        stopped_owners: list[str] = []
        for index in decode:
            if not self._stop_engine(index):
                LOGGER.warning(
                    "Decode engine %s did not stop cleanly; its handoff owner is "
                    "not attested as terminated",
                    self._engine_records[index].instance_id,
                )
                continue
            owner = owners[index]
            if owner:
                stopped_owners.append(owner)
        # Releases the KV blocks prefill still pins for the stopped decode
        # engines; best effort at full teardown, since prefill stops next.
        if stopped_owners and prefill:
            self._confirm_decode_terminated(stopped_owners)
        for index in prefill:
            self._stop_engine(index)
        # Engines that failed before their record existed.
        for index in range(len(self._engine_records), len(self._engines)):
            self._stop_engine(index)
        for cluster in self._clusters:
            try:
                cluster.shutdown()
            except Exception:
                LOGGER.exception("Could not remove an engine placement group")
        self._engine_records.clear()
        self._engines.clear()
        self._clusters.clear()
        self._endpoints_started = False
        # After the engines: they leave discovery as their processes exit.
        runtime, self._dynamo_runtime = self._dynamo_runtime, None
        if runtime is not None:
            runtime.shutdown()
        return True

    def _handoff_owners(self, indices: Sequence[int]) -> dict[int, str | None]:
        """Each decode engine's handoff owner; None before its endpoint started.

        Read from every engine at once, under one control-timeout deadline. A
        read that fails or runs past it is logged and counts as no owner.
        """
        deadline = time.monotonic() + self._control_timeout_s
        pending = [
            (
                index,
                self._engine_records[index].leader.dynamo_endpoint_metadata.remote(),
            )
            for index in indices
        ]
        owners: dict[int, str | None] = {}
        for index, future in pending:
            try:
                metadata = ray.get(future, timeout=_seconds_left(deadline))
            except Exception:
                LOGGER.exception(
                    "Could not read the handoff owner of %s",
                    self._engine_records[index].instance_id,
                )
                metadata = None
            owners[index] = None if metadata is None else metadata.get("handoff_owner")
        return owners

    def _confirm_decode_terminated(self, owners: Sequence[str]) -> None:
        """Have a prefill engine's rank zero attest the stopped decode owners.

        The attester is the first prefill engine whose coordinator address was
        read, and only the coordinators read are attested to, so a prefill
        engine that does not answer does not keep the others from releasing
        their pins.
        """
        try:
            addresses = self._prefill_coordinator_addresses()
        except Exception:
            LOGGER.exception("Could not read prefill coordinator addresses")
            return
        if not addresses:
            LOGGER.warning(
                "No prefill coordinator address could be read; decode owners %s "
                "are not attested as terminated",
                list(owners),
            )
            return
        attester = self._engine_records[next(iter(addresses))]
        coordinator_addresses = list(addresses.values())
        timeout_s = self._control_timeout_s
        for owner in owners:
            try:
                accepted = ray.get(
                    attester.leader.confirm_decode_terminated.remote(
                        coordinator_addresses=coordinator_addresses,
                        owner=owner,
                        timeout_s=timeout_s,
                    ),
                    # The worker bounds its own wait at timeout_s plus a margin.
                    timeout=timeout_s + 10.0,
                )
            except Exception:
                LOGGER.exception("Could not attest decode owner %s termination", owner)
                continue
            if not accepted:
                LOGGER.warning(
                    "Prefill coordinators did not accept decode owner %s "
                    "termination; its pinned KV blocks stay until the prefill "
                    "engines stop (see the prefill rank zero's log)",
                    owner,
                )

    def _stop_engine(self, index: int) -> bool:
        """Stop one engine's Dynamo endpoint and policy; whether both stopped cleanly.

        A hung or failed endpoint stop is logged and the engine is shut down
        anyway, which kills its workers.
        """
        stopped = True
        if index < len(self._engine_records):
            try:
                ray.get(
                    self._engine_records[index].leader.stop_dynamo_endpoint.remote(),
                    timeout=self._control_timeout_s,
                )
            except Exception:
                LOGGER.exception("Could not stop a Megatron engine's Dynamo endpoint")
                stopped = False
        if index < len(self._engines):
            engine = self._engines[index]
            try:
                engine_stopped = engine.shutdown()
            except Exception:
                LOGGER.exception("Could not stop a Megatron engine")
                stopped = False
            else:
                if not engine_stopped:
                    LOGGER.error("A Megatron engine did not shut down cleanly")
                    stopped = False
        return stopped

    def __del__(self) -> None:
        """Safety net so the engines and services do not outlive the object that built them."""
        if getattr(self, "_engines", None) or getattr(self, "_dynamo_runtime", None):
            self.shutdown()
