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
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.generation.megatron.config import MCoreGenerationConfig
from nemo_rl.models.generation.megatron.dynamo_config import (
    MegatronDynamoCfg,
    MegatronDynamoRole,
    disaggregation_cfg,
    engine_roles,
    engine_widths,
    megatron_dynamo_cfg,
    validate_megatron_dynamo_generation,
)
from nemo_rl.models.generation.megatron.megatron_generation import MegatronGeneration
from nemo_rl.models.megatron.router_replay import router_replay_enabled
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
    decode engine, each over
    its own one-node virtual cluster and built with ``skip_weight_load=True`` so
    its inference engine comes up at the first weight refit. Every engine has
    the layout the policy config gives plus its role's
    ``dynamo_cfg.prefill_engine`` / ``decode_engine`` overrides, so prefill and
    decode engines may differ in TP and PP (the KV handoff reshards between
    them). An engine's role, carried in its config, also picks the engine
    variant and KV-transfer agents its ranks build.

    This object is also the engines' refit side: every refit call the trainer
    makes here (collective init, prepare the weights, transfer, suspend,
    prepare the KV cache, resume) is fanned out over the engines' Ray workers
    in engine order. Rank placement comes from the shared ``RefitMembership``
    arithmetic: engine ``i`` starts at the trainers plus the widths of the
    engines before it, so a prefill engine at TP 4 may sit next to decode
    engines at TP 2. After a failed suspend the fleet attempts a best-effort
    resume of every engine and the original error still propagates. The object
    owns no policy and no worker group of its own.

    Pickling keeps the engine records (plain data plus the workers' actor
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
        same way. ``cluster`` contributes the node shape, port range and node
        pins only; each engine gets its own one-node cluster. Its placement
        groups stay unmaterialized, or the engines' GPUs would be booked twice.
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
        # Every control call is bounded: suspend and resume by their timeout_s
        # (control_timeout_s when None), one deadline per call, and
        # prepare_refit_info by control_timeout_s.
        self._control_timeout_s = float(self._dynamo_cfg.control_timeout_s)
        # One record per engine, in launch (refit rank) order; what refit calls use.
        self._engine_records: list[MegatronDynamoEngine] = []
        # The owning objects, in the same order; empty on a copy.
        self._engines: list[MegatronGeneration] = []
        self._clusters: list[RayVirtualCluster] = []
        # True on a copy that was unpickled (the SingleController actor's).
        self._is_client_copy = False

        validate_megatron_dynamo_generation(self.cfg)
        widths = engine_widths(self._policy_config)
        roles = engine_roles(disaggregation_cfg(self.cfg), cluster.world_size(), widths)
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
        would shut the engine down, so the copy keeps the engine records (plain
        data plus worker handles) and drives refits through them.
        """
        state = self.__dict__.copy()
        state["_engines"] = []
        state["_clusters"] = []
        state["_tokenizer"] = None
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
        self.check_full_fleet(membership)
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

    def _restore_engines(self, indices: Sequence[int], *, deadline: float) -> list[str]:
        """Resume ``indices``; return the failures."""
        errors: list[str] = []
        self._attempt(
            indices,
            lambda record: record.resume_after_refit(),
            what="resume_after_refit",
            deadline=deadline,
            errors=errors,
        )
        return errors

    def suspend_for_refit(self, *, timeout_s: float | None = None) -> None:
        """Suspend every engine before a weight update.

        The caller prepares the weights afterwards, as the Megatron weight
        synchronizer does for one dedicated engine.

        If any step fails or the deadline passes, the fleet attempts a
        best-effort resume of every engine and the original error still
        propagates, with any resume failures added to it as notes. A worker
        that failed mid-transition may need the next refit to recover.

        Args:
            timeout_s: One deadline for the whole suspend; ``None`` uses
                ``dynamo_cfg.control_timeout_s``.
        """
        timeout_s = self._control_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        indices = list(range(self.num_engines))
        try:
            self._call_engines(
                indices,
                lambda record: record.suspend_for_refit(),
                what="suspend_for_refit",
                deadline=deadline,
            )
        except Exception as error:
            # The weights are unchanged, so a plain resume puts each engine back
            # in service. A fresh deadline: the suspend may have used up its own.
            restore_errors = self._restore_engines(
                indices, deadline=time.monotonic() + self._control_timeout_s
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
        """Resume serving on every engine; the first resume is when the engines start serving.

        Every engine is attempted even if another fails, so one bad engine
        does not keep the rest down.

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

    def shutdown(self) -> bool:
        """Stop every engine, then release their placement groups.

        Best effort and idempotent: every failure is logged and the teardown
        continues. A copy that owns no engine does nothing.
        """
        if self._is_client_copy:
            return True
        # Includes engines that failed before their record existed.
        for index in range(len(self._engines)):
            self._stop_engine(index)
        for cluster in self._clusters:
            try:
                cluster.shutdown()
            except Exception:
                LOGGER.exception("Could not remove an engine placement group")
        self._engine_records.clear()
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
