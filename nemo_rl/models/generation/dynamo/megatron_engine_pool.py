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

"""NeMo-RL-owned Megatron prefill/decode engines behind a Dynamo frontend.

One engine is one ``MegatronGeneration`` over its own worker group (DP = 1),
exactly the dedicated inference policy the Megatron backend already builds.
Next to each engine runs an adapter sidecar: Megatron-LM's Dynamo backend in
external-launch mode, which registers the engine with Dynamo, talks to its
coordinator through the ordinary ``InferenceClient`` protocol, and carries the
KV handoff between prefill and decode. The engine process group is created
here; the sidecar only waits for rank zero to report readiness on the parent
event socket, which happens when the engine comes up at the first refit.
"""

import copy
import logging
import os
import signal
import socket
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_LOW,
    RayVirtualCluster,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.models.generation.dynamo.arguments import (
    build_managed_worker_env,
    redact_argv,
    redact_environment,
)
from nemo_rl.models.generation.dynamo.config import (
    MEGATRON_ROLE_LAYOUT_KEYS,
    DynamoConfig,
)
from nemo_rl.models.generation.dynamo.role_plan import (
    DisaggRolePlan,
    EngineRecord,
    EngineSpec,
)
from nemo_rl.models.generation.dynamo.venv import (
    get_dynamo_python,
    get_dynamo_venv_dir,
)

LOGGER = logging.getLogger(__name__)

_ADAPTER_ACTOR_FQN = (
    "nemo_rl.models.generation.dynamo.megatron_engine_pool.DynamoMegatronAdapterActor"
)
_ADAPTER_MODULE = "megatron.inference.integrations.dynamo"
_HANDOFF_RECOVERY_MODULE = "megatron.inference.integrations.dynamo.handoff_recovery"

# Flags the pool owns. A user-supplied value would either change which process
# builds the engine (launch mode, rank count) or break the event/ownership
# contract between the sidecar and the engine this pool started.
_RESERVED_ADAPTER_FLAGS = frozenset(
    {
        "--engine-launch-mode",
        "--nproc-per-node",
        "--parent-event-host",
        "--parent-event-port",
        "--coordinator-host",
        "--coordinator-port",
        "--handoff-owner",
        "--role",
        "--component",
        "--model",
        "--served-model-name",
        "--namespace",
        "--worker-id-file",
        "--megatron-root",
    }
)


def default_megatron_root() -> Path:
    """Megatron-LM nested in the Megatron-Bridge submodule."""
    repo_root = Path(__file__).resolve().parents[4]
    return (
        repo_root
        / "3rdparty"
        / "Megatron-Bridge-workspace"
        / "Megatron-Bridge"
        / "3rdparty"
        / "Megatron-LM"
    )


def build_megatron_adapter_argv(
    *,
    role: str,
    component: str,
    model_name: str,
    namespace: str,
    parent_event_host: str,
    parent_event_port: int,
    coordinator_host: str,
    megatron_root: str,
    engine_start_timeout_s: float,
    handoff_owner: str,
    endpoint_types: Sequence[str],
    engine_argv: Sequence[str],
    extra_cli_args: Sequence[str] = (),
) -> list[str]:
    """Arguments for ``python -m megatron.inference.integrations.dynamo``.

    External launch mode: the adapter binds the parent event socket at
    ``parent_event_host:parent_event_port`` and waits for an engine it did not
    start. The adapter's parser still requires a Megatron argument list after
    ``--``; in external mode that list is only recorded, never executed, so the
    engine's layout flags are passed there for the record.
    """
    if role not in ("prefill", "decode"):
        raise ValueError(f"role must be 'prefill' or 'decode', got {role!r}")
    for arg in extra_cli_args:
        flag = str(arg).split("=", 1)[0]
        if flag in _RESERVED_ADAPTER_FLAGS:
            raise ValueError(
                "dynamo_cfg.worker_args.extra_cli_args may not set the adapter "
                f"flag {flag}; the Megatron engine pool owns it"
            )
    if "--" in extra_cli_args:
        raise ValueError(
            "dynamo_cfg.worker_args.extra_cli_args may not contain '--'; the "
            "Megatron engine is launched by NeMo RL, not by the adapter"
        )
    if not engine_argv:
        raise ValueError("engine_argv must describe the engine layout")
    argv = [
        "--role",
        role,
        "--component",
        component,
        "--endpoint",
        "generate",
        "--endpoint-types",
        ",".join(endpoint_types),
        "--model",
        model_name,
        "--served-model-name",
        model_name,
        "--namespace",
        namespace,
        "--discovery-backend",
        "etcd",
        "--request-plane",
        "tcp",
        "--event-plane",
        "nats",
        "--engine-launch-mode",
        "external",
        "--parent-event-host",
        parent_event_host,
        "--parent-event-port",
        str(parent_event_port),
        "--coordinator-host",
        coordinator_host,
        "--megatron-root",
        megatron_root,
        "--engine-start-timeout",
        str(float(engine_start_timeout_s)),
    ]
    if role == "decode":
        argv.extend(["--handoff-owner", handoff_owner])
    argv.extend(str(arg) for arg in extra_cli_args)
    argv.append("--")
    argv.extend(str(arg) for arg in engine_argv)
    return argv


def engine_layout_argv(megatron_cfg: Mapping[str, Any]) -> list[str]:
    """Megatron layout flags recorded after ``--`` for one engine."""
    argv = [
        "--tensor-model-parallel-size",
        str(int(megatron_cfg["tensor_model_parallel_size"])),
        "--pipeline-model-parallel-size",
        str(int(megatron_cfg.get("pipeline_model_parallel_size", 1))),
    ]
    expert = megatron_cfg.get("expert_model_parallel_size")
    if expert is not None and int(expert) > 1:
        argv.extend(["--expert-model-parallel-size", str(int(expert))])
    return argv


def engine_policy_config(
    policy_config: Mapping[str, Any], spec: EngineSpec
) -> dict[str, Any]:
    """The ``PolicyConfig`` one engine's dedicated inference policy runs with.

    The role's layout overrides win over both ``policy.megatron_cfg`` and any
    layout key inherited in ``mcore_generation_config``, so the engine's world
    size is exactly what the plan allocated for it.
    """
    config = copy.deepcopy(dict(policy_config))
    megatron_cfg = dict(config["megatron_cfg"])
    generation = dict(config["generation"])
    mcore = dict(generation["mcore_generation_config"])
    role_megatron = dict(spec.engine_cfg.get("megatron_cfg") or {})
    role_megatron.setdefault("context_parallel_size", 1)
    for key, value in role_megatron.items():
        megatron_cfg[key] = value
        if key in MEGATRON_ROLE_LAYOUT_KEYS:
            mcore[key] = value
    for key, value in (spec.engine_cfg.get("mcore_generation_config") or {}).items():
        mcore[key] = value
    generation["mcore_generation_config"] = mcore
    config["megatron_cfg"] = megatron_cfg
    config["generation"] = generation
    return config


@ray.remote(num_cpus=0)
class DynamoMegatronAdapterActor:  # pragma: no cover
    """Own one Megatron Dynamo adapter subprocess next to an engine's rank zero.

    Scheduled on the engine's first placement-group bundle with no GPU, so the
    parent event socket it binds is on the node rank zero connects from, and the
    coordinator host it advertises is routable from the other engines.
    """

    def __init__(
        self,
        *,
        instance_id: str,
        role: str,
        component: str,
        model_name: str,
        namespace: str,
        megatron_root: str,
        manager_env: dict[str, str],
        engine_start_timeout_s: float,
        endpoint_types: list[str],
        extra_cli_args: list[str],
        engine_argv: list[str],
        excluded_ports: list[int],
    ) -> None:
        self._instance_id = instance_id
        self._role = role
        self._node_ip = _get_node_ip_local()
        self._handoff_owner = uuid.uuid4().hex
        self._process: subprocess.Popen | None = None
        used = set(excluded_ports)
        self._event_port = _get_free_port_local(
            DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_LOW,
            DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_HIGH,
            max_retries=None,
            excluded_ports=used,
        )
        used.add(self._event_port)
        self._system_port = _get_free_port_local(
            DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_LOW,
            DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_HIGH,
            max_retries=None,
            excluded_ports=used,
        )
        if not Path(megatron_root).is_dir():
            raise FileNotFoundError(
                f"dynamo_cfg.megatron_root {megatron_root!r} is not a directory "
                f"on {self._node_ip}"
            )
        dynamo_python = get_dynamo_python()
        dynamo_venv = str(get_dynamo_venv_dir())
        env = build_managed_worker_env(
            base_env=os.environ,
            configured_env={},
            manager_env={
                **manager_env,
                "DYN_SYSTEM_PORT": str(self._system_port),
                # The sidecar never touches a GPU; the engine ranks own them.
                "CUDA_VISIBLE_DEVICES": "",
                "VIRTUAL_ENV": dynamo_venv,
                "UV_PROJECT_ENVIRONMENT": dynamo_venv,
            },
        )
        env["PYTHONPATH"] = os.pathsep.join(
            path for path in (megatron_root, env.get("PYTHONPATH", "")) if path
        )
        argv = build_megatron_adapter_argv(
            role=role,
            component=component,
            model_name=model_name,
            namespace=namespace,
            parent_event_host=self._node_ip,
            parent_event_port=self._event_port,
            coordinator_host=self._node_ip,
            megatron_root=megatron_root,
            engine_start_timeout_s=engine_start_timeout_s,
            handoff_owner=self._handoff_owner,
            endpoint_types=endpoint_types,
            engine_argv=engine_argv,
            extra_cli_args=extra_cli_args,
        )
        command = [dynamo_python, "-m", _ADAPTER_MODULE, *argv]
        relevant_env = {
            key: value
            for key, value in env.items()
            if key.startswith(("DYN_", "ETCD_", "NATS_", "PYTHONPATH"))
        }
        print(
            f"  [Dynamo:{instance_id}] launching adapter argv={redact_argv(command)!r} "
            f"env={redact_environment(relevant_env)!r} "
            f"parent_event={self.parent_event_address}",
            flush=True,
        )
        self._command = command
        self._env = env
        try:
            self._process = subprocess.Popen(command, env=env, start_new_session=True)
            self._wait_for_port(self._event_port, engine_start_timeout_s)
        except Exception:
            self._stop_process()
            raise

    @property
    def parent_event_address(self) -> str:
        host = f"[{self._node_ip}]" if ":" in self._node_ip else self._node_ip
        return f"tcp://{host}:{self._event_port}"

    def _wait_for_port(self, port: int, timeout_s: float) -> None:
        deadline = time.monotonic() + min(timeout_s, 120.0)
        while time.monotonic() < deadline:
            process = self._process
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"Dynamo Megatron adapter for {self._instance_id} exited with "
                    f"code {process.returncode} before binding its event socket."
                )
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError(
            f"Dynamo Megatron adapter for {self._instance_id} did not bind its "
            f"event socket on port {port}."
        )

    def metadata(self) -> dict[str, Any]:
        process = self._process
        if process is None:
            raise RuntimeError(
                f"Dynamo Megatron adapter for {self._instance_id} is not running."
            )
        return {
            "instance_id": self._instance_id,
            "role": self._role,
            "node_ip": self._node_ip,
            "parent_event_address": self.parent_event_address,
            "system_port": self._system_port,
            "event_port": self._event_port,
            "handoff_owner": self._handoff_owner,
            "process_pid": process.pid,
        }

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def confirm_terminated(
        self, coordinator_addresses: list[str], owner: str, timeout_s: float
    ) -> bool:
        """Tell every prefill coordinator that a decode attempt is gone for good.

        Runs Megatron's supervisor hook in the adapter environment. Only valid
        after the decode adapter and every decode rank have exited.
        """
        if not coordinator_addresses:
            return True
        command = [self._command[0], "-m", _HANDOFF_RECOVERY_MODULE]
        for address in coordinator_addresses:
            command.extend(["--coordinator-address", address])
        command.extend(["--confirm-terminated-owner", owner])
        result = subprocess.run(
            command,
            env=self._env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            LOGGER.error(
                "handoff recovery for owner %s failed: stdout=%r stderr=%r",
                owner,
                result.stdout,
                result.stderr,
            )
            return False
        return True

    def _stop_process(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        self._process = None

    def shutdown(self) -> bool:
        process = self._process
        if process is None:
            return True
        self._stop_process()
        print(
            f"  [Dynamo:{self._instance_id}] adapter stopped pid={process.pid}",
            flush=True,
        )
        return True


class MegatronEnginePool:
    """``DynamoEnginePool`` of NeMo-RL-owned Megatron engines plus adapter sidecars."""

    def __init__(
        self,
        *,
        policy_config: Mapping[str, Any],
        tokenizer: Any,
        config: DynamoConfig,
        namespace: str,
        manager_env: Mapping[str, str],
        num_gpus_per_node: int,
    ) -> None:
        if not config.is_megatron_engine:
            raise ValueError("MegatronEnginePool requires dynamo_cfg.engine='megatron'")
        self._policy_config = policy_config
        self._tokenizer = tokenizer
        self._config = config
        self._namespace = namespace
        self._manager_env = dict(manager_env)
        self._num_gpus_per_node = num_gpus_per_node
        megatron_root = config.dynamo_cfg.megatron_root
        self._megatron_root = str(
            Path(megatron_root).expanduser()
            if megatron_root
            else default_megatron_root()
        )
        self._plan: DisaggRolePlan | None = None
        self._records: list[EngineRecord] = []
        self._clusters: list[RayVirtualCluster] = []
        self._engines: list[Any] = []
        self._adapters: list[ray.actor.ActorHandle] = []
        self._adapter_metadata: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # DynamoEnginePool
    # ------------------------------------------------------------------

    @property
    def records(self) -> list[EngineRecord]:
        return list(self._records)

    @property
    def plan(self) -> DisaggRolePlan:
        if self._plan is None:
            raise RuntimeError("MegatronEnginePool has not been started")
        return self._plan

    def expected_endpoints(self) -> dict[str, set[str]]:
        """Each engine registers one ``generate`` endpoint on its component."""
        return {component: {"generate"} for component in self.plan.expected_components}

    def start(self, plan: DisaggRolePlan) -> list[EngineRecord]:
        if self._records:
            raise RuntimeError("MegatronEnginePool is already started")
        self._plan = plan
        # Imported here: the driver process may lack Megatron, and the generation
        # facade only ships Ray handles to it.
        from nemo_rl.models.generation.megatron.megatron_generation import (
            MegatronGeneration,
        )

        ports_by_node: dict[str, set[int]] = {}
        try:
            for spec in plan.engines:
                if spec.world_size > self._num_gpus_per_node:
                    raise ValueError(
                        f"engine {spec.name} spans {spec.world_size} GPUs but nodes "
                        f"have {self._num_gpus_per_node}; multi-node engines are not "
                        "supported behind Dynamo"
                    )
                engine_cfg = engine_policy_config(self._policy_config, spec)
                cluster = RayVirtualCluster(
                    name=f"dynamo_{self._namespace}_{spec.name}",
                    bundle_ct_per_node_list=[spec.world_size],
                    use_gpus=True,
                    num_gpus_per_node=self._num_gpus_per_node,
                    max_colocated_worker_groups=1,
                    placement_group_strategy="PACK",
                )
                self._clusters.append(cluster)
                engine = MegatronGeneration(
                    config=engine_cfg,
                    tokenizer=self._tokenizer,
                    cluster=cluster,
                    name_prefix=f"dynamo_{spec.name}",
                    skip_weight_load=True,
                )
                self._engines.append(engine)
                adapter, metadata = self._start_adapter(
                    spec, engine_cfg, cluster, ports_by_node
                )
                self._adapters.append(adapter)
                self._adapter_metadata.append(metadata)
                engine.attach_dynamo_adapter(
                    parent_event_address=metadata["parent_event_address"],
                    role=spec.role,
                )
                self._records.append(
                    EngineRecord(
                        role=spec.role,
                        world_size=spec.world_size,
                        instance_id=spec.name,
                        node_ip=metadata["node_ip"],
                        control=engine,
                    )
                )
                print(
                    f"  [Dynamo:{spec.name}] engine ready for refit on "
                    f"{metadata['node_ip']} ({spec.world_size} ranks), adapter "
                    f"owner={metadata['handoff_owner']}",
                    flush=True,
                )
        except Exception:
            self.shutdown()
            raise
        return self.records

    def _start_adapter(
        self,
        spec: EngineSpec,
        engine_cfg: Mapping[str, Any],
        cluster: RayVirtualCluster,
        ports_by_node: dict[str, set[int]],
    ) -> tuple[ray.actor.ActorHandle, dict[str, Any]]:
        placement_group = cluster.get_placement_groups()[0]
        strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=0,
            placement_group_capture_child_tasks=True,
        )
        runtime_env = {
            "py_executable": get_actor_python_env(_ADAPTER_ACTOR_FQN),
            "env_vars": {
                **os.environ,
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            },
        }
        # Ports are node-local; the first engine on a node has nothing to avoid,
        # later ones avoid what earlier sidecars on that node reported.
        node_ip = ray.get(
            cluster_node_ip.options(scheduling_strategy=strategy).remote()
        )
        excluded = sorted(ports_by_node.setdefault(node_ip, set()))
        worker_args = self._config.dynamo_cfg.worker_args
        adapter = DynamoMegatronAdapterActor.options(
            num_gpus=0,
            runtime_env=runtime_env,
            scheduling_strategy=strategy,
            name=f"{self._namespace}-dynamo-adapter-{spec.name}",
        ).remote(
            instance_id=spec.name,
            role=spec.role,
            component=spec.component,
            model_name=str(self._policy_config["model_name"]),
            namespace=self._namespace,
            megatron_root=self._megatron_root,
            manager_env=self._manager_env,
            engine_start_timeout_s=self._config.dynamo_cfg.startup_timeout_s,
            endpoint_types=list(worker_args.endpoint_types),
            extra_cli_args=list(worker_args.extra_cli_args),
            engine_argv=engine_layout_argv(engine_cfg["megatron_cfg"]),
            excluded_ports=excluded,
        )
        metadata = dict(ray.get(adapter.metadata.remote()))
        ports_by_node[node_ip].update(
            {int(metadata["event_port"]), int(metadata["system_port"])}
        )
        return adapter, metadata

    def is_alive(self) -> bool:
        if not self._records:
            return False
        adapters_alive = ray.get(
            [adapter.is_alive.remote() for adapter in self._adapters]
        )
        if not all(adapters_alive):
            return False
        return all(ray.get(engine.shard_liveness_ref(0)) for engine in self._engines)

    def validate(self, expected: Sequence[EngineRecord]) -> list[EngineRecord]:
        if not self.is_alive():
            raise RuntimeError(
                "A Megatron engine or its Dynamo adapter exited behind the frontend."
            )
        current = ray.get([adapter.metadata.remote() for adapter in self._adapters])
        if [m["process_pid"] for m in current] != [
            m["process_pid"] for m in self._adapter_metadata
        ]:
            raise RuntimeError(
                "Dynamo adapter membership changed after collective initialization."
            )
        identities = [
            (r.role, r.world_size, r.instance_id, r.node_ip) for r in expected
        ]
        live = [(r.role, r.world_size, r.instance_id, r.node_ip) for r in self._records]
        if identities != live:
            raise RuntimeError(
                "Megatron engine membership changed after collective "
                f"initialization: expected={identities}, current={live}."
            )
        return self.records

    def prefill_coordinator_addresses(self) -> list[str]:
        """Coordinator addresses of the prefill engines that have come up."""
        addresses = []
        for record, engine in zip(self._records, self._engines, strict=True):
            if record.role != "prefill":
                continue
            address = engine.dynamo_coordinator_address()
            if address:
                addresses.append(address)
        return addresses

    def shutdown(self) -> None:
        """Decode first, then attest their termination to prefill, then prefill.

        Prefill engines keep KV blocks pinned for decode attempts they have not
        heard back from. Once every decode rank is gone, their owner IDs are
        reported to each surviving prefill coordinator so the pins are released
        before the prefill engines themselves go down.
        """
        indexed = list(enumerate(self._records))
        decode = [i for i, record in indexed if record.role == "decode"]
        prefill = [i for i, record in indexed if record.role == "prefill"]
        for index in decode:
            self._stop_engine(index)
        if decode and prefill:
            try:
                addresses = self.prefill_coordinator_addresses()
            except Exception:
                LOGGER.exception("Could not read prefill coordinator addresses")
                addresses = []
            attester = self._adapters[prefill[0]]
            for index in decode:
                owner = self._adapter_metadata[index].get("handoff_owner")
                if not addresses or not owner:
                    continue
                try:
                    ray.get(
                        attester.confirm_terminated.remote(
                            addresses,
                            owner,
                            self._config.dynamo_cfg.control_timeout_s,
                        ),
                        timeout=self._config.dynamo_cfg.control_timeout_s + 30,
                    )
                except Exception:
                    LOGGER.exception(
                        "Could not attest decode owner %s termination", owner
                    )
        for index in prefill:
            self._stop_engine(index)
        # Engines that failed before their record existed.
        for index in range(
            len(self._records), max(len(self._engines), len(self._adapters))
        ):
            self._stop_engine(index)
        for cluster in self._clusters:
            try:
                cluster.shutdown()
            except Exception:
                LOGGER.exception("Could not remove an engine placement group")
        self._records.clear()
        self._engines.clear()
        self._adapters.clear()
        self._adapter_metadata.clear()
        self._clusters.clear()

    def _stop_engine(self, index: int) -> None:
        if index < len(self._adapters):
            adapter = self._adapters[index]
            try:
                ray.get(adapter.shutdown.remote(), timeout=30)
            except Exception:
                LOGGER.exception("Could not stop a Dynamo Megatron adapter")
            try:
                ray.kill(adapter, no_restart=True)
            except Exception:
                pass
        if index < len(self._engines):
            try:
                self._engines[index].shutdown()
            except Exception:
                LOGGER.exception("Could not stop a Megatron engine")


@ray.remote(num_cpus=0)
def cluster_node_ip() -> str:  # pragma: no cover
    """IP of the node a placement-group bundle lives on."""
    return _get_node_ip_local()
