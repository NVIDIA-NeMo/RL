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
import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import AsyncGenerator, Mapping
from pathlib import Path
from typing import Any, Dict, List, NotRequired, Optional, Protocol, TypedDict

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from transformers import PreTrainedTokenizerBase

from nemo_rl.data.multimodal_utils import (
    attach_image_model_inputs_to_message,
    extract_input_media_sources_from_responses_messages,
    media_sources_equal,
    uses_image_placeholder,
)
from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GYM_PORT_RANGE_HIGH,
    DEFAULT_GYM_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.environments.gym_checkpoint import (
    GYM_AGENT_CONTINUATION_INDEX_FEATURE,
    GYM_AGENT_COMPLETION_ACK_PATH,
    GYM_AGENT_COMPLETION_RECEIPT_PATH,
    GYM_AGENT_CHECKPOINT_PREFIX,
    GYM_AGENT_DISCARD_RESTORED_CONTINUATION_PATH,
    GYM_CHECKPOINT_CAPABILITIES_PATH,
    GYM_CHECKPOINT_CONTROL_PREFIX,
    GYM_EXTERNAL_STORAGE_REFERENCE_INDEX_FEATURE,
    GYM_MODEL_ADMISSION_PREFIX,
    GYM_MODEL_CHECKPOINT_PREFIX,
    GYM_RESOURCES_CHECKPOINT_PREFIX,
    GymAgentCheckpointDirectoryRequest,
    GymAgentDiscardRestoredContinuationRequest,
    GymAgentDiscardRestoredContinuationResponse,
    GymActorExecutionRegistry,
    GymAgentCommitResponse,
    GymAgentPrepareResponse,
    GymAgentRestoreResponse,
    GymAgentResumeResponse,
    GymCheckpointArtifactReference,
    GymCheckpointCommitResult,
    GymCheckpointControlRequest,
    GymCheckpointDirectoryRequest,
    GymCheckpointPrepareResult,
    GymCheckpointResumeResult,
    GymCheckpointRestoreResult,
    GymCheckpointTopology,
    GymCompletionReceipt,
    GymCompletedExecution,
    GymCompletedExecutionAcknowledgementRequest,
    GymCompletedExecutionAcknowledgementResponse,
    GymCoordinatorModelStatusResponse,
    GymControlCapabilities,
    GymDiscoveredParticipant,
    GymExecutionIdentity,
    GymModelCheckpointCommitRequest,
    GymModelCheckpointRestoreRequest,
    GymModelCommitResponse,
    GymModelPrepareResponse,
    GymModelRestoreResponse,
    GymModelResumeResponse,
    GymParticipantCommitResult,
    GymParticipantIdentity,
    GymParticipantManifestReference,
    GymParticipantPrepareResult,
    GymParticipantRestoreResult,
    GymParticipantResumeResult,
    GymResourcesCommitResponse,
    GymResourcesPrepareResponse,
    GymResourcesRestoreResponse,
    GymResourcesResumeResponse,
    GymSingleWorkerModelStatusResponse,
    gym_capture_key,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.nemo_gym_multimodal import (
    _index_per_turn_images,
    _is_trainable_output_item,
    _without_initial_media_sources,
    normalize_media_in_examples,
)
from nemo_rl.experience.failures import (
    GymTransportError,
    RolloutDataFailure,
    http_status_is_infra,
)
from nemo_rl.models.generation.interfaces import (
    resolve_routed_experts_dtype_name_for_model,
    should_use_async_rollouts,
)
from nemo_rl.models.policy import PolicyConfig, TokenizerConfig
from nemo_rl.utils.routed_experts_codec import decode_routed_experts
from nemo_rl.utils.timer import Timer
from nemo_rl.utils.venvs import make_actor_runtime_env

NEMO_GYM_ACTOR_FQN = "nemo_rl.environments.nemo_gym.NemoGym"

# Kept local so the Gym actor does not depend on model-config dtype resolution.
# Must cover every name resolve_routed_experts_dtype can produce.
_ROUTED_EXPERTS_DTYPES = {
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
}

DEFAULT_INVALID_TOOL_CALL_PATTERNS = [
    "<tool_call>",
    "</tool_call>",
    "<function_call>",
    "</function_call>",
]
DEFAULT_THINKING_TAGS = ["<think>", "</think>"]


class NemoGymCompatibleConfig(Protocol):
    """Configuration fields required to select the NeMo Gym rollout path."""

    @property
    def env(self) -> dict[str, Any]: ...

    @property
    def policy(self) -> PolicyConfig: ...


def should_use_nemo_gym(master_config: NemoGymCompatibleConfig) -> bool:
    """Determine whether NeMo Gym should handle rollouts and validation."""
    should_use_gym = bool(master_config.env.get("should_use_nemo_gym"))
    if not should_use_gym:
        return False

    generation_config = master_config.policy["generation"]
    assert should_use_async_rollouts(generation_config), (
        "❌ Error: In order to use NeMo-Gym, you must use a generation "
        "backend with `async_engine: true`!"
    )

    if generation_config["backend"] == "vllm":
        should_expose_http_server = generation_config.get("vllm_cfg", {}).get(
            "expose_http_server"
        )
    elif generation_config["backend"] == "megatron":
        should_expose_http_server = generation_config.get(
            "mcore_generation_config", {}
        ).get("expose_http_server")
    elif generation_config["backend"] == "trtllm":
        should_expose_http_server = generation_config.get("trtllm_cfg", {}).get(
            "expose_http_server"
        )
    elif generation_config["backend"] == "dynamo":
        should_expose_http_server = generation_config.get("vllm_cfg", {}).get(
            "expose_http_server"
        )
    else:
        should_expose_http_server = False
    assert should_expose_http_server, (
        "In order to use NeMo-Gym, you must expose the generation server via "
        "`expose_http_server: true`!"
    )

    return True


def _has_nan_generation_logprobs(result: dict) -> bool:
    """Return whether a postprocessed rollout contains NaN policy logprobs."""
    return any(
        message.get("generation_logprobs") is not None
        and torch.isnan(message["generation_logprobs"]).any()
        for message in result["message_log"]
    )


def _typed_gym_failure(error: Exception) -> Optional[Exception]:
    """Map a NeMo-Gym HTTP failure onto a typed, PICKLABLE failure, or None if not one.

    Classification has to happen here, on the raising side, because ``run_rollouts`` runs
    inside the ``NemoGym`` Ray actor and the exception must survive the actor boundary to
    reach the retry policy on the driver.

    It does not survive. aiohttp's ``raise_for_status`` passes ``headers=self.headers``,
    and those are a ``CIMultiDictProxy``, which cloudpickle cannot serialize -- so Ray
    drops the cause and the driver receives a bare ``RayTaskError`` with no type and no
    ``.status``. Every gym HTTP failure then classified DATA, capping the gym path at
    ``max_data_attempts_per_prompt`` (2) and leaving ``max_attempts_per_prompt`` (5)
    unreachable on the very path whose dead-endpoint scenario motivates it. Two things
    made that the dominant case rather than a corner: Gym's middleware turns inner-server
    failures into 500 -- exactly the status the INFRA branch is for -- and its transport
    layer retries disconnects in an uncapped loop, so those never arrive at all.

    ``GymTransportError`` and ``RolloutDataFailure`` take a single str, so they pickle
    cleanly and ``classify_rollout_failure``'s explicit-class fast path wins on the far
    side.

    Returns None when the exception carries no HTTP status, leaving the caller to
    re-raise it untouched.
    """
    status = getattr(error, "status", None)
    if not isinstance(status, int):
        return None
    detail = f"NeMo-Gym /run failed with HTTP {status}: {error}"
    if http_status_is_infra(status):
        return GymTransportError(detail)
    return RolloutDataFailure(detail)


def get_nemo_gym_uv_cache_dir() -> str | None:
    """Return the uv cache directory inside a container, or None outside one.

    Inside a container (NRL_CONTAINER=1), returns the uv cache location so Gym
    stores its caches in the expected shared path. Returns None outside a
    container, meaning the caller should omit this arg and let Gym create the
    cache locally (the default when you may not be able to write to /opt).
    """
    if not os.environ.get("NRL_CONTAINER"):
        return None
    return subprocess.check_output(["uv", "cache", "dir"]).decode().strip()


def get_nemo_gym_venv_dir() -> str | None:
    """Return the NeMo Gym venv directory from NEMO_GYM_VENV_DIR, or None.

    Returns the value of NEMO_GYM_VENV_DIR if set, otherwise None. When None
    the caller should omit this arg and let Gym create venvs locally (the
    default when a container is not used since you may not be able to write
    to /opt).
    """
    return os.environ.get("NEMO_GYM_VENV_DIR")


class NemoGymConfig(TypedDict):
    model_name: str
    base_urls: List[str]
    initial_global_config_dict: Dict[str, Any]
    # Port range for Gym HTTP servers (head server + subprocess servers).
    # Defaults to DEFAULT_GYM_PORT_RANGE_LOW/HIGH (5000-5999) from
    # nemo_rl.distributed.virtual_cluster.  See the port layout there.
    port_range_low: NotRequired[int]
    port_range_high: NotRequired[int]
    invalid_tool_call_patterns: NotRequired[
        List[str] | None
    ]  # Substrings in assistant text content that indicate an invalid tool call
    thinking_tags: NotRequired[
        List[str] | None
    ]  # Thinking tags to check for malformed usage
    require_routed_experts: NotRequired[
        bool
    ]  # Require Gym output items to carry R3 routed_experts
    routed_experts_dtype: NotRequired[
        str
    ]  # Carry dtype name for routed_experts tensors ("int8"/"int16"/"int32"), resolved from the model's expert count
    # Forwarded from policy.tokenizer.use_fastokens so rollout actors patch their
    # tokenizer consistently with the driver. Defaults to off when absent.
    use_fastokens: NotRequired[bool]
    # Multimodal fields (populated by `setup_nemo_gym_config` when VLM is enabled).
    tokenizer_config: NotRequired[
        Optional[TokenizerConfig]
    ]  # For processor reconstruction inside the actor
    pad_dynamic_image_shapes: NotRequired[
        bool
    ]  # Normalize heterogeneous image tensors while retaining exact imgs_sizes
    # Ledger-authoritative token capture (token_capture.enabled): the dumped
    # TokenCaptureConfig. Turns on external staging in Gym's policy model
    # server, switches run_rollouts to receipt mode, and assembles receipts
    # from the manifest control route. None/absent = legacy token-echo path.
    token_capture: NotRequired[Dict[str, Any] | None]


# Gym control-plane server name (the model server hosting the ledger) and the
# opaque run-body key rollout ids ride on (Gym's ROLLOUT_ID_KEY_NAME): the
# agent derives the id from the run body and stamps /ng-rollout/<id> on every
# model call, so the TQ sample id IS the capture key end to end.
_POLICY_SERVER_NAME = "policy_model"
_NG_ROLLOUT_ID_BODY_KEY = "_ng_rollout_id"
_NG_ATTEMPT_INDEX_BODY_KEY = "_ng_attempt_index"
_TOKEN_CAPTURE_CONTROL_PREFIX = "/training-token-capture/control"
_TOKEN_CAPTURE_CONTROL_ENV = "NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN"
_CHECKPOINT_CONTROL_ENV = "NEMO_GYM_CHECKPOINT_CONTROL_TOKEN"
_GYM_COMPONENT_KEYS = frozenset(
    {"responses_api_models", "responses_api_agents", "resources_servers"}
)


class GymControlRequestError(RuntimeError):
    """Typed non-success response from one Gym checkpoint control route."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        error_code: Optional[str],
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_code = error_code


def _checkpoint_server_names(global_config: Mapping[str, Any]) -> list[str]:
    """Return deterministic Gym service routing names from resolved config."""
    server_names: list[str] = []
    for server_name, entry in global_config.items():
        if not isinstance(server_name, str) or not isinstance(entry, Mapping):
            continue
        if len(_GYM_COMPONENT_KEYS.intersection(entry)) == 1:
            server_names.append(server_name)
    return sorted(server_names)


def _adapt_execution_identity_for_installed_gym(
    row: dict[str, Any],
    *,
    stable_execution_identity_enabled: bool,
) -> None:
    """Translate the new identity pair for Gym revisions that predate it."""
    if stable_execution_identity_enabled or _NG_ATTEMPT_INDEX_BODY_KEY not in row:
        return
    row[_NG_ROLLOUT_ID_BODY_KEY] = gym_capture_key(
        row[_NG_ROLLOUT_ID_BODY_KEY],
        row.pop(_NG_ATTEMPT_INDEX_BODY_KEY),
    )


def _detect_invalid_tool_call_and_malformed_thinking(
    output_item_dict: dict[str, Any],
    invalid_tool_call_patterns: list[str] | None = None,
    thinking_tags: list[str] | None = None,
) -> tuple[bool, bool]:
    """Flag a NeMo-Gym output item as an invalid tool call / malformed thinking.

    Inspects the final output item of a model turn. For a final *content*
    message, any thinking tag is malformed (thinking should never leak into the
    answer); for a *reasoning* summary, only a repeated tag (count > 1) is
    malformed (a single pair is expected). A textual tool-call pattern in either
    indicates an invalid (unexecuted) tool call.

    Returns:
        (is_invalid_tool_call, has_malformed_thinking).
    """
    invalid_tool_call_patterns = (
        invalid_tool_call_patterns or DEFAULT_INVALID_TOOL_CALL_PATTERNS
    )
    thinking_tags = thinking_tags or DEFAULT_THINKING_TAGS

    content = output_item_dict.get("content")
    is_output_message = (
        isinstance(content, list)
        and bool(content)
        and isinstance(content[0], dict)
        and isinstance(content[0].get("text"), str)
    )
    # NeMo-Gym only attaches generation_token_ids to the last output item of a
    # model call (see vllm_model/app.py postprocess_chat_response). So this item
    # is guaranteed to be the final thing the model produced for this turn.
    # If it's a reasoning item, the model output only reasoning (no content/tool calls).
    summary = output_item_dict.get("summary")
    is_reasoning_message = (
        output_item_dict.get("type") == "reasoning"
        and isinstance(summary, list)
        and bool(summary)
        and isinstance(summary[0], dict)
        and isinstance(summary[0].get("text"), str)
    )

    is_invalid_tool_call = False
    has_malformed_thinking = False
    if is_output_message:
        assistant_message_content = output_item_dict["content"][0]["text"]
        if any(
            pattern in assistant_message_content
            for pattern in invalid_tool_call_patterns
        ):
            is_invalid_tool_call = True
        if any(tag in assistant_message_content for tag in thinking_tags):
            has_malformed_thinking = True
    elif is_reasoning_message:
        assistant_message_content = output_item_dict["summary"][0]["text"]
        if any(
            pattern in assistant_message_content
            for pattern in invalid_tool_call_patterns
        ):
            is_invalid_tool_call = True
        if any(assistant_message_content.count(tag) > 1 for tag in thinking_tags):
            has_malformed_thinking = True

    return is_invalid_tool_call, has_malformed_thinking


def get_pad_dynamic_image_shapes(env_config: Mapping[str, Any]) -> bool:
    """Return nemo_gym's pad_dynamic_image_shapes from an env config, or False.

    Takes ``master_config.env`` rather than the whole config: interpreting
    NeMo-Gym settings belongs with the environment, and callers outside it only
    need the resolved boolean.

    The NemoGym actor reads the same key from its own config for the per-turn
    attach. The initial-payload attach runs in the driver instead, so it has to
    be read here and passed down, or multi-image prompts would be processed
    under different rules on the two paths.
    """
    nemo_gym_config = env_config.get("nemo_gym") if env_config else None
    if not nemo_gym_config:
        return False
    return bool(nemo_gym_config.get("pad_dynamic_image_shapes"))


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class NemoGym(EnvironmentInterface):
    """This environment class isn't really used for training. It's really meant as an integration wrapper around NeMo-Gym that hooks into the existing NeMo RL resource management via ray. So there is still one source of truth for resource management in NeMo RL."""

    def __init__(self, cfg: NemoGymConfig):
        self.cfg = cfg
        # Populated by _spinup. Declared here so a restarted actor -- Ray recreates it
        # through __init__, which does not start the Gym servers -- reports what
        # actually happened instead of an AttributeError from deep inside a rollout.
        self.rh: Any = None
        self.rch: Any = None
        self.head_server_config: Any = None
        self.node_ip: Optional[str] = None
        self.head_server_port: Optional[int] = None
        # Installed by set_tokenizer at spinup, not passed per rollout call. Declared
        # here rather than in _spinup so a second spinup cannot wipe an installed
        # tokenizer and then report that set_tokenizer was never called.
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        # _spinup replaces this from cfg. Keep restarted/unspun actors internally
        # complete so diagnostics and focused tests do not fail with AttributeError.
        self._token_capture_enabled = False
        self._stable_execution_identity_enabled = False
        self._server_client: Any = None
        self._token_capture_control_headers: Dict[str, str] = {}
        self._checkpoint_control_headers: Dict[str, str] = {}
        self._control_timeout_s = 60.0
        self._gym_checkpoint_participants: tuple[GymDiscoveredParticipant, ...] = ()
        self._gym_checkpoint_topology: Optional[GymCheckpointTopology] = None
        self._active_gym_checkpoint_id: Optional[str] = None
        self._gym_execution_registry = GymActorExecutionRegistry()
        self._pad_dynamic_image_shapes = bool(cfg.get("pad_dynamic_image_shapes"))
        # Reconstruct the processor inside the actor (rather than serializing it
        # per rollout call) for full-trajectory multimodal postprocessing.
        self._processor: Optional[Any] = None
        tokenizer_config = cfg.get("tokenizer_config")
        if tokenizer_config:
            from nemo_rl.algorithms.utils import get_tokenizer

            self._processor = get_tokenizer(tokenizer_config, get_processor=True)
            # attach_image_model_inputs_to_message assumes a placeholder-style
            # processor (imgs_sizes / num_frames reconstruction + pad_to_max_shape
            # PackedTensor build). A non-placeholder VLM would silently produce
            # wrong multimodal tensors — fail at actor construction instead.
            assert uses_image_placeholder(self._processor), (
                "NemoGym multimodal path assumes a placeholder-style processor "
                "(see _PLACEHOLDER_STYLE_PROCESSOR_NAMES in nemo_rl/data/multimodal_utils.py); "
                f"got {type(self._processor).__name__}. Update "
                "attach_image_model_inputs_to_message before enabling."
            )

    def _require_spinup(self) -> None:
        """Raise a diagnosable error if this instance never ran :meth:`_spinup`."""
        if self.rh is None:
            raise RuntimeError(
                "NeMo-Gym actor has no running servers: _spinup() was never called on "
                "this instance. Ray recreates a restarted actor through __init__ only, "
                "so an actor that died and came back reaches this state and cannot "
                "serve rollouts until it is spun up again."
            )

    def health_check(self) -> None:
        """Raise if the Gym head server or any subprocess server has died.

        Thin wrapper over NeMo-Gym's own ``RunHelper.poll``, which is what ``gym env
        start`` calls every 60s from ``run_forever``. NeMo-RL only calls ``rh.start``,
        so without this the check Gym already implements never runs and a dead tool
        server surfaces as unexplained rollout timeouts instead of a named process.
        """
        self._require_spinup()
        self.rh.poll()

    def _spinup(self) -> None:
        """Start the NeMo-Gym head server and rollout collection helper.

        Deferred from __init__ so the actor can be created cheaply (and
        scheduled onto reserved nodes) and spun up explicitly once the vLLM
        server URLs are available, overlapping with vLLM model loading.
        """
        self.node_ip = _get_node_ip_local()
        _gym_port_low = self.cfg.get("port_range_low", DEFAULT_GYM_PORT_RANGE_LOW)
        _gym_port_high = self.cfg.get("port_range_high", DEFAULT_GYM_PORT_RANGE_HIGH)
        self.head_server_port = _get_free_port_local(_gym_port_low, _gym_port_high)

        from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
        from nemo_gym import global_config as gym_global_config
        from nemo_gym.rollout_collection import RolloutCollectionHelper
        from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME, BaseServerConfig
        from omegaconf import DictConfig

        RELATIVE_PATH = "nemo_rl/environments/nemo_gym.py"
        assert __file__.endswith(RELATIVE_PATH)

        # Make a shallow copy so that NeMo-RL-side keys we pop or add below
        # do not mutate the caller's config dict (config.env["nemo_gym"]).
        initial_global_config_dict = dict(
            self.cfg.get("initial_global_config_dict") or {}
        )
        self._stable_execution_identity_enabled = (
            getattr(gym_global_config, "ATTEMPT_INDEX_KEY_NAME", None)
            == _NG_ATTEMPT_INDEX_BODY_KEY
        )
        # Strip NeMo-RL-only training knobs that must not be forwarded to the
        # NeMo-Gym server (same pattern as the pops in run_grpo_nemo_gym.py).
        initial_global_config_dict.pop("effort_levels", None)
        initial_global_config_dict.pop("pad_dynamic_image_shapes", None)
        # Policy information
        initial_global_config_dict["policy_model_name"] = self.cfg["model_name"]
        initial_global_config_dict["policy_api_key"] = (
            "dummy_key"  # No key necessary for training.
        )
        initial_global_config_dict["policy_base_url"] = self.cfg["base_urls"]
        # In multinode runs, Gym-managed service configs must advertise a real node IP
        # rather than falling back to localhost, or remote workers will connect to
        # their own loopback interface instead of the actor-hosted service.
        initial_global_config_dict.setdefault("default_host", self.node_ip)

        _gym_port_low = self.cfg.get("port_range_low", DEFAULT_GYM_PORT_RANGE_LOW)
        _gym_port_high = self.cfg.get("port_range_high", DEFAULT_GYM_PORT_RANGE_HIGH)
        if (
            _gym_port_low < DEFAULT_GYM_PORT_RANGE_LOW
            or _gym_port_high > DEFAULT_GYM_PORT_RANGE_HIGH
        ):
            print(
                f"WARNING: Gym port range [{_gym_port_low}, {_gym_port_high}) is outside "
                f"the default [{DEFAULT_GYM_PORT_RANGE_LOW}, {DEFAULT_GYM_PORT_RANGE_HIGH}). "
                f"Check the port layout in virtual_cluster.py for conflicts."
            )
        initial_global_config_dict["port_range_low"] = _gym_port_low
        initial_global_config_dict["port_range_high"] = _gym_port_high

        initial_global_config_dict.setdefault(
            "global_aiohttp_connector_limit_per_host", 16_384
        )
        initial_global_config_dict.setdefault("global_aiohttp_connector_limit", 65_536)
        print(
            f"""Set global_aiohttp_connector_limit_per_host={initial_global_config_dict["global_aiohttp_connector_limit_per_host"]} and global_aiohttp_connector_limit={initial_global_config_dict["global_aiohttp_connector_limit"]}.
Depending on your data shape, you may want to change these values."""
        )

        # Get Ray head node address if Ray is initialized
        assert ray.is_initialized(), (
            "Ray must be initialized before using NeMo-Gym environment"
        )
        ray_context = ray.get_runtime_context()
        assert ray_context.gcs_address, "Ray must have a GCS address"

        initial_global_config_dict["ray_head_node_address"] = ray_context.gcs_address
        print(f"Ray head node address: {ray_context.gcs_address}")

        # Head server
        initial_global_config_dict[HEAD_SERVER_KEY_NAME] = {
            "host": "0.0.0.0",
            "port": self.head_server_port,
        }

        # Ledger-authoritative token capture: enable external staging in the
        # policy model server (via the policy_model global-config override
        # block the env yamls already use) and disable the legacy token echo.
        token_capture = self.cfg.get("token_capture") or None
        self._token_capture_enabled = bool(
            token_capture and token_capture.get("enabled")
        )
        self._server_client = None
        checkpoint_control_token = os.environ.get(_CHECKPOINT_CONTROL_ENV)
        self._token_capture_control_headers = {}
        self._checkpoint_control_headers = (
            {"Authorization": f"Bearer {checkpoint_control_token}"}
            if checkpoint_control_token
            else {}
        )
        self._control_timeout_s = 60.0
        self._gym_checkpoint_participants = ()
        self._gym_checkpoint_topology = None
        if self._token_capture_enabled:
            policy_overrides = (
                initial_global_config_dict.setdefault("policy_model", {})
                .setdefault("responses_api_models", {})
                .setdefault("vllm_model", {})
            )
            policy_overrides["return_token_id_information"] = False
            capture_dir = os.path.abspath(token_capture["capture_dir"])
            initial_global_config_dict["token_id_capture"] = {
                "enabled": True,
                "all_agents": True,
                "rebuild_response": False,
                "dir": capture_dir,
                # The lineage store is process-shared and doubles as the
                # per-rollout capture ledger. Every uvicorn worker builds its
                # own handle over the same root, so token-in ancestry remains
                # valid when consecutive calls land on different workers.
                "lineage_store": ("nemo_gym.token_id_capture.lineage:FileLineageStore"),
                "lineage_store_kwargs": {"root": os.path.join(capture_dir, "lineage")},
                "external_staging": True,
                "control_auth_token_env": _TOKEN_CAPTURE_CONTROL_ENV,
            }
            # Gym resolves the credential inside each serving process. Keep
            # only the variable name in serialized config and inherit the
            # secret through the server process environment.
            os.environ[_TOKEN_CAPTURE_CONTROL_ENV] = token_capture["control_auth_token"]
            self._token_capture_control_headers = {
                "Authorization": f"Bearer {token_capture['control_auth_token']}"
            }
            # Gym's checkpoint routes reuse the token-capture bearer only when
            # no dedicated checkpoint bearer is configured.
            if not checkpoint_control_token:
                self._checkpoint_control_headers = dict(
                    self._token_capture_control_headers
                )
            self._control_timeout_s = float(
                token_capture.get("control_timeout_s") or 60.0
            )

        self.rh = RunHelper()
        self.rh.start(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(
                dotenv_path=Path(__file__.removesuffix(RELATIVE_PATH)).absolute()
                / "nemo_gym_env.yaml",
                initial_global_config_dict=DictConfig(initial_global_config_dict),
                skip_load_from_cli=True,
            )
        )

        # Setup for rollout collection
        self.head_server_config = BaseServerConfig(
            host=self.node_ip,
            port=self.head_server_port,
        )
        self.rch = RolloutCollectionHelper()

    def set_tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Install the tokenizer run_rollouts postprocesses with.

        Called once per actor at spinup. It used to be a run_rollouts argument,
        which meant Ray deserialized a tokenizer per prompt on this actor's
        task-execution thread. That thread holds the GIL, so it blocked the
        actor's event loop and no rollout could issue its first HTTP request
        until its own copy finished loading.

        The cost is not marginal. A tokenizer of this shape measured 7.45 MB on
        the wire, 286 ms to serialize and 1052 ms to deserialize, so a
        SingleController recipe admitting ~1000 prompts per step spends roughly
        18 minutes deserializing a single admission burst, against a step that
        should take minutes. Runs on that shape stalled without completing a
        rollout.

        Measured on one CPU node with 1024 concurrent calls against one actor:
        153 of 1024 prompts finished in 300 s passing the tokenizer per call,
        versus all 1024 in 16 s holding it here. Passing an ObjectRef instead
        does not help -- Ray caches the object buffer, not the deserialized
        value, so it still pays per task.
        """
        self._tokenizer = tokenizer

    # ── Gym token-capture and checkpoint control planes ─────────────────────

    def _control_client(self):
        """Gym ServerClient resolving servers by name from the head server."""
        if self._server_client is None:
            from nemo_gym.server_utils import ServerClient

            self._server_client = ServerClient.load_from_global_config(
                self.head_server_config
            )
        return self._server_client

    async def _control(
        self,
        method: str,
        path: str,
        *,
        server_name: str = _POLICY_SERVER_NAME,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        default_headers = (
            self._checkpoint_control_headers
            if path.startswith(GYM_CHECKPOINT_CONTROL_PREFIX)
            else self._token_capture_control_headers
        )
        headers = {**kwargs.pop("headers", {}), **default_headers}
        request_timeout_s = self._control_timeout_s if timeout_s is None else timeout_s
        try:
            response = await asyncio.wait_for(
                self._control_client().request(
                    server_name=server_name,
                    url_path=path,
                    method=method,
                    headers=headers,
                    **kwargs,
                ),
                timeout=request_timeout_s,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Gym control call to {server_name!r} {method} {path} exceeded "
                f"{request_timeout_s}s (control plane unreachable or stalled)"
            ) from None
        if response.status != 200:
            response_text = await response.text()
            error_code: Optional[str] = None
            try:
                error_payload = json.loads(response_text)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(error_payload, dict):
                    error = error_payload.get("error")
                    if isinstance(error, dict) and isinstance(error.get("code"), str):
                        error_code = error["code"]
            raise GymControlRequestError(
                f"Gym control call to {server_name!r} {method} {path} failed: "
                f"HTTP {response.status} {response_text}",
                status=response.status,
                error_code=error_code,
            )
        payload = await response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Gym control call to {server_name!r} {method} {path} returned "
                f"{type(payload).__name__}, expected an object"
            )
        return payload

    @staticmethod
    def _checkpoint_request_timeout(deadline_ts: float) -> float:
        """Bound one control RPC by the transaction's absolute deadline."""
        return max(0.001, deadline_ts - time.time())

    async def discover_checkpoint_capabilities(
        self,
        server_names: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Discover and validate checkpoint capabilities without enabling saves.

        The result is cached for later explicit control calls on this actor.
        """
        self._require_spinup()
        candidates = (
            _checkpoint_server_names(self._control_client().global_config_dict)
            if server_names is None
            else sorted(set(server_names))
        )
        if not candidates:
            raise RuntimeError("NeMo-Gym exposes no checkpoint participants")

        participants: list[GymDiscoveredParticipant] = []
        seen: set[tuple[str, str]] = set()
        for server_name in candidates:
            raw = await self._control(
                "GET",
                GYM_CHECKPOINT_CAPABILITIES_PATH,
                server_name=server_name,
            )
            capabilities = GymControlCapabilities.model_validate(raw)
            if capabilities.component == "responses_api_models":
                if capabilities.instance_role is None:
                    raise RuntimeError(
                        f"Gym model participant {server_name!r} did not declare "
                        "its policy or auxiliary role"
                    )
            elif capabilities.instance_role is not None:
                raise RuntimeError(
                    f"non-model Gym participant {server_name!r} unexpectedly "
                    f"declared instance_role={capabilities.instance_role!r}"
                )
            if capabilities.multi_process.mode == "unmanaged":
                raise RuntimeError(
                    f"Gym checkpoint participant {server_name!r} runs "
                    f"{capabilities.multi_process.num_workers} unmanaged workers"
                )
            participant = capabilities.participant(server_name)
            identity_key = (participant.component, participant.participant_name)
            if identity_key in seen:
                raise RuntimeError(
                    f"duplicate Gym checkpoint participant identity {identity_key!r}"
                )
            seen.add(identity_key)
            participants.append(
                GymDiscoveredParticipant(
                    participant=participant,
                    capabilities=capabilities,
                )
            )
        if not any(
            item.capabilities.component == "responses_api_models"
            and item.capabilities.instance_role == "policy"
            for item in participants
        ):
            raise RuntimeError(
                "NeMo-Gym checkpoint topology has no policy model participant"
            )
        self._gym_checkpoint_participants = tuple(participants)
        self._gym_checkpoint_topology = GymCheckpointTopology.from_discovered(
            participants
        )
        return self._gym_checkpoint_topology.model_dump(mode="json")

    def _checkpoint_participants(self) -> tuple[GymDiscoveredParticipant, ...]:
        if not self._gym_checkpoint_participants:
            raise RuntimeError(
                "discover_checkpoint_capabilities must succeed before Gym "
                "checkpoint control calls"
            )
        return self._gym_checkpoint_participants

    def _agent_checkpoint_participant(
        self,
        agent_name: str,
    ) -> GymDiscoveredParticipant:
        matches = [
            item
            for item in self._checkpoint_participants()
            if item.participant.component == "responses_api_agents"
            and item.participant.participant_name == agent_name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "resolved Gym agent must match exactly one checkpoint "
                f"participant: agent={agent_name!r}, matches={len(matches)}"
            )
        participant = matches[0]
        if "completed_result_acknowledgement" not in participant.capabilities.features:
            raise RuntimeError(
                f"Gym agent {agent_name!r} does not advertise completed-result "
                "acknowledgement"
            )
        return participant

    async def _completion_receipt_for(
        self,
        execution: GymExecutionIdentity,
        *,
        agent_name: str,
    ) -> GymCompletionReceipt:
        """Fetch the exact Gym-issued receipt before publishing a completion."""
        discovered = self._agent_checkpoint_participant(agent_name)
        return GymCompletionReceipt.model_validate(
            await self._control(
                "GET",
                GYM_AGENT_COMPLETION_RECEIPT_PATH,
                server_name=discovered.participant.server_name,
                params={
                    "rollout_id": execution.rollout_id,
                    "attempt_index": execution.attempt_index,
                },
            )
        )

    async def acknowledge_completed_executions(
        self,
        executions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Release terminal results after RL records checkpointable ownership."""
        validated = [
            GymCompletedExecution.model_validate(execution) for execution in executions
        ]
        if not validated:
            return {"acknowledged": []}
        keys = [
            (
                item.agent_name,
                item.receipt.rollout_id,
                item.receipt.attempt_index,
            )
            for item in validated
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("completed Gym execution acknowledgements must be unique")

        by_agent: dict[str, list[GymCompletionReceipt]] = {}
        for item in validated:
            by_agent.setdefault(item.agent_name, []).append(item.receipt)

        acknowledged: list[GymCompletedExecution] = []
        for agent_name, receipts in sorted(by_agent.items()):
            discovered = self._agent_checkpoint_participant(agent_name)
            request = GymCompletedExecutionAcknowledgementRequest(executions=receipts)
            response = GymCompletedExecutionAcknowledgementResponse.model_validate(
                await self._control(
                    "POST",
                    GYM_AGENT_COMPLETION_ACK_PATH,
                    server_name=discovered.participant.server_name,
                    json=request.model_dump(mode="json"),
                )
            )
            requested_keys = {
                (receipt.rollout_id, receipt.attempt_index) for receipt in receipts
            }
            requested_receipts = {
                (receipt.rollout_id, receipt.attempt_index): receipt
                for receipt in receipts
            }
            response_receipts = {
                (receipt.rollout_id, receipt.attempt_index): receipt
                for receipt in response.acknowledged
            }
            if response_receipts != requested_receipts:
                raise RuntimeError(
                    "Gym completed-result acknowledgement did not cover the "
                    f"requested executions for agent {agent_name!r}: "
                    f"requested={sorted(requested_keys)!r}, "
                    f"acknowledged={sorted(response_receipts)!r}"
                )
            acknowledged.extend(
                GymCompletedExecution(receipt=receipt, agent_name=agent_name)
                for receipt in response.acknowledged
            )
        return {"acknowledged": [item.model_dump(mode="json") for item in acknowledged]}

    def _ordered_checkpoint_participants(
        self,
        *,
        resume: bool = False,
    ) -> tuple[GymDiscoveredParticipant, ...]:
        """Order policy fencing first and policy reopening last."""
        component_order = {
            "responses_api_models": 0,
            "responses_api_agents": 1,
            "resources_servers": 2,
        }
        return tuple(
            sorted(
                self._checkpoint_participants(),
                key=lambda item: component_order[item.participant.component],
                reverse=resume,
            )
        )

    @staticmethod
    def _participant_checkpoint_path(
        participant: GymParticipantIdentity,
    ) -> str:
        if participant.component == "responses_api_models":
            safe_name = GymExecutionIdentity(
                rollout_id=participant.participant_name,
                attempt_index=0,
            ).rollout_id
            return f"model-ledger/{safe_name}/manifest.json"
        if participant.component == "resources_servers":
            safe_name = GymExecutionIdentity(
                rollout_id=participant.participant_name,
                attempt_index=0,
            ).rollout_id
            return f"resources/{safe_name}/manifest.json"
        name_digest = hashlib.sha256(
            participant.participant_name.encode("utf-8")
        ).hexdigest()
        return f"agent/instance-{name_digest}/manifest.json"

    async def prepare_checkpoint(
        self,
        checkpoint_id: str,
        deadline_ts: float,
    ) -> dict[str, Any]:
        """Prepare every participant or fail closed and reopen live work.

        Gym's agent and resources prepare routes wait for their safe boundary.
        Policy admission pause is intentionally non-blocking, so a draining
        policy server is observed through its status route until the common
        deadline. No incomplete cut is returned to a checkpoint coordinator.
        """
        if self._active_gym_checkpoint_id not in (None, checkpoint_id):
            raise RuntimeError(
                f"Gym checkpoint {self._active_gym_checkpoint_id!r} is already active"
            )
        self._gym_execution_registry.freeze(checkpoint_id)
        self._active_gym_checkpoint_id = checkpoint_id
        request = GymCheckpointControlRequest(
            checkpoint_id=checkpoint_id,
            deadline_ts=deadline_ts,
        ).model_dump(mode="json")
        results: list[GymParticipantPrepareResult] = []
        touched: list[GymDiscoveredParticipant] = []
        try:
            for discovered in self._ordered_checkpoint_participants():
                participant = discovered.participant
                capabilities = discovered.capabilities
                payload: (
                    GymModelPrepareResponse
                    | GymAgentPrepareResponse
                    | GymResourcesPrepareResponse
                )
                if participant.component == "responses_api_models":
                    if capabilities.instance_role != "policy":
                        continue
                    if "paused" not in capabilities.admission_states:
                        raise RuntimeError(
                            f"Gym policy model {participant.server_name!r} cannot pause"
                        )
                    touched.append(discovered)
                    payload = GymModelPrepareResponse.model_validate(
                        await self._control(
                            "POST",
                            f"{GYM_MODEL_ADMISSION_PREFIX}/pause",
                            server_name=participant.server_name,
                            timeout_s=self._checkpoint_request_timeout(deadline_ts),
                            json=request,
                        )
                    )
                    ready = (
                        payload.state == "paused"
                        and payload.inflight_total == 0
                        and payload.workers.acknowledged == payload.workers.expected
                    )
                elif participant.component == "responses_api_agents":
                    if capabilities.checkpoint_mode != "export_restore":
                        continue
                    touched.append(discovered)
                    payload = await self._prepare_agent_checkpoint(
                        discovered,
                        request=request,
                        deadline_ts=deadline_ts,
                    )
                    ready = payload.ready_to_commit
                else:
                    if capabilities.checkpoint_mode != "export_restore":
                        continue
                    touched.append(discovered)
                    payload = GymResourcesPrepareResponse.model_validate(
                        await self._control(
                            "POST",
                            f"{GYM_RESOURCES_CHECKPOINT_PREFIX}/prepare",
                            server_name=participant.server_name,
                            timeout_s=self._checkpoint_request_timeout(deadline_ts),
                            json=request,
                        )
                    )
                    ready = payload.state == "prepared"
                results.append(
                    GymParticipantPrepareResult(
                        participant=participant,
                        ready=ready,
                        payload=payload,
                    )
                )

            for index, result in enumerate(results):
                if (
                    result.ready
                    or result.participant.component != "responses_api_models"
                ):
                    continue
                discovered = next(
                    item for item in touched if item.participant == result.participant
                )
                payload = await self._wait_for_policy_model_pause(
                    discovered,
                    checkpoint_id=checkpoint_id,
                    deadline_ts=deadline_ts,
                )
                results[index] = GymParticipantPrepareResult(
                    participant=result.participant,
                    ready=(payload.state == "paused" and payload.inflight_total == 0),
                    payload=payload,
                )

            if not all(item.ready for item in results):
                raise TimeoutError(
                    f"Gym checkpoint {checkpoint_id!r} did not prepare before its deadline"
                )
        except (Exception, asyncio.CancelledError) as prepare_error:
            abort_deadline_ts = max(
                deadline_ts,
                time.time() + self._control_timeout_s,
            )
            try:
                await self._resume_checkpoint_participants(
                    checkpoint_id,
                    abort_deadline_ts,
                    tuple(touched),
                )
            except (Exception, asyncio.CancelledError) as abort_error:
                raise BaseExceptionGroup(
                    "Gym checkpoint prepare failed and participant resume also failed",
                    [prepare_error, abort_error],
                ) from prepare_error
            self._gym_execution_registry.unfreeze(checkpoint_id)
            if time.time() >= deadline_ts and not isinstance(
                prepare_error, asyncio.CancelledError
            ):
                self._active_gym_checkpoint_id = None
                raise TimeoutError(
                    f"Gym checkpoint {checkpoint_id!r} exceeded its prepare deadline; "
                    "participants were resumed and the previous checkpoint remains valid"
                ) from prepare_error
            self._active_gym_checkpoint_id = None
            raise

        return GymCheckpointPrepareResult(
            checkpoint_id=checkpoint_id,
            ready=True,
            participants=results,
        ).model_dump(mode="json")

    async def _prepare_agent_checkpoint(
        self,
        discovered: GymDiscoveredParticipant,
        *,
        request: dict[str, Any],
        deadline_ts: float,
    ) -> GymAgentPrepareResponse:
        """Retry Gym's non-ready 409 until terminal results are acknowledged."""
        participant = discovered.participant
        while True:
            remaining = deadline_ts - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Gym agent {participant.server_name!r} missed the prepare deadline"
                )
            try:
                raw = await self._control(
                    "POST",
                    f"{GYM_AGENT_CHECKPOINT_PREFIX}/prepare",
                    server_name=participant.server_name,
                    timeout_s=remaining,
                    json=request,
                )
            except GymControlRequestError as error:
                if not (
                    error.status == 409
                    and error.error_code == "agent_prepare_incomplete"
                ):
                    raise
                await asyncio.sleep(min(0.1, max(0.0, deadline_ts - time.time())))
                continue
            payload = GymAgentPrepareResponse.model_validate(raw)
            if not payload.ready_to_commit:
                raise RuntimeError(
                    f"Gym agent {participant.server_name!r} returned HTTP 200 "
                    "without a complete checkpoint boundary"
                )
            return payload

    async def _wait_for_policy_model_pause(
        self,
        discovered: GymDiscoveredParticipant,
        *,
        checkpoint_id: str,
        deadline_ts: float,
    ) -> GymModelPrepareResponse:
        """Long-poll one policy model until its accepted calls have drained."""
        participant = discovered.participant
        remaining = deadline_ts - time.time()
        if remaining <= 0:
            raise TimeoutError(
                f"Gym policy model {participant.server_name!r} missed the prepare deadline"
            )
        raw = await self._control(
            "GET",
            f"{GYM_MODEL_ADMISSION_PREFIX}/status",
            server_name=participant.server_name,
            timeout_s=remaining,
            params={
                "checkpoint_id": checkpoint_id,
                "deadline_ts": deadline_ts,
                "wait_state": "paused",
                "timeout_s": remaining,
            },
        )
        if discovered.capabilities.multi_process.mode == "coordinator":
            status = GymCoordinatorModelStatusResponse.model_validate(raw)
            workers = {
                "acknowledged": status.workers.acknowledged,
                "expected": status.workers.expected,
            }
            ready = (
                status.state == "paused"
                and status.inflight_total == 0
                and status.missing_workers == 0
                and status.workers.acknowledged == status.workers.expected
            )
        else:
            status = GymSingleWorkerModelStatusResponse.model_validate(raw)
            workers = {
                "acknowledged": len(status.per_worker),
                "expected": discovered.capabilities.multi_process.num_workers,
            }
            ready = status.state == "paused" and status.inflight_total == 0
        payload = GymModelPrepareResponse(
            state=status.state,
            workers=workers,
            inflight_total=status.inflight_total,
            waiters_total=status.waiters_total,
        )
        if not ready:
            raise TimeoutError(
                f"Gym policy model {participant.server_name!r} remained "
                f"{payload.state!r} with {payload.inflight_total} in-flight request(s)"
            )
        return payload

    async def commit_checkpoint(
        self,
        checkpoint_id: str,
        deadline_ts: float,
        checkpoint_dir: str,
    ) -> dict[str, Any]:
        """Commit every stateful participant into a caller-owned temp directory."""
        common_request = GymCheckpointDirectoryRequest(
            checkpoint_id=checkpoint_id,
            deadline_ts=deadline_ts,
            checkpoint_dir=checkpoint_dir,
        ).model_dump(mode="json")
        results: list[GymParticipantCommitResult] = []
        continuation_indexes: list[GymCheckpointArtifactReference] = []
        component_order = {
            "responses_api_agents": 0,
            "responses_api_models": 1,
            "resources_servers": 2,
        }
        participants = sorted(
            self._checkpoint_participants(),
            key=lambda item: component_order[item.participant.component],
        )
        for discovered in participants:
            participant = discovered.participant
            capabilities = discovered.capabilities
            if capabilities.checkpoint_mode != "export_restore":
                continue
            payload: (
                GymModelCommitResponse
                | GymAgentCommitResponse
                | GymResourcesCommitResponse
            )
            if participant.component == "responses_api_models":
                if capabilities.instance_role != "policy":
                    continue
                if (
                    GYM_EXTERNAL_STORAGE_REFERENCE_INDEX_FEATURE
                    not in capabilities.features
                ):
                    raise RuntimeError(
                        "Gym policy model does not support external-storage "
                        "reference indexes"
                    )
                request = GymModelCheckpointCommitRequest(
                    checkpoint_id=checkpoint_id,
                    deadline_ts=deadline_ts,
                    checkpoint_dir=checkpoint_dir,
                    continuation_indexes=continuation_indexes,
                ).model_dump(mode="json")
                payload = GymModelCommitResponse.model_validate(
                    await self._control(
                        "POST",
                        f"{GYM_MODEL_CHECKPOINT_PREFIX}/commit",
                        server_name=participant.server_name,
                        timeout_s=self._checkpoint_request_timeout(deadline_ts),
                        json=request,
                    )
                )
            elif participant.component == "responses_api_agents":
                if GYM_AGENT_CONTINUATION_INDEX_FEATURE not in capabilities.features:
                    raise RuntimeError(
                        "Gym agent does not support continuation indexes"
                    )
                request = GymAgentCheckpointDirectoryRequest(
                    checkpoint_id=checkpoint_id,
                    deadline_ts=deadline_ts,
                    checkpoint_dir=checkpoint_dir,
                ).model_dump(mode="json")
                payload = GymAgentCommitResponse.model_validate(
                    await self._control(
                        "POST",
                        f"{GYM_AGENT_CHECKPOINT_PREFIX}/commit",
                        server_name=participant.server_name,
                        timeout_s=self._checkpoint_request_timeout(deadline_ts),
                        json=request,
                    )
                )
                continuation_indexes.append(payload.continuation_index)
            else:
                request = common_request
                payload = GymResourcesCommitResponse.model_validate(
                    await self._control(
                        "POST",
                        f"{GYM_RESOURCES_CHECKPOINT_PREFIX}/commit",
                        server_name=participant.server_name,
                        timeout_s=self._checkpoint_request_timeout(deadline_ts),
                        json=request,
                    )
                )
            manifest = GymParticipantManifestReference(
                participant=participant,
                relative_path=self._participant_checkpoint_path(participant),
                manifest_digest=payload.manifest_digest,
            )
            results.append(
                GymParticipantCommitResult(
                    participant=participant,
                    payload=payload,
                    manifest=manifest,
                )
            )
        return GymCheckpointCommitResult(
            checkpoint_id=checkpoint_id,
            participants=results,
        ).model_dump(mode="json")

    async def restore_checkpoint(
        self,
        checkpoint_id: str,
        deadline_ts: float,
        checkpoint_dir: str,
        source_checkpoint_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Restore every stateful participant but leave admission paused."""
        if self._active_gym_checkpoint_id not in (None, checkpoint_id):
            raise RuntimeError(
                f"Gym checkpoint {self._active_gym_checkpoint_id!r} is already active"
            )
        self._gym_execution_registry.freeze(checkpoint_id)
        self._active_gym_checkpoint_id = checkpoint_id
        common_request = GymCheckpointDirectoryRequest(
            checkpoint_id=checkpoint_id,
            deadline_ts=deadline_ts,
            checkpoint_dir=checkpoint_dir,
        ).model_dump(mode="json")
        results: list[GymParticipantRestoreResult] = []
        for discovered in self._ordered_checkpoint_participants():
            participant = discovered.participant
            capabilities = discovered.capabilities
            if capabilities.checkpoint_mode != "export_restore":
                continue
            payload: (
                GymModelRestoreResponse
                | GymAgentRestoreResponse
                | GymResourcesRestoreResponse
            )
            if participant.component == "responses_api_models":
                if capabilities.instance_role != "policy":
                    continue
                if (
                    GYM_EXTERNAL_STORAGE_REFERENCE_INDEX_FEATURE
                    not in capabilities.features
                ):
                    raise RuntimeError(
                        "Gym policy model does not support external-storage "
                        "reference indexes"
                    )
                request = GymModelCheckpointRestoreRequest(
                    checkpoint_id=checkpoint_id,
                    deadline_ts=deadline_ts,
                    checkpoint_dir=checkpoint_dir,
                ).model_dump(mode="json")
                payload = GymModelRestoreResponse.model_validate(
                    await self._control(
                        "POST",
                        f"{GYM_MODEL_CHECKPOINT_PREFIX}/restore",
                        server_name=participant.server_name,
                        timeout_s=self._checkpoint_request_timeout(deadline_ts),
                        json=request,
                    )
                )
            elif participant.component == "responses_api_agents":
                if GYM_AGENT_CONTINUATION_INDEX_FEATURE not in capabilities.features:
                    raise RuntimeError(
                        "Gym agent does not support continuation indexes"
                    )
                request = GymAgentCheckpointDirectoryRequest(
                    checkpoint_id=checkpoint_id,
                    deadline_ts=deadline_ts,
                    checkpoint_dir=checkpoint_dir,
                ).model_dump(mode="json")
                payload = GymAgentRestoreResponse.model_validate(
                    await self._control(
                        "POST",
                        f"{GYM_AGENT_CHECKPOINT_PREFIX}/restore",
                        server_name=participant.server_name,
                        timeout_s=self._checkpoint_request_timeout(deadline_ts),
                        json=request,
                    )
                )
            else:
                request = common_request
                payload = GymResourcesRestoreResponse.model_validate(
                    await self._control(
                        "POST",
                        f"{GYM_RESOURCES_CHECKPOINT_PREFIX}/restore",
                        server_name=participant.server_name,
                        timeout_s=self._checkpoint_request_timeout(deadline_ts),
                        json=request,
                    )
                )
            results.append(
                GymParticipantRestoreResult(
                    participant=participant,
                    payload=payload,
                )
            )
            if source_checkpoint_id is not None:
                actual_source_checkpoint_id = (
                    payload.checkpoint_id
                    if isinstance(payload, GymModelRestoreResponse)
                    else payload.source_checkpoint_id
                )
                if actual_source_checkpoint_id != source_checkpoint_id:
                    raise RuntimeError(
                        "Gym participant restored the wrong source checkpoint: "
                        f"participant={participant.participant_name!r}, "
                        f"expected={source_checkpoint_id!r}, "
                        f"actual={actual_source_checkpoint_id!r}"
                    )
        return GymCheckpointRestoreResult(
            checkpoint_id=checkpoint_id,
            participants=results,
        ).model_dump(mode="json")

    async def discard_restored_agent_continuations(
        self,
        checkpoint_id: str,
        deadline_ts: float,
        executions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Prevent selected replacement attempts from consuming saved turns."""
        if self._active_gym_checkpoint_id != checkpoint_id:
            raise RuntimeError(
                f"Gym checkpoint {checkpoint_id!r} is not the active restore"
            )
        identities = [GymExecutionIdentity.model_validate(item) for item in executions]
        keys = [(item.rollout_id, item.attempt_index) for item in identities]
        if len(keys) != len(set(keys)):
            raise ValueError("restored Gym continuations to retire must be unique")

        agent_participants = [
            discovered
            for discovered in self._checkpoint_participants()
            if discovered.participant.component == "responses_api_agents"
            and discovered.capabilities.checkpoint_mode == "export_restore"
        ]
        discarded = 0
        for discovered in agent_participants:
            for identity in identities:
                request = GymAgentDiscardRestoredContinuationRequest(
                    checkpoint_id=checkpoint_id,
                    deadline_ts=deadline_ts,
                    rollout_id=identity.rollout_id,
                    attempt_index=identity.attempt_index,
                ).model_dump(mode="json")
                response = GymAgentDiscardRestoredContinuationResponse.model_validate(
                    await self._control(
                        "POST",
                        GYM_AGENT_DISCARD_RESTORED_CONTINUATION_PATH,
                        server_name=discovered.participant.server_name,
                        timeout_s=self._checkpoint_request_timeout(deadline_ts),
                        json=request,
                    )
                )
                discarded += int(response.discarded)
        return {
            "executions": len(identities),
            "agent_participants": len(agent_participants),
            "discarded": discarded,
        }

    async def resume_checkpoint(
        self,
        checkpoint_id: str,
        deadline_ts: float,
    ) -> dict[str, Any]:
        """Resume participants after commit, restore, or an aborted prepare."""
        results = await self._resume_checkpoint_participants(
            checkpoint_id,
            deadline_ts,
            self._checkpoint_participants(),
        )
        self._gym_execution_registry.unfreeze(checkpoint_id)
        self._active_gym_checkpoint_id = None
        return GymCheckpointResumeResult(
            checkpoint_id=checkpoint_id,
            participants=results,
        ).model_dump(mode="json")

    async def _resume_checkpoint_participants(
        self,
        checkpoint_id: str,
        deadline_ts: float,
        participants: tuple[GymDiscoveredParticipant, ...],
    ) -> list[GymParticipantResumeResult]:
        """Resume a prepared subset, reopening policy admission last."""
        request = GymCheckpointControlRequest(
            checkpoint_id=checkpoint_id,
            deadline_ts=deadline_ts,
        ).model_dump(mode="json")
        results: list[GymParticipantResumeResult] = []
        component_order = {
            "responses_api_models": 0,
            "responses_api_agents": 1,
            "resources_servers": 2,
        }
        ordered = sorted(
            participants,
            key=lambda item: component_order[item.participant.component],
            reverse=True,
        )
        errors: list[BaseException] = []
        for discovered in ordered:
            try:
                result = await self._resume_checkpoint_participant(
                    discovered,
                    request=request,
                    deadline_ts=deadline_ts,
                )
            except (Exception, asyncio.CancelledError) as error:
                errors.append(error)
                continue
            if result is not None:
                results.append(result)
        if errors:
            raise BaseExceptionGroup(
                "one or more Gym checkpoint participants failed to resume",
                errors,
            )
        return results

    async def _resume_checkpoint_participant(
        self,
        discovered: GymDiscoveredParticipant,
        *,
        request: dict[str, Any],
        deadline_ts: float,
    ) -> Optional[GymParticipantResumeResult]:
        """Resume one participant, returning None when it does not participate."""
        participant = discovered.participant
        capabilities = discovered.capabilities
        payload: (
            GymModelResumeResponse | GymAgentResumeResponse | GymResourcesResumeResponse
        )
        if participant.component == "responses_api_models":
            if capabilities.instance_role != "policy":
                return None
            payload = GymModelResumeResponse.model_validate(
                await self._control(
                    "POST",
                    f"{GYM_MODEL_ADMISSION_PREFIX}/resume",
                    server_name=participant.server_name,
                    timeout_s=self._checkpoint_request_timeout(deadline_ts),
                    json=request,
                )
            )
        elif participant.component == "responses_api_agents":
            if capabilities.checkpoint_mode != "export_restore":
                return None
            payload = GymAgentResumeResponse.model_validate(
                await self._control(
                    "POST",
                    f"{GYM_AGENT_CHECKPOINT_PREFIX}/resume",
                    server_name=participant.server_name,
                    timeout_s=self._checkpoint_request_timeout(deadline_ts),
                    json=request,
                )
            )
        else:
            if capabilities.checkpoint_mode != "export_restore":
                return None
            payload = GymResourcesResumeResponse.model_validate(
                await self._control(
                    "POST",
                    f"{GYM_RESOURCES_CHECKPOINT_PREFIX}/resume",
                    server_name=participant.server_name,
                    timeout_s=self._checkpoint_request_timeout(deadline_ts),
                    json=request,
                )
            )
        return GymParticipantResumeResult(
            participant=participant,
            payload=payload,
        )

    async def abort_checkpoint(
        self,
        checkpoint_id: str,
        deadline_ts: float,
    ) -> dict[str, Any]:
        """Abort an unpublished checkpoint by reopening every participant.

        Gym's current wire protocol uses the same resume routes for successful
        completion and abort; it does not expose a separate abort endpoint.
        """
        return await self.resume_checkpoint(checkpoint_id, deadline_ts)

    async def run_rollouts(
        self,
        nemo_gym_examples: list[dict],
        timer_prefix: str,
        deduplicate_multimodal_data: bool = False,
    ) -> AsyncGenerator[tuple[int, dict, dict, dict | None], None]:
        """Stream postprocessed rollouts as NeMo-Gym tasks complete."""
        self._require_spinup()
        if not nemo_gym_examples:
            raise ValueError("NeMo-Gym rollout batch must not be empty")
        if self._tokenizer is None:
            raise RuntimeError(
                "NemoGym.set_tokenizer must be called before run_rollouts"
            )
        tokenizer = self._tokenizer

        from nemo_rl.utils.fastokens import maybe_patch_fastokens

        maybe_patch_fastokens(bool(self.cfg.get("use_fastokens")))

        if not self._stable_execution_identity_enabled:
            # Compatibility with Gym revisions predating the stable execution
            # identity contract. They understand only one physical rollout ID,
            # so qualify it here exactly as the newer Gym middleware would.
            for row in nemo_gym_examples:
                _adapt_execution_identity_for_installed_gym(
                    row,
                    stable_execution_identity_enabled=False,
                )

        registered_executions: list[GymExecutionIdentity] = []
        if self._gym_checkpoint_participants:
            try:
                for row in nemo_gym_examples:
                    execution = GymExecutionIdentity(
                        rollout_id=row[_NG_ROLLOUT_ID_BODY_KEY],
                        attempt_index=row[_NG_ATTEMPT_INDEX_BODY_KEY],
                    )
                    self._gym_execution_registry.register(execution)
                    registered_executions.append(execution)
            except (KeyError, TypeError, ValueError, RuntimeError):
                for execution in registered_executions:
                    self._gym_execution_registry.release(execution)
                raise

        # Normalize local media before shipping requests to vLLM. Helper is a no-op
        # for text-only rows and already-qualified URLs.
        # Megatron's HTTP backend consumes the same normalized Responses payload.
        normalize_media_in_examples(nemo_gym_examples)

        timer = Timer()
        timer.start("_run_rollouts_total")
        nemo_gym_result_iterator = self.rch.run_examples(
            examples=nemo_gym_examples, head_server_config=self.head_server_config
        )
        # Gym resolves task_source to agent_ref synchronously in run_examples().
        # Build the counter afterward so completion rows use the resolved identity.
        counts_left = Counter(row["agent_ref"]["name"] for row in nemo_gym_examples)

        num_results = 0
        for task in nemo_gym_result_iterator:
            with timer.time(label=f"{timer_prefix}/await_results"):
                try:
                    nemo_gym_row, nemo_gym_result = await task
                except Exception as error:
                    for execution in registered_executions:
                        self._gym_execution_registry.release(execution)
                    if hasattr(error, "response_content"):
                        print(
                            "EXCEPTION RESULT",
                            error.response_content,
                            file=sys.stderr,
                        )
                    typed = _typed_gym_failure(error)
                    if typed is not None:
                        # `from None`, deliberately: chaining the original would put the
                        # unpicklable exception back on the wire as __cause__ and undo
                        # the whole point. The status and message are already in `detail`.
                        raise typed from None
                    raise

            execution: GymExecutionIdentity | None = None
            if self._gym_checkpoint_participants:
                execution = GymExecutionIdentity(
                    rollout_id=nemo_gym_row[_NG_ROLLOUT_ID_BODY_KEY],
                    attempt_index=nemo_gym_row[_NG_ATTEMPT_INDEX_BODY_KEY],
                )
                self._gym_execution_registry.mark_terminal(execution)

            with timer.time(label=f"{timer_prefix}/postprocess_results"):
                if self._token_capture_enabled:
                    # Receipt mode: fetch the ledger manifest and assemble the
                    # receipt locally; token-free result. The canonical row is
                    # rebuilt by the finalizer, so no message_log walk (and no
                    # NaN check) applies here.
                    nemo_rl_result = await self._postprocess_receipt_mode(
                        nemo_gym_row, nemo_gym_result
                    )
                else:
                    nemo_rl_result = self._postprocess_nemo_gym_to_nemo_rl_result(
                        nemo_gym_row,
                        nemo_gym_result,
                        tokenizer,
                        include_initial_multimodal_data=not deduplicate_multimodal_data,
                    )
                    if _has_nan_generation_logprobs(nemo_rl_result):
                        raise RuntimeError("Generation logprobs contain NaN")
                if self._gym_checkpoint_participants:
                    assert execution is not None
                    completion_receipt = await self._completion_receipt_for(
                        execution,
                        agent_name=nemo_gym_row["agent_ref"]["name"],
                    )
                    nemo_rl_result["gym_completion_receipt"] = (
                        completion_receipt.model_dump(mode="json")
                    )
            num_results += 1
            timing_metrics = None
            if num_results == len(nemo_gym_examples):
                timer.stop("_run_rollouts_total")
                timing_metrics = timer.get_timing_metrics("sum")
                total_time = timing_metrics.pop("_run_rollouts_total")
                timing_metrics[f"{timer_prefix}/postprocess_results_pct"] = (
                    100
                    * timing_metrics[f"{timer_prefix}/postprocess_results"]
                    / total_time
                )

            agent_name = nemo_gym_row["agent_ref"]["name"]
            counts_left[agent_name] -= 1
            if counts_left[agent_name] <= 0:
                counts_left.pop(agent_name)
            if num_results % 10 == 0 and counts_left:
                top_left = counts_left.most_common(5)
                top_left_str = "\n".join(
                    f"{index + 1}. {name}: {count}"
                    for index, (name, count) in enumerate(top_left)
                )
                print(
                    "Top 5 NeMo Gym agent refs left in this rollout batch: "
                    f"{top_left_str}",
                    file=sys.stderr,
                )

            # task_source is resolved to agent_ref inside this Ray actor, after
            # the caller's row was serialized. Return the resolved ref explicitly
            # so the caller can hydrate its own row copy before postprocessing.
            try:
                yield (
                    nemo_gym_row["_rowidx"],
                    nemo_gym_row["agent_ref"],
                    nemo_rl_result,
                    timing_metrics,
                )
            finally:
                if execution is not None:
                    self._gym_execution_registry.release(execution)

    async def _postprocess_receipt_mode(
        self, nemo_gym_row: dict, nemo_gym_result: dict
    ) -> dict:
        """Fetch the ledger manifest and assemble the receipt locally.

        The legacy token walk (and its contiguity assert) does not run: the
        capture ledger owns lineage, output items carry no token arrays, and
        the canonical row is rebuilt by the finalizer from staged deltas. The
        Ray return carries only the receipt (~100 B/call) beside the
        agent-level result.
        """
        assert isinstance(nemo_gym_result, dict), (
            f"Hit a non-successful response when querying NeMo Gym for rollouts: {nemo_gym_result}"
        )
        logical_rollout_id = nemo_gym_row[_NG_ROLLOUT_ID_BODY_KEY]
        attempt_index = nemo_gym_row.get(_NG_ATTEMPT_INDEX_BODY_KEY, 0)
        execution = GymExecutionIdentity(
            rollout_id=logical_rollout_id,
            attempt_index=attempt_index,
        )
        rollout_id = execution.capture_key
        # Gym's TERMINAL_RESPONSE_ID_KEY: the served response envelope id the
        # harness kept (``response.id``), not the logical-request header.
        terminal_response_id = nemo_gym_result.get("terminal_response_id")
        if not (isinstance(terminal_response_id, str) and terminal_response_id):
            # A harness that reports no terminal id still gets its manifest
            # fetched; receipt assembly attributes the terminal from the
            # scored response, falling back to heuristic selection.
            terminal_response_id = None
        scored_response = nemo_gym_result.get("response")
        if not isinstance(scored_response, dict):
            scored_response = None
        receipt = None
        try:
            manifest = await self._control(
                "GET",
                f"{_TOKEN_CAPTURE_CONTROL_PREFIX}/rollouts/{rollout_id}/manifest",
            )
            receipt = self._assemble_receipt(
                rollout_id,
                manifest,
                terminal_response_id=terminal_response_id,
                scored_response=scored_response,
                reward=float(nemo_gym_result.get("reward") or 0.0),
            )
        except (RuntimeError, OSError) as error:
            # An unfetchable manifest finalizes as a placeholder row.
            print(f"manifest({rollout_id}) fetch failed: {error}", flush=True)
        return {
            "message_log": [],
            "input_message_log": [],
            "full_result": nemo_gym_result,
            "rollout_id": rollout_id,
            "receipt": receipt,
        }

    @staticmethod
    def _assemble_receipt(
        rollout_id: str,
        manifest: dict,
        *,
        terminal_response_id: Optional[str],
        scored_response: Optional[dict] = None,
        reward: float,
    ) -> dict:
        """Build the token-free RolloutReceipt payload from a ledger manifest.

        ``terminal_response_id`` is the served response envelope id the
        harness reports for the completion it kept (Gym's
        ``terminal_response_id`` result key), forwarded to ``resolve_terminal``
        as ``declared_response_id``. It is not the logical-request header.

        Terminal selection is staged, fail-closed at every stage:

        1. Witness attribution (``resolve_terminal``): the declared response
           id (``terminal_response_id``), the scored response's
           envelope id, and its content fingerprints each independently name
           a manifest row through ``CallRecord.response_id`` and the recorded
           fingerprints. Agreeing witnesses attribute; a declared id that
           matches no row masks and never falls back; disagreeing witnesses
           attribute nothing.
        2. Heuristic fallback (``select_terminal_call``): with no witness,
           the manifest's explicit parent links infer the terminal
           (fail-closed: ambiguity masks).

        The receipt records the resolving stage in ``terminal_selection``
        (``declared``/``response_id``/``content``/``heuristic`` — failed
        selections stamp the last stage attempted) and the witness trail in
        ``terminal_attribution_reason``. Retry duplicates are dead-branch
        rows: they stay in the manifest (their staged rows are fetched,
        verified, and cleaned) but never join the terminal chain —
        ``verify_and_linearize`` tolerates rows unreferenced by the terminal
        chain.

        Poisoning is fail-closed with one carve-out. A failure row whose
        reason is ``request_finished_without_staged_coordinates`` is a call
        that never returned a completion (the ledger commit precedes the
        response leaving the server) and can never be a lineage parent (an
        uncommitted call has no row to resolve against) — e.g. the doomed
        final call of a rollout that exhausted the model's context window.
        Such rows are structurally off-chain and do not poison; if the
        *terminal* request itself died this way, the missing-terminal-row
        check below still masks the rollout. Every other failure reason
        (``worker_capture_failed``, ``invalid_worker_commit_coordinates``)
        marks a call whose completion WAS served — a hole in the chain —
        and poisons.
        """
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture import UNCOMMITTED_CALL_REASON
        from nemo_gym.token_id_capture.staging import resolve_terminal
        from nemo_gym.token_id_capture.staging.records import CallRecord
        from nemo_gym.token_id_capture.staging.terminal import select_terminal_call

        records = [dict(record) for record in manifest.get("records") or []]
        failures = list(manifest.get("failures") or [])
        deduped: dict[str, dict] = {}
        for record in records:
            deduped.setdefault(str(record.get("model_call_id")), record)
        terminal_record = None
        selection_reason = None
        attribution_reason = None
        terminal_selection = "heuristic"
        parsed_records = None
        try:
            parsed_records = [
                CallRecord.model_validate(record) for record in deduped.values()
            ]
        except ValueError:
            selection_reason = "invalid_manifest_row"
        if parsed_records is not None:
            attribution = resolve_terminal(
                parsed_records,
                scored_response,
                declared_response_id=terminal_response_id,
            )
            attribution_reason = attribution.reason or None
            if attribution.attributed:
                terminal_selection = attribution.method
                terminal_record = deduped[attribution.model_call_id]
            elif terminal_response_id is not None:
                # A declaration is authoritative: a declared id the ledger
                # cannot confirm masks and never falls back to the heuristic.
                terminal_selection = "declared"
                selection_reason = None
            else:
                selection = select_terminal_call(parsed_records)
                if selection.terminal_model_call_id is not None:
                    terminal_record = deduped[selection.terminal_model_call_id]
                else:
                    selection_reason = selection.reason
        poisoning_failures = [
            failure
            for failure in failures
            if str(failure.get("reason") or "") != UNCOMMITTED_CALL_REASON
        ]
        failure_reason = None
        if poisoning_failures:
            failure_reason = str(
                poisoning_failures[0].get("reason") or "capture_failed"
            )
        elif terminal_record is None:
            failure_reason = selection_reason or "missing_terminal_row"
        return {
            "rollout_id": rollout_id,
            "reward": reward,
            "terminal_model_call_id": (
                terminal_record.get("model_call_id")
                if terminal_record is not None
                else None
            ),
            "manifest": list(deduped.values()),
            "capture_poisoned": failure_reason is not None,
            "failure_reason": failure_reason,
            "terminal_selection": terminal_selection,
            "terminal_attribution_reason": attribution_reason,
        }

    def _postprocess_nemo_gym_to_nemo_rl_result(
        self,
        nemo_gym_row: dict,
        nemo_gym_result: dict,
        tokenizer: PreTrainedTokenizerBase,
        *,
        include_initial_multimodal_data: bool = True,
    ) -> dict:
        assert isinstance(nemo_gym_result, dict), (
            f"Hit a non-successful response when querying NeMo Gym for rollouts: {nemo_gym_result}"
        )

        processor = getattr(self, "_processor", None)
        response = nemo_gym_result["response"]
        result_input = nemo_gym_result["responses_create_params"].get("input", [])
        request_input = nemo_gym_row.get("responses_create_params", {}).get("input")
        raw_input = (
            request_input
            if isinstance(request_input, list) and request_input
            else result_input
        )
        initial_input = response.get("agent_input")
        if not isinstance(initial_input, list) or not initial_input:
            initial_input = raw_input

        seed_obs = response.get("seed_obs")
        media_messages = (
            seed_obs if isinstance(seed_obs, list) and seed_obs else initial_input
        )
        raw_initial_sources = extract_input_media_sources_from_responses_messages(
            raw_input
        )
        agent_initial_sources = extract_input_media_sources_from_responses_messages(
            initial_input
        )
        returned_media_sources = extract_input_media_sources_from_responses_messages(
            media_messages
        )
        initial_media_matches_raw_input = (
            bool(raw_initial_sources)
            and len(agent_initial_sources) == len(raw_initial_sources)
            and all(
                media_sources_equal(agent_source, raw_source)
                for agent_source, raw_source in zip(
                    agent_initial_sources, raw_initial_sources
                )
            )
        )
        returned_media_matches_raw_input = len(returned_media_sources) == len(
            raw_initial_sources
        ) and all(
            media_sources_equal(returned_source, raw_source)
            for returned_source, raw_source in zip(
                returned_media_sources, raw_initial_sources
            )
        )
        initial_multimodal_data_omitted = (
            not include_initial_multimodal_data
            and initial_media_matches_raw_input
            and returned_media_matches_raw_input
        )
        if initial_multimodal_data_omitted:
            media_messages, _ = _without_initial_media_sources(
                media_messages, raw_initial_sources
            )
        per_turn_images = (
            _index_per_turn_images(
                response["output"],
                input_messages=media_messages,
            )
            if processor is not None
            else []
        )
        turn_idx = 0

        nemo_rl_message_log = []
        seen_token_ids: List[int] = []
        batch_decode_items = []
        for output_item_dict in nemo_gym_result["response"]["output"]:
            # Nemo RL really only has two types of messages: assistant and not assistant since that is all that it is concerned with (i.e. to train or not to train)
            # Here we map all the trainable messages to assistant and all the non-trainable messages to user.
            # Eventually we can maybe be smarter about this, but this is functional for now.

            # Note that NeMo-Gym will only return token ids on "assistant" messages and not other message types.
            # Also skip if generation_token_ids is present but empty, e.g. all-EOS generation stripped to [] — torch.tensor([]) defaults to float32 and breaks batch dtype consistency.
            if not _is_trainable_output_item(output_item_dict):
                continue

            assert (
                seen_token_ids
                == output_item_dict["prompt_token_ids"][: len(seen_token_ids)]
            ), f"""Non-contiguous messages found! This may be a tokenization issue where certain tokens are combined when messages are concatenated, or it may be due to part of the chat history being truncated (like if super long history is truncated or if reasoning is stripped out).
Seen token IDs: {seen_token_ids}
Output prompt token IDs: {output_item_dict["prompt_token_ids"]}
output prompt token ids till seen: {output_item_dict["prompt_token_ids"][: len(seen_token_ids)]}
"""

            prompt_token_ids = output_item_dict.pop("prompt_token_ids")
            generation_token_ids = output_item_dict.pop("generation_token_ids")
            generation_log_probs = output_item_dict.pop("generation_log_probs")
            routed_experts_raw = output_item_dict.pop("routed_experts", None)
            new_prompt_token_ids = prompt_token_ids[len(seen_token_ids) :]

            routed_experts = None
            if routed_experts_raw is not None:
                routed_experts_dtype = _ROUTED_EXPERTS_DTYPES[
                    self.cfg.get("routed_experts_dtype", "int16")
                ]
                routed_experts = decode_routed_experts(
                    routed_experts_raw, dtype=routed_experts_dtype
                )
                if routed_experts.dim() != 3:
                    raise ValueError(
                        "NeMo Gym returned routed_experts with invalid shape. "
                        "Expected [tokens, num_moe_layers, topk], got "
                        f"{tuple(routed_experts.shape)}."
                    )
                expected_tokens = len(prompt_token_ids) + len(generation_token_ids)
                if routed_experts.shape[0] < expected_tokens:
                    raise ValueError(
                        "NeMo Gym returned too few routed_experts rows for a "
                        "trainable output item: "
                        f"routes={routed_experts.shape[0]}, expected_at_least="
                        f"{expected_tokens}."
                    )
            elif self.cfg.get("require_routed_experts", False):
                # Routes can be legitimately unrecoverable on the echo path
                # (e.g. a context-overflow rollout whose only persisted
                # completion record is the gate's synthetic empty response).
                # Leave the message routeless: backfill_missing_routed_experts
                # sentinel-fills it at flatten and Megatron self-routes those
                # tokens; total absence across a batch still fails loudly at
                # the rollout actor's ROUTED_EXPERTS_FIELD guard.
                print(
                    "router_replay: trainable Gym output item without "
                    "routed_experts — falling back to the missing-route "
                    "sentinel for this message "
                    f"[item_idx={len(nemo_rl_message_log) // 2}, "
                    f"item_type={output_item_dict.get('type')!r}, "
                    f"n_prompt={len(prompt_token_ids)}, "
                    f"n_gen={len(generation_token_ids)}]",
                    flush=True,
                )

            # The next prompt prefill supplies the real route for the previous
            # turn's final token, whose decode route was padded.
            if routed_experts is not None and seen_token_ids:
                previous_routes = nemo_rl_message_log[-1].get("routed_experts")
                if isinstance(previous_routes, torch.Tensor):
                    previous_routes[-1] = routed_experts[len(seen_token_ids) - 1]

            prompt_start = len(seen_token_ids)
            prompt_end = len(prompt_token_ids)
            generation_start = prompt_end
            generation_end = prompt_end + len(generation_token_ids)

            user_message = {
                "role": "user",
                "content": "",
                "token_ids": torch.tensor(new_prompt_token_ids),
            }
            if routed_experts is not None:
                user_message["routed_experts"] = routed_experts[prompt_start:prompt_end]
            nemo_rl_message_log.append(user_message)

            if processor is not None:
                images_this_turn = (
                    per_turn_images[turn_idx] if turn_idx < len(per_turn_images) else []
                )
                attach_image_model_inputs_to_message(
                    user_message,
                    images=images_this_turn,
                    processor=processor,
                    # Read with a default, like _processor above: this method is
                    # called unbound against lightweight stand-ins that define
                    # only what they exercise, so a bare attribute access turns
                    # an unrelated test into an AttributeError.
                    pad_dynamic_image_shapes=getattr(
                        self, "_pad_dynamic_image_shapes", False
                    ),
                )
            # Valid tool calls go through the structured API (tool_calls field) and get
            # executed by NeMo-Gym. If tool call patterns appear in the text content instead,
            # the call was invalid and never executed — flag it so training can penalize it.
            is_invalid_tool_call, has_malformed_thinking = (
                _detect_invalid_tool_call_and_malformed_thinking(
                    output_item_dict,
                    invalid_tool_call_patterns=self.cfg.get(
                        "invalid_tool_call_patterns"
                    ),
                    thinking_tags=self.cfg.get("thinking_tags"),
                )
            )

            assistant_message = {
                "role": "assistant",
                "content": "",
                "token_ids": torch.tensor(generation_token_ids),
                "generation_logprobs": torch.tensor(generation_log_probs),
                "is_invalid_tool_call": is_invalid_tool_call,
                "has_malformed_thinking": has_malformed_thinking,
            }
            if routed_experts is not None:
                assistant_message["routed_experts"] = routed_experts[
                    generation_start:generation_end
                ]
            nemo_rl_message_log.append(assistant_message)

            seen_token_ids.extend(new_prompt_token_ids)
            seen_token_ids.extend(generation_token_ids)

            # We pop to remove larger tensors from logging.
            batch_decode_items.append(
                (output_item_dict, prompt_token_ids, generation_token_ids)
            )
            turn_idx += 1

        if batch_decode_items:
            prompt_strs = tokenizer.batch_decode(
                [item[1] for item in batch_decode_items]
            )
            generation_strs = tokenizer.batch_decode(
                [item[2] for item in batch_decode_items]
            )

            for (output_item_dict, _, _), prompt_str, generation_str in zip(
                batch_decode_items, prompt_strs, generation_strs
            ):
                output_item_dict["prompt_str"] = prompt_str
                output_item_dict["generation_str"] = generation_str

        if not nemo_rl_message_log:
            input_messages = nemo_gym_result["responses_create_params"]["input"]
            try:
                prompt_token_ids = tokenizer.apply_chat_template(
                    input_messages, tokenize=True
                )
                prompt_len_str = f"{len(prompt_token_ids)} tokens"
            except Exception as e:
                prompt_len_str = (
                    f"<unknown — apply_chat_template failed: {type(e).__name__}: {e}>"
                )
            output_item_types = [
                o.get("type") for o in nemo_gym_result["response"]["output"]
            ]
            raise ValueError(
                f"NeMo Gym returned a result with no generation data. "
                f"Possible causes: (1) the prompt for the first turn already exceeds the vLLM max_model_len, "
                f"so vLLM rejected the request before any tokens could be generated; "
                f"(2) all response output items were reasoning/tool-call items with no assistant generation.\n"
                f"  Prompt length: {prompt_len_str}.\n"
                f"  response.output item types ({len(output_item_types)} items): {output_item_types}.\n"
                f"  → If (1): increase `policy.max_total_sequence_length` and `policy.generation.vllm_cfg.max_model_len` "
                f"above the prompt length above.\n"
                f"  → If (2): inspect why no assistant content was produced for this rollout."
            )

        if initial_multimodal_data_omitted:
            for container, key in (
                (nemo_gym_result["responses_create_params"], "input"),
                (response, "agent_input"),
                (response, "seed_obs"),
            ):
                if key in container:
                    container[key], _ = _without_initial_media_sources(
                        container[key], raw_initial_sources
                    )

        result = {
            "message_log": nemo_rl_message_log,
            "input_message_log": nemo_rl_message_log[:1],
            "full_result": nemo_gym_result,
        }
        if not include_initial_multimodal_data:
            result["_initial_multimodal_data_omitted"] = initial_multimodal_data_omitted
        return result

    def shutdown(self) -> None:
        # Teardown runs in a finally block, so it must not turn a real training error
        # into a confusing AttributeError from a never-spun-up (e.g. restarted) actor.
        if self.rh is None:
            return
        run_helper = self.rh
        self.rh = None
        run_helper.shutdown()

    def step(self, message_log_batch, metadata):
        # This is not used since NeMo-Gym will handle the rollouts entirely.
        raise NotImplementedError

    def global_post_process_and_metrics(self, batch):
        # Similar to the step function, this is not used.
        raise NotImplementedError


def extract_reward_components(nemo_gym_result: dict) -> Dict[str, float] | None:
    """Return per-component rewards from a NeMo Gym verify result, or None.

    Single-reward NeMo Gym environments return only a scalar ``reward``. Multi-reward
    environments additionally return ``reward_components``: a mapping of
    component-name -> score. These are surfaced as ``reward/<name>`` batch keys and
    consumed by GDPO (see ``nemo_rl.algorithms.advantage_estimator.GDPOAdvantageEstimator``).

    Returns ``None`` when the environment is single-reward (no ``reward_components``),
    so callers fall back to the scalar ``reward`` path unchanged.
    """
    components = nemo_gym_result.get("reward_components")
    if not components:
        return None
    return {str(name): float(score) for name, score in components.items()}


def build_reward_component_columns(
    component_dicts: List[Dict[str, float] | None],
) -> Dict[str, torch.Tensor]:
    """Build ``reward/<name>`` batch columns from per-sample reward-component dicts.

    Takes the union of component names across the batch in sorted (deterministic) order
    and, for each, emits a ``reward/<name>`` tensor with one entry per sample. A
    component absent on a given sample is filled with ``0.0`` so every column covers all
    samples (the per-prompt baseline requires each component present for all responses).

    Keys are prefixed ``reward/`` so they are exactly what
    ``nemo_rl.algorithms.utils.get_gdpo_reward_component_keys`` selects (it matches
    ``startswith("reward/")`` and sorts by name); the name carries the component identity,
    so no positional index is needed. Returns an empty dict when no sample has components.
    """
    component_names = sorted(
        {name for c in component_dicts if c is not None for name in c}
    )
    return {
        f"reward/{name}": torch.tensor(
            [c[name] if c is not None and name in c else 0.0 for c in component_dicts]
        )
        for name in component_names
    }


def validate_reward_components_match_scalar(nemo_gym_results: List[dict]) -> None:
    """Assert each multi-reward result sets ``reward == sum(reward_components)``.

    A multi-reward verifier must set the scalar ``reward`` to the sum of its
    ``reward_components`` so single-reward (GRPO) consumers and GDPO read the same
    aggregate. We keep the verifier's scalar ``reward`` as ``total_reward`` rather than
    silently overwriting it with the component sum, so a verifier that violates this
    contract must be surfaced here instead of masked.

    Raises ``ValueError`` on the first violating result. A no-op for single-reward
    results (those without ``reward_components``).
    """
    for idx, result in enumerate(nemo_gym_results):
        components = extract_reward_components(result)
        if components is None:
            continue
        scalar_reward = float(result["reward"])
        component_sum = sum(components.values())
        if not math.isclose(scalar_reward, component_sum, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(
                f"NeMo Gym verify result {idx} has reward={scalar_reward} but its "
                f"reward_components sum to {component_sum} ({components}). A multi-reward "
                "verifier must set reward = sum(reward_components.values()) so single-reward "
                "(GRPO) consumers and GDPO read the same aggregate."
            )


########################################
# Global config utils
########################################


def setup_nemo_gym_config(config, tokenizer) -> None:
    generation_config = config.policy["generation"]

    backend = generation_config.get("backend")
    if backend == "vllm":
        # Enable the http server. Requires both async engine and the expose_http_server flag
        generation_config["vllm_cfg"]["async_engine"] = True
        generation_config["vllm_cfg"]["expose_http_server"] = True
    elif backend == "megatron":
        # Enable the http server for Gym dispatch over the Megatron generation backend.
        generation_config["mcore_generation_config"]["expose_http_server"] = True
    else:
        raise ValueError(f"NeMo Gym does not support generation backend {backend!r}.")

    # Stop strings or token ids are not supported
    generation_config["stop_strings"] = None
    generation_config["stop_token_ids"] = None

    # For VLM runs, plumb the tokenizer config into the gym env config so the
    # NemoGym actor can reconstruct the processor inside itself (needed for
    # multi-turn multimodal postprocessing).
    if config.policy.get("is_vlm"):
        env_cfg = config.env.setdefault("nemo_gym", {})
        env_cfg.setdefault("tokenizer_config", dict(config.policy["tokenizer"]))


def build_nemo_gym_config(
    env_configs: dict[str, Any],
    *,
    base_urls: list[str],
    model_name: str,
    enable_router_replay: bool,
    use_fastokens: bool,
    token_capture: Optional[dict[str, Any]] = None,
) -> NemoGymConfig:
    """Build the ``NemoGymConfig`` for a NeMo-Gym actor.

    Splits ``env_configs["nemo_gym"]`` into the NeMo-RL-side fields the actor
    reads directly and the remainder, which is forwarded verbatim as NeMo-Gym's
    initial global config.

    Args:
        env_configs: The master_config.env mapping; env_configs["nemo_gym"] supplies
            the Gym global config plus NeMo-RL detection knobs (invalid_tool_call_patterns,
            thinking_tags, num_gpu_nodes).
        base_urls: Per-DP-rank OpenAI-compatible server base URLs from the generation backend.
        model_name: Served model name the Gym rollouts should target.
        enable_router_replay: Sets ``require_routed_experts`` and selects the
            routed-experts carry dtype ("int8"/"int16"/"int32") for the model.
        use_fastokens: Forwarded from ``policy.tokenizer.use_fastokens`` so the
            actor patches its tokenizer the same way the driver does.

    Returns:
        A ``NemoGymConfig`` with NeMo-RL fields at the top level and the
        remaining ``env_configs["nemo_gym"]`` keys under
        ``initial_global_config_dict``. The caller's ``env_configs`` is not mutated.
    """
    nemo_gym_dict = dict(env_configs["nemo_gym"])

    # NeMo-RL-side detection knobs are top-level NemoGymConfig fields
    # (where the detector reads them), not part of Gym's global config.
    invalid_tool_call_patterns = nemo_gym_dict.pop("invalid_tool_call_patterns", None)
    thinking_tags = nemo_gym_dict.pop("thinking_tags", None)
    tokenizer_config = nemo_gym_dict.pop("tokenizer_config", None)
    # Same treatment for the multimodal knobs: NemoGymConfig declares them as
    # top-level fields, so populate them here instead of leaving the actor to
    # read them back out of Gym's global config dict.
    multimodal_flags: dict[str, bool] = {}
    for _flag in ("pad_dynamic_image_shapes",):
        _value = nemo_gym_dict.pop(_flag, None)
        if _value is not None:
            multimodal_flags[_flag] = bool(_value)

    # Pass prebuilt cache + venv dirs through the global config so the gym reuses
    # image-baked venvs instead of rebuilding them.
    uv_cache_dir = get_nemo_gym_uv_cache_dir()
    if uv_cache_dir is not None:
        nemo_gym_dict.setdefault("uv_cache_dir", uv_cache_dir)
    uv_venv_dir = get_nemo_gym_venv_dir()
    if uv_venv_dir is not None:
        nemo_gym_dict.setdefault("uv_venv_dir", uv_venv_dir)

    routed_experts_dtype = (
        resolve_routed_experts_dtype_name_for_model(model_name)
        if enable_router_replay
        else "int16"
    )

    return NemoGymConfig(
        model_name=model_name,
        base_urls=base_urls,
        invalid_tool_call_patterns=invalid_tool_call_patterns,
        thinking_tags=thinking_tags,
        tokenizer_config=tokenizer_config,
        require_routed_experts=enable_router_replay,
        routed_experts_dtype=routed_experts_dtype,
        use_fastokens=use_fastokens,
        initial_global_config_dict=nemo_gym_dict,
        token_capture=token_capture,
        **multimodal_flags,
    )


def spinup_nemo_gym_actor(
    env_configs: dict[str, Any],
    *,
    base_urls: list[str],
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    enable_router_replay: bool,
    use_fastokens: bool,
    token_capture: Optional[dict[str, Any]] = None,
) -> Any:
    """Spin up the NeMo-Gym actor against the given generation server URLs.

    When ``env_configs["nemo_gym"]["num_gpu_nodes"] > 0``, the actor is
    scheduled with soft NodeAffinity to the caller's Ray node so its colocated
    GPU resources land where the caller expects.

    Args:
        tokenizer: Installed on the actor once, here, rather than passed per
            rollout call. See ``NemoGym.set_tokenizer`` for why that
            distinction is the difference between a working run and a stalled
            one.
        token_capture: Dumped ``TokenCaptureConfig`` when ledger-authoritative
            token capture is enabled, else ``None``. Forwarded to
            ``build_nemo_gym_config``.

    Returns:
        The spun-up ``NemoGym`` Ray actor handle (``_spinup`` already awaited).
    """
    nemo_gym_cfg = build_nemo_gym_config(
        env_configs,
        base_urls=base_urls,
        model_name=model_name,
        enable_router_replay=enable_router_replay,
        use_fastokens=use_fastokens,
        token_capture=token_capture,
    )

    nemo_gym_opts: dict[str, Any] = {
        "runtime_env": make_actor_runtime_env(NEMO_GYM_ACTOR_FQN)
    }
    if env_configs["nemo_gym"].get("num_gpu_nodes", 0):
        nemo_gym_opts["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=True,
        )

    actor = NemoGym.options(**nemo_gym_opts).remote(nemo_gym_cfg)
    ray.get(actor._spinup.remote())
    ray.get(actor.set_tokenizer.remote(tokenizer))
    return actor
