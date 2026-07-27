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
"""Schema for splitting one NeMo-Gym stack across several actors.

A single ``NemoGym`` actor runs every environment in the job on one node, which
caps a run at what that node can host. Sharding partitions the Gym config into
groups that each get their own actor on their own node.

NeMo RL composes shard configs; it never interprets them. Everything forwarded
into a shard's merge is opaque Gym config, and validation compares entry
*names*, never meanings.
"""

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping

from omegaconf import OmegaConf

# Keys under ``env.nemo_gym`` that NeMo RL consumes itself. They must never
# reach NeMo Gym: the merged config is serialized into NEMO_GYM_CONFIG_DICT for
# every one of the ~60 child processes, and Gym treats unrecognized dict-shaped
# top-level keys as server instance configs, which can trip its
# error_on_almost_servers path.
SHARDING_CONFIG_KEYS = frozenset(
    {
        "shards",
        "common_overrides",
        "allowed_duplicate_entries",
        "placement_strategy",
    }
)

# Ray placement-group strategies a shard plan may ask for. STRICT_SPREAD is the
# point of sharding -- one actor per node -- and anything else colocates shards
# and gives up the capacity isolation. PACK exists so the mechanism can be
# exercised on a single machine.
PLACEMENT_STRATEGIES = frozenset({"STRICT_SPREAD", "SPREAD", "STRICT_PACK", "PACK"})
DEFAULT_PLACEMENT_STRATEGY = "STRICT_SPREAD"

# Gym's log-directory key (``nemo_gym.global_config.NEMO_GYM_LOG_DIR_KEY_NAME``).
# Restated rather than imported: this module is imported on the driver, and
# nemo_gym is installed only in the actor's venv.
GYM_LOG_DIR_KEY = "nemo_gym_log_dir"

# Per-shard keys consumed by NeMo RL rather than forwarded to Gym.
SHARD_SPEC_KEYS = frozenset(
    {
        "name",
        "config_paths",
        "overrides",
        "replicas",
        "actor_cpus",
        "port_range_low",
        "port_range_high",
    }
)


class ShardConfigError(ValueError):
    """Raised for a malformed ``env.nemo_gym.shards`` block."""


class ShardSetupError(RuntimeError):
    """Raised when a sharded stack fails to start or fails its startup checks."""


@dataclass(frozen=True)
class ShardSpec:
    """One shard: a slice of the Gym config that gets its own actor.

    ``replicas`` instances are created from a single merge, so replicas are
    identical apart from the node they land on.
    """

    name: str
    config_paths: list[str]
    overrides: dict[str, Any] = field(default_factory=dict)
    replicas: int = 1
    actor_cpus: float | None = None
    port_range_low: int | None = None
    port_range_high: int | None = None


@dataclass(frozen=True)
class ShardPlan:
    """The parsed ``shards`` block plus the settings that apply to all shards."""

    shards: list[ShardSpec]
    common_overrides: dict[str, Any] = field(default_factory=dict)
    allowed_duplicate_entries: frozenset[str] = frozenset()
    placement_strategy: str = DEFAULT_PLACEMENT_STRATEGY

    @property
    def total_instances(self) -> int:
        """Actors to create, counting replicas. One node is needed per instance."""
        return sum(shard.replicas for shard in self.shards)


def _as_plain_dict(value: Any, *, context: str) -> dict[str, Any]:
    """Coerce a mapping (possibly OmegaConf-backed) to a plain dict."""
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise ShardConfigError(
            f"{context} must be a mapping, got {type(value).__name__}"
        )
    return dict(value)


def _parse_shard(raw: Any, index: int) -> ShardSpec:
    entry = _as_plain_dict(raw, context=f"shards[{index}]")

    unknown = set(entry) - SHARD_SPEC_KEYS
    if unknown:
        raise ShardConfigError(
            f"shards[{index}] has unrecognized keys {sorted(unknown)}. "
            f"Gym config overlays belong under this shard's 'overrides'."
        )

    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ShardConfigError(f"shards[{index}] needs a non-empty string 'name'")

    config_paths = entry.get("config_paths")
    if OmegaConf.is_config(config_paths):
        config_paths = OmegaConf.to_container(config_paths, resolve=True)
    if not isinstance(config_paths, list) or not config_paths:
        raise ShardConfigError(
            f"shard '{name}' needs at least one entry in 'config_paths'"
        )

    replicas = entry.get("replicas", 1)
    if not isinstance(replicas, int) or isinstance(replicas, bool) or replicas < 1:
        raise ShardConfigError(
            f"shard '{name}' has replicas={replicas!r}; must be an integer >= 1"
        )

    port_low = entry.get("port_range_low")
    port_high = entry.get("port_range_high")
    if (port_low is None) != (port_high is None):
        raise ShardConfigError(
            f"shard '{name}' must set both port_range_low and port_range_high, or neither"
        )
    if port_low is not None and port_low >= port_high:
        raise ShardConfigError(
            f"shard '{name}' has an empty port range [{port_low}, {port_high})"
        )

    return ShardSpec(
        name=name,
        config_paths=[str(path) for path in config_paths],
        overrides=_as_plain_dict(
            entry.get("overrides"), context=f"shard '{name}' overrides"
        ),
        replicas=replicas,
        actor_cpus=entry.get("actor_cpus"),
        port_range_low=port_low,
        port_range_high=port_high,
    )


def find_gym_config_entries(nemo_gym_config: Mapping[str, Any]) -> list[str]:
    """Return top-level keys that Gym would read as server instance configs.

    Mirrors Gym's own rule (``filter_for_server_instance_configs``): any
    dict-shaped top-level key that is not a known setting is an entry overlay.
    Under sharding these are ambiguous — they have no shard to belong to — so
    the caller rejects them and asks the author to place them explicitly.
    """
    return sorted(
        key
        for key, value in nemo_gym_config.items()
        if key not in SHARDING_CONFIG_KEYS
        and (isinstance(value, Mapping) or OmegaConf.is_dict(value))
    )


def parse_shard_plan(nemo_gym_config: Mapping[str, Any]) -> ShardPlan | None:
    """Parse ``env.nemo_gym.shards``, or return None when the job is unsharded.

    Returning None is the backward-compatible path: a config with
    ``config_paths`` and no ``shards`` behaves exactly as it did before
    sharding existed.

    Raises:
        ShardConfigError: the shards block is malformed, or it is combined with
            config that has no unambiguous shard to belong to.
    """
    if "shards" not in nemo_gym_config:
        return None

    raw_shards = nemo_gym_config["shards"]
    if OmegaConf.is_config(raw_shards):
        raw_shards = OmegaConf.to_container(raw_shards, resolve=True)
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ShardConfigError("env.nemo_gym.shards must be a non-empty list")

    # Anything whose scope would be implicit has to be placed by hand: a
    # top-level config_paths or entry overlay could plausibly belong to one
    # shard or to all of them, and guessing wrong silently changes what each
    # actor hosts.
    # A null reads as absent throughout this block. A sharded config is
    # normally written as a Hydra override of an unsharded recipe, and an
    # override can only blank an inherited key, never delete it -- so
    # `config_paths: null` is how one says "the shards own the paths now".
    if nemo_gym_config.get("config_paths") is not None:
        raise ShardConfigError(
            "env.nemo_gym.config_paths cannot be combined with 'shards'. "
            "Move each path into the shard that should host it, and set the "
            "inherited config_paths to null."
        )
    stray_entries = find_gym_config_entries(nemo_gym_config)
    if stray_entries:
        raise ShardConfigError(
            f"Gym config overlays {stray_entries} sit at the top level alongside "
            f"'shards', so their scope is ambiguous. Move each one into "
            f"'common_overrides' (to apply to every shard) or into the "
            f"'overrides' of the shard that hosts it."
        )

    shards = [_parse_shard(raw, index) for index, raw in enumerate(raw_shards)]

    duplicate_names = sorted(
        {
            shard.name
            for shard in shards
            if [s.name for s in shards].count(shard.name) > 1
        }
    )
    if duplicate_names:
        raise ShardConfigError(f"Duplicate shard names: {duplicate_names}")

    placement_strategy = (
        nemo_gym_config.get("placement_strategy") or DEFAULT_PLACEMENT_STRATEGY
    )
    if placement_strategy not in PLACEMENT_STRATEGIES:
        raise ShardConfigError(
            f"env.nemo_gym.placement_strategy must be one of "
            f"{sorted(PLACEMENT_STRATEGIES)}, got {placement_strategy!r}"
        )

    raw_allowed = nemo_gym_config.get("allowed_duplicate_entries") or []
    if OmegaConf.is_config(raw_allowed):
        raw_allowed = OmegaConf.to_container(raw_allowed, resolve=True)
    if not isinstance(raw_allowed, list) or not all(
        isinstance(entry, str) for entry in raw_allowed
    ):
        raise ShardConfigError(
            "env.nemo_gym.allowed_duplicate_entries must be a list of strings"
        )

    return ShardPlan(
        shards=shards,
        common_overrides=_as_plain_dict(
            nemo_gym_config.get("common_overrides"), context="common_overrides"
        ),
        allowed_duplicate_entries=frozenset(raw_allowed),
        placement_strategy=placement_strategy,
    )


def apply_shard_overlay(
    base_config: dict[str, Any], plan: ShardPlan, shard: ShardSpec
) -> dict[str, Any]:
    """Build one shard's Gym config: base, then common overrides, then the shard's.

    The ``config_paths`` merge itself still happens inside Gym, exactly as for
    an unsharded job; this only settles which paths and overlays that shard
    starts from. Per-shard overrides win over ``common_overrides`` on conflict.

    Top-level keys left null are dropped rather than forwarded. Blanking a key
    is the only way a Hydra override can retract one it inherited (see
    ``parse_shard_plan``), and Gym should not receive a null where it expects
    an entry.
    """
    merged = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.create(base_config),
            OmegaConf.create(plan.common_overrides),
            OmegaConf.create(shard.overrides),
        ),
        resolve=True,
    )
    assert isinstance(merged, dict)
    merged = {key: value for key, value in merged.items() if value is not None}
    merged["config_paths"] = list(shard.config_paths)
    if shard.port_range_low is not None:
        merged["port_range_low"] = shard.port_range_low
        merged["port_range_high"] = shard.port_range_high
    return merged


def build_agent_shard_map(
    entries_by_shard: Mapping[str, Mapping[str, list[str]]],
    allowed_duplicate_entries: frozenset[str] | set[str] = frozenset(),
) -> dict[str, str]:
    """Map each agent entry to its shard, rejecting duplicates across shards.

    Takes what each shard reported from ``NemoGym.list_entries()`` and returns
    ``{agent_entry_name: shard_name}``, the lookup the router dispatches on.

    Two failures are caught here rather than at first dispatch. An agent hosted
    by two shards is always an error: rows naming it could go to either, so
    routing would be silently nondeterministic. Any other entry in two shards
    has to be allowlisted, because duplication is usually accidental — a shared
    YAML dropped into two shards' path lists quietly brings its judge along and
    doubles that judge's GPU claim.

    Only names are compared. What an entry means is Gym's business.
    """
    agent_to_shard: dict[str, str] = {}
    for shard_name, entries in entries_by_shard.items():
        for entry, types in entries.items():
            if "responses_api_agents" not in types:
                continue
            if entry in agent_to_shard:
                raise ShardSetupError(
                    f"Agent '{entry}' is hosted by both shard "
                    f"'{agent_to_shard[entry]}' and shard '{shard_name}'. An "
                    f"agent must live in exactly one shard so rows naming it "
                    f"have one destination."
                )
            agent_to_shard[entry] = shard_name

    hosting_shard: dict[str, str] = {}
    for shard_name, entries in entries_by_shard.items():
        for entry in entries:
            if entry in agent_to_shard:
                continue
            if entry in hosting_shard and entry not in allowed_duplicate_entries:
                raise ShardSetupError(
                    f"Config entry '{entry}' appears in shard "
                    f"'{hosting_shard[entry]}' and shard '{shard_name}' but is "
                    f"not listed in allowed_duplicate_entries. If this entry "
                    f"starts a model engine, duplicating it doubles its GPU "
                    f"claim; if the duplication is intended, allowlist it."
                )
            hosting_shard.setdefault(entry, shard_name)

    return agent_to_shard


def apply_shard_log_dir(
    gym_config: Mapping[str, Any],
    shard_name: str,
    *,
    replica_index: int | None = None,
) -> dict[str, Any]:
    """Give one shard instance its own subdirectory of the configured log dir.

    Gym names each server's log file after the server and appends to it with
    ``tee -a`` (``cli/setup_command.py``). One log dir shared by every instance
    therefore interleaves output whenever two instances host the same server
    name — which replicas always do, since they are stamped from one merge — and
    on shared storage that happens across nodes without any error.

    Returns the config unchanged when no log dir is set, which is the default.
    """
    config = dict(gym_config)
    log_dir = config.get(GYM_LOG_DIR_KEY)
    if not log_dir:
        return config

    # Gym applies the same sanitization to server names before using them as
    # path components.
    suffix = shard_name.replace("/", "_")
    if replica_index is not None:
        suffix = f"{suffix}/{replica_index}"
    config[GYM_LOG_DIR_KEY] = str(PurePosixPath(str(log_dir)) / suffix)
    return config
