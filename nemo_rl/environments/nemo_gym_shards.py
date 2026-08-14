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

import math
import re
from dataclasses import dataclass, field, replace
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
        "common_inherited_overlays",
        "common_overrides",
        "allowed_duplicate_entries",
    }
)

# Dict-shaped NeMo RL integration settings that are not Gym server entries.
# The actor factory or rollout code consumes these separately.
NEMO_RL_DICT_CONFIG_KEYS = frozenset({"effort_levels", "tokenizer_config"})

DEFAULT_REPLICAS = 1
SHARD_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Gym's log-directory key (``nemo_gym.global_config.NEMO_GYM_LOG_DIR_KEY_NAME``).
# Restated rather than imported: this module is imported on the driver, and
# nemo_gym is installed only in the actor's venv.
GYM_LOG_DIR_KEY = "nemo_gym_log_dir"

# Per-shard keys consumed by NeMo RL rather than forwarded to Gym.
SHARD_SPEC_KEYS = frozenset(
    {
        "name",
        "config_paths",
        "inherited_overlays",
        "overrides",
        "replicas",
        "actor_cpus",
        "port_range_low",
        "port_range_high",
    }
)


class ShardConfigError(ValueError):
    """Raised for a malformed ``env.nemo_gym.shards`` block."""


@dataclass(frozen=True)
class ShardSpec:
    """One shard: a slice of the Gym config that gets its own actor.

    ``replicas`` instances are created from a single merge, so replicas are
    identical apart from the node they land on.
    """

    name: str
    config_paths: list[str]
    inherited_overlays: frozenset[str] = frozenset()
    overrides: dict[str, Any] = field(default_factory=dict)
    replicas: int = DEFAULT_REPLICAS
    actor_cpus: float | None = None
    port_range_low: int | None = None
    port_range_high: int | None = None


@dataclass(frozen=True)
class ShardPlan:
    """The parsed ``shards`` block plus the settings that apply to all shards."""

    shards: list[ShardSpec]
    common_inherited_overlays: frozenset[str] = frozenset()
    common_overrides: dict[str, Any] = field(default_factory=dict)
    allowed_duplicate_entries: frozenset[str] = frozenset()

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


def _as_string_set(value: Any, *, context: str) -> frozenset[str]:
    """Coerce a list of unique strings to a set."""
    if value is None:
        return frozenset()
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, list) or not all(
        isinstance(entry, str) and entry for entry in value
    ):
        raise ShardConfigError(f"{context} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ShardConfigError(f"{context} contains duplicate keys")
    return frozenset(value)


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
    if not SHARD_NAME_PATTERN.fullmatch(name) or name in {".", ".."}:
        raise ShardConfigError(
            f"shard name {name!r} must be one safe path component containing "
            "only letters, numbers, dots, underscores, and hyphens"
        )

    config_paths = entry.get("config_paths")
    if OmegaConf.is_config(config_paths):
        config_paths = OmegaConf.to_container(config_paths, resolve=True)
    if not isinstance(config_paths, list) or not config_paths:
        raise ShardConfigError(
            f"shard '{name}' needs at least one entry in 'config_paths'"
        )
    if not all(isinstance(path, str) and path for path in config_paths):
        raise ShardConfigError(f"shard '{name}' config_paths must be non-empty strings")

    replicas = entry.get("replicas", DEFAULT_REPLICAS)
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
    if port_low is not None and (
        not isinstance(port_low, int)
        or isinstance(port_low, bool)
        or not isinstance(port_high, int)
        or isinstance(port_high, bool)
        or port_low < 1
        or port_high > 65_536
    ):
        raise ShardConfigError(
            f"shard '{name}' port range must use integers within [1, 65536]"
        )
    if port_low is not None and port_low >= port_high:
        raise ShardConfigError(
            f"shard '{name}' has an empty port range [{port_low}, {port_high})"
        )

    actor_cpus = entry.get("actor_cpus")
    if actor_cpus is not None and (
        not isinstance(actor_cpus, (int, float))
        or isinstance(actor_cpus, bool)
        or not math.isfinite(actor_cpus)
        or actor_cpus <= 0
    ):
        raise ShardConfigError(
            f"shard '{name}' has actor_cpus={actor_cpus!r}; "
            "must be a positive finite number"
        )

    return ShardSpec(
        name=name,
        config_paths=list(config_paths),
        inherited_overlays=_as_string_set(
            entry.get("inherited_overlays"),
            context=f"shard '{name}' inherited_overlays",
        ),
        overrides=_as_plain_dict(
            entry.get("overrides"), context=f"shard '{name}' overrides"
        ),
        replicas=replicas,
        actor_cpus=float(actor_cpus) if actor_cpus is not None else None,
        port_range_low=port_low,
        port_range_high=port_high,
    )


def find_gym_config_entries(nemo_gym_config: Mapping[str, Any]) -> list[str]:
    """Return top-level keys that Gym would read as server instance configs.

    Mirrors Gym's own rule (``filter_for_server_instance_configs``): any
    dict-shaped top-level key that is not a known setting is an entry overlay.
    Under sharding these values are inherited from the resolved parent config.
    A shard claims them by key so their YAML bodies do not need to be copied.
    """
    return sorted(
        key
        for key, value in nemo_gym_config.items()
        if key not in SHARDING_CONFIG_KEYS
        and key not in NEMO_RL_DICT_CONFIG_KEYS
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

    # A top-level config_paths could plausibly belong to one shard or all of
    # them, so its scope must remain explicit.
    if "config_paths" in nemo_gym_config:
        raise ShardConfigError(
            "env.nemo_gym.config_paths cannot be combined with 'shards'. "
            "Move each path into the shard that should host it."
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

    inherited_inventory = frozenset(find_gym_config_entries(nemo_gym_config))
    common_inherited = _as_string_set(
        nemo_gym_config.get("common_inherited_overlays"),
        context="env.nemo_gym.common_inherited_overlays",
    )
    claims: dict[str, list[str]] = {}
    for key in common_inherited:
        claims.setdefault(key, []).append("common_inherited_overlays")
    for shard in shards:
        for key in shard.inherited_overlays:
            claims.setdefault(key, []).append(f"shard '{shard.name}'")

    unknown = sorted(set(claims) - inherited_inventory)
    if unknown:
        raise ShardConfigError(
            f"Inherited overlay keys {unknown} do not exist in the resolved "
            f"env.nemo_gym config. Available overlays: {sorted(inherited_inventory)}."
        )

    duplicated = {
        key: owners for key, owners in sorted(claims.items()) if len(owners) > 1
    }
    if duplicated:
        raise ShardConfigError(
            "Each inherited Gym overlay must be claimed once; duplicate claims: "
            f"{duplicated}."
        )

    unclaimed = inherited_inventory - set(claims)
    if len(shards) == 1:
        if unclaimed:
            shards[0] = replace(
                shards[0],
                inherited_overlays=shards[0].inherited_overlays | unclaimed,
            )
    elif unclaimed:
        raise ShardConfigError(
            f"Inherited Gym overlays {sorted(unclaimed)} have no owner. List each "
            f"key under one shard's 'inherited_overlays' or under "
            f"'common_inherited_overlays'. Complete overlay inventory: "
            f"{sorted(inherited_inventory)}."
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
        common_inherited_overlays=common_inherited,
        common_overrides=_as_plain_dict(
            nemo_gym_config.get("common_overrides"), context="common_overrides"
        ),
        allowed_duplicate_entries=frozenset(raw_allowed),
    )


def apply_shard_overlay(
    base_config: dict[str, Any], plan: ShardPlan, shard: ShardSpec
) -> dict[str, Any]:
    """Build one shard's Gym config from shared settings and its claimed overlays.

    The ``config_paths`` merge itself still happens inside Gym, exactly as for
    an unsharded job. Dict-shaped entries inherited from a parent recipe are
    removed from the shared base and restored only for the shards that claim
    them. Explicit overrides win over inherited values.
    """
    inherited_entries = set(find_gym_config_entries(base_config))
    shared_base = {
        key: value
        for key, value in base_config.items()
        if key not in inherited_entries and key not in SHARDING_CONFIG_KEYS
    }
    selected_inherited = {
        key: base_config[key]
        for key in plan.common_inherited_overlays | shard.inherited_overlays
    }
    merged = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.create(shared_base),
            OmegaConf.create(selected_inherited),
            OmegaConf.create(plan.common_overrides),
            OmegaConf.create(shard.overrides),
        ),
        resolve=True,
    )
    assert isinstance(merged, dict)
    merged["config_paths"] = list(shard.config_paths)
    if shard.port_range_low is not None:
        merged["port_range_low"] = shard.port_range_low
        merged["port_range_high"] = shard.port_range_high
    return merged


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
