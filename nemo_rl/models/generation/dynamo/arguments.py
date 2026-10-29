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

"""Argument construction and log redaction for the managed Dynamo frontend."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from nemo_rl.models.generation.dynamo.config import DynamoCfg

_CREDENTIAL_FLAGS = {
    "--access-token",
    "--api-key",
    "--apikey",
    "--auth-token",
    "--hf-token",
    "--password",
    "--secret",
    "--token",
}


def _normalise_flag(flag: str) -> str:
    flag = flag.split("=", 1)[0].replace("_", "-")
    if flag.startswith("--no-"):
        return "--" + flag[5:]
    return flag


def _serialise_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)


class _ArgvBuilder:
    def __init__(self) -> None:
        self.argv: list[str] = []
        self.sources: dict[str, str] = {}

    def add(self, flag: str, value: Any = None, *, source: str) -> None:
        normalised = _normalise_flag(flag)
        prior = self.sources.get(normalised)
        if prior is not None:
            raise ValueError(
                f"Dynamo frontend option {normalised} is set by both {prior} and {source}."
            )
        self.sources[normalised] = source
        if isinstance(value, bool):
            self.argv.append(flag if value else f"--no-{flag.removeprefix('--')}")
        else:
            self.argv.append(flag)
            if value is not None:
                self.argv.append(_serialise_value(value))

    def add_raw(self, args: Sequence[str], *, source: str) -> None:
        if not args:
            return
        current_flag: str | None = None
        current_has_value = False
        for token in args:
            if token.startswith("--"):
                current_flag = token
                current_has_value = "=" in token
                normalised = _normalise_flag(token)
                if normalised in self.sources:
                    raise ValueError(
                        f"Dynamo frontend option {normalised} is set by both "
                        f"{self.sources[normalised]} and {source}."
                    )
                self.sources[normalised] = source
            elif current_flag is None or current_has_value:
                raise ValueError(
                    f"{source} contains invalid positional argument {token!r}; "
                    "values must immediately follow one --option."
                )
            else:
                current_has_value = True
        self.argv.extend(args)


def build_dynamo_frontend_argv(
    *,
    host: str,
    port: int,
    namespace: str,
    dynamo_cfg: DynamoCfg,
) -> list[str]:
    """Build the managed ``dynamo.frontend`` argument list."""
    builder = _ArgvBuilder()
    builder.add("--http-host", host, source="managed runtime")
    builder.add("--http-port", port, source="managed runtime")
    builder.add("--namespace-prefix", namespace, source="managed runtime")
    builder.add("--discovery-backend", "etcd", source="managed runtime")
    builder.add("--request-plane", "tcp", source="managed runtime")
    builder.add("--event-plane", "nats", source="managed runtime")
    builder.add(
        "--router-mode",
        dynamo_cfg.frontend_args.router_mode,
        source="dynamo_cfg.frontend_args.router_mode",
    )
    builder.add_raw(
        dynamo_cfg.frontend_args.extra_cli_args,
        source="dynamo_cfg.frontend_args.extra_cli_args",
    )
    return builder.argv


def redact_argv(argv: Sequence[str]) -> list[str]:
    """Redact values following explicit credential options for safe logging."""
    redacted = list(argv)
    for idx, token in enumerate(redacted):
        is_sensitive = token.startswith("--") and (
            _normalise_flag(token.lower()) in _CREDENTIAL_FLAGS
        )
        if is_sensitive and "=" in token:
            redacted[idx] = token.split("=", 1)[0] + "=<redacted>"
        elif is_sensitive and idx + 1 < len(redacted):
            if not redacted[idx + 1].startswith("--"):
                redacted[idx + 1] = "<redacted>"
    return redacted


def redact_environment(env: Mapping[str, str]) -> dict[str, str]:
    """Redact credential-like environment values before logging."""
    sensitive_fragments = ("token", "password", "secret", "api_key", "apikey")
    return {
        key: "<redacted>"
        if any(part in key.lower() for part in sensitive_fragments)
        else value
        for key, value in env.items()
    }
