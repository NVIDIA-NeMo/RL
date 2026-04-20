"""Redaction helpers used before logging k8s objects.

Manifests carry secret material in two common shapes: ConfigMap/Secret
``data`` blocks and container ``env`` entries whose name matches a
well-known credential pattern. Surface-level log lines should never show
either, even on error paths where we dump the full body for context.
"""

from __future__ import annotations

import copy
import re
from typing import Any

_SECRET_NAME_RE = re.compile(r"(TOKEN|KEY|PASSWORD|SECRET|CRED|PASSWD)", re.IGNORECASE)
_REDACTED = "***REDACTED***"


def _redact_env_list(envs: list[Any]) -> None:
    for entry in envs:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "") or ""
        is_secretish = bool(_SECRET_NAME_RE.search(name))
        if "valueFrom" in entry and isinstance(entry["valueFrom"], dict):
            vf = entry["valueFrom"]
            if "secretKeyRef" in vf:
                entry["valueFrom"] = {"secretKeyRef": _REDACTED}
        if is_secretish and "value" in entry:
            entry["value"] = _REDACTED


def redact(obj: Any) -> Any:
    """Return a deep copy of ``obj`` with secret-looking fields masked.

    Handles ConfigMap/Secret ``data`` + ``stringData`` blocks and
    ``env[].value`` / ``env[].valueFrom.secretKeyRef`` across every
    pod template (head + worker groups).
    """
    if not isinstance(obj, (dict, list)):
        return obj

    scrubbed = copy.deepcopy(obj)
    _walk(scrubbed)
    return scrubbed


def _walk(node: Any) -> None:
    if isinstance(node, dict):
        kind = node.get("kind")
        if kind in ("ConfigMap", "Secret"):
            for key in ("data", "stringData"):
                if isinstance(node.get(key), dict):
                    node[key] = {k: _REDACTED for k in node[key]}
        if "env" in node and isinstance(node["env"], list):
            _redact_env_list(node["env"])
        for v in node.values():
            _walk(v)
    elif isinstance(node, list):
        for item in node:
            _walk(item)


__all__ = ["redact"]
