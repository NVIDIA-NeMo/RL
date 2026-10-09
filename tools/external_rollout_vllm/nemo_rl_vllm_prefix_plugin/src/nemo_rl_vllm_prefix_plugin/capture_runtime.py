"""External-staging client and per-request state for NeMo Gym token capture."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def _service_url(base_url: str, path: str) -> str:
    parsed = urlsplit(base_url)
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


class CaptureBridgeClient:
    """Blocking stdlib client for the controller-side staging bridge."""

    def __init__(self, *, base_url: str, auth_token: str) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("bridge_url must start with http:// or https://")
        if not auth_token:
            raise ValueError("auth_token must be non-empty")
        self._base_url = base_url.rstrip("/")
        self._auth_token = auth_token

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            _service_url(self._base_url, path),
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._auth_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=1800.0) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"token-capture bridge returned HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"could not reach token-capture bridge: {error.reason}"
            ) from error
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise RuntimeError("token-capture bridge returned a non-object response")
        return decoded

    def fetch_prefix(self, staging_chain: list[str]) -> list[int]:
        """Fetch and concatenate an admitted chain of staged token deltas."""
        response = self._post(
            "/v1/nemo-rl/token-capture/prefix",
            {"staging_chain": staging_chain},
        )
        token_ids = response.get("prefix_token_ids")
        if not isinstance(token_ids, list):
            raise RuntimeError("token-capture bridge returned no prefix_token_ids list")
        return [int(token_id) for token_id in token_ids]

    def commit(self, body: dict[str, Any]) -> dict[str, Any]:
        """Stage a completed call and return its lightweight commit coordinates."""
        response = self._post("/v1/nemo-rl/token-capture/commit", body)
        coords = response.get("ng_commit_coords")
        if not isinstance(coords, dict):
            raise RuntimeError(
                "token-capture bridge returned no ng_commit_coords object"
            )
        return coords


@dataclass(frozen=True)
class _PendingCapture:
    admission: dict[str, Any]
    prefix_token_ids: list[int]
    prompt_token_ids: list[int]


class CaptureRuntime:
    """Resolve staged prefixes and commit completed non-streaming requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bridge: CaptureBridgeClient | None = None
        self._pending: dict[int, _PendingCapture] = {}

    def configure(self, *, bridge_url: str, auth_token: str) -> None:
        """Replace the staging bridge used for subsequent requests."""
        bridge = CaptureBridgeClient(base_url=bridge_url, auth_token=auth_token)
        with self._lock:
            self._bridge = bridge

    def _require_bridge(self) -> CaptureBridgeClient:
        with self._lock:
            bridge = self._bridge
        if bridge is None:
            raise RuntimeError(
                "ng_capture request arrived before the token-capture bridge was configured"
            )
        return bridge

    def resolve_prefix(
        self,
        admission: dict[str, Any],
        *,
        prefill_prompt_token_ids: list[int] | None,
    ) -> list[int]:
        """Resolve inline or staged-chain token-in admission to exact prefix IDs."""
        mode = admission.get("mode")
        if mode == "text":
            return []
        if mode != "token_in":
            raise ValueError(f"unsupported ng_capture mode {mode!r}")
        prev_len = admission.get("prev_len")
        if type(prev_len) is not int or prev_len < 0:
            raise ValueError("ng_capture.prev_len must be a non-negative integer")
        inline = admission.get("required_prefix_token_ids") or []
        if not isinstance(inline, list):
            raise ValueError("ng_capture.required_prefix_token_ids must be a list")
        if prefill_prompt_token_ids is not None:
            prefix = list(prefill_prompt_token_ids[:prev_len])
        elif inline:
            prefix = [int(token_id) for token_id in inline]
        else:
            staging_chain = admission.get("staging_chain") or []
            if not isinstance(staging_chain, list) or not staging_chain:
                raise ValueError(
                    "token_in admission must carry required_prefix_token_ids "
                    "or a non-empty staging_chain"
                )
            prefix = self._require_bridge().fetch_prefix(staging_chain)
        if len(prefix) != prev_len:
            raise ValueError(
                f"resolved prefix length {len(prefix)} does not equal prev_len {prev_len}"
            )
        if inline and prefix != [int(token_id) for token_id in inline]:
            raise ValueError("resolved prefix conflicts with inline ng_capture prefix")
        return prefix

    def record_prompt(
        self,
        request: Any,
        *,
        admission: dict[str, Any],
        prefix_token_ids: list[int],
        prompt_token_ids: list[int],
    ) -> None:
        """Retain the exact engine prompt until the full response is available."""
        pending = _PendingCapture(
            admission=dict(admission),
            prefix_token_ids=list(prefix_token_ids),
            prompt_token_ids=list(prompt_token_ids),
        )
        with self._lock:
            self._pending[id(request)] = pending

    @staticmethod
    def _extract_generation(
        content: dict[str, Any],
    ) -> tuple[list[int], list[float], dict[str, Any] | None]:
        choices = content.get("choices") or []
        if len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("token capture requires exactly one response choice")
        choice = choices[0]
        logprob_content = (choice.get("logprobs") or {}).get("content")
        if not isinstance(logprob_content, list):
            raise ValueError("vLLM response carries no choice.logprobs.content")
        token_ids: list[int] = []
        logprobs: list[float] = []
        for item in logprob_content:
            if not isinstance(item, dict):
                raise ValueError("vLLM logprob entries must be objects")
            token_ids.append(int(str(item["token"]).removeprefix("token_id:")))
            logprobs.append(float(item["logprob"]))
        message = choice.get("message") or {}
        routed_experts = (
            message.get("routed_experts") if isinstance(message, dict) else None
        )
        extras = (
            {"routed_experts": routed_experts} if routed_experts is not None else None
        )
        return token_ids, logprobs, extras

    def finish(self, request: Any, content: dict[str, Any]) -> dict[str, Any]:
        """Stage one completed call, attach coordinates, and drop token payloads."""
        with self._lock:
            pending = self._pending.pop(id(request), None)
        if pending is None:
            return content
        generated_ids, generated_logprobs, extras = self._extract_generation(content)
        coords = self._require_bridge().commit(
            {
                "admission": pending.admission,
                "prefix_token_ids": pending.prefix_token_ids,
                "prompt_token_ids": pending.prompt_token_ids,
                "generated_token_ids": generated_ids,
                "generated_logprobs": generated_logprobs,
                "extras": extras,
            }
        )
        for choice in content.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            choice.pop("logprobs", None)
            message = choice.get("message")
            if isinstance(message, dict):
                message.pop("routed_experts", None)
        content["ng_commit_coords"] = coords
        return content

    def discard(self, request: Any) -> None:
        """Forget capture state for a request that did not produce a response."""
        with self._lock:
            self._pending.pop(id(request), None)

    def capability(self) -> dict[str, Any]:
        """Return whether a controller bridge is currently configured."""
        with self._lock:
            return {"token_capture_configured": self._bridge is not None}
