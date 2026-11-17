"""[CUSTOM] Enable proxy connection via env variable passthrough.

This module's only job is, when explicitly enabled in config, to read the proxy's URL
from the environment (PROXY_URL env var) and export it as HTTP_PROXY/HTTPS_PROXY for
this process -- nemo_rl's own actor creation propagates it to every Ray actor's
runtime_env from there.
"""

import logging
import os
from typing import NotRequired, TypedDict

logger = logging.getLogger(__name__)


class ProxyConfig(TypedDict):
    enabled: bool
    url_env_var: NotRequired[str]


DEFAULT_PROXY_CONFIG: ProxyConfig = {
    "enabled": False,
    "url_env_var": "PROXY_URL",
}

# [FIXED 2026-11-04] Ports this loudly: our old 0.6.0-era fork's proxy.py (domyn-rl, not this
# repo) already carries this exact fix dated [CUSTOM-NOPROXY-WILDCARD-HOST 2026-10-24] -- it was
# simply never carried over to this repo's own copy of the file during the version bump. Root
# cause (confirmed independently via a real squidward outage before finding the existing fix):
# our forward proxy (squidward, see its own squid.conf ACLs) only allows ports 80/443 -- any
# intra-cluster call (NemoGym's own head-server liveness check, a locally-spawned subprocess
# like brave_search's MCP server) runs on some other port and gets rejected outright, proxy
# healthy or not. Without this exemption, EVERY such call depends on squidward being reachable
# for no reason -- the concrete symptom is NemoGym hanging forever on "Head server is not up
# yet (status `connection_error`)" instead of just connecting directly.
#   - 0.0.0.0 is NOT a general networking standard -- it's specific, already-diagnosed knowledge
#     about NemoGym's own code: its head server binds to 0.0.0.0 and then polls that literal
#     string (not a resolved real IP) via requests.get("http://0.0.0.0:<port>") to check
#     readiness. Without this entry that poll goes through the proxy, which cannot forward to
#     0.0.0.0, causing exactly the hang above.
#   - 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16 are the full RFC1918 private-address space, not
#     just the one block (10.0.0.0/8) this particular cluster happens to use (confirmed via
#     squidward's own squid.conf: `acl hpc_nodes src 10.0.0.0/8`) -- covering all three is free
#     (none of them are ever real internet destinations) and keeps this correct even if this
#     cluster's internal addressing ever changes, or this code runs somewhere else.
# NOTE: NO_PROXY is not a universal fix by itself -- `requests` (used by
# nemo_gym/server_utils.py's ServerClient) has real CIDR-aware NO_PROXY matching (confirmed via
# its own is_valid_cidr/address_in_network), but `httpx` (used by brave_search's MCP client)
# does NOT despite its docstring's claim -- a CIDR entry there falls through to being treated
# as a literal domain-suffix wildcard and never matches a real IP (confirmed empirically). This
# only helps requests-based call sites; httpx-based local clients need trust_env=False at
# construction time instead (see brave_search/app.py's _connect_mcp_session).
_NO_PROXY_EXEMPTIONS = (
    "localhost,127.0.0.1,::1,0.0.0.0,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
)


def maybe_enable_proxy(config: dict) -> None:
    """Export HTTP_PROXY/HTTPS_PROXY from the configured env var, if `config["proxy"]["enabled"]`.

    No-op if the `proxy` block is absent or `enabled` is false. Call this once, early
    in the main training script, before anything that needs outbound internet access.
    """
    proxy_config: ProxyConfig = {**DEFAULT_PROXY_CONFIG, **config.get("proxy", {})}
    if not proxy_config["enabled"]:
        return

    url_env_var = proxy_config["url_env_var"]
    proxy_url = os.environ.get(url_env_var)
    if not proxy_url:
        raise ValueError(
            f"proxy.enabled=true requires {url_env_var} to be set (e.g. in .env, see "
            ".env.example) -- the forward proxy itself is a standing job managed outside this "
            "repo; this only needs its address."
        )

    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url

    for var in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(var, "")
        entries = [e for e in existing.split(",") if e] + _NO_PROXY_EXEMPTIONS.split(",")
        # dedup, preserve order
        os.environ[var] = ",".join(dict.fromkeys(entries))

    logger.info(
        "Forward-proxy enabled: HTTP_PROXY/HTTPS_PROXY exported from %s, "
        "NO_PROXY/no_proxy exempts %s (plus any pre-existing entries).",
        url_env_var,
        _NO_PROXY_EXEMPTIONS,
    )
