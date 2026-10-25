# External rollout vLLM

See [external vLLM rollout and NCCL refit](../../docs/guides/external-vllm.md)
for configuration, service startup, exact multi-turn prefixes, token capture,
and functional tests.

This directory contains the version-pinned vLLM endpoint plugin and its offline
wheel builder. Deployment-specific scheduler launchers and experiment recipes
are intentionally kept outside the upstream feature.
