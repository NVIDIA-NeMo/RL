# NeMo-RL vLLM endpoint plugin

This package extends vLLM 0.29.0 chat completions with exact multi-turn prefix
tokens and external token capture. It preserves the stock serving handler and
loads through the `vllm.endpoint_plugins` entry point `nemo_rl_prefix_api`.

See the [external vLLM guide](../../../docs/guides/external-vllm.md) for building
the wheel, activating the plugin, choosing inline versus staged tokens, and
running the functional tests.
