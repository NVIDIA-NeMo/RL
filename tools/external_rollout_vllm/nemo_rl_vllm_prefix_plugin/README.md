# NeMo RL vLLM prefix-token plugin

This package extends the vLLM 0.29.0 OpenAI chat endpoint with NeMo RL's
multi-turn request fields:

```json
{"required_prefix_token_ids": [1, 2, 3]}
```

When present, the API server preserves those exact token IDs through the last
assistant turn and appends only the newly rendered suffix. This avoids
retokenization drift in multi-turn RL trajectories.

For ledger-authoritative capture, the request may instead carry Gym's
`ng_capture` admission. The controller configures the plugin with a private HTTP
bridge to TransferQueue. The plugin fetches staged prefix deltas, uses them as
the exact engine prefix, and commits the completed token delta before returning
`ng_commit_coords` to Gym. This works even though the external vLLM deployment
and the NeMo RL controller belong to separate Ray clusters.

The package intentionally pins vLLM 0.29.0 because it wraps internal serving
and renderer objects that are not part of vLLM's stable plugin interface. Load
it explicitly:

```bash
export VLLM_PLUGINS=nemo_rl_prefix_api
vllm serve ...
```

The plugin deliberately shadows `/v1/chat/completions`. Its implementation
otherwise delegates response generation, tool parsing, and reasoning parsing
to the stock vLLM serving handler. Ledger capture is non-streaming, matching
Gym's staging contract.
