# External rollout vLLM prototype

This prototype runs async Single Controller GRPO with rollout generation in a
vLLM server outside NeMo-RL's training Ray cluster.

The launch topology is:

```text
one Slurm heterogeneous job
├── hetgroup 0: 4-node Megatron policy + 1-node Gym safety judge
└── hetgroup 1: policy rollout vLLM + GenRM + NL2Bash pools
```

A site-specific launcher can reuse
`tools/external_gym_vllm/run_in_allocation.sh`. It should start the external
pools, wait for health, substitute their OpenAI base URLs into the training
command, launch `ray.sub` only on the training hetgroup, and tear down both
sides when either exits.

## Stock vLLM API usage

The launcher sets `VLLM_SERVER_DEV_MODE=1`, which enables vLLM's development
control endpoints. The controller uses:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Verify the engine is live. |
| `GET` | `/v1/models` | Verify the served model alias. |
| `GET` | `/server_info` | Verify development endpoints are enabled. |
| `POST` | `/pause?mode=keep&clear_cache=true` | Freeze in-flight work and clear stale KV state before refit. |
| `POST` | `/collective_rpc` | Invoke `reload_weights(weights_path=...)` on all engine workers. |
| `POST` | `/resume` | Resume request processing after a successful reload. |

NeMo-Gym sends rollout requests directly to the server's standard
OpenAI-compatible API. During refit, Megatron-Bridge exports the live policy to
a unique HF checkpoint directory on shared storage. vLLM then reloads that
directory globally through `/collective_rpc`.

The `keep` pause mode preserves pending HTTP requests across the refit. Cache
clearing moves active requests back to vLLM's waiting queue, so they recompute
their prefix with the new weights after `/resume` rather than retaining stale
KV state.

These development endpoints are powerful and must remain on the job's private
network. A failed reload leaves the server paused because some workers may
already contain the new version.

## Multi-turn prefix-token extension

Stock vLLM can return prompt and generation token IDs, but it does not accept
NeMo Gym's `required_prefix_token_ids` request field. Multi-turn workloads that
need exact token continuity load the endpoint plugin under
`nemo_rl_vllm_prefix_plugin/`. The plugin is pinned to vLLM 0.29.0, shadows
only `/v1/chat/completions`, and delegates the rest of request handling to the
stock vLLM serving implementation.

Build the pure-Python wheel onto shared storage and expose it to the vLLM API
server process:

```bash
PLUGIN_WHEEL=$(tools/external_rollout_vllm/build_prefix_plugin.sh /shared/plugin)
export PYTHONPATH="${PLUGIN_WHEEL}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_PLUGINS=nemo_rl_prefix_api
vllm serve ...
```

The wheel path on `PYTHONPATH` supplies both the Python package and its entry
point metadata; nothing is installed into the container. Check activation at
`GET /v1/nemo-rl/prefix-token-capability`.

The legacy inline-prefix mode enables token IDs in both directions:

```yaml
return_token_id_information: true
request_prompt_and_generation_token_ids: true
supply_prefix_token_ids: true
```

For SingleController external staging, enable `token_capture.enabled=true`
instead. NeMo RL then configures every plugin backend with a controller-hosted
staging bridge. Gym sends `ng_capture` admissions, and the plugin returns
`ng_commit_coords` only after the exact token delta is durable in TransferQueue.
This mode sets `return_token_id_information=false` and
`supply_prefix_token_ids=false`; the staged delta is the authoritative token
transport.

## Launching the Nano 3.5 RLVR smoke

The smoke configuration is
`examples/nemo_gym/nemotron-3.5-nano/rlvr_sc_smoke_small_external_vllm.yaml`.
A launcher must supply deployment-specific artifact locations rather than
placing them in Python or YAML:

```bash
export EXTERNAL_ROLLOUT_VLLM_URL=http://rollout-service:8000/v1
export EXTERNAL_ROLLOUT_HF_EXPORT_DIR=/shared/path/to/hf_exports

uv run examples/run_grpo_external_vllm_single_controller.py \
  --config examples/nemo_gym/nemotron-3.5-nano/rlvr_sc_smoke_small_external_vllm.yaml \
  policy.model_name=/shared/path/to/policy \
  data.train.data_path=/shared/path/to/train.jsonl \
  data.validation.data_path=/shared/path/to/validation.jsonl
```

Container images, optional prebuilt Gym environments, reward-model checkpoints,
and cluster scheduler settings likewise belong in the site-specific launcher.
External vLLM servers can use the prebuilt `VllmAsyncGenerationWorker` virtual
environment from the NeMo RL image.

HF refit exports default to `$BASE_LOG_DIR/hf_exports`. Override
`EXTERNAL_ROLLOUT_HF_EXPORT_DIR` when needed. That path must be mounted at the
same absolute location in both containers. Version directories are retained
for this prototype so vLLM never reloads an overwritten path.

## Scaling boundary

One native vLLM deployment may contain multiple TP/PP/DP workers;
`/collective_rpc` applies to every worker owned by that engine. A separate
router in front of multiple independent `vllm serve` deployments does not make
control calls global. This launcher therefore exposes a separate Python
control-plane proxy that fans lifecycle and refit calls out to every backend,
while generation remains on the Rust router. Preflight accepts multiple
backends only when that control proxy reports `control_fanout=true`.

## AnyTerminal multi-harness deployment

`launch_anyterminal_multi_harness.sh` is the full-scale Terminal-Bench entry
point. It combines the external rollout backend with the Gym multi-harness
fan-out and launches:

- eight 4-GPU policy-training nodes;
- 16 independent TP1 vLLM servers, packed four per rollout node;
- `vllm-router` with `consistent_hash` session routing;
- a 196,608-token model context, prefix caching, and async vLLM scheduling;
- 128 prompt groups x 16 generations = 2,048 rollouts per GRPO step; and
- Kubernetes task containers through Gym's OpenSandbox profile.

The 128 prompt groups are 32 source Terminal-Bench rows fanned out to OpenCode,
OpenClaw, Pi, and Hermes. Each source row therefore runs through all four
harnesses; this recipe does not randomly select one harness.

Build the prefix plugin, provide a compatible `vllm-router` wheel, and point the
launcher at the two PR checkouts:

```bash
export MODEL_PATH=/shared/checkpoints/policy
export TRAIN_PATH=/shared/data/terminal_bench_multi_harness.train.jsonl
export CONTAINER=/shared/images/nemo-rl.sqsh
export ROLLOUT_VLLM_CONTAINER=/shared/images/vllm.sqsh
export PREFIX_PLUGIN_WHEEL=/shared/wheels/nemo_rl_vllm_prefix_plugin.whl
export VLLM_ROUTER_WHEEL=/shared/wheels/vllm_router.whl
export NEMO_GYM_ROOT=/shared/checkouts/Gym
export OPENSANDBOX_DOMAIN=https://opensandbox.example
export OPENSANDBOX_API_KEY=...
tools/external_rollout_vllm/launch_anyterminal_multi_harness.sh
```

For a non-submitting validation, set `DRY_RUN=1`. The launcher prints the exact
NeMo RL and Gym commit IDs before submission; record those two IDs with the W&B
run. A commit cannot pin its own eventual hash, so deployment automation should
checkout the reviewed PR heads by SHA before invoking the launcher.

The external rollout implementation in this branch was integrated from Guyue
Huang's public `codex/external-vllm-rollout` series, through source commit
`2b82e71ac41a26a883e2054ad61cc0a66a7d023b`, on top of NeMo RL main commit
`d8b376aa9f48bebb19e8fab0860a4784a0666847`. The older OpenCode-only run used
NeMo RL `5e3d27cbd0208f9abc4c1d8f414d306c1f6af7c4` and Gym
`12678a69942a1482b002da55b15451d291b258c4`; those hashes are provenance only
and must not be used for this four-harness recipe.
