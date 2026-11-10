# Captured TITO

Captured TITO reuses Gym's source-history matching and NeMo RL's exact token capture to construct subsequent model inputs from a stored prefix plus newly encoded observations. The initial implementation supports the official Qwen3-0.6B chat template on native async vLLM 0.29.0. It renders only appended user/tool messages, including grouped tool responses and the assistant opener. It preserves captured reasoning and tool-call tokens even when the harness's text representation differs.

## Configuration

| Key | Values | Behavior |
| --- | --- | --- |
| `policy.generation.vllm_cfg.tito_model` | `null` (default), `qwen3` | Enable the qualified incremental suffix renderer. |
| `token_capture.enabled` | `true` | Use existing worker-side capture and storage. |
| `token_capture.forest_training` | `true` | Use existing source-history matching and returned-history training selection. |

The generation backend must be native async vLLM with its HTTP server exposed. The opt-in path requires a candidate from Gym's source-history matcher; it does not trust assistant token metadata as a substitute for that proof. Uncaptured requests and ordinary fingerprint capture retain the existing renderer.

The template is qualified by its SHA-256, not by a model name substring. Custom templates, non-text requests, unfinished assistant generations, tokenization-only requests, prompt truncation, and unsupported request options use the existing full-history path. Invalid capture evidence and storage failures remain errors. The module checks the vLLM version because it delegates to the versioned OnlineRenderer interface.

## Gym and training compatibility

Gym continues to own history matching, identities, render-option compatibility and accepted-output evidence. RL continues to own converted-message boundary validation, original token retrieval and capture registration. The request-local renderer replaces only the text-to-input construction inside vLLM preprocessing; native request adjustment, token validation and engine preparation still execute.

Successful continuations record exactly the prompt submitted to the engine with the original `token_in` parent and prefix. The synthetic newline and user/tool observations are conditioning context with zero loss and zero action logprob. Generated spans retain their sampled logprobs and loss masks. Existing capture finalization selects returned-history calls, reconstructs training rows and counts shared generated ancestors once. This change introduces no new receipt schema, reward policy or training-row format.

The default path is unchanged. Cross-session history matching, a new matcher, a new capture transport and broader model/template support are outside this initial implementation. Miles provides the reference behavior (checkpoint reuse plus model-specific incremental suffix construction); existing NeMo RL/Gym mechanisms supply checkpoint selection and training ownership. This overlaps with PR #3581, but uses capture-selected prefixes and a qualified local template adapter rather than echoed token metadata and full marked-history rendering.

## SWE compatibility smoke

Use `examples/nemo_gym/grpo_qwen3_0_6b_swe_tito.yaml`: Qwen/Qwen3-0.6B, one learner GPU and one vLLM GPU, two short GRPO steps, and real multi-turn SWE-bench Verified tasks through Gym's sandboxed OpenCode agent. The small model checks protocol/data compatibility; successful issue resolution is not an acceptance criterion. These tasks are for smoke validation, not a benchmark score or training-quality claim.

1. Initialize the pinned Gym and training dependencies according to the normal NeMo RL setup. Configure the Gym OpenSandbox provider and required sandbox images/agent binaries; the recipe does not provision infrastructure.
2. In the pinned Gym checkout, prepare the official rows:

   ```bash
   uv run gym eval prepare --config benchmarks/swebench/verified/opencode.yaml
   ```

3. Select a few prepared rows and add the agent reference consumed by NemoGymDataset:

   ```bash
   head -n 4 benchmarks/swebench/data/swebench_verified_benchmark.jsonl | \
     jq -c '.agent_ref = {type: "responses_api_agents", name: "swebench_verified_opencode_sandboxed_agent"}' \
     > /path/to/swe_tito_smoke.jsonl
   ```

4. From NeMo RL, run in the normal two-GPU environment:

   ```bash
   NRL_LOG_LEVEL=DEBUG uv run examples/run_grpo_single_controller.py \
     --config examples/nemo_gym/grpo_qwen3_0_6b_swe_tito.yaml \
     data.train.data_path=/path/to/swe_tito_smoke.jsonl \
     data.validation.data_path=/path/to/swe_tito_smoke.jsonl
   ```

Confirm that the worker logs `Captured Qwen3 TITO reused ... prefix tokens`; a successful rollout alone may have used the native fallback. Inspect accepted multi-turn calls, actual prompt IDs, reconstructed masks/logprobs and the resulting training batch. If the model produces no tool calls, the run has not exercised incremental TITO. Compare against the same recipe with `policy.generation.vllm_cfg.tito_model=null`. Ensure at least one matched continuation before drawing any compatibility or performance conclusion.

## Validation scope

The focused CPU tests use the real Qwen tokenizer, compare incremental prompts against the existing splice path, verify only suffix text is encoded, and round-trip two synthetic SWE-style calls through Gym's production capture and receipt reconstruction. The optional vLLM parser boundary is substituted in the adapter unit tests.

A separate real run used Qwen3-0.6B, vLLM 0.29.0, two H100 GPUs and two SWE-bench Verified astropy tasks through sandboxed OpenCode. Both the enabled recipe and the `tito_model=null` control completed two training steps. The enabled run recorded five incremental renderer hits, reusing 8,216–8,358 prefix tokens across real tool continuations. The control retained existing capture-selected `token_in` behavior without incremental renderer hits. These short sampled runs establish path compatibility, not output equivalence, speedup or issue-resolution quality.

That validation used Apptainer with an independent Gym compatibility patch to propagate serialized sandbox descriptors from the SWE resource server to OpenCode. The pinned Gym submodule is unchanged; this is not evidence that an unpatched Apptainer deployment works. The default recipe still requires configured OpenSandbox infrastructure. Full supported-environment CI remains required before marking the PR ready.
