# CUA environment and data

## Runtime

Start with the repository's AutoModel/vLLM GPU environment. Add its Gym extra:

```bash
# Run from research/logra; does not change pyproject.toml or uv.lock.
uv sync --project ../.. --locked --inexact --extra automodel --extra nemo_gym --group test
```

The desktop adapter additionally imports these separately installed packages:

- `desktop_env.desktop_env.DesktopEnv`, with AgentENV sandbox support;
- `desktop_env.execution_feedback`, including `CommandExecutionFeedback`;
- `mm_agents.nvidia.nemotron_agent.NemotronAgent`, supporting `rfc0015_v1` prompts.

The benchmark desktop runtime and the model checkpoint are external prerequisites,
not bundled dependencies or automatic downloads. Install their compatible releases
in the driver/desktop-actor environment or add their source root to `PYTHONPATH`.
Make the same runtime, tokenizer and asset paths available on every node. No
changes to NeMo RL, AutoModel or Gym source are required.

For an **already prepared** environment, the native registry can use system Python:

```bash
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
# uv selects this existing interpreter; it does not resolve the workspace again.
uv run --no-project --python /path/to/prepared/python --no-sync run_cua.py \
  --config configs/cua_smoke.yaml osworld.setup_cache=/writable/setup_cache
```

This mode requires the repository, AutoModel, Gym and `research/logra` to be
importable in that interpreter on every node. It does not install dependencies.
The measured runs used Python 3.12.3, PyTorch 2.13.0+cu130, Transformers 5.12.1,
vLLM 0.29.0 and Ray 2.55.x in a prepared ARM64 GB200 container. Workspace Python
3.13.14 and private desktop-package compatibility should be checked separately.

## External assets

The supplied ScaleCUA recipe expects this layout under `OSWORLD_ROOT`:

```text
models/rfc0040_iter881_hf/                         Model and tokenizer
data/scalecua-train/rfc0040-step881/
   train_no5bc63fb9.hsg.jsonl                      Training manifest
   ...                                           Referenced task JSONs
data/osworld-eval/validation_nogdrive_361.jsonl    Evaluation manifest
data/osworld_setup_cache/                        Preprovisioned setup assets
code/Nemo_GYM/responses_api_agents/nemotron_agent/
   ...                                           Referenced evaluation task JSONs
code/RPGA/scripts/osworld/data/
   template_caps.json                            Desktop-template capabilities
   scalecua_task_reqs.json                        Task requirements
```

These names describe the validated external asset export, not files included in
this PR. Other layouts work by overriding the following YAML paths:

| Config key | Required content |
|---|---|
| `policy.model_name`, `policy.tokenizer.name` | Compatible model/tokenizer directory |
| `osworld.train_data`, `osworld.validation_data` | JSONL manifests |
| `osworld.train_task_base`, `osworld.validation_task_base` | Roots for relative task-config paths |
| `osworld.setup_cache` | Writable setup cache, outside the repository |
| `osworld.template_capabilities` | Template-capability JSON with a `templates` mapping |
| `osworld.task_requirements` | Per-task requirements JSON |

Each nonempty manifest line has this structure (the instruction must match the
referenced task JSON):

```json
{"id":"task-id","verifier_metadata":{"instruction":"Open the application and ...","domain":"chrome","task_config_path":"tasks/task-id.json"}}
```

Alternatively, `verifier_metadata.osworld_task` can contain the task JSON directly.
Input order and repeated rows are preserved, allowing deliberate oversampling.
Desktop setup, actions and rewards use the task-specific OSWorld verifier.
Changing the desktop image/template changes the experiment; `osworld-v2` names
a desktop template, not a different benchmark release.

## Sandbox credentials

Put the authorized service hostname and API key in an external private JSON file:

```json
{"endpoint":"your-agentenv-hostname","apikey":"YOUR_KEY"}
```

```bash
chmod 600 /private/sandbox.json
export OSWORLD_CREDENTIAL_FILE=/private/sandbox.json
export AENV_TTL=4500
# Optional when the service uses a private certificate:
export AENV_TLS_CA=/private/agentenv-ca.pem
```

Do not export `AENV_API_KEY` or `E2B_API_KEY` into the model-worker environment.
Only desktop CPU actors read the private file while executing an episode.
The TTL must cover the task timeout plus setup and cleanup. Keep artifacts under
a writable external path using `OSWORLD_TASK_ARTIFACT_ROOT` when needed.

## Outputs and restart

Set a fresh `CUA_OUTPUT_DIR` for every new experiment. It contains `metrics.jsonl`,
TensorBoard events, per-episode artifacts, consumed-ticket manifests in `batches/`,
and native checkpoints in `checkpoints/step_N/`. Checkpoints also retain the
baseline, sampler and unconsumed rollout tickets.

A restart uses `training.resume=/path/to/checkpoints/step_N`, the same parallel
layout and unchanged `cua`/`osworld` settings. You may extend `training.max_steps`
without resetting the KL-annealing schedule. Completed episodes are not replayed;
unconsumed tasks are rerun because live desktops are not checkpointed.
