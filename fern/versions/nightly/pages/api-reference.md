---
title: "API Reference"
---

This API reference documents the public Python interfaces of NeMo RL. It is generated from the docstrings and type annotations in the [`nemo_rl`](https://github.com/NVIDIA-NeMo/RL/tree/main/nemo_rl) package.

## Key Modules

<table>
  <thead>
    <tr>
      <th>Module</th>
      <th>What it contains</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{ whiteSpace: 'nowrap' }}><code>nemo_rl.algorithms</code></td>
      <td>Training algorithms (GRPO, PPO, DPO, SFT, reward modeling, and distillation), their loss functions, and advantage estimators</td>
    </tr>
    <tr>
      <td style={{ whiteSpace: 'nowrap' }}><code>nemo_rl.data</code></td>
      <td>Datasets, processors, chat templates, collation, and dataloaders</td>
    </tr>
    <tr>
      <td style={{ whiteSpace: 'nowrap' }}><code>nemo_rl.data_plane</code></td>
      <td>A backend-agnostic client for moving rollout and training data between algorithms, rollout workers, and training workers</td>
    </tr>
    <tr>
      <td style={{ whiteSpace: 'nowrap' }}><code>nemo_rl.distributed</code></td>
      <td>Ray worker groups, virtual clusters, batched data containers, and collective utilities</td>
    </tr>
    <tr>
      <td style={{ whiteSpace: 'nowrap' }}><code>nemo_rl.environments</code></td>
      <td>The environment interface and the built-in environments, such as math, code, games, and reward-model scoring, that return rewards for model outputs</td>
    </tr>
    <tr>
      <td style={{ whiteSpace: 'nowrap' }}><code>nemo_rl.evals</code></td>
      <td>Environment-based evaluation with pass@k and cons@k metrics, plus answer parsing</td>
    </tr>
    <tr>
      <td style={{ whiteSpace: 'nowrap' }}><code>nemo_rl.experience</code></td>
      <td>Multi-turn rollouts that run generation backends against environments, plus the rollout actors and managers that drive them</td>
    </tr>
    <tr>
      <td style={{ whiteSpace: 'nowrap' }}><code>nemo_rl.modelopt</code></td>
      <td>Quantization-aware policy and generation workers built on NVIDIA Model Optimizer</td>
    </tr>
    <tr>
      <td style={{ whiteSpace: 'nowrap' }}><code>nemo_rl.models</code></td>
      <td>Policy, value, and generation backends, including the Hugging Face, Automodel, Megatron, vLLM, and SGLang integrations</td>
    </tr>
    <tr>
      <td style={{ whiteSpace: 'nowrap' }}><code>nemo_rl.telemetry</code></td>
      <td>Optional OpenTelemetry tracing and metrics</td>
    </tr>
    <tr>
      <td style={{ whiteSpace: 'nowrap' }}><code>nemo_rl.utils</code></td>
      <td>Configuration, logging, checkpointing, timers, memory and FLOPs tracking, and other shared utilities</td>
    </tr>
    <tr>
      <td style={{ whiteSpace: 'nowrap' }}><code>nemo_rl.weight_sync</code></td>
      <td>Transferring model weights between training and generation components over several transports</td>
    </tr>
  </tbody>
</table>
