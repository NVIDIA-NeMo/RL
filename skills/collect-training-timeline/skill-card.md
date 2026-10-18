## Description: <br>
Playbook and tools for collecting, validating, analyzing, archiving, and rendering end-to-end NeMo-RL training timelines: trainer phases, rollout lifecycle, NeMo Gym/OpenHands model calls with per-turn token, prefix-cache and LLM-server scheduling metrics, tool actions, and runtime lifecycle. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to add timeline tracing to a NeMo-RL GRPO training or generation-only run with NeMo Gym/OpenHands rollouts, validate the resulting JSONL traces, extract per-step and per-turn metrics (prefix-cache reuse, scheduler queueing, TPOT, generation-replica attribution), package restorable timeline archives, and render interactive HTML timelines. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>
Risk: Instrumented overlays replace source files inside the training container. <br>
Mitigation: The skill requires revision-specific overlays built from fresh target files, best-effort tracing that never changes control flow, and compile, wiring, and smoke validation before launch. <br>
Risk: Rollout traces could capture sensitive payloads. <br>
Mitigation: The trace contract records only IDs, phase names, status, and numeric metadata, and archives exclude prompts, responses, sandbox contents, credentials, and environment dumps. <br>

## Reference(s): <br>
- [NeMo RL Documentation](https://docs.nvidia.com/nemo/rl/latest/index.html) <br>
- [NeMo RL GitHub Repository](https://github.com/NVIDIA-NeMo/RL) <br>
- [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Code changes, Analysis, Interactive HTML] <br>
**Output Format:** [Markdown with inline bash code blocks; JSONL traces; HTML timelines] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
Pending NVSkills-Eval. <br>

## Evaluation Tasks: <br>
`evals/evals.json` defines 5 tasks (3 positive skill-activation, 2 negative); results pending NVSkills-Eval. <br>

## Evaluation Results: <br>
Pending NVSkills-Eval. <br>

## Testing Completed: <br>
**[ ] Agent Red-Teaming** <br>
**[ ] Network Security** <br>
**[ ] Product Security** <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
