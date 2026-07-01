# Dynamo examples

These examples require the Dynamo operator and `DynamoGraphDeployment` CRD to
already be installed in the target cluster. `nrl-k8s` does not install them.

Each versioned directory contains the three files needed to run one NeMo-RL
example with Dynamo on Kubernetes:

- `<name>.yaml`: the NeMo-RL recipe.
- `<name>.<platform>.infra.yaml`: the Ray and Kubernetes topology. Its
  `dynamoGraphDeployment.manifest` field references the DGD file in the same
  directory.
- `<name>.dgd.yaml`: the DynamoGraphDeployment (DGD), including the frontend
  and vLLM worker configuration.

The current examples are:

- `V1`: Qwen2.5 1.5B DTensor GRPO on the math task.
- `V2`: Llama 3.1 8B Instruct Megatron async GRPO.
- `V3`: Qwen2.5 1.5B Instruct DTensor GRPO on the sliding-puzzle task.
- `V5`: Nemotron Nano v2 9B Megatron GRPO on the workplace-assistant task.

All four examples use Dynamo direct generation and vLLM's native NCCL weight
transfer on GB300. See each infra file's header for validation and launch
commands.
