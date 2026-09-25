# Overview

**NeMo RL** is an open-source post-training library within the [NeMo Framework](https://github.com/NVIDIA-NeMo), designed to streamline and scale reinforcement learning methods for multimodal models (LLMs, VLMs, etc.). Designed for flexibility, reproducibility, and scale, NeMo RL enables both small-scale experiments and massive multi-GPU, multi-node deployments for fast experimentation in research and production environments.

## What You Can Expect

- **Flexibility** with a modular design that allows easy integration and customization.
- **Efficient resource management using Ray**, enabling scalable and flexible deployment across different hardware configurations.
- **Hackable** with native PyTorch-only paths for quick research prototypes.
- **High performance with Megatron Core**, supporting various parallelism techniques for large models and large context lengths.
- **Seamless integration with Hugging Face** for ease of use, allowing users to leverage a wide range of pre-trained models and tools.
- **Comprehensive documentation** that is both detailed and user-friendly, with practical examples.

For more details on the architecture and design philosophy, see the [design documents](../design-docs/design-and-philosophy.md).

## NeMo Gym Integration

NeMo RL and [NeMo Gym](https://docs.nvidia.com/nemo/gym/) provide complementary parts of an environment-driven post-training workflow:

- **NeMo RL** runs scalable model training, including policy updates, rollout generation, and checkpointing.
- **NeMo Gym** defines reusable datasets, tools, verifiers, and single-step or multi-step interactions. Verifier scores can be used as evaluation metrics or as rewards during training.

Together, NeMo RL serves the model for rollouts and trains it from the rewards produced by NeMo Gym environments. See the [NeMo Gym integration guide](../design-docs/nemo-gym-integration.md) for supported algorithms, configuration, and architecture details.

## Releases

For a complete list of releases and detailed changelogs, visit the [GitHub Releases page](https://github.com/NVIDIA-NeMo/RL/releases).
