---
description: "High-level architecture overview of NeMo RL's modular, scalable design for distributed reinforcement learning with Ray-based coordination"
categories: ["concepts-architecture"]
tags: ["architecture-overview", "distributed", "ray", "virtual-cluster", "worker-groups", "reinforcement-learning", "scalability"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "universal"
---

# Core Architecture

NeMo RL is built on a modular, scalable architecture designed to handle the complexities of distributed reinforcement learning while maintaining simplicity and flexibility. This overview provides a high-level understanding of NeMo RL's architectural components and design principles.

## Design Philosophy

NeMo RL coordinates various software components (RL Actors) through a unified interface that handles resource allocation, isolation, coordination, and communication. This design enables seamless scaling from 1 to 1000+ GPUs while remaining independent of specific RL Actor implementations.

For detailed design philosophy, implementation specifics, and advanced architectural concepts, see the [Core Design and Architecture](../core-design/index) documentation.

## Core Components

### RL Actors

RL Actors are the fundamental building blocks of NeMo RL. Each actor represents a specific component of the RL system:

- **Policy Model/Training Framework**: Handles model training and updates
- **Generation Framework**: Provides high-throughput inference using VLLM backend
- **Reward Environments**: Implements reward computation and environment simulation
- **Critics**: Evaluates model performance and provides feedback

### Virtual Cluster

The `RayVirtualCluster` manages resource allocation and provides a unified interface for accessing distributed resources:

- **Dynamic Resource Allocation**: Allocates GPUs, CPUs, and memory as needed
- **Scalability**: Supports from single-node to multi-node clusters
- **Flexibility**: Works with various cluster managers (Slurm, Kubernetes, etc.)

### Worker Groups

`RayWorkerGroup` provides process isolation and dependency management:

- **Process Isolation**: Each RL Actor runs in its own isolated process
- **Dependency Management**: Configurable dependencies to avoid conflicts
- **Resource Mapping**: Maps actors to specific hardware resources

### Controller

A single-process controller coordinates all RL Actors:

- **Centralized Coordination**: Manages the lifecycle of all actors
- **Ray Integration**: Uses Ray for distributed coordination
- **State Management**: Maintains global state and synchronization

## Communication Architecture

Data flows through multiple communication channels:

### Direct Controller Communication
- **Synchronous Communication**: Direct method calls to the controller
- **State Synchronization**: Centralized state management
- **Configuration Updates**: Dynamic configuration changes

### Distributed Communication
- **NCCL Collectives**: High-performance GPU-to-GPU communication
- **Multiprocess Queues**: Inter-process communication for data transfer
- **Ray Object Store**: Efficient data sharing between actors

## Training Backends

NeMo RL supports multiple training backends to accommodate different model sizes and requirements:

| Backend         | Model Support | Key Features                                   | Best Use Case              |
|----------------|---------------|------------------------------------------------|----------------------------|
| **Hugging Face** | 1-32B parameters | Easy integration, custom architectures | Broad model support        |
| **Megatron Core** | Up to 70B+ parameters | Advanced parallelism, long context | Extremely large models     |
| **DTensor (FSDP2)** | Variable | Memory efficiency, PyTorch native | Memory-constrained setups  |

Backends are configurable and interchangeable without altering core algorithm logic.

## Data Flow

### Training Pipeline
1. **Data Loading**: Load and preprocess training data
2. **Generation**: Generate responses using the current policy
3. **Evaluation**: Compute rewards and metrics
4. **Loss Computation**: Calculate training loss with proper normalization
5. **Backward Pass**: Update model parameters
6. **Synchronization**: Synchronize weights across workers

### Inference Pipeline
1. **Request Handling**: Receive inference requests
2. **Load Balancing**: Distribute requests across workers
3. **Generation**: Generate responses using vLLM or other backends
4. **Response Collection**: Gather and format responses
5. **Weight Updates**: Update model weights if needed

## Scalability Design

### Horizontal Scaling
- **Multi-Node Support**: Scale across multiple machines
- **Load Balancing**: Automatic load distribution
- **Fault Tolerance**: Automatic recovery from failures

### Vertical Scaling
- **Multi-GPU Support**: Utilize multiple GPUs per node
- **Memory Optimization**: Efficient memory usage patterns
- **Parallelism**: Multiple types of parallelism for optimal performance

## Configuration System

### YAML Configuration
- **Hierarchical Structure**: Nested configuration options
- **Type Safety**: Comprehensive type checking
- **Validation**: Automatic configuration validation

### CLI Overrides
- **Flexible Overrides**: Override any configuration parameter
- **Dot Notation**: Easy access to nested parameters
- **Type Conversion**: Automatic type conversion for overrides

## Monitoring and Observability

### Logging
- **Structured Logging**: JSON-formatted logs for easy parsing
- **Multi-Level**: Debug, info, warning, and error levels
- **Context Preservation**: Maintain context across distributed components

### Metrics
- **Performance Metrics**: Training and inference performance
- **Resource Metrics**: GPU, CPU, and memory utilization
- **Custom Metrics**: User-defined metrics and KPIs

### Visualization
- **WandB Integration**: Real-time experiment tracking
- **TensorBoard Support**: Training visualization
- **Custom Dashboards**: User-defined monitoring dashboards

## Security and Isolation

### Process Isolation
- **Independent Processes**: Each actor runs in its own process
- **Resource Isolation**: Separate memory and CPU allocation
- **Dependency Isolation**: Independent dependency environments

### Data Security
- **Secure Communication**: Encrypted communication channels
- **Access Control**: Role-based access control
- **Audit Logging**: Comprehensive audit trails

## Extensibility

### Plugin Architecture
- **Custom Actors**: Easy addition of new RL Actors
- **Custom Environments**: Support for custom environments
- **Custom Algorithms**: Implementation of new RL algorithms

### API Design
- **Interface-Based**: Clean interfaces for all components
- **Type Hints**: Full type annotation support
- **Documentation**: Comprehensive API documentation

## Further Reading

This overview provides a high-level understanding of NeMo RL's architecture. For detailed design philosophy, implementation specifics, and advanced architectural concepts, see the [Core Design and Architecture](../core-design/index) documentation. 