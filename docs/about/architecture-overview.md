---
description: "High-level architecture overview of the modular, scalable design of NeMo RL for distributed reinforcement learning with Ray-based coordination"
categories: ["concepts-architecture"]
tags: ["architecture-overview", "distributed", "ray", "virtual-cluster", "worker-groups", "reinforcement-learning", "scalability"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "universal"
---

# Architecture Overview

NeMo RL uses a modular, scalable architecture designed to handle the complexities of distributed reinforcement learning while maintaining simplicity and flexibility. This overview provides a high-level understanding of NeMo RL architectural components and design principles.

## Design Philosophy

NeMo RL coordinates various software components (RL Actors) through a unified interface that handles resource allocation, isolation, coordination, and communication. This design enables seamless scaling from 1 to 1,000+ graphics processing units (GPUs) while remaining independent of specific RL Actor implementations.

For detailed design philosophy, implementation specifics, and advanced architectural concepts, see the [Core Design](../core-design/index) documentation.

## Core Components

### Reinforcement Learning Actors

RL Actors are the fundamental building blocks of NeMo RL. Each actor represents a specific component of the RL system:

- **Policy Model/Training Framework**: Handles model training and updates
- **Generation Framework**: Provides high-throughput inference using the vLLM back end
- **Reward Environments**: Implements reward computation and environment simulation
- **Critics**: Evaluates model performance and provides feedback

### Virtual Cluster

The `RayVirtualCluster` manages resource allocation and provides a unified interface for accessing distributed resources:

- **Dynamic Resource Allocation**: Allocates graphics processing units (GPUs), central processing units (CPUs), and memory as needed
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

### Core Components Diagram

```{mermaid}
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#E8F4F8','primaryTextColor':'#000','primaryBorderColor':'#0066CC','lineColor':'#333','secondaryColor':'#FFFDE7','secondaryTextColor':'#000','secondaryBorderColor':'#FFC107','tertiaryColor':'#F0F0F0','tertiaryTextColor':'#000','tertiaryBorderColor':'#666'}}}%%
graph TB
  Controller["Controller (single process)"]:::controllerStyle

  subgraph cluster["RayVirtualCluster"]
    subgraph workers["Worker Groups"]
      Policy["RL Actor: Policy / Trainer"]:::actorStyle
      Generation["RL Actor: Generation (vLLM)"]:::actorStyle
      Critics["RL Actor: Critics"]:::actorStyle
      Envs["RL Actor: Reward Environments"]:::actorStyle
    end
  end

  Controller --> Policy
  Controller --> Generation
  Controller --> Critics
  Controller --> Envs

  classDef controllerStyle fill:#4A90E2,stroke:#0066CC,stroke-width:3px,color:#fff
  classDef actorStyle fill:#FFF9C4,stroke:#FDD835,stroke-width:2px,color:#000
```

*High-level view of the `Controller`, `RL Actors`, `RayWorkerGroup`s, and `RayVirtualCluster`.*

## Communication Architecture

Data flows through multiple communication channels:

### Direct Controller Communication

- **Synchronous Communication**: Direct method calls to the controller
- **State Synchronization**: Centralized state management
- **Configuration Updates**: Dynamic configuration changes

### Distributed Communication

- **NCCL Collectives**: High-performance GPU-to-GPU communication
- **Multi-process queues**: Inter-process communication for data transfer
- **Ray Object Store**: Efficient data sharing between actors

## Training Back Ends

NeMo RL supports several training back ends to accommodate different model sizes and requirements:

| Back end        | Model Support | Key Features                                   | Best Use Case              |
|----------------|---------------|------------------------------------------------|----------------------------|
| **Hugging Face** | 1-32B parameters | Easy integration, custom architectures | Broad model support        |
| **Megatron Core** | Up to 70B+ parameters | Advanced parallelism, long context | Large-scale models         |
| **DTensor (FSDP2)** | Variable | Memory efficiency, PyTorch native | Memory-constrained setups  |

Back ends are configurable and interchangeable without altering core algorithm logic.

## Data Flow

### Training Pipeline

1. **Data Loading**: Load and prepare training data
2. **Generation**: Generate responses using the current policy
3. **Evaluation**: Compute rewards and metrics
4. **Loss Computation**: Calculate training loss with proper normalization
5. **Backward Pass**: Update model parameters
6. **Synchronization**: Synchronize weights across workers

#### Training Flow Diagram

```{mermaid}
%%{init: {'theme':'base', 'themeVariables': { 'actorBkg':'#4A90E2','actorBorder':'#0066CC','actorTextColor':'#fff','actorLineColor':'#0066CC','signalColor':'#333','signalTextColor':'#000','labelBoxBkgColor':'#F39C12','labelBoxBorderColor':'#D68910','labelTextColor':'#000','loopTextColor':'#000','activationBorderColor':'#666','activationBkgColor':'#E8F4F8','sequenceNumberColor':'#fff'}}}%%
sequenceDiagram
  participant Client
  participant Controller
  participant Generator as "Generation (vLLM)"
  participant Critics
  participant Policy as "Policy/Trainer"

  Client->>Controller: Start training step
  Controller->>Generator: Generate samples
  Generator-->>Controller: Sample batches
  Controller->>Critics: Compute rewards/metrics
  Critics-->>Controller: Rewards/metrics
  Controller->>Policy: Compute loss + backward
  Policy-->>Controller: Weights updated & synchronized (NCCL)
```

*Sequence of a training step from generation through rewards to weight synchronization.*

### Inference Pipeline

1. **Request Handling**: Receive inference requests
2. **Load Balancing**: Distribute requests across workers
3. **Generation**: Generate responses using vLLM or other back ends
4. **Response Collection**: Gather and format responses
5. **Weight Updates**: Update model weights if needed

## Scalability Design

### Horizontal Scaling

- **Multi-Node Support**: Scale across multiple machines
- **Load Balancing**: Automatic load distribution
- **Fault Tolerance**: Automatic recovery from failures

### Vertical Scaling

- **Multi-GPU Support**: Use several GPUs per node
- **Memory Optimization**: Efficient memory usage patterns
- **Parallelism**: Several types of parallelism for optimal performance

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
- **Resource Metrics**: GPU, CPU, and memory use
- **Custom Metrics**: User-defined metrics and key performance indicators (KPIs)

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

This overview provides a high-level understanding of the architecture of NeMo RL. For detailed design philosophy, implementation specifics, and advanced architectural concepts, see the [Core Design](../core-design/index) documentation.