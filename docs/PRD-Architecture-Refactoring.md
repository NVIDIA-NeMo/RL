# Product Requirements Document: NeMo RL Architecture Refactoring

**Document Version:** 1.0  
**Date:** January 15, 2026  
**Project Duration:** 6 months  
**Target Audience:** Engineering Team, Researchers, Product Management

---

## Executive Summary

NeMo RL is a powerful post-training library for reinforcement learning on large language models, but its current architecture presents significant barriers to adoption by developers and researchers. This PRD outlines a comprehensive 6-month refactoring initiative to improve usability, maintainability, and extensibility while preserving the library's performance characteristics.

### Key Objectives
1. Reduce time-to-first-run from hours to minutes
2. Enable researchers to implement new algorithms in days, not weeks
3. Improve code maintainability and reduce duplication by 40%+
4. Create clear abstraction boundaries between components

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Current Architecture Analysis](#2-current-architecture-analysis)
3. [Target Architecture](#3-target-architecture)
4. [Detailed Requirements](#4-detailed-requirements)
5. [Implementation Phases](#5-implementation-phases)
6. [Success Metrics](#6-success-metrics)
7. [Risks and Mitigations](#7-risks-and-mitigations)
8. [Appendix](#8-appendix)

---

## 1. Problem Statement

### 1.1 User Pain Points

**For Researchers:**
- Implementing a new RL algorithm requires understanding 5+ modules and 10+ files
- The GRPO implementation spans 2600+ lines in a single file
- Configuration requires understanding multiple inheritance patterns and TypedDict schemas
- Modifying training loops requires changes in deeply nested code paths

**For MLOps Engineers:**
- Configuration sprawl: 300+ line YAML files with complex interdependencies
- Unclear separation between algorithm, infrastructure, and model concerns
- Debugging distributed failures requires understanding Ray internals
- Checkpoint format complexity with multiple backends (DCP, Megatron, HF)

**For Application Developers:**
- No simple "hello world" path - minimum viable example requires ~260 lines
- Entry point scripts (`run_*.py`) have substantial duplicated logic
- API surface area is large and poorly documented for programmatic use

### 1.2 Technical Debt Indicators

| Metric | Current State | Target |
|--------|--------------|--------|
| Lines in `grpo.py` | 2,621 | <500 per module |
| Config YAML avg size | 306 lines | <100 lines |
| Duplicated setup code across algorithms | ~60% | <15% |
| Time to implement new algorithm | 2-3 weeks | 2-3 days |
| Files to modify for new model support | 8-12 | 2-3 |

---

## 2. Current Architecture Analysis

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Entry Points (run_*.py)                      │
├─────────────────────────────────────────────────────────────────────┤
│                    Algorithms (grpo.py, sft.py, dpo.py)            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌───────────┐ │
│  │   Policy    │  │  Generation  │  │Environment │  │   Data    │ │
│  │  (lm_policy)│  │   (vllm)     │  │  (math,etc)│  │ (datasets)│ │
│  └─────────────┘  └──────────────┘  └────────────┘  └───────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                     Distributed Layer (Ray)                         │
│  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │ RayVirtualCluster│  │  RayWorkerGroup │  │ BatchedDataDict  │  │
│  └──────────────────┘  └─────────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│              Training Backends (DTensor/FSDP2, Megatron)            │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Identified Issues by Component

#### 2.2.1 Algorithm Layer (`nemo_rl/algorithms/`)

**Problems:**
- `grpo.py` is a 2600+ line monolithic file containing:
  - Configuration TypedDicts (6 different config classes)
  - Setup functions
  - Training loops (both sync and async)
  - Validation logic
  - Metric computation
  - Memory management
- Significant code duplication between `grpo.py`, `sft.py`, `dpo.py`, and `distillation.py`
- Each algorithm re-implements: cluster setup, checkpointing, logging, data loading

**Evidence:**
```python
# Pattern repeated in grpo.py, sft.py, dpo.py:
def setup(...):
    # 1. Logger initialization (repeated)
    logger = Logger(logger_config)
    # 2. Checkpointing setup (repeated)
    checkpointer = CheckpointManager(...)
    # 3. Data loading (repeated)
    train_dataloader = StatefulDataLoader(...)
    # 4. Cluster setup (repeated)
    cluster = RayVirtualCluster(...)
    # 5. Policy creation (repeated)
    policy = Policy(...)
```

#### 2.2.2 Policy Layer (`nemo_rl/models/policy/`)

**Problems:**
- `Policy` class in `lm_policy.py` violates Single Responsibility Principle
  - Handles both training AND generation via `GenerationInterface`
  - Manages two different worker types (DTensor and Megatron)
  - Contains 888 lines with mixed concerns
- Abstraction leak: Backend selection logic (`megatron_cfg.enabled` vs `dtensor_cfg.enabled`) is scattered
- The `PolicyInterface` and `ColocatablePolicyInterface` have overlapping concerns

**Evidence:**
```python
class Policy(ColocatablePolicyInterface, GenerationInterface):
    # Inherits from two interfaces with different concerns
    # Training interface methods
    def train(self, ...): ...
    def get_logprobs(self, ...): ...
    # Generation interface methods
    def generate(self, ...): ...  
    # Colocation methods (infrastructure concern)
    def offload_before_refit(self): ...
```

#### 2.2.3 Configuration System

**Problems:**
- Configuration complexity with multiple patterns:
  1. YAML files with inheritance (`defaults:` key)
  2. TypedDict classes for type hints (not validation)
  3. Hydra overrides for CLI
  4. OmegaConf interpolation (`${...}`)
- Same configuration can be specified in multiple places with unclear precedence
- No centralized validation - errors discovered at runtime deep in execution

**Evidence:**
```yaml
# grpo_math_1B.yaml - 306 lines with nested configs
policy:
  dtensor_cfg:
    enabled: true
    # ... 10+ parameters
  megatron_cfg:
    enabled: false
    # ... 40+ parameters (even when disabled!)
    optimizer: { ... }  # 15 parameters
    scheduler: { ... }  # 8 parameters
```

#### 2.2.4 Data Layer (`nemo_rl/data/`)

**Problems:**
- Tight coupling between data formats and algorithms
- Multiple dataset abstractions (`ResponseDataset`, `PreferenceDataset`, `ProcessedDataset`)
- Data processing functions are algorithm-specific but live in generic module
- `TaskDataSpec` pattern requires understanding multiple levels of indirection

#### 2.2.5 Environment Layer (`nemo_rl/environments/`)

**Problems:**
- `EnvironmentInterface` is too generic - doesn't capture common patterns
- Environments must be Ray actors (implementation detail leaks to interface)
- No standard way to compose or chain environments
- Reward computation is mixed with environment logic

#### 2.2.6 Distributed Layer (`nemo_rl/distributed/`)

**Strengths:**
- `RayVirtualCluster` and `RayWorkerGroup` provide clean abstractions
- `BatchedDataDict` is a useful primitive

**Problems:**
- Complex initialization flow spread across multiple files
- Worker builder pattern (`RayWorkerBuilder`) has too many responsibilities
- Colocation logic is scattered between `virtual_cluster.py`, `worker_groups.py`, and `lm_policy.py`

---

## 3. Target Architecture

### 3.1 Design Principles

1. **Separation of Concerns**: Each module has one clear responsibility
2. **Composition over Inheritance**: Use dependency injection and composition
3. **Progressive Disclosure**: Simple use cases should be simple; complex use cases should be possible
4. **Configuration as Code**: Type-safe, validated configuration with sensible defaults
5. **Plugin Architecture**: Easy extension points for new algorithms, environments, and backends

### 3.2 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      High-Level API (nemo_rl.api)                   │
│  train_grpo(model, dataset, reward_fn)  # 3 lines to run           │
│  train_sft(model, dataset)                                          │
└─────────────────────────────────────────────────────────────────────┤
│                      Trainer Framework (nemo_rl.trainers)           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ GRPOTrainer │  │ SFTTrainer  │  │ DPOTrainer  │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         └────────────────┼────────────────┘                         │
│                    ┌─────┴─────┐                                    │
│                    │BaseTrainer│  (shared training loop logic)      │
│                    └───────────┘                                    │
├─────────────────────────────────────────────────────────────────────┤
│                         Core Components                             │
│  ┌──────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │ PolicyModule │  │  Rollout   │  │Environment │  │DataModule  │ │
│  │  (training)  │  │  Engine    │  │  Manager   │  │            │ │
│  └──────────────┘  └────────────┘  └────────────┘  └────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                     Backend Abstraction                             │
│  ┌────────────────────┐  ┌────────────────────────┐                │
│  │   TrainingBackend  │  │   GenerationBackend    │                │
│  │  - DTensorBackend  │  │  - VLLMBackend         │                │
│  │  - MegatronBackend │  │  - MegatronInference   │                │
│  └────────────────────┘  └────────────────────────┘                │
├─────────────────────────────────────────────────────────────────────┤
│                Infrastructure Layer (nemo_rl.infra)                 │
│  ┌──────────┐  ┌───────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │ Cluster  │  │ResourceManager│  │Checkpointing │  │ Logging  │ │
│  └──────────┘  └───────────────┘  └──────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Key Architectural Changes

#### 3.3.1 Unified Trainer Framework

```python
# Target API
from nemo_rl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    model="Qwen/Qwen2.5-1.5B",
    num_prompts_per_step=32,
    num_generations_per_prompt=16,
)

trainer = GRPOTrainer(config)
trainer.fit(
    train_dataset=my_dataset,
    reward_fn=my_reward_function,  # Simple callable
)
```

#### 3.3.2 Configuration Redesign

```python
# Target: Type-safe configuration with validation
from nemo_rl.config import GRPOConfig, PolicyConfig, ClusterConfig

config = GRPOConfig(
    policy=PolicyConfig(
        model_name="Qwen/Qwen2.5-1.5B",
        precision="bfloat16",
        backend="dtensor",  # Single field, not two boolean flags
    ),
    cluster=ClusterConfig.auto_detect(),  # Sensible defaults
    # Most parameters have good defaults, only specify what you need
)
```

#### 3.3.3 Backend Plugin System

```python
# Target: Clean backend registration
from nemo_rl.backends import register_training_backend, TrainingBackend

@register_training_backend("custom")
class CustomTrainingBackend(TrainingBackend):
    def setup(self, config, cluster): ...
    def train_step(self, batch, loss_fn): ...
    def save_checkpoint(self, path): ...
```

---

## 4. Detailed Requirements

### 4.1 Phase 1: Foundation (Months 1-2)

#### REQ-1.1: Configuration System Redesign

**Current State:**
- TypedDict classes scattered across modules
- No runtime validation
- Complex YAML inheritance

**Requirements:**

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|---------------------|
| REQ-1.1.1 | Create unified config module with Pydantic-based dataclasses | P0 | All configs validated at load time |
| REQ-1.1.2 | Implement config builder with sensible defaults | P0 | 80% reduction in required config lines |
| REQ-1.1.3 | Support YAML, JSON, and Python dict configs | P0 | All three formats work interchangeably |
| REQ-1.1.4 | Provide config templates for common use cases | P1 | 5+ templates for each algorithm |
| REQ-1.1.5 | Generate config documentation from code | P1 | Auto-generated docs match implementation |

**Proposed Structure:**
```
nemo_rl/
  config/
    __init__.py          # Public API exports
    base.py              # BaseConfig with validation
    policy.py            # PolicyConfig, OptimizerConfig
    training.py          # GRPOConfig, SFTConfig, DPOConfig
    generation.py        # GenerationConfig, VLLMConfig
    cluster.py           # ClusterConfig
    defaults.py          # Default configurations
    validation.py        # Custom validators
```

#### REQ-1.2: Shared Infrastructure Module

**Requirements:**

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|---------------------|
| REQ-1.2.1 | Extract common setup logic to `BaseTrainer` | P0 | <100 lines of setup code per algorithm |
| REQ-1.2.2 | Create `ResourceManager` for cluster allocation | P0 | Single source of truth for resources |
| REQ-1.2.3 | Unify logging interface across all components | P1 | All logs go through single interface |
| REQ-1.2.4 | Create checkpoint abstraction layer | P0 | Backend-agnostic checkpoint save/load |

### 4.2 Phase 2: Core Refactoring (Months 2-4)

#### REQ-2.1: Algorithm Decomposition

**Requirements:**

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|---------------------|
| REQ-2.1.1 | Split `grpo.py` into focused modules | P0 | No file >500 lines |
| REQ-2.1.2 | Extract training loop to `BaseTrainer` | P0 | Algorithms only define loss + data transform |
| REQ-2.1.3 | Create `RolloutEngine` abstraction | P0 | Generation logic separate from training |
| REQ-2.1.4 | Implement callback system for extensibility | P1 | Hooks for all lifecycle events |
| REQ-2.1.5 | Unify validation logic across algorithms | P1 | Single validation runner |

**Proposed GRPO Structure:**
```
nemo_rl/
  trainers/
    base.py              # BaseTrainer (~300 lines)
    callbacks.py         # Callback system
  algorithms/
    grpo/
      __init__.py
      config.py          # GRPOConfig (~50 lines)
      loss.py            # Loss functions (~150 lines)
      data.py            # Data transforms (~100 lines)
      trainer.py         # GRPOTrainer extends BaseTrainer (~200 lines)
      utils.py           # Algorithm-specific utilities (~100 lines)
```

#### REQ-2.2: Policy Layer Refactoring

**Requirements:**

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|---------------------|
| REQ-2.2.1 | Separate training and generation responsibilities | P0 | Two distinct classes |
| REQ-2.2.2 | Create `TrainingBackend` abstraction | P0 | Backend-agnostic policy training |
| REQ-2.2.3 | Create `GenerationBackend` abstraction | P0 | Backend-agnostic generation |
| REQ-2.2.4 | Implement backend factory pattern | P0 | Single string selects backend |
| REQ-2.2.5 | Remove colocation logic from policy layer | P1 | Colocation handled by infrastructure |

**Target Interface:**
```python
class TrainingBackend(Protocol):
    def setup(self, config: PolicyConfig, cluster: Cluster) -> None: ...
    def train_step(self, batch: BatchedDataDict, loss_fn: LossFunction) -> dict: ...
    def get_logprobs(self, batch: BatchedDataDict) -> torch.Tensor: ...
    def save_checkpoint(self, path: Path) -> None: ...
    def load_checkpoint(self, path: Path) -> None: ...

class GenerationBackend(Protocol):
    def setup(self, config: GenerationConfig, cluster: Cluster) -> None: ...
    def generate(self, prompts: BatchedDataDict, sampling_params: dict) -> BatchedDataDict: ...
    def update_weights(self, state_dict: dict) -> None: ...
```

#### REQ-2.3: Data Layer Simplification

**Requirements:**

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|---------------------|
| REQ-2.3.1 | Create unified `DataModule` interface | P0 | Single way to provide data |
| REQ-2.3.2 | Support HuggingFace datasets natively | P0 | Zero-code data loading for HF |
| REQ-2.3.3 | Simplify dataset registration | P1 | Decorator-based registration |
| REQ-2.3.4 | Remove TaskDataSpec indirection | P1 | Direct prompt templates |

### 4.3 Phase 3: High-Level API (Months 4-5)

#### REQ-3.1: Simple API Layer

**Requirements:**

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|---------------------|
| REQ-3.1.1 | Create `nemo_rl.train()` single-function API | P0 | 5-line training script |
| REQ-3.1.2 | Implement `Trainer.from_pretrained()` pattern | P0 | Load configs from HF/local |
| REQ-3.1.3 | Support functional reward functions | P0 | No need for Environment class |
| REQ-3.1.4 | Add progress bars and status updates | P1 | Clear feedback during training |

**Target API:**
```python
# Minimal example (5 lines)
import nemo_rl

trainer = nemo_rl.GRPOTrainer.from_pretrained("Qwen/Qwen2.5-1.5B")
trainer.fit(
    dataset="nvidia/OpenMathInstruct-2",
    reward_fn=lambda prompt, response: math_verify(prompt, response),
)

# Full control example
from nemo_rl import GRPOTrainer, GRPOConfig
from nemo_rl.config import PolicyConfig, ClusterConfig

config = GRPOConfig(
    policy=PolicyConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        backend="megatron",
        tensor_parallel_size=4,
    ),
    cluster=ClusterConfig(num_nodes=4, gpus_per_node=8),
    num_prompts_per_step=256,
    num_generations_per_prompt=8,
)

trainer = GRPOTrainer(config)
trainer.fit(dataset, reward_fn, callbacks=[WandbCallback(), CheckpointCallback()])
```

#### REQ-3.2: Environment Simplification

**Requirements:**

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|---------------------|
| REQ-3.2.1 | Support simple callable as reward function | P0 | No class required for basic rewards |
| REQ-3.2.2 | Create `Environment` base with sensible defaults | P0 | Easy custom environment creation |
| REQ-3.2.3 | Implement async environment pool | P1 | Built-in parallelization |
| REQ-3.2.4 | Add environment composition utilities | P2 | Chain/combine environments |

### 4.4 Phase 4: Documentation & Polish (Months 5-6)

#### REQ-4.1: Documentation Overhaul

**Requirements:**

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|---------------------|
| REQ-4.1.1 | Create quickstart tutorial (10 min to first run) | P0 | New user success in 10 minutes |
| REQ-4.1.2 | Document all public APIs with examples | P0 | 100% API coverage |
| REQ-4.1.3 | Create architecture guide for contributors | P0 | Clear contribution path |
| REQ-4.1.4 | Add troubleshooting guide | P1 | Common issues documented |
| REQ-4.1.5 | Create migration guide from old API | P0 | Existing users can migrate |

#### REQ-4.2: Developer Experience

**Requirements:**

| ID | Requirement | Priority | Acceptance Criteria |
|----|------------|----------|---------------------|
| REQ-4.2.1 | Add comprehensive error messages | P0 | Errors include fix suggestions |
| REQ-4.2.2 | Create debug mode with verbose logging | P1 | Easy debugging enabled |
| REQ-4.2.3 | Add configuration diff tool | P2 | Compare configs easily |
| REQ-4.2.4 | Implement dry-run mode | P1 | Validate config without training |

---

## 5. Implementation Phases

### Phase 1: Foundation (Weeks 1-8)

```
Week 1-2: Configuration System
├── Create nemo_rl/config/ module structure
├── Implement BaseConfig with Pydantic validation
├── Port PolicyConfig, ClusterConfig
└── Write migration utilities from TypedDict

Week 3-4: Infrastructure Module
├── Create nemo_rl/infra/ module
├── Extract ResourceManager from lm_policy.py
├── Unify checkpoint abstraction
└── Create logging facade

Week 5-6: Base Trainer
├── Create BaseTrainer class
├── Extract common setup logic
├── Implement callback system
└── Add lifecycle hooks

Week 7-8: Testing & Stabilization
├── Update unit tests for new structure
├── Integration testing
├── Performance benchmarking
└── Documentation updates
```

### Phase 2: Core Refactoring (Weeks 9-16)

```
Week 9-10: GRPO Decomposition
├── Split grpo.py into modules
├── Create algorithms/grpo/ package
├── Extract loss functions
└── Create GRPOTrainer

Week 11-12: SFT/DPO Refactoring
├── Create SFTTrainer
├── Create DPOTrainer
├── Unify validation logic
└── Extract shared utilities

Week 13-14: Backend Abstraction
├── Create TrainingBackend protocol
├── Implement DTensorBackend
├── Implement MegatronBackend
├── Backend factory pattern

Week 15-16: Generation Refactoring
├── Create GenerationBackend protocol
├── Refactor VLLMBackend
├── Separate from Policy class
└── Testing & stabilization
```

### Phase 3: High-Level API (Weeks 17-20)

```
Week 17-18: Simple API
├── Create nemo_rl.api module
├── Implement from_pretrained pattern
├── Functional reward support
└── Dataset auto-loading

Week 19-20: Environment Simplification
├── Callable reward support
├── Environment composition
├── Async environment pool
└── Testing & examples
```

### Phase 4: Documentation & Polish (Weeks 21-24)

```
Week 21-22: Documentation
├── Quickstart tutorial
├── API reference generation
├── Architecture guide
└── Migration guide

Week 23-24: Polish
├── Error message improvements
├── Debug mode implementation
├── Final testing
└── Release preparation
```

---

## 6. Success Metrics

### 6.1 Quantitative Metrics

| Metric | Current | 3 Month Target | 6 Month Target |
|--------|---------|----------------|----------------|
| Time to first successful run | ~2 hours | 30 minutes | 10 minutes |
| Lines of code for minimal GRPO | 261 | 50 | 10 |
| Max file size (lines) | 2,621 | 800 | 500 |
| Config file size for basic run | 306 | 100 | 30 |
| New algorithm implementation time | 2-3 weeks | 1 week | 2-3 days |
| Code duplication (setup logic) | ~60% | ~30% | <15% |
| Unit test coverage | ~45% | ~60% | ~75% |

### 6.2 Qualitative Metrics

| Metric | Measurement Method | Target |
|--------|-------------------|--------|
| Developer satisfaction | Survey (1-5 scale) | 4.0+ |
| Documentation quality | User testing | 90% task completion |
| API discoverability | Time to find feature | <5 minutes |
| Error message helpfulness | User testing | 80% self-resolution |

### 6.3 Compatibility Metrics

| Metric | Requirement |
|--------|------------|
| Backward compatibility | All existing configs work with deprecation warnings |
| Performance regression | <5% throughput impact |
| Memory regression | <5% memory increase |

---

## 7. Risks and Mitigations

### 7.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Performance regression from abstraction layers | Medium | High | Continuous benchmarking, performance CI |
| Breaking changes for existing users | High | Medium | Maintain compatibility shims, clear migration guide |
| Distributed training edge cases | Medium | High | Extensive integration testing, phased rollout |
| Backend-specific quirks break abstraction | Medium | Medium | Allow backend-specific escape hatches |

### 7.2 Project Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Scope creep | High | Medium | Strict scope control, phase gates |
| Insufficient testing resources | Medium | High | Automated testing, community beta testing |
| Documentation lag | Medium | Medium | Doc-as-code, automated doc generation |
| Knowledge silos | Low | High | Pair programming, design reviews |

---

## 8. Appendix

### 8.1 Glossary

| Term | Definition |
|------|-----------|
| Backend | Implementation of training or generation (DTensor, Megatron, vLLM) |
| Trainer | High-level class orchestrating training loop |
| Policy | Model being trained (actor in RL terminology) |
| Rollout | Process of generating responses from prompts |
| Environment | Component computing rewards from responses |

### 8.2 File Structure Comparison

**Current Structure:**
```
nemo_rl/
├── algorithms/
│   ├── grpo.py          # 2,621 lines (config + setup + train + async)
│   ├── sft.py           # 641 lines
│   ├── dpo.py           # 769 lines
│   └── ...
├── models/
│   └── policy/
│       └── lm_policy.py # 888 lines (training + generation)
└── ...
```

**Target Structure:**
```
nemo_rl/
├── api/
│   ├── __init__.py      # Public API (train_grpo, train_sft, etc.)
│   └── functional.py    # Functional interface
├── config/
│   ├── __init__.py
│   ├── base.py          # BaseConfig (~100 lines)
│   ├── training.py      # Training configs (~200 lines)
│   └── defaults.py      # Default values (~100 lines)
├── trainers/
│   ├── base.py          # BaseTrainer (~300 lines)
│   ├── callbacks.py     # Callback system (~200 lines)
│   └── hooks.py         # Lifecycle hooks (~100 lines)
├── algorithms/
│   ├── grpo/
│   │   ├── __init__.py
│   │   ├── trainer.py   # GRPOTrainer (~200 lines)
│   │   ├── loss.py      # Loss functions (~150 lines)
│   │   └── data.py      # Data transforms (~100 lines)
│   ├── sft/
│   │   └── ...
│   └── dpo/
│       └── ...
├── backends/
│   ├── training/
│   │   ├── base.py      # TrainingBackend protocol
│   │   ├── dtensor.py   # DTensor implementation
│   │   └── megatron.py  # Megatron implementation
│   └── generation/
│       ├── base.py      # GenerationBackend protocol
│       └── vllm.py      # vLLM implementation
├── infra/
│   ├── cluster.py       # Cluster management
│   ├── resources.py     # Resource allocation
│   ├── checkpointing.py # Checkpoint abstraction
│   └── logging.py       # Unified logging
└── data/
    ├── module.py        # DataModule interface
    └── datasets/        # Dataset implementations
```

### 8.3 Migration Examples

**Before (Current API):**
```python
# ~70 lines of imports
from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
# ... many more imports

def main():
    args, overrides = parse_args()
    config = load_config(args.config)
    config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    
    init_ray()
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(...)
    
    dataset, val_dataset, task_to_env, val_task_to_env = setup_data(...)
    
    policy, policy_generation, cluster, dataloader, ... = setup(
        config, tokenizer, dataset, val_dataset
    )
    
    grpo_train(policy, policy_generation, dataloader, ...)
```

**After (Target API):**
```python
from nemo_rl import GRPOTrainer

def main():
    trainer = GRPOTrainer.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        num_prompts_per_step=32,
        num_generations_per_prompt=16,
    )
    
    trainer.fit(
        dataset="nvidia/OpenMathInstruct-2",
        reward_fn=math_verify,
    )
```

### 8.4 Backward Compatibility Strategy

1. **Deprecation Warnings**: Old APIs emit warnings pointing to new APIs
2. **Shim Layer**: `nemo_rl.compat` module provides old interfaces wrapping new ones
3. **Config Migration**: Tool to auto-convert old YAML configs to new format
4. **Documentation**: Clear migration guide with before/after examples
5. **Timeline**: 2 release cycles before removing deprecated APIs

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-15 | Claude | Initial draft |

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Tech Lead | | | |
| Product Manager | | | |
| Engineering Manager | | | |
