# Examples

Welcome to the NeMo RL Examples! These complete, working examples demonstrate real-world applications and provide hands-on experience with NeMo RL.

## What You'll Find Here

Our examples are complete, working implementations that walk you through entire training workflows using specific datasets and configurations. Each example includes:

- **Complete Code**: Ready-to-run training scripts and configurations
- **Dataset Integration**: Real datasets and data processing pipelines
- **Configuration Files**: Optimized settings for different model sizes and scenarios
- **Expected Results**: Performance benchmarks and expected outcomes
- **Troubleshooting**: Common issues and solutions

## Examples by Category

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` SFT on OpenMathInstruct-2
:link: sft-openmathinstruct2
:link-type: doc

Complete example of supervised fine-tuning on math instruction dataset.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` GRPO on DeepScaleR
:link: grpo-deepscaler
:link-type: doc

Large-scale distributed training example with DeepScaleR integration.

+++
{bdg-warning}`Advanced`
:::



::::

## Example Descriptions

### **SFT on OpenMathInstruct-2**
- **Domain**: Mathematical reasoning and instruction following
- **Dataset**: OpenMathInstruct-2 with step-by-step math solutions
- **Model Size**: Configurable for different model sizes (1B to 70B parameters)
- **Training**: Supervised fine-tuning with domain-specific data
- **Evaluation**: Math problem-solving accuracy and reasoning quality
- **Use Case**: Educational AI, mathematical reasoning assistants

### **GRPO on DeepScaleR**
- **Domain**: Large-scale distributed training
- **Dataset**: DeepScaleR dataset for mathematical reasoning
- **Model Size**: Large models (1.5B+ parameters)
- **Training**: Group Relative Policy Optimization with distributed computing
- **Scaling**: Multi-GPU and multi-node training strategies
- **Use Case**: Large-scale distributed training on mathematical reasoning dataset





## Example Scripts

### **Core Training Scripts**

The following scripts are available in the `examples/` directory:

#### **GRPO Training**
- `examples/run_grpo_math.py` - GRPO training on math problems
- `examples/configs/grpo_math_1B.yaml` - Configuration for 1B parameter model
- `examples/configs/grpo_deepscaler-1.5b-24K.yaml` - DeepScaleR configuration

#### **DPO Training**
- `examples/run_dpo.py` - DPO training implementation
- `examples/configs/dpo.yaml` - DPO configuration template

#### **SFT Training**
- `examples/run_sft.py` - Supervised fine-tuning
- `examples/configs/sft.yaml` - SFT configuration template

#### **Evaluation**
- `examples/run_eval.py` - Model evaluation script
- `examples/configs/eval.yaml` - Evaluation configuration

### **Usage Examples**

```bash
# Run GRPO training
uv run examples/run_grpo_math.py --config examples/configs/grpo_math_1B.yaml

# Run DPO training
uv run examples/run_dpo.py --config examples/configs/dpo.yaml

# Run evaluation
uv run examples/run_eval.py --config examples/configs/eval.yaml
```

## Getting Started with Examples

### **Prerequisites**
- **Installation**: Complete NeMo RL installation (see [Installation Guide](../../get-started/installation))
- **Hardware**: GPU with sufficient memory for your chosen model size
- **Data Access**: Download required datasets (instructions provided in each example)
- **Dependencies**: Install additional dependencies as specified

### **Running Examples**

1. **Choose an Example**: Start with basic examples if you're new to NeMo RL
2. **Review Requirements**: Check hardware and software requirements
3. **Download Data**: Follow dataset preparation instructions
4. **Configure Training**: Adjust configuration files for your setup
5. **Run Training**: Execute the training script
6. **Monitor Progress**: Track training metrics and performance
7. **Evaluate Results**: Assess model performance and quality

### **Customization**

Each example can be customized for your specific needs:

- **Model Architecture**: Change model backends and configurations
- **Dataset**: Adapt for your own datasets and domains
- **Training Parameters**: Adjust learning rates, batch sizes, and other hyperparameters
- **Hardware**: Optimize for your available GPU memory and compute resources
- **Evaluation**: Add custom evaluation metrics and benchmarks

## Example Workflows

### **Basic Workflow**
1. **Data Preparation**: Download and preprocess the dataset
2. **Model Setup**: Configure the model architecture and parameters
3. **Training Configuration**: Set up training parameters and optimization
4. **Training Execution**: Run the training script
5. **Evaluation**: Assess model performance on test data
6. **Analysis**: Analyze results and identify areas for improvement

### Advanced Workflow

For distributed training, see the cluster setup documentation.

## Best Practices

### **Data Management**
- **Data Quality**: Ensure high-quality, well-preprocessed datasets
- **Validation**: Use proper train/validation/test splits
- **Augmentation**: Apply appropriate data augmentation techniques
- **Monitoring**: Track data quality and distribution shifts

### **Training Optimization**
- **Hyperparameter Tuning**: Systematically optimize training parameters
- **Memory Management**: Efficient memory usage for large models
- **Gradient Accumulation**: Handle large batch sizes with limited memory
- **Mixed Precision**: Use FP16/BF16 for faster training

### **Evaluation and Monitoring**
- **Comprehensive Metrics**: Use multiple evaluation metrics
- **Human Evaluation**: Incorporate human feedback where appropriate
- **Performance Tracking**: Monitor training progress and identify issues
- **Reproducibility**: Ensure reproducible results across runs

### **Performance Optimization**

For comprehensive performance optimization strategies, see the training optimization guides.

## Troubleshooting

### **Common Issues**
- **Memory Errors**: Reduce batch size or use gradient accumulation
- **Training Instability**: Adjust learning rate and optimizer settings
- **Poor Performance**: Check data quality and model configuration
- **Distributed Issues**: Verify cluster setup and network connectivity

### **Performance Optimization**
- **GPU Utilization**: Monitor and optimize GPU usage
- **Memory Efficiency**: Implement memory optimization techniques
- **Throughput**: Maximize training speed and efficiency
- **Scalability**: Scale training to multiple GPUs and nodes

## Next Steps

After running the examples:

1. **Experiment**: Modify parameters and configurations
2. **Customize**: Adapt examples for your own datasets and domains
3. **Scale Up**: Move to larger models and distributed training
4. **Production**: Deploy models in production environments
5. **Contribute**: Share your improvements and customizations

For additional learning resources, visit the main [Learning Resources](../index) page.

---

::::{toctree}
:hidden:
:caption: Examples
:maxdepth: 2
sft-openmathinstruct2
grpo-deepscaler
grpo-sliding-puzzle

:::: 
 