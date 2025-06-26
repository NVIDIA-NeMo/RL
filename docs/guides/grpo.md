# An in-depth Walkthrough of GRPO in NeMo RL

This guide details the Group Relative Policy Optimization(GRPO) implementation within NeMo RL. We'll walk through essential aspects including data handling, policy model training, fast generation, and the specifics of the GRPO loss function and its enhancements. 

## Quickstart: Launch a GRPO Run

To get started quickly, use the script [examples/run_grpo_math.py](../../examples/run_grpo_math.py), which demonstrates how to train a model on math problems using GRPO. You can launch this script locally or via Slurm. For detailed instructions on setting up Ray and launching a job with Slurm, refer to the [cluster documentation](../cluster.md).

We recommend launching the job using `uv`:

```bash
uv run examples/run_grpo_math.py --config <PATH TO YAML CONFIG> {overrides}
```

If not specified, `config` will default to [examples/configs/grpo.yaml](../../examples/configs/grpo_math_1B.yaml).

**Reminder**: Don't forget to set your HF_HOME, WANDB_API_KEY, and HF_DATASETS_CACHE (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

In this guide, we'll walk through how we handle:

* Data
* Model training
* Fast generation
* Overall Resource Flow
* Loss

### Data

We support training with multiple RL "Environments" at the same time.

An [Environment](../../nemo_rl/environments/interfaces.py) is an object that accepts a state/action history and returns an update state and rewards for the step. They run as Ray Remote Actors. Example [MathEnvironment](../../nemo_rl/environments/math_environment.py).

To support this, we need to know:

* What environments you have
* Which data should go to which environments
* How to prepare the data from your dataset into a form we can use

#### Common Data Format

We define a [DatumSpec](../../nemo_rl/data/interfaces.py) that holds all relevant information for each training example:

```python
class DatumSpec(TypedDict):
    message_log: LLMMessageLogType
    length: int  # total (concatenated) length of the message tensors
    extra_env_info: Dict[str, Any] # anything your environment requires goes here, for example the 'answer' of a math problem
    loss_multiplier: float  # multiplier for the loss for this datum. 0 to mask out (say the sample is invalid)
    idx: int
    task_name: Optional[str] = "default"
    __extra__: Any  # This allows additional fields of any type
```

#### Data Processors

We refer to each distinct environment your model aims to optimize against as a "task." For example, you might define tasks like "math" or "code."

For each task, you should provide a data processor that reads from your dataset and returns a [DatumSpec](../../nemo_rl/data/interfaces.py)

```python
def my_data_processor(
    datum_dict: Dict[str, Any], # loaded directly from your dataset (i.e. single line of jsonl data)
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
```

We have an example of this as `math_data_processor` in [run_grpo_math.py](../../examples/run_grpo_math.py)

#### Putting it all together

GRPO expects datasets to have the following form:

```json
{"task_name": "math", /* actual data */}
```

Then, you can set the data up as follows:

```python
base_dataset = load_dataset("json", data_files=data_config["dataset_name"])["train"]
tokenizer = AutoTokenizer.from_pretrained(policy_config["model_name"])

task_data_processors = defaultdict(lambda: (math_task_spec, math_data_processor))
task_data_processors["math"] = (math_task_spec, math_data_processor)

math_env = MathEnvironment.remote(env_configs["math"]) # ray remote actor

dataset = AllTaskProcessedDataset(
    base_dataset,
    tokenizer,
    math_task_spec,
    task_data_processors,
    max_seq_length=data_config["max_input_seq_length"],
)
```

Ensure you provide a mapping of tasks to their processors so the dataset knows which processor to use when handling samples.

## Policy Model

We define a {py:class}`PolicyInterface]() <nemo_rl.models.interfaces>` that contains everything you need to train a Policy model.

This Policy object holds a [RayWorkerGroup](../../nemo_rl/distributed/worker_groups.py) of SPMD (1 proc/gpu) processes that run HF/MCore, all coordinated by this object so it appears to you like 1 GPU!

## Fast Generation

We support vLLM through the [VllmGeneration](../../nemo_rl/models/generation/vllm.py) class right now.

The function [grpo_train](../../nemo_rl/algorithms/grpo.py) contains the core GRPO training loop.

## Loss
We use the [ClippedPGLossFn](../../nemo_rl/algorithms/loss_functions.py) to calculate the loss for GRPO. Formally,

$$
L(\theta) = E_{x \sim \pi_{\theta_{\text{old}}}} \Big[ \min \Big(\frac{\pi_\theta(x)}{\pi_{\theta_{\text{old}}}(x)}A_t, \text{clip} \big( \frac{\pi_\theta(x)}{\pi_{\theta_{\text{old}}}(x)}, 1 - \varepsilon, 1 + \varepsilon \big) A_t \Big) \Big] - \beta D_{\text{KL}} (\pi_\theta \| \pi_\text{ref})
$$

where:

- $\pi_\theta$ is the policy model we are currently optimizing
- $\pi_{\theta_{\text{old}}}$ is the previous policy model (from the beginning of this step)
- $A_t$ is the advantage estimate
- $\varepsilon$ is a clipping hyperparameter
- $\beta$ is the KL penalty coefficient
- $\pi_{\text{ref}}$ is the reference policy

Also supports "Dual-Clipping" from https://arxiv.org/pdf/1912.09729, which
imposes an additional upper bound on the probability ratio when advantages are negative.
This prevents excessive policy updates. $rA \ll 0$ -> $cA$(clipped).
The loss function is modified to the following when A_t < 0:

$$
L(\theta) = E_t \Big[ \max \Big( \min \big(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) A_t \big), c A_t \Big) \Big] - \beta D_{\text{KL}} (\pi_\theta \| \pi_\text{ref})
$$

where:
- c is the dual-clip parameter (ratio_clip_c), which must be greater than 1 and is
    usually set as 3 empirically
- $r_t(\theta)$ is the ratio $\frac{\pi_\theta(x)}{\pi_{\theta_{\text{old}}}(x)}$ that measures how much the policy has change

### Improvements to the GRPO loss formulation for stability and accuracy

#### On-Policy KL Approximation (use_on_policy_kl_approximation)

In practice, we calculate the KL divergence using the estimator from Schulman 2020 (http://joschu.net/blog/kl-approx.html), which is unbiased and guaranteed to be positive.

$$
D_{\text{KL}} (\pi_\theta || \pi_\text{ref}) \approx E_{x \sim \pi_{\theta}} \Big[ \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - \log \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - 1 \Big]
$$

Note that the loss function above samples from $\pi_{\theta_{\text{old}}}$ instead of $\pi_\theta$, meaning that the KL approximation is off-policy if we use samples from $\pi_{\theta_{\text{old}}}$. This is the default formulation used in the [original GRPO paper](https://arxiv.org/abs/2402.03300). In order to use an _on-policy_ KL approximation while sampling from $\pi_{\theta_{\text{old}}}$, we can incorporate importance weights:

$$
\begin{align*}
D_{\text{KL}} (\pi_\theta || \pi_\text{ref}) &\approx E_{x \sim \pi_{\theta}} \Big[ \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - \log \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - 1 \Big] \\
&= \sum_x \pi_{\theta}(x) \Big[ \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - \log \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - 1 \Big] \\
&= \sum_x \pi_{\theta_{\text{old}}}(x) \frac{\pi_{\theta}(x)}{\pi_{\theta_{\text{old}}}(x)} \Big[ \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - \log \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - 1 \Big] \\
&= E_{x \sim \pi_{\theta_\text{old}}} \frac{\pi_{\theta}(x)}{\pi_{\theta_{\text{old}}}(x)} \Big[ \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - \log \frac{\pi_\text{ref}(x)}{\pi_\theta(x)} - 1 \Big] \\
\end{align*}
$$

To enable the on-policy KL approximation, set the config `use_on_policy_kl_approximation=True` in the `ClippedPGLossConfig`. By default, we set this config to False to align with standard GRPO.


#### Importance Sampling Correction (use_importance_sampling_correction)
The policy we use to draw samples, $\pi_{\theta_{\text{old}}}$, is used in both the inference framework and the training framework. To account for this distinction, we refer to the inference framework policy as $\pi_{\text{inference}}$ and the training framework policy as $\pi_{\text{training}}$. As noted in [Adding New Models](../adding-new-models.md#understand-discrepancies-between-backends), it is possible for the token probabilities from $\pi_{\text{training}}$ and $\pi_{\text{inference}}$ to have discrepancies (from numerics, precision differences, bugs, etc.), leading to off-policy samples. We can correct for this by introducing importance weights between $\pi_{\text{training}}$ and $\pi_{\text{inference}}$ to the first term of the loss function. 

Let $f_\theta(x) = \min \Big(\frac{\pi_\theta(x)}{\pi_{\theta_{\text{old}}}(x)}A_t, \text{clip} \big( \frac{\pi_\theta(x)}{\pi_{\theta_{\text{old}}}(x)}, 1 - \varepsilon, 1 + \varepsilon \big) A_t \Big)$ represent the first term of loss function. Then,

$$
\begin{align*}
E_{x \sim \pi_\text{training}} f_\theta(x) &= \sum_x \pi_\text{training}(x) f_\theta(x) \\
&= \sum_x \pi_\text{inference}(x) \frac{\pi_\text{training}(x)}{\pi_\text{inference}(x)} f_\theta(x) \\
&= E_{x \sim \pi_\text{inference}} \frac{\pi_\text{training}(x)}{\pi_\text{inference}(x)} f_\theta(x)
\end{align*}
$$

By multiplying the first term of the loss function by the importance weights $\frac{\pi_\text{training}(x)}{\pi_\text{inference}(x)}$, we can correct for the distribution mismatch between $\pi_{\text{training}}$ and $\pi_{\text{inference}}$ while still sampling from $\pi_{\text{inference}}$.

To enable the importance sampling correction, set the config `use_importance_sampling_correction=True` in the `ClippedPGLossConfig`. By default, we set this config to False to align with standard GRPO.


## Metrics ({wandb, tb}_name)
We track a few metrics during training for scientific experimentation and to validate correctness as the run progresses.

### Multiplicative Token Probability Error (token_mult_prob_error)
This is equal to the 'Logprob consistency metric' defined in [Adding New Models](../adding-new-models.md#importance-of-log-probability-consistency-in-training-and-inference):

$$
\text{token-mult-prob-error} = \frac{1}{n}\sum_{i=1}^{n\text{(tokens)}}\exp\left(\left\|\text{log-train-fwk}_i - \text{logprobs-inference-fwk}_i\right\|\right)
$$

Intuitively, this measures the average multiplicative probability error for sampled tokens, where samples are drawn as $x \sim \pi_{\text{inference-framework}}$. The purpose of this is to highlight any obvious sampling errors or discrepencies between the inference backend and training framework. If it trends upward steeply over the course of training past $\sim 1-2\%$, there is usually a problem with how your weights are being updated. If very spiky, it can indicate a bug in the inference framework or buggy weight refitting.

### Sampling Importance Ratio (sampling_importance_ratio)
Not to be confused with the clipped importance ratio in PPO/GRPO, this is the importance ratio between $\pi_{\text{training}}$ and $\pi_{\text{inference}}$.

This is simply $\frac{1}{|T|}\sum_{t \in \text{tokens}}\text{exp}(\text{log}(\pi_{\text{training}}(t)) - \text{log}(\pi_{\text{inference}}(t)))$

Similar to [Multiplicative Token Probability Error](#multiplicative-token-probability-error-token_mult_prob_error), this is a measure of how far off your inference backend is from your training framework. However, this metric is meant to find the bias in that error instead of loosely the variance as it does not take the absolute value of the error. With some noise, this should hover around 1.

This metric is always calculated and the per-token version (without the mean) is used in the loss function when [Importance Sampling Correction](#importance-sampling-correction-use_importance_sampling_correction) is enabled.

### Entropy (approx_entropy)
We roughly approximate the entropy of the LLM's distribution throughout training by calculating:

$$
E_{s \sim \pi_{\text{inference}}(x)}[-\frac{\pi_{\text{training}}(x)}{\pi_{\text{inference}}(x)}log(\pi_{\text{training}}(x))]
$$
using the rollouts in each training global batch as Monte-Carlo samples. The ratio of $\pi$ is in the formula to importance-correct for the mismatch between the policy over the course of training in a singular GRPO step and the inference framework.

We use this to track if our models are entropy-collapsing too quickly during training (as is quite common). This is a pretty rough monte-carlo approximation, so we wouldn't recommend using this directly for an entropy bonus or otherwise backpropagating through this. You can take a look at NeMo-Aligner's [implementation](https://github.com/NVIDIA/NeMo-Aligner/blob/main/nemo_aligner/utils/distributed.py#L351) of a full entropy calculation if you're interested (WIP efficient calculation in NeMo-RL).


## Evaluate the Trained Model

Upon completion of the training process, you can refer to our [evaluation guide](eval.md) to assess model capabilities.
