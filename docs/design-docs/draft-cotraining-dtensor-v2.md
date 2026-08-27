# DTensor-v2 上的 Draft 联合训练（Draft Co-Training）

本分支在 DTensor-v2（Automodel FSDP2）后端上，为三种 speculative decoding 草稿模型
（drafter）家族——**DSpark**、**DFlash**、**EAGLE3**——增加了与 policy 联合训练
（co-training）的能力。Draft 模型使用与 policy 相同的 rollout batch 进行训练，
以 policy 自身训练前向过程中产生的 hidden states / logits 作为教师信号（teacher
signal），并在权重 refit 时与 policy 权重一起同步给 vLLM 生成 worker，使下一轮
rollout 能够用最新的 draft 权重做投机采样。

本文聚焦 **co-training 这一层**：它如何复用 Automodel 的 speculative decoding
基础组件（`nemo_automodel.components.speculative.*`，Automodel r0.6.0）；对于每一处
**没有**复用的功能，说明为什么这部分职责更适合放在 NeMo-RL 这一侧。

## 0. 一图看懂：训练循环中 Draft 处于哪个位置

```mermaid
flowchart TB
    A["Rollout: policy 采样生成响应"] --> B["policy.train 前向传播"]
    B --> C["DSparkHiddenCapture 钩子<br/>捕获 policy 指定层的 hidden states"]
    B --> D["stash_teacher_logits<br/>在温度缩放前截获 policy 原始 logits"]
    C --> E["Draft Runtime<br/>DSparkRuntime / Eagle3Runtime"]
    D --> E
    E --> F["Draft 前向：cross-attend hidden states<br/>以 policy logits 为教师蒸馏"]
    F --> G["DraftRuntimeLossWrapper<br/>policy_loss + loss_weight × draft_loss"]
    B --> H["policy loss（不变）"]
    H --> G
    G --> I["backward：按 dp*cp / cp_gradient_fanout 缩放"]
    I --> J["optimizer.step<br/>policy 组 + draft 组（各自学习率）"]
    J --> K["权重 refit：policy.* 与 draft.* 一起<br/>同步给 vLLM 生成 worker"]
    K --> A
```

policy 的 loss 计算路径完全不受影响；draft 只是"顺路"读取 policy 训练前向中已经
产生的中间结果，训练出的 draft 权重再随 policy 权重一起被 refit 到推理引擎，
形成一个闭环：rollout → 训练 → refit → 下一轮 rollout 用新 draft 加速采样。

## 1. Automodel 提供了什么

Automodel r0.6.0 提供了 DSpark/DFlash/EAGLE3 预训练所需的模型定义、attention
mask 构造和 loss 数学计算（`nemo_automodel.components.speculative.dspark.*`、
`nemo_automodel.components.attention.dflash_mask`）。这些代码是纯架构 + 纯数学：
不关心训练数据从哪来、梯度如何分布，因此可以**原样字节级导入**到
`nemo_rl/models/automodel/draft/` 中：

| Automodel 符号 | 使用方式 |
| --- | --- |
| `dspark._sampling.sample_tokens` | draft 阶段的 token 采样，原样 re-export |
| `dspark.config.build_draft_config` | draft config 构建辅助函数，原样 re-export |
| `attention.dflash_mask.{create_dflash_block_mask, create_dflash_sdpa_mask}` | `draft_qwen3.py` 直接调用的 block-causal attention mask |
| `dspark.common.{AcceptRatePredictor, context_doc_ids, create_noise_embed, create_position_ids, extract_context_feature, pin_rope_inv_freq_fp32, validate_target_layer_ids}` | 未改动的辅助函数，从 `nemo_rl/models/automodel/draft/common.py` re-export |
| `dspark.markov_head.{GatedMarkovHead, RNNHead}` | markov head 类，未改动 |
| `eagle.draft_llama.{LlamaEagle3DraftModel, Eagle3LlamaAttention, Eagle3LlamaDecoderLayer, Eagle3LlamaModel}` | eagle3 的整套网络实现（attention/decoder layer/embedding/fc/lm_head）——`eagle3_llama.py` 直接子类化 `LlamaEagle3DraftModel`，不重新实现 attention 数学（详见 §2.5，这是本次改动新增的复用面） |

`nemo_rl/models/automodel/draft/__init__.py` 把这一层定位为"thin extension
layer"：上表内容全部直接导入，只有下面几个文件包含本地代码，且每个文件顶部都
写明了它在 extend 什么、为什么（详见各文件 header 注释中对上游来源的完整引用）。

```mermaid
flowchart LR
    subgraph AM["Automodel r0.6.0（原样复用）"]
        direction TB
        AM1["dspark._sampling"]
        AM2["dspark.config"]
        AM3["attention.dflash_mask"]
        AM4["dspark.common（大部分）"]
        AM5["dspark.markov_head（基类）"]
        AM6["eagle.draft_llama<br/>LlamaEagle3DraftModel 网络实现"]
    end
    subgraph RL["nemo_rl/models/automodel/draft/（本地扩展）"]
        direction TB
        RL1["common.py<br/>anchor 采样 gate 在 RL response 边界"]
        RL2["draft_qwen3.py<br/>教师信号=policy 实时 logits"]
        RL3["loss.py<br/>DP/CP 归一化约定"]
        RL4["markov_head.py<br/>speculators 缩减词表适配"]
        RL5["eagle3_llama.py<br/>子类化 automodel 的网络<br/>+ RL 形态的 TTT 训练循环"]
        RL6["integration.py<br/>RL 训练循环胶水层"]
    end
    AM --> RL
```

## 2. NeMo-RL 在哪些地方做了扩展，以及为什么

### 2.1 `common.py` —— anchor 采样要 gate 在 RL 的 response 边界上

上游 `build_anchor_candidate_mask` 要求 **anchor token 自身**落在 loss mask
内。但在 RL rollout 里，response 的第一个 token 恰好锚定在**最后一个 prompt
token** 上，这个位置按构造就不在 loss mask 里——如果照搬上游的 gate 逻辑，会
直接丢掉 prompt→response 这一转折点（恰恰是推理时开始做投机解码的地方），并让
只有一个 token 的短 response 完全拿不到 draft 训练信号。NeMo-RL 版本的
`build_anchor_candidate_mask` 改为 gate 在**第一个目标 token**（`anchor + 1`）
的 mask 上。

```mermaid
flowchart LR
    subgraph U["上游（预训练语料）"]
        U1["anchor token 本身<br/>必须在 loss mask 内"] --> U2["丢弃 prompt→response<br/>过渡位置的 anchor"]
    end
    subgraph N["NeMo-RL（RL rollout）"]
        N1["anchor+1（首个目标 token）<br/>必须在 loss mask 内"] --> N2["保留 prompt→response<br/>过渡位置，单 token response 也有信号"]
    end
```

`sample_anchor_positions` 算法本身和上游一致，但仍然保留了完整的本地副本：
因为上游函数内部直接调用**上游自己的** `build_anchor_candidate_mask`，如果
直接 import 上游的 `sample_anchor_positions`，会悄悄地把这里刚修好的 RL anchor
gating 又丢回去。

`build_eval_mask` 把上游针对单一 layout 的 per-slot label 偏移逻辑做了泛化，
并新增 `supervised_from_slot` 处理，使同一个函数能同时服务 dspark 的
next-token layout 和 dflash 的 bonus-anchor layout（见 §2.3）。

这段代码本质上是"贴近架构"的代码，只是恰好编码了一个 RL 特有的不变量
（loss mask 来自 rollout 的 token 边界，而不是预训练语料的文档边界）——它离
模型很近，但不可能是上游的职责，因为上游根本没有"RL loss mask"这个概念。

### 2.2 `draft_qwen3.py` —— 用 teacher logits 代替 drafter 自己的 lm_head

上游 DSpark 的 `forward()` 计算蒸馏目标 `aligned_target_logits` 的方式，是把
`target_last_hidden_states` 重新过一遍**它自己的** `lm_head`。在预训练场景下
这没问题：教师是一个冻结的、独立产出的 verifier 模型。但在 RL 联合训练里，
draft 自己的 `lm_head` 恰恰是被训练的两个部分之一（`train_embed_and_head`），
如果教师信号来自自身，就等于让 draft 去拟合一个由自己不断变化产生的目标——
蒸馏目标退化了。

NeMo-RL 的 fork 版本额外接受一个可选的 `teacher_logits: [B, S, V]` 张量，
一旦提供，就从这个张量（已 detach）中 gather 出 `aligned_target_logits`，
而不再重新计算。这个张量正是 **policy 自身训练前向产生的原始 logits**
（见 §3.1）。这是唯一一处"运行 drafter"与"提供一个良定义的教师"两件事
交织得足够紧密、需要把整个 `forward()` 保留在本地而不是外面套一层 wrapper
的地方。

### 2.3 `loss.py` —— DP 归一化约定不同，而非上游的 world-size 缩放

上游 `compute_dspark_loss` 对分母做了一次 all-reduce 归一化后，又额外乘以
`dist.get_world_size()`——这对一个自己拥有 optimizer step、只需要对整个
world 缩放一次的 trainer 来说是正确的。但 NeMo-RL 的
`automodel_forward_backward` 在 backward 之前，已经把每个 microbatch 的
loss 按 `dp_size * cp_size` 缩放过（用来抵消 FSDP2 梯度平均的效果），如果这里
再套用上游的 `world_size` 乘法，就会在 `dp * cp` 个 rank 上重复修正——因此
fork 版本的 `compute_dspark_loss` 改为在一个**显式传入的 DP process group**
上做 all-reduce，返回的 loss 已经按 nemo-rl 的约定归一化好，去掉了上游的
world-size 乘法。这纯粹是分布式训练集成层面的问题：数学本身完全一样，
区别只在于最终归一化这一步由哪个系统负责——而这个归属只有在做缩放的那个
trainer 内部才能判断清楚。

`loss.py` 里的 chunked KL/TV 计算还在本分支内部收到过一次修复
（commit `e5715b63`）：把 eagle3 TTT 的 KL 计算沿序列维度分块，用来控制
长（18k token）RL rollout 序列上的 fp32 激活内存——这是固定短 block 的
预训练语料不会触发的量级。

### 2.4 `markov_head.py` —— 为 speculators checkpoint 拆分 embedding/output 词表

上游的 markov head 假设"上一个 token 的 embedding"和"输出 logits"共享同一个
词表。而本分支要支持的 speculators 格式 checkpoint（例如
`RedHatAI/*-speculator.*`）使用了一个**缩减过的 draft 词表**作为输出投影，
但输入侧的"上一个 token id"仍然是 target 词表空间的 id（draft 从始至终只在
输入端看到 target-space token id）。NeMo-RL 的 `VanillaMarkov` 子类为此增加了
独立的 `embed_vocab_size`；这是一个 checkpoint 格式兼容性的 shim，并非 RL
特有的决策，但因为它是加载本集成所面向的具体 checkpoint 所必需的，也就只能
放在这里。

### 2.5 `eagle3_llama.py` —— 复用 Automodel 的网络实现，但训练循环仍是本地代码

这一节是本文档更新的重点，专门回答一个问题：**既然 eagle3 现在 import 了
Automodel 的模型代码，为什么 `eagle3_llama.py`／`integration.py` 里还有
好几百行本地代码，不能直接把 Automodel 的 eagle3 全套原样拿来用？**

早期版本的 `eagle3_qwen3.py` 是从 `vllm-project/speculators@0b08a89`
（而非 Automodel）整段 vendor 过来、手写 attention 的一个自包含文件——原因是
"当时没有做过 Automodel-vs-vendor 的成本对比"，而不是 Automodel 真的没有
这块能力。这一版重新做了对比：Automodel `nemo_automodel.components
.speculative.eagle.draft_llama` 其实**已经**提供了一套通用的
`LlamaEagle3DraftModel` 网络实现（配套的 `Eagle3TrainerModule` 训练循环见
`eagle.core`），而且这套 attention 实现明显优于手写版本——step 0 的
causal block 只算一次，之后每个 TTT step 只需要对**自己上一步**的 K/V
做一次 O(T) 的 einsum "对角线"注意力（靠一个共享的 `cache_hidden` 列表），
而不是像手写版本那样每一步都要对整条不断增长的 KV cache 重新做一次
O(T² × step) 的矩阵乘法。`eagle3_llama.py` 现在直接子类化
`LlamaEagle3DraftModel`，**网络本身（attention/decoder layer/embedding/
fc/lm_head）完全是 import 过来的，不再重新实现**。

但"网络实现"和"训练循环"是两回事——Automodel 的 `Eagle3TrainerModule`
（`eagle.core`）是配合它自己的 `TrainEagle3Recipe`（从零训练一个 drafter
的离线 SFT recipe）设计的，跟 RL co-training 的形态在好几个维度上都对不上：

- **它不知道怎么加载别人发布的 checkpoint。** `Eagle3TrainerModule` /
  `LlamaEagle3DraftModel` 面向的场景是"用 Automodel 自己的 recipe 从零训一个
  drafter"，整个 `speculative/eagle/` 目录里**没有任何一处** `from_pretrained`
  是用来加载第三方发布的 draft checkpoint 的（唯二的 `from_pretrained` 调用
  都是加载 target/verifier 模型）。而 RL co-training 的前提恰恰是**从一个
  已经训练好、别人发布的 drafter 开始**（RedHatAI 的 speculators 格式
  checkpoint、lmsys/SGLang SpecForge 的原生扁平格式 checkpoint），
  co-train 让它跟上 RL 训练中漂移的 policy——这个能力 Automodel 完全没有，
  只能自己写：`_adapt_speculators_eagle3_config` /
  `_adapt_native_flat_eagle3_config` 把两种第三方格式翻译成
  `LlamaEagle3DraftModel` 期望的 config，`_load_eagle3_weights` 把两种
  格式各自的扁平 key（`layers.0.*` 或 SpecForge 的 `midlayer.*`）重映射到
  Automodel 的 `model.*` 前缀布局（详见 §3.5）。
- **`Eagle3TrainerModule.forward()` 的归一化约定跟 RL 的梯度累积对不上。**
  它返回的是一个已经按 TTT 衰减权重合并好的标量 loss、和一个已经除好的
  accuracy 比率，只对 CP group 做过 all-reduce——这跟 §2.3／§3.2 里
  DSpark 遇到的问题是同一类：RL 需要按**整个 global batch** 的
  microbatch-slot 数量做归一化，而不是 `Eagle3TrainerModule` 内置的那种
  CP-only 归一化。所以 `eagle3_llama.py` 没有调用
  `Eagle3TrainerModule.forward()`，而是在 `Eagle3DraftModel.forward()`
  里重写了一份等价的 TTT 循环，返回每个 TTT step 的 loss 分子/分母
  （而非合并好的标量），交给 `Eagle3Runtime.compute_loss` 去做
  DP-reduced、按 microbatch slot 归一化的组合。
- **RL 的 microbatch 打包方式跟它自带的 mask 构造对不上。**
  `LlamaEagle3DraftModel.forward()` 这个"一键跑完整个 TTT step"的封装，
  只会从一个 2D padding mask，或者一个按 `seq_lens` 表示的定长 packing
  方案里构造 attention mask；而 RL 训练把一整个 microbatch（多条 rollout
  序列）打包进**一行**、每条样本占一个固定宽度的 slot、padding 在 slot
  尾部——这个具体的打包约定跟它自带的两条路径都不完全一致。因此
  `Eagle3DraftModel.forward()` 没有走这层封装，而是直接调用模型暴露的、
  明显是设计给外部 trainer 用的分件 API（`embed_input_ids` /
  `project_hidden_states` / `model.layers[0]` / `compute_logits`），自己
  拼出一份贴合 RL 打包方式的 mask（eager 路径）或 `seq_lens`/`cu_seqlens`
  （flash-attn 路径）。flash-attn 这条路径上有个不算显然的小结论：把
  **每个固定宽度的 slot（含尾部 padding）整体声明成一个 varlen "document"**
  是安全的——padding 总是在 slot 尾部，causal mask 本来就不会让真实 token
  往后看到自己的 padding，而 padding 位置自己的输出又会被 `loss_mask`
  排除在 loss 之外，所以不需要在 `seq_lens` 里再区分 slot 内的真实长度和
  padding 长度。这个结论已经写进了一个 GPU 单测
  （`test_flash_attention_2_packing_matches_eager_dense_mask`），逐位对比
  flash-attn 打包路径和 eager dense-mask 路径的 loss/梯度。
- **它没有为 20k token 级别的 RL rollout 做过显存优化。** eager attention
  在 `Eagle3TrainerModule`/`LlamaEagle3DraftModel` 里会构造一个
  `[B, H, T, T]` 的 fp32 softmax 中间张量——在 32 个 head、1 万 token 的
  打包行下就是单次 ~12 GiB 的分配，在完整 4n8g 训练规模下直接 OOM，这也是
  `attn_implementation` 现在切到 `flash_attention_2` 的原因（eager 只是
  迁移初期为了降低风险留的保底路径，配合上面提到的单测跟 flash-attn 版本
  做数值对比）。`compute_logits`（也就是 `lm_head`）同样没做过分块——
  预训练语料一般远短于 RL 生成的响应长度，不会暴露这个问题；但在
  18-20k token 的 RL rollout、64000 词表的 draft 词表下，单次
  `[1, T, draft_vocab]` 的 logits 张量就有 ~2.4 GiB，每个 TTT step 都会
  materialize 一次。`eagle3_llama.py` 把 `compute_logits` 沿序列维度分块
  调用（`_KL_CHUNK_TOKENS`），这两处都是 RL 长响应场景特有的问题，
  Automodel 的 SFT 语料不会触发。

除了"训练循环怎么接"之外，把 Automodel 的网络原样接进 RL 训练循环时还
踩到了两个**真实的正确性 bug**，都不是 RL 特有的设计取舍，而是 Automodel
这套代码本身在"被非它自己的 recipe 调用"时暴露出来的缺陷，只能在本地打
补丁：

- `Eagle3LlamaDecoderLayer.forward()` 把 `norm_before_residual`
  这个残差卷积方式**写死成 False**，完全没有开关；而 RedHatAI 的
  speculators checkpoint 在 config 里明确声明 `norm_before_residual: true`。
  如果直接用未修改的 Automodel 层去加载这份 checkpoint，会悄悄用错残差
  约定，训练数值不会报错但是错的。`_Eagle3LlamaDecoderLayerWithResidualToggle`
  把这个开关加回来，只重写了 `forward()`，state dict 的 key/shape
  跟父类完全一致，checkpoint 加载不受影响。
- `LlamaRotaryEmbedding` 的 cos/sin cache dtype 取自 `config.torch_dtype`，
  没设置时默认 fp32——这会让 RoPE 之后的 q/k 被提升成 fp32，而没被 RoPE
  处理过的、缓存起来的 v 还留在训练 dtype（bf16），eager attention 里
  `attn_probs @ v0` 直接类型不匹配报错。`build_eagle3_draft_model` 里
  现在显式设置 `draft_hf_config.torch_dtype = torch_dtype`。

最后，vLLM serving 侧对 draft checkpoint 权重流（refit）里 key 命名的
预期，也是按照 Automodel **之前**没有的东西设计的：vLLM 自己的 eagle3
serving 端模型同样是"把东西包在一个 `model` 子模块下面"的写法，refit
校验逻辑（`_expected_draft_keys`）预期训练端发来的是**去掉 `model.` 前缀
之后**的扁平 key（跟旧版手写 vendor 代码的扁平命名一致）。Automodel 的
`LlamaEagle3DraftModel` 把网络包在 `self.model` 下面，导致训练端
state_dict 的 key 天然带 `model.` 前缀，直接对不上 vLLM 的预期——
`dtensor_policy_worker_v2.py` 里新增的 `_draft_refit_export_name` 就是
在导出 refit 权重流时把这个前缀剥掉。

也踩过的另一处，是 FSDP2 相关但更偏工程实现细节：`fully_shard` 的
unshard/reshard 钩子只挂在 `nn.Module.__call__` 上，如果像上面说的那样
直接调用 `embed_input_ids` / `compute_logits` 这些"裸方法"（不经过顶层
`__call__`），这些子模块的参数会在 forward 中途仍然是切分后的 DTensor，
导致矩阵乘法报 DTensor/Tensor 混用的错误。所以整个 TTT 循环现在整体放在
`Eagle3DraftModel.forward()` 里，调用方永远是 `self.draft_model(...)`
这样的整体调用，而不是拆开的分件调用。

**小结**：Automodel 提供的是"网络长什么样、怎么算 attention"这一层，
`eagle3_llama.py` 复用的正是这一层；但"怎么加载第三方发布的 checkpoint、
怎么归一化 loss、怎么打包 RL 的 microbatch、怎么跟 FSDP2/refit 对接"这几层，
本质上都是 §3 里反复出现的同一个理由——Automodel 的 Trainer 完全没有
"RL rollout / refit / 第三方 checkpoint 兼容"这些概念，这些代码只能留在
NeMo-RL 这一侧。

## 3. `integration.py` —— Automodel 没有位置安放的 RL 专属胶水层

`integration.py` 中的内容（checkpoint config 适配除外）之所以存在，是因为
RL 联合训练有一些监督式 drafter trainer 根本不会遇到的需求。以下内容都不是
"照原样上游化"就能解决的问题——它们**就是** RL 与 policy 的集成本身。

### 3.1 从 policy 自己的前向过程中截获教师信号

`DSparkHiddenCapture` 在 policy 的 decoder 层（以及可选的 embedding 层）上
挂 forward hook，用来捕获 draft cross-attend 所需的目标 hidden states——
捕获发生在 **policy 自己的训练前向过程中**，而不是单独跑一次教师前向。
`stash_teacher_logits` 同理，在 NeMo-RL 的温度缩放对 logits 做原地修改**之前**
把 policy 的原始 logits 截获下来。两者都会 detach，因此 draft loss 永远不会
反传进 policy 主干。

```mermaid
sequenceDiagram
    participant Rollout as Rollout Batch
    participant Policy as Policy 前向
    participant Hook as DSparkHiddenCapture<br/>forward hooks
    participant Stash as stash_teacher_logits
    participant Runtime as Draft Runtime
    participant Loss as DraftRuntimeLossWrapper

    Rollout->>Policy: 输入 token
    Policy->>Hook: 逐层输出（armed 状态下捕获，detach）
    Policy->>Stash: 温度缩放前的原始 logits（detach）
    Policy->>Loss: policy loss（正常路径，不受影响）
    Hook->>Runtime: target_hidden_states
    Stash->>Runtime: teacher_logits
    Runtime->>Runtime: draft 前向 + compute_loss
    Runtime->>Loss: draft_loss
    Loss->>Loss: policy_loss + loss_weight × draft_loss
```

一个普通的预训练 trainer 不会遇到这个问题：它只需要加载一次冻结的教师
checkpoint，想跑就跑。但这里的"教师"**就是**正在被 RL 训练的那个模型，
每一步都要重新取值，所以 hidden/logit 的捕获必须接入 NeMo-RL 已经在跑的、
用于计算 policy loss 的那一次前向——没有别的地方可以单独跑这次捕获，而
Automodel 的 Trainer 也完全没有"复用别人前向里的 hidden state"这种 hook
接口。

这些 hook 在设计上是按 microbatch"上膛（armed）"、在 backward 阶段的
activation-checkpointing 重放时"卸膛（disarmed）"的——这个细节完全是为了
配合 NeMo-RL 自己的、带 checkpoint 的 FSDP2 训练循环，通用 drafter trainer
不需要考虑这个问题。

### 3.2 跨梯度累积与 CP 的梯度缩放记账

`begin_global_batch` 以及 `/ self._num_microbatch_slots` 的除法之所以存在，
是因为 NeMo-RL 在梯度累积（gradient accumulation）的多个 microbatch 上对
loss 做求和，而 policy loss 是按**整个 global batch** 的 token 数做归一化的
均值。要让 draft loss 跟上同样的有效缩放比例——避免 `mbs`/`gbs` 的变化悄悄
改变 draft 梯度相对 policy 梯度的比例——纯粹是 NeMo-RL 梯度累积与 CP fanout
约定带来的产物（`nemo_rl/algorithms/loss/wrapper.py` 中
`DraftRuntimeLossWrapper.draft_loss_scale = cp_gradient_fanout`）；
Automodel 通用训练循环里根本没有这样的约定需要对齐。

### 3.3 跨 TP 副本的确定性 anchor 采样

`anchor_sampling_seed` 是 `(dp_rank, global_batch_index, microbatch_index)`
的纯函数，保证同一个 DP 切片下的每个 TP/CP peer 都采样出相同的 anchor——
draft 只在 `dp_cp` 维度上做 FSDP2 切分，在 TP 维度上是被复制的，一旦 anchor
采样在各副本间出现分歧，就会悄悄让副本状态发散，并使 refit 导出的 draft
权重变成"和 rank 相关"（此问题在 commit `db050ec9` 中被 review 发现并修复）。
这是一个和 NeMo-RL 如何切分 draft 相关的分布式并行正确性问题，通用的、单一
拓扑结构的 Automodel trainer 根本不会遇到。

### 3.4 checkpoint 配对、resume 与元数据校验

`PolicyWithDraft`、`optimizer_layout_record`、`draft_meta_record`、
`save_draft_checkpoint` / `load_draft_checkpoint` / `load_dspark_checkpoint`
以及 `draft_checkpoint_dir` 实现了：

- 一个组合 `nn.Module`，让一个 optimizer（拥有按固定顺序命名的
  `["policy", "draft"]` 参数组）可以作为一个整体通过 Automodel 的
  checkpointer 保存——该 checkpointer 的约定是"一个 model 对应一个
  optimizer"；
- 一个作为 policy 权重目录**兄弟节点**的 DCP 目录（而不是嵌套在 policy
  权重目录下面），原因是 policy checkpoint 加载器通过递归遍历权重目录来
  自动判断格式，如果把 draft 的 `.distcp` 文件嵌套进去，会被误判；
- 一份带版本号的元数据记录，并在 resume 时做硬性校验（算法类型、架构字段、
  optimizer 参数组布局），防止一次运行悄悄地用 DFlash 配置去 resume DSpark
  权重（或反过来）。

```mermaid
flowchart TB
    subgraph Save["保存"]
        S1["policy 权重<br/>weights/model/"] 
        S2["draft 权重（DCP）<br/>weights/draft/"]
        S3["dspark_meta.json<br/>algo/block_size/layout 等"]
        S4["optimizer 状态<br/>PolicyWithDraft 组合模块<br/>param groups: [policy, draft]"]
    end
    Save -.兄弟目录，避免被<br/>递归格式探测误判.- S1
    S2 --> S3
    subgraph Resume["恢复"]
        R1["load_dspark_checkpoint"] --> R2["policy 权重正常加载"]
        R1 --> R3["load_draft_checkpoint<br/>校验 meta 后加载 draft 权重"]
        R1 --> R4["optimizer 按 composite_model 恢复<br/>校验 param group 布局"]
    end
```

这类"某个具体 trainer 如何用一个 optimizer 为两个耦合模块做 checkpoint"的
配管工作，是标准问题，但完全依赖于 NeMo-RL 自己的 `CheckpointManager` 和
Automodel 的 DCP checkpointer API，无论模型代码多么忠实于上游，这部分逻辑
都只能放在集成层。

### 3.5 第三方发布 checkpoint 的 config／权重适配

`load_draft_hf_config` / `_adapt_speculators_dspark_config` /
`_adapt_speculators_eagle3_config` 把 speculators 格式的 checkpoint
（例如为 vLLM serving 生产/使用的 `RedHatAI/*-speculator.*`）翻译成 vendor
模型所期望的扁平 config 结构——重新映射嵌套的 `transformer_layer_config`，
并在 vLLM 的"id j = 第 j-1 层的输出"捕获约定和 trainer 的"输出自第 i 层"约定
之间转换 `aux_hidden_state_layer_ids`。这样做纯粹是为了让 NeMo-RL 能够
**联合训练与 vLLM 已经在 serving 的同一种 checkpoint 格式**，从而打通
"训练 → refit → serving"这个闭环；一个从零开始的 drafter trainer 完全没有
理由需要理解一个 serving 引擎的 checkpoint 方言。

eagle3 这边还多了一层：`_adapt_native_flat_eagle3_config` 处理**不是**
speculators 格式、而是 SGLang SpecForge 直接导出的原生扁平 checkpoint
（例如 `lmsys/SGLang-EAGLE3-*`）——这类 checkpoint 既不记录
`norm_before_residual`，也不记录 aux 捕获层 id，需要按跟 speculators
适配器一致的方式补全默认值；同时它不带自己的 `embed_tokens`（设计上是要
共享 target 模型的 embedding），`_load_eagle3_weights` 会从 policy 的
embedding 拷贝一份初始值（不是真正的权重共享——draft 和 policy 分别挂在
不同的 FSDP2 mesh 上，没法共享同一个 `nn.Parameter`），并把它的单层
`midlayer.*` key 重映射到 Automodel 的 `model.layers.0.*`。这两条适配
路径（speculators、SpecForge）连同 §2.5 提到的 `model.` 前缀重映射，
被统一到一个 `_load_eagle3_weights` 里。同样地，能这样做的前提是
Automodel 自己完全没有理解任何一种第三方 checkpoint 方言的理由——它只
认得自己训出来的 checkpoint。

## 4. RL 训练循环中的接线（`draft/` 目录之外）

上述内容被 NeMo-RL 中真正让 co-training 成为"一个 RL 特性"（而不只是
"一个模型特性"）的部分所调用：

- **`nemo_rl/models/automodel/setup.py`**：在 optimizer 构建**之前**先构建
  draft 模型（这样它的参数才能从一开始就加入 optimizer），把它作为第二个
  命名 optimizer 参数组加入（通常配一个远高于 policy 的学习率），并让 policy
  与 draft 一起 resume。
- **`nemo_rl/models/automodel/train.py`**：在训练前向中恰好还未做温度缩放的
  那个时间点（`forward_with_post_processing_fn`）把 policy 的原始 logits
  截获为 draft 的教师信号，并在训练前向外围套上 `draft_capture_ctx`。
- **`nemo_rl/algorithms/loss/wrapper.py`** 的 `DraftRuntimeLossWrapper`
  与 policy loss wrapper 并列存在：policy loss 的计算方式和没有 draft 时
  完全一样，draft loss 按 `cp_gradient_fanout` 和 `loss_weight` 缩放后加在
  上面——这和 NeMo-RL 组合其他 RL loss（例如蒸馏）的方式一致，而不是
  Automodel 组织训练 loss 的方式。
- **`nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py`**：负责每个
  worker 上 draft 模型的整个生命周期——构建 `DSparkRuntime` 或
  `Eagle3Runtime`、把 hidden capture 挂到 policy 模型上，并扩展
  `prepare_refit_info` / `_refit_params_generator`，加入 `draft.<name>`
  这些 key，使**权重 refit 路径把 draft 权重和 policy 权重一起同步给 vLLM
  生成 worker**——这正是让 co-training 真正对 RL 有用的一步（下一轮 rollout
  会用更新后的 draft 权重做投机采样）。Automodel 的 Trainer 完全没有
  "refit"这个概念；这纯粹是 RL rollout 生成侧的集成点。eagle3 改用
  Automodel 的 `LlamaEagle3DraftModel` 之后，这里还多了一步
  `_draft_refit_export_name`：把 state_dict 天然带的 `model.` 前缀剥掉，
  对齐 vLLM serving 端 `_expected_draft_keys` 的扁平 key 预期（详见
  §2.5 最后一段）。
- **`nemo_rl/models/policy/lm_policy.py`**：校验 `policy.draft.*` 配置——
  后端限制（DSpark/DFlash 要求 DTensor-v2；EAGLE3 还可以走 Megatron 上
  已有的独立路径）、与 sequence packing 的不兼容（暂不支持）、以及在
  `policy.generation.refit_transport` 为 `nccl_reshard` 时报硬错误（它基于
  后缀的批量 key 路由机制不认识 `draft.*` 这些 key，会把它们错误地分发——
  对应 vLLM 侧的收窄修复见 commit `fcb8c266`）。

## 5. 一图总结：复用边界

```mermaid
flowchart TB
    subgraph L1["架构 / 纯数学 —— Automodel 拥有"]
        A1["模型架构、attention mask、纯 loss 数学"]
        A2["确定性计算，教师无关：<br/>预训练 trainer 或 NeMo-RL 调用结果完全一致"]
    end
    subgraph L2["RL 形态的教师信号与归一化约定 —— NeMo-RL 拥有"]
        B1["policy 自身的实时 logits/hidden 作为教师"]
        B2["anchor gate 在 rollout response 边界上"]
        B3["DP/CP 归一化约定"]
        B4["draft/{common,draft_qwen3,loss,eagle3_llama}.py"]
    end
    subgraph L3["训练循环 & 分布式拓扑胶水 —— NeMo-RL 拥有"]
        C1["hidden/logit 捕获钩子"]
        C2["跨梯度累积一致的 loss 缩放"]
        C3["跨 TP 确定性 anchor 采样"]
        C4["policy+draft 耦合 checkpoint"]
        C5["第三方 checkpoint 格式适配<br/>（speculators / SpecForge）"]
        C6["draft/integration.py"]
    end
    subgraph L4["把 co-training 接入 RL 主循环 —— NeMo-RL 拥有"]
        D1["optimizer/参数组构建"]
        D2["教师 logits 截获时机"]
        D3["组合 loss wrapper"]
        D4["权重 refit 同步 draft.* 给 vLLM"]
        D5["config 校验"]
        D6["automodel/{setup,train}.py<br/>algorithms/loss/wrapper.py<br/>policy/{lm_policy,workers/dtensor_policy_worker_v2}.py"]
    end
    L1 -->|"依赖 policy 如何跑前向<br/>与如何分布梯度，<br/>Automodel 对此不可见"| L2
    L2 --> L3
    L3 -->|"Automodel 的 Trainer<br/>没有 rollout/refit 概念可以接入"| L4
```

| 复用边界 | 归属 | 原因 |
| --- | --- | --- |
| 模型架构、attention mask、纯 loss 数学 | Automodel（`nemo_automodel.components.speculative.*`） | 确定性、教师无关的计算；无论调用方是预训练 trainer 还是 NeMo-RL，结果完全一致 |
| RL 形态的教师信号（policy 自身实时 logits/hidden）、anchor gate 在 rollout response 边界上、DP/CP 归一化约定 | NeMo-RL（`draft/{common,draft_qwen3,loss,eagle3_llama}.py`） | 依赖 NeMo-RL **如何**跑 policy 前向、**如何**分布梯度——Automodel 对这两者都不可见 |
| hidden/logit 捕获钩子、跨梯度累积一致的 loss 缩放、跨 TP 确定性 anchor 采样、policy+draft 耦合 checkpoint、第三方 checkpoint 格式适配（speculators / SpecForge） | NeMo-RL（`draft/integration.py`） | RL 训练循环与分布式拓扑相关的问题，上游没有对应场景 |
| optimizer/参数组搭建、教师 logits 截获时机、组合 loss wrapper、权重 refit 同步（含 eagle3 的 `model.` 前缀剥离）、config 校验 | NeMo-RL（`automodel/{setup,train}.py`、`algorithms/loss/wrapper.py`、`policy/{lm_policy,workers/dtensor_policy_worker_v2}.py`） | 让 co-training 真正成为 RL 主循环（rollout → 训练 → refit）的一部分；Automodel 的 Trainer 没有 rollout/refit 概念可以接入 |
