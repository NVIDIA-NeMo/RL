# Search-R1 reproduction TODO

本清单记录用 NeMo RL 复现开源
[Search-R1](https://github.com/PeterGriffinJin/Search-R1) 的当前进度。
基线固定在 Search-R1 提交
`2d0e225716fe3ccc071c9d020f5561548fdefc54`。详细对齐约束见
[`examples/nemo_gym/ai_search/comparison/SEARCH_R1_PARITY.md`](examples/nemo_gym/ai_search/comparison/SEARCH_R1_PARITY.md)。

## 已完成

- [x] 实现可运行的 NeMo RL AI-search 示例，包括检索服务、多轮 agent、GRPO
  训练配置、测试、文档和性能分析。
- [x] 对齐 Search-R1 的核心训练协议：
  - Qwen2.5-7B base 模型；
  - NQ + HotpotQA 官方训练问题；
  - 区分大小写的 `<search>`、`<information>` 和 `<answer>` 文本协议；
  - 每个问题五条 rollout、最多四次可执行动作；
  - observation token 不参与 loss；
  - normalized exact-match reward 和对应的 GRPO 分组优势计算。
- [x] 下载、校验并转换完整问题数据：169,615 条训练数据和 51,713 条验证数据。
- [x] 固定并校验 Qwen2.5-7B 模型 revision 及全部权重分片。
- [x] 准备官方 2018 Wikipedia E5 检索资产：
  - 21,015,324 篇文档；
  - 21,015,324 x 768 的 FlatIP 索引；
  - `intfloat/e5-base-v2` 官方 revision；
  - top-3 检索协议。
- [x] 为单张 80GB H100 实现有界内存的 Faiss GPU 索引加载路径，避免上游一次性
  clone 的启动峰值 OOM；完整索引、已知向量探针和真实查询均已通过。
- [x] 支持把 E5 检索服务放在独立 GPU 节点，并通过
  `AI_SEARCH_RETRIEVER_URL` 连接训练节点。
- [x] 修复预检暴露的兼容问题：passthrough tokenizer 配置、eager vLLM、无 EOS
  的多轮 prefix 拼接、vLLM Triton selected-token logprob 卡死，以及首次 Ray
  worker 注册超时。
- [x] 完成 BM25 小规模端到端门槛：Slurm `3592559` 在四张 H100 上完成
  20/20 条轨迹、reward、advantage、policy/reference logprob、反向传播和一次
  `optimizer.step()`。
- [x] 完成官方 E5 小规模端到端门槛：
  - E5 服务 `3594555` 加载完整索引并正常退出；
  - 四卡训练 `3594627` 完成 20/20 条多轮轨迹、18 次真实搜索和一次参数更新；
  - search error 为 0，format validity 为 0.95；
  - loss 为 -0.0202614，gradient norm 为 1.83843，说明确实执行了非零更新；
  - 训练和检索作业均以退出码 0 完成。
- [x] 保存 manifest、Slurm 日志、资源采样、TensorBoard 指标和 tokenized
  trajectory，能够重建模型搜索、读取 Wikipedia 结果并继续回答的过程。

## 当前结论

NeMo RL 版本的 Search-R1 完整训练链路已经跑通，包括真实 E5 检索和参数更新。
当前证据只是 4 个 prompt、20 条 trajectory、1 个训练 step 的系统验证，不能当作
500 步训练效果或正式 Search-R1 parity 结果。

## 下一步：正式训练前

- [ ] 确认正式训练预算并预留 8 张训练 H100 和 1 张独立检索 H100。
- [ ] 把 E5 索引、语料、模型 revision 信息和训练 checkpoint 目标移到可靠的持久
  存储；当前完整 E5 资产只存在于 `ipp2-0715` 的 node-local `/tmp`。
- [ ] 冻结正式 launcher、容器/uv 环境、模型和数据 checksum，并保存可复现
  manifest。
- [ ] 最后核对正式 recipe：500 steps、512 prompts x 5 rollouts、每个 optimizer
  mini-batch 256 条 trajectory、8 卡切分、每 50 steps 验证、每 100 steps 保存。
- [ ] 打开有限的响应文本或等价 token-ID 审计，定期抽查真实 query、Wikipedia
  observation、最终答案和奖励，避免只看汇总指标。
- [ ] 做一次正式资源布局的短预热，确认 checkpoint 目录、跨节点网络和 E5 服务在
  正式 allocation 中可用；该预热仍单独标记为非 parity 结果。

## 下一步：500 步训练

- [ ] 先启动并验证独立 E5 服务，再启动八卡 NeMo RL 训练。
- [ ] 持续监控 rollout 完成率、search error、无效格式、reward、gradient norm、
  GPU/host memory 和 step time。
- [ ] 按计划保存 checkpoint 和验证结果；失败时保留最后一个完整 step 的证据，
  从已验证 checkpoint 恢复，不静默跳过坏样本或检索错误。
- [ ] 保存完整的训练曲线、抽样轨迹、检索统计、硬件信息、版本和资源消耗。

## 下一步：当前 NeMo RL 复现验收

- [ ] 使用与开源 Search-R1 一致的数据、E5 检索器、动作预算和 exact-match 指标
  评估最终 checkpoint。
- [ ] 对比开源 veRL Search-R1 基线与 NeMo RL 的 reward/EM 曲线、搜索次数、搜索
  有效性和多轮行为。
- [ ] 同时报告端到端时间、生成/训练吞吐、训练 GPU 内存、检索 GPU 内存和
  checkpoint 成本。
- [ ] 清楚披露框架实现差异、随机性、失败/恢复记录和任何无法完全对齐的配置。
- [ ] 只有在 500 步训练、最终评估和证据报告全部完成后，才把任务标记为正式
  Search-R1 reproduction 完成。

## 后续：四套实现横向对比

完成上述 NeMo RL 正式复现并确认结果无误后，再启动一次独立的横向比较。比较对象
固定为：

1. 本仓库的 NeMo RL Search-R1；
2. [原版 Search-R1](https://github.com/PeterGriffinJin/Search-R1)，即论文使用的
   veRL fork；
3. [当前 veRL](https://github.com/verl-project/verl) 的 Search-R1-like 多轮搜索
   路径；
4. [slime Search-R1 lite](https://github.com/THUDM/slime/tree/main/examples/search-r1)。

### 比较准备

- [ ] 为四套实现固定准确的仓库 URL、commit、依赖锁文件、容器和启动命令。
- [ ] 先确认当前 veRL 版本中可运行的 Search-R1-like recipe。当前主线保留了数据、
  reward 和接入文档，但不再内置完整 `SearchTool`；如需固定到最后一个完整 commit
  或加入最小 adapter，必须记录并公开差异。
- [ ] 把 slime 默认的 Qwen2.5-3B、两轮搜索、八条 rollout 等参数改成共同协议；
  不把它的默认 “lite” 配置直接当作论文 parity 配置。
- [ ] 选择四套实现都支持的同一模型和精确 revision，并从完全相同的初始权重开始。
- [ ] 固定同一训练/验证问题、prompt 文本、Wikipedia corpus、E5 index、top-k、
  action budget、最大长度、采样参数、reward、GRPO group、batch 和 optimizer 配置。
- [ ] 使用同一检索服务或经过逐 query 等价性验证的副本，避免把检索结果差异误认为
  RL 框架差异。
- [ ] 固定同一 GPU 型号、GPU 数量、节点布局、精度、CPU/内存配额和持久存储；为
  四套实现分别记录 framework-native 必要差异。
- [ ] 在正式比较前审批四套完整训练的计算预算。效果比较与短性能 benchmark 分开，
  不能用几步性能测试替代训练效果结论。

### 效果比较

- [ ] 给四套实现相同的训练数据量、trajectory 数、optimizer update 数和验证频率。
- [ ] 至少使用同一个固定 seed 完成可重复对照；预算允许时运行三个 seed，并报告
  均值、标准差和失败率。
- [ ] 比较训练 reward、held-out exact match、answer F1、检索召回、格式正确率、
  search error、每题搜索次数、重复 query 比例和自然终止率。
- [ ] 比较达到相同 EM/reward 所需的 samples、tokens、optimizer updates、GPU-hours
  和 wall time，衡量 sample efficiency 与 time-to-quality。
- [ ] 抽查并分类多轮轨迹：直接回答、有效搜索、无效搜索、重复搜索、检索后修正、
  格式错误、截断和最终答案错误。
- [ ] 同时报告各框架最终 checkpoint 的统一离线评估，以及训练过程中的验证曲线；
  不只比较各仓库自己发布的单个最好数字。

### 性能比较与阶段 breakdown

- [ ] 每套实现先执行独立 warm-up，再在相同 workload 上测量多个稳定 step；报告
  median、p95 和离散程度，不使用第一个包含编译/缓存的 step 代表稳态性能。
- [ ] 把初始化单独报告：环境启动、Ray/worker 启动、模型加载、检索索引加载、
  optimizer 创建和首次 JIT，不混入训练 step 占比。
- [ ] 对每个训练 step 记录以下阶段的绝对耗时和占总 wall time 的比例：
  - rollout 准备与权重同步；
  - 模型生成，包括每轮生成和多轮 agent 调度；
  - 搜索排队、query encode、Faiss/index search、文档 fetch、序列化和跨节点网络；
  - trajectory 后处理和 reward；
  - policy logprob；
  - reference logprob；
  - advantage 计算和训练数据准备；
  - actor forward、backward、gradient reduce 和 optimizer step；
  - checkpoint 与 validation，二者从核心训练 step 中分开报告；
  - 无法归类的同步、等待和 idle 时间。
- [ ] 定义一套不重叠的顶层 critical-path buckets，使各阶段占比加总为 100%；嵌套、
  异步或并发阶段另外报告 inclusive time，避免重复计时导致占比超过 100%。
- [ ] 同时采集实际 prompt/generated/search-observation token 数，报告 samples/s、
  generated tokens/s、trained tokens/s 和每百万 token 的 GPU-hours，避免随机输出长度
  让 wall-time 对比失真。
- [ ] 分别记录训练 GPU 与检索 GPU 的峰值/平均显存、利用率、功耗，及 host RSS、
  CPU load、object store、跨节点流量、retrieval QPS 和 latency。
- [ ] 记录 rollout、训练和检索之间的并行度与等待关系，找出关键路径，而不是只列
  各函数累计时间。

### 横向比较交付物

- [ ] 保存每套实现的 manifest、原始日志、TensorBoard/W&B、资源 CSV、阶段 trace、
  checkpoint 和抽样轨迹。
- [ ] 生成统一的机器可读 JSON/TSV，包含每个 measured step 的 workload、质量、
  阶段耗时、占比、吞吐、内存和硬件信息。
- [ ] 发布四套实现的总结果表、逐阶段 breakdown 表和时间线图，分别解释生成、
  搜索、logprob、反向训练、同步与 I/O 的瓶颈。
- [ ] 给出效果与性能的联合结论：谁最终效果最好、谁达到目标效果最快、谁吞吐最高、
  谁最省显存，以及这些差异来自哪个训练阶段。
- [ ] 只有四套实现都完成受控 workload、质量评估、阶段计时校验和原始证据归档后，
  才把横向比较标记为完成。
