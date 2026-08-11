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

## 下一步：效果对标

- [ ] 使用与开源 Search-R1 一致的数据、E5 检索器、动作预算和 exact-match 指标
  评估最终 checkpoint。
- [ ] 对比开源 veRL Search-R1 基线与 NeMo RL 的 reward/EM 曲线、搜索次数、搜索
  有效性和多轮行为。
- [ ] 同时报告端到端时间、生成/训练吞吐、训练 GPU 内存、检索 GPU 内存和
  checkpoint 成本。
- [ ] 清楚披露框架实现差异、随机性、失败/恢复记录和任何无法完全对齐的配置。
- [ ] 只有在 500 步训练、最终评估和证据报告全部完成后，才把任务标记为正式
  Search-R1 reproduction 完成。
