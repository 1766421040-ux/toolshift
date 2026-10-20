# ToolShift 参考实现审查

核验日期：2026-10-20。此次重新获取了 7 个公开仓库的 30 份 README、实际源码及数据示例，逐文件记录 URL、抓取时间和 SHA-256，见 [references.json](../literature/references.json)。链接中的 `main` 是可变分支；内容哈希标识本次读到的版本，旧 Git tree SHA 不冒充 commit SHA。没有运行这些外部项目，没有沿用其论文结果作为本项目结果。

## 1. 从参考实现到本工程的要求

| 参考 | 核验的真实内容 | ToolShift 对应要求 | 没有覆盖的外部复杂度 |
|---|---|---|---|
| [ToolSandbox execution_context.py](https://github.com/apple-aiml-research/ToolSandbox/blob/main/tool_sandbox/common/execution_context.py) 与 [evaluation.py](https://github.com/apple-aiml-research/ToolSandbox/blob/main/tool_sandbox/common/evaluation.py) | `dbs_schemas` 定义 SANDBOX、SETTING、CONTACT、MESSAGING、REMINDER；实际用 Polars，按 `sandbox_message_index` 保留历史状态。增删改相似度、guardrail、milestone/minefield 判定过程和效果 | SQLite 独立实现；每次查询验证无写入，每次动作记录前后状态；过程不变量和最终目标分别判断 | 自然语言用户模拟、多轮澄清、外部搜索、工具增强、更多长程依赖 |
| [AppWorld task.py](https://github.com/StonyBrookNLP/appworld/blob/main/src/appworld/task.py) 与 [evaluator.py](https://github.com/StonyBrookNLP/appworld/blob/main/src/appworld/evaluator.py) | `Task.load` 区分任务规格、数据库和 ground truth；`evaluate_task` 构造 `ModelCollectionPair`，把私有数据和答案传给任务测试 | `public_context` 使用白名单序列化；WorldSpec、数据库句柄和 oracle 仅供生成/评估；真实执行后按数据库判定 | 多应用、多用户、鉴权和复杂真实业务 API；本项目不是 AppWorld 复现 |
| [AgentGym types.py](https://github.com/WooooDyy/AgentGym/blob/main/agentenv/agentenv/controller/types.py)、[env.py](https://github.com/WooooDyy/AgentGym/blob/main/agentenv/agentenv/controller/env.py)、[bc_trainer.py](https://github.com/WooooDyy/AgentGym/blob/main/agentenv/agentenv/trainer/bc_trainer.py) | `ConversationMessage` 含 `from/loss/value`；`StepOutput` 含 `state/reward/done`；环境提供 observe/step/reset。所读 BC 代码用硬编码 `[INST]`，human token 标为 -100，其他消息训练 | 轨迹、公开观测、动作目标分开；导出模型自己的 chat template；测试只有目标 assistant token 进入 loss | 14 类环境与多种交互动作；本项目先是固定候选动作选择，不是通用 ReAct agent |
| [AgentLab reproducibility_util.py](https://github.com/ServiceNow/AgentLab/blob/main/src/agentlab/experiments/reproducibility_util.py) 与 [study.py](https://github.com/ServiceNow/AgentLab/blob/main/src/agentlab/experiments/study.py) | 记录 Git、模块、基准版本及机器环境；检查已有实验配置兼容性，区分未完成和错误任务 | 每个结果绑定代码、生成器、split、模型、seed、配置；运行错误单列，不能静默删失败 episode | 浏览器实例重置、多个外部基准、并发服务冲突；当前不部署浏览器或 VM |
| [ToolTalk tool_executor.py](https://github.com/microsoft/ToolTalk/blob/main/src/tooltalk/evaluation/tool_executor.py) 与 [数据示例](https://github.com/microsoft/ToolTalk/blob/main/data/easy/ModifyEvent-easy.json) | JSON 包含会话、参考 API request/response/exception。成功要求参考调用召回完整且无未匹配的成功副作用动作 | 任务完成之外还要检测多余副作用；工具返回成功不能直接算 episode 成功 | ToolTalk 是参考调用匹配；ToolShift 是状态语义判分，两者分数不可直接比较 |
| [AgentCompat analyzer.py](https://github.com/armanzareian/agentcompat/blob/main/src/agentcompat/analyzer.py) 与 [models.py](https://github.com/armanzareian/agentcompat/blob/main/src/agentcompat/models.py) | 历史调用参数对 baseline/candidate schema 做静态验证；排除旧版本就无效的调用，计算加权兼容率及迁移提示 | 对比其静态兼容检查的边界；本项目实际实现旧语义加显式 scope 控制，没有集成 schema 差分引擎 | 不执行工具、无主动查询策略训练；其轨迹抽样不是序贯信息获取 |
| [agent-tool-drift diff.py](https://github.com/fykong/agent-tool-drift/blob/main/src/agent_tool_drift/diff.py)、[models.py](https://github.com/fykong/agent-tool-drift/blob/main/src/agent_tool_drift/models.py)、[mcp_client.py](https://github.com/fykong/agent-tool-drift/blob/main/src/agent_tool_drift/mcp_client.py) | `ToolContract`、`Snapshot` 归一化前后契约；比较默认值、schema、环境。MCP 客户端分页 `tools/list`，建议由规则产生 | 版本快照和默认值变更检测不能称原创；为完整新契约可得时建立直接修复基线 | 不等于任务相关、预算约束的查询策略；未知新契约不能免费提供给本项目基线 |

## 2. 规模只按证据陈述

- AppWorld 当前 [README](https://github.com/StonyBrookNLP/appworld) 声明 9 个应用、457 个 API、100+ 数据表及约 100 个模拟用户；这些是**作者声明的引擎规模**，本次没有安装并逐项运行计数。
- AgentGym 当前 [README](https://github.com/WooooDyy/AgentGym) 声明 14 类环境，表格逐环境列 Traj/Eval，并给出 `AgentEval` 与 `AgentTraj-L` 的结构示例；本次未下载完整 Hugging Face 数据，不把表格数量当作已验证的可训练样本总量。
- ToolTalk [README](https://github.com/microsoft/ToolTalk) 声明 28 个 easy、50 个 hard 会话、28 个工具、7 个插件。已缓存源码树中 `data/easy/*.json` 与 `data/tooltalk/*.json` 的文件数分别为 28、50；重新读取了两个会话和 Calendar 数据示例，未运行整个基准。
- ToolSandbox 的 [场景聚合器](https://github.com/apple-aiml-research/ToolSandbox/blob/main/tool_sandbox/scenarios/__init__.py) 会组合基础场景与干扰工具/描述增强。源码文件数、模板数、展开后场景数不是同一个量，本次不提供未经运行确认的总场景数。
- AgentLab 是管理多个基准的框架，README 中各基准任务数不能相加称为“AgentLab 数据集规模”。

ToolShift 第一版是三个模拟域、每域两个必要写步骤的有限世界研究环境；三个域共享数值赋值与集合更新两种底层算子。增加 seed 或数值变体只增加实例，不自动增加工具家族或业务复杂度。当前足以验证状态隔离、动作充分性、查询成本和不可恢复错误；不足以证明真实企业 Agent、GUI Agent 或开放工具库上的泛化。

## 3. 最近邻与可主张的范围

[Skill Drift 全文](https://arxiv.org/html/2605.10990v1) 已包含成对漂移/无漂移样本、运行契约过滤、查询验证及局部提示修复。因此“发现工具升级”“构造版本对”“局部修复”“先确认再行动”均不是本项目原创主张。

本项目的待验证对象是：**当多个版本仍可能成立，但存在共同可完成任务的动作时，不必识别唯一版本；只有动作决策仍有歧义才继续查询。** 在有限、可枚举的任务模型上生成集合监督，再检查它是否比版本识别监督、纯贪心监督和强静态策略更有用。主动诊断和动态规划本身是已有技术。

[ToolDriftBench 的 IEEE 原始摘要](https://ieeexplore.ieee.org/abstract/document/11677805/) 描述 manifest 漂移监控和 FAISS 索引的局部重建；DOI 为 `10.1109/SmartIoT70864.2026.00056`。全文页面要求 JavaScript 人机验证，本次未取得全文。因此不能据摘要证明其没有其他查询/训练机制，也不作穷尽性的原创认证。

## 4. 许可与数据来源

所有 ToolShift SQLite fixtures、目标、模板和轨迹由本项目独立生成；不复制外部基准的任务、用户数据、解决方案或数据库快照。

- ToolSandbox 主许可证允许使用、修改与再分发，须保留适用声明；它是 Apple 自定义许可，不是标准 SPDX，也不额外声称 OSI 认证。
- AppWorld 的普通基础设施是 Apache-2.0；[README 的 License 节](https://github.com/StonyBrookNLP/appworld#lock_with_ink_pen-license) 对 `.bundle` 中任务、应用/API 实现等另有公开再分发保持加密的要求。本工程不解包或再分发这些资产。公共评估设计的阅读不构成对其任务资产的复制授权。
- AgentGym 主库许可不替代其上游环境、模型和数据的许可证。将来接外部数据必须单独登记版本、许可及 split。
- 当前参考文件仅用于审查，没有把外部源码批量纳入工程。若后续实际复制代码，应追加对应版权与许可证，不能仅靠本文件致谢替代。

## 5. 必须能被测试推翻的实现要求

1. 同一个公开任务、不同隐藏 WorldSpec，初始 public context 与候选列表完全一致。
2. 查询不改变业务数据库；写动作真正改变 SQLite，错误写入不能靠重置后继续的方式伪装成功。
3. 至少存在“当前不触发保护对象不变量、但破坏未来完成可能性”的错误动作。
4. 至少存在两个版本无法被唯一识别、却可用同一动作完成的对照。
5. 至少存在预算内不可分辨且没有共同安全完成策略的任务，正确输出暂停且不修改状态。
6. 候选评分器和 QLoRA 只能使用公开输入；oracle 动作集合只作训练标签或评估诊断。
7. 实际域留出、前缀扰动和变更组合留出分开报告；当前三域共享算子、前缀只改文字，不能称为未见操作机制或强语言模板泛化。
8. CPU candidate scorer 的实际结果单独命名，不写成“大模型微调提升”；没有 CUDA/权重时，QLoRA 状态明确为未运行。
