# Repo Structure And Documentation Plan

这份文档的目标不是重新设计 OmniTrack 的业务代码，而是把“主流程代码”“运行产物”“研究性材料”分层，减少仓库根目录和 `docs/` 的杂乱度。

## 1. 当前问题

当前仓库混放了三类东西：

1. 主业务代码和 canonical entrypoint  
   例如 `projects/`、`tools/`、`scripts/`、`jrdb_toolkit/`。
2. 主流程生成的运行产物  
   例如 `work_dirs/`、`results/`、以及过去默认落在根目录的 `evaluation_workspace*`。
3. 与业务逻辑解耦的研究/汇报材料  
   例如 `logs/`、`paper/`、`docs/baseline_reporting/`、`tools/baseline_reporting/`。
真正的问题不是这些内容不重要，而是它们缺少一套明确的“放哪里、谁算主流程、谁只是研究资产”的规则。

## 2. 建议的分层

### 2.1 业务代码

这部分保留在当前主树，不建议为了“看起来整洁”而大挪：

- `projects/`
- `tools/`
- `scripts/`
- `jrdb_toolkit/`
- `mmcv-full-1.7.1/`
- `data/`
- `ckpt/`
- `work_dirs/`
- `results/`

其中建议继续保留的原则：

- `tools/` 只放训练、测试、导出、评测转换等主流程工具。
- `scripts/` 只放 canonical shell entrypoint。
- `docs/` 只放与主流程直接相关、需要长期维护的稳定文档。

### 2.2 运行产物

所有“模型跑出来的东西”统一往 `results/` 和 `work_dirs/` 收，不再在仓库根部新开目录。

推荐约定：

```text
results/
  eval/
    jrdb_e2e/
    jrdb_tbd_hybridsort/
  submission/
  submission_tbd_hybridsort/
  test_submission/
  test_submission_tbd_hybridsort/
```

当前建议已经先落地了第一步：

- `evaluation_workspace/` 的默认职责迁到 `results/eval/jrdb_e2e/`
- `evaluation_workspace_tbd_hybridsort/` 的默认职责迁到 `results/eval/jrdb_tbd_hybridsort/`

这样做的好处：

- 根目录更干净。
- 所有推理/评测中间产物和最终 JSON、submission zip 在同一命名空间下。
- 后续如果再加 `bytetrack`、`ocsort` 或新 split，不会继续制造新的根目录。

### 2.3 非业务代码与研究资产

这类内容不建议继续散落在根目录，建议统一放到独立命名空间，例如：

```text
research/
  papers/
  logs/
  reporting/
    package/
    tools/
```

当前目录到目标目录的建议映射：

- `logs/` -> `research/logs/`
- `paper/` -> `research/papers/`
- `docs/baseline_reporting/` -> `research/reporting/package/`
- `tools/baseline_reporting/` -> `research/reporting/tools/`

说明：

- `docs/baseline_reporting/` 虽然叫 `docs`，但本质上是报告包和分析产物，不是主流程文档入口。
- `tools/baseline_reporting/` 虽然叫 `tools`，但本质上是研究汇报生成工具，不是训练/测试/eval 业务链路的一部分。

## 3. 文档建设建议

### 3.1 `docs/` 只保留四类文档

- Onboarding：安装、数据准备、快速开始。
- Code map：论文到代码、主流程结构说明。
- Baseline/reproduction：冻结结论、复现实验说明。
- Audit/governance：评测审计、仓库治理、结构约定。

### 3.2 给 `docs/` 增加统一入口

建议固定一个 [README.md](/mnt/sdb/ym/OmniTrack/docs/README.md) 作为索引页，避免后续文档继续平铺在目录里但没有入口。

### 3.3 把“稳定文档”和“分析资产”分开

以下内容不要继续直接塞进 `docs/` 主文档树：

- 大量截图
- 运行期导出的 CSV
- config snapshot 批量产物
- 个人调研卡片
- 临时验证记录

这类内容更适合归到 `research/reporting/package/` 或其他 analysis 命名空间。

## 4. 实施顺序

推荐按下面顺序收敛，而不是一次性大搬家：

1. 先收敛主流程默认输出路径  
   已把本地 eval workspace 默认收进 `results/eval/`。
2. 再收敛文档入口  
   给 `docs/` 增加索引页，并把仓库结构约定文档化。
3. 最后迁移非业务目录  
   等你准备整理本地研究资产时，再把 `logs/`、`paper/`、`baseline_reporting/` 统一迁到 `research/`。

## 5. 对当前仓库最实用的判断标准

如果一个目录满足下面任一条件，就应当视为业务主树的一部分：

- 被 `tools/train.py`、`tools/test.py`、`scripts/run_eval*.sh`、`scripts/run_test_submission*.sh` 直接调用
- 会影响训练、推理、导出、评测、submission 正确性
- 是主流程配置、模型、数据集或 evaluator 的实现代码

如果一个目录主要满足下面特征，就应归到非业务研究区：

- 不参与训练/推理/导出主链路
- 主要用于人工分析、复盘、截图汇报、论文背景对照
- 内容可以整体忽略而不影响 baseline 复现与提交流程

## 6. 这次调整后的建议结论

- 业务代码层面：不要重排 `projects/`、`tools/`、`scripts/` 主结构，只收敛输出路径和文档入口。
- 非业务代码层面：把 `paper/`、`logs/`、`baseline_reporting` 从根目录和 `docs/tools` 主树里剥离出去，独立成 `research/`。
- 文档建设层面：把 `docs/` 明确收敛成“主流程文档树”，不要再同时承担分析包、截图仓和个人研究笔记仓的角色。
