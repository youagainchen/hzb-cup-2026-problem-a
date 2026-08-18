# 2026 华中杯 A 题：城市绿色物流配送调度

三人团队、三天冲刺项目。仓库只保留一套数据口径和一套路线评估器，问题一至问题三共享同一套成本与约束计算。

## 三人分工

| 角色 | 主要任务 | 首日交付 |
|---|---|---|
| 1 号：优化实现 | 评估器接口、初始解、局部搜索与后续求解器 | `src/solver/` 代码骨架与可运行示例 |
| 2 号：模型验证 | 目标与约束口径、手算样例、评估器测试 | `src/model/model_spec.md` 与 `tests/` |
| 3 号：数据与整合 | 数据审计、清洗、实验记录、图表与论文整合 | `data/processed/` 与数据审计记录 |

详细启动说明见 [`docs/team/华中杯A题_三人启动分工_两页版.docx`](docs/team/华中杯A题_三人启动分工_两页版.docx)。

问题一的数学模型、约束、算法伪代码和最新结果说明见 [`paper/问题一模型与算法说明.md`](paper/问题一模型与算法说明.md)。

## 项目结构

```text
data/raw/            原始附件，只读
data/processed/      统一清洗后的数据
docs/problem/        题目、承诺书与提交说明
docs/team/           团队分工与启动说明
paper/               论文正文与写作素材
results/             路线、表格和图片
src/data/            数据读取与清洗
src/model/           模型口径、成本和约束
src/solver/          初始解与优化算法
src/visualization/   绘图与结果导出
tests/               手算样例和自动检查
tools/               非求解代码的辅助工具
```

## 第一天启动顺序

1. 统一单位、缺失值、拆单、绿色区和分时速度口径。
2. 3 号只生成一套 `data/processed/` 数据；其他成员不得各自清洗。
3. 2 号给出 1—2 个客户的手算样例和模型口径。
4. 1 号实现 `evaluator.py`，先通过手算与可行性检查，再开始优化。
5. 路线、图表与三问结果都调用同一评估器。

## 问题一优化代码

当前版本统一按完整成本进行搜索，包含：不受全天车辆数限制的成本感知 Clarke-Wright、同客户需求重新拆分、低装载路线消除、2-opt、relocate、swap、route merge、车型与物理车辆联合分配、10 分钟粒度发车时间优化，以及“一辆物理车执行多趟任务”的排班优化。`vehicle.count` 只约束同一时刻可启用的物理车辆数，不再错误限制全天趟次数；车辆返回配送中心后可再次出车，400 元启动成本每天只对该物理车辆计算一次。所有趟次必须在当日 24:00 前返回配送中心，跨天配送作为不可行解处理。

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m src.main
```

程序默认严格读取 `data/processed/team_cleaned/` 中的队友清洗数据，并输出：

- `results/routes/question1_optimized_routes.csv`：逐车、逐站配送路线与到达时间；
- `results/tables/question1_optimized_route_summary.csv`：物理车辆、趟次、发车时间和分项成本；
- `results/tables/question1_optimized_totals.json`：总成本、分项成本、总里程、碳排放、车辆数下界及建模假设。
- `results/figures/question1_optimized_*.svg`：最终方案的成本构成、路线、装载率、车辆甘特图和优化轨迹。

加入“所有趟次必须在当日 24:00 前返场”的硬约束后，当前清洗数据的确定性最优结果为 98 趟、38 辆、43,397.26 元。最优方案采用 24 个新能源趟次，并在节约值中同时考虑空间距离和时间窗中点差异。论文还应明确说明“允许车辆返回配送中心后多趟配送”属于模型假设。

可用 `--initial greedy` 或 `--initial savings` 固定初始解，使用 `--no-local-search` 做初始解消融；也可用 `--data-dir data/raw` 回跑原始数据。当前仍是确定性启发式结果，不保证全局最优；下一阶段可在同一评估器上升级为 ALNS。

## 问题二：实例基线与ALNS优化

2号模块由 `tools/build_q2_startup.py` 生成绿色客户清单、Q2实例摘要、政策敏感性表和确定性合规基线。政策判定集中在 `src/model/policy_q2.py`，统一评估器返回政策违规、漏单、容量、迟到和24:00返场指标。

1号模块在 `src/solver/q2_search.py` 中实现车型重选、发车修复、整趟交换、整趟迁移、车辆数压缩和ALNS，通过 `src/q2_adapter.py` 只读调用2号评估器，并输出路线、收敛日志和5种子统计。

在2号交接完成后，1号又增加了 `src/solver/q2_scheduling.py` 与 `tools/run_q2_optimized.py`：直接继承问题一的98趟配送任务，联合搜索绿色节点访问顺序、合规发车时刻、车型和物理车辆多趟复用。正式结果写入 `results/question2_optimized/`，同时给出问题一/问题二的成本、车辆结构和碳排放对比。

问题二的论文级建模说明、唯一问题一基准、政策边界、容量复核、算法局限、最终结果和作图规范见 `paper/问题二模型与算法说明.md`。正式运行会输出问题一证据哈希、逐站政策检查、绿色客户审计、容量审计和五车型候选表，正式结论只保留满足全部约束后的最低总成本方案。

```powershell
# 只重建2号实例与确定性基线
run_question2.cmd --baseline-only

# 接入2号实例与评估器，运行1号5种子ALNS
run_question2.cmd

# 基于问题一正式方案进行限行重排，并生成问题二正式对比结果
run_question2.cmd --optimized
```

## 协作约定

- `main` 始终保持可运行；功能分支使用 `role1/`、`role2/`、`role3/` 前缀。
- 原始 Excel 不修改；清洗结果写入 `data/processed/`。
- 提交信息使用简短中文，例如：`数据：完成订单缺失值审计`。
- 合并前至少检查漏单、重复、超载、时间窗、限行和成本汇总。

## 问题三动态调度

1 号动态调度器位于 `src/solver/q3_dynamic.py`，只调用 2 号维护的 `apply_events`、`extract_freeze_state` 和 `evaluate_dynamic`，不复制事件判定或动态成本口径。运行：

```powershell
python tools/run_q3_optimized.py
```

输出 `results/question3_optimized/`，包括未来路线、逐事件响应日志和动态成本汇总。正式四事件场景通过冻结、需求完整、容量、政策、排程和 24:00 返场检查后才会写出结果。
