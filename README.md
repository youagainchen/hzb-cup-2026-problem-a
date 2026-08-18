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
- `results/routes/question1_balanced_49_routes.csv` 与对应表格：49 辆低迟到稳健方案；
- `results/figures/question1_*.svg`：成本构成、路线、装载率、车辆甘特图和优化轨迹。

加入“所有趟次必须在当日 24:00 前返场”的硬约束后，当前清洗数据的确定性结果为：总成本最优方案 98 趟、38 辆、43,397.26 元；49 辆对照方案同为 98 趟、48,266.20 元。最优方案采用 24 个新能源趟次，并在节约值中同时考虑空间距离和时间窗中点差异。由于第一问的目标是总成本最小，应选择 38 辆方案作为最终方案，49 辆方案仅用于敏感性分析。论文还应明确说明“允许车辆返回配送中心后多趟配送”属于模型假设。

可用 `--initial greedy` 或 `--initial savings` 固定初始解，使用 `--no-local-search` 做初始解消融；也可用 `--data-dir data/raw` 回跑原始数据。当前仍是确定性启发式结果，不保证全局最优；下一阶段可在同一评估器上升级为 ALNS。

## 协作约定

- `main` 始终保持可运行；功能分支使用 `role1/`、`role2/`、`role3/` 前缀。
- 原始 Excel 不修改；清洗结果写入 `data/processed/`。
- 提交信息使用简短中文，例如：`数据：完成订单缺失值审计`。
- 合并前至少检查漏单、重复、超载、时间窗、限行和成本汇总。
