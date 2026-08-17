# 2026 华中杯 A 题：城市绿色物流配送调度

三人团队、三天冲刺项目。仓库只保留一套数据口径和一套路线评估器，问题一至问题三共享同一套成本与约束计算。

## 三人分工

| 角色 | 主要任务 | 首日交付 |
|---|---|---|
| 1 号：优化实现 | 评估器接口、初始解、局部搜索与后续求解器 | `src/solver/` 代码骨架与可运行示例 |
| 2 号：模型验证 | 目标与约束口径、手算样例、评估器测试 | `src/model/model_spec.md` 与 `tests/` |
| 3 号：数据与整合 | 数据审计、清洗、实验记录、图表与论文整合 | `data/processed/` 与数据审计记录 |

详细启动说明见 [`docs/team/华中杯A题_三人启动分工_两页版.docx`](docs/team/华中杯A题_三人启动分工_两页版.docx)。

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

## 问题一基线代码

当前版本实现“软时间窗感知的拆分配送贪心 + 路线内 2-opt”：

```powershell
python -m src.main
```

程序会读取 `data/raw/`，并输出：

- `results/routes/question1_baseline_routes.csv`：逐车、逐站配送路线与到达时间；
- `results/tables/question1_baseline_route_summary.csv`：车辆使用和分项成本；
- `results/tables/question1_baseline_totals.json`：总成本、总里程、碳排放及建模假设。

这是一套可复现的初始可行解，不是最终最优解。下一步应在同一评估器上加入跨路线客户搬移、交换、车辆类型重分配，再升级为 ALNS。

## 协作约定

- `main` 始终保持可运行；功能分支使用 `role1/`、`role2/`、`role3/` 前缀。
- 原始 Excel 不修改；清洗结果写入 `data/processed/`。
- 提交信息使用简短中文，例如：`数据：完成订单缺失值审计`。
- 合并前至少检查漏单、重复、超载、时间窗、限行和成本汇总。
