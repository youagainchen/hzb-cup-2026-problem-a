# -*- coding: utf-8 -*-
"""问题三论文作图脚本（独立设计，非旧脚本复刻）。

围绕问题三要回答评审的四个问题组织 6 张必要图：

- 图1 发车级冻结（方法：τ=10:00 前已发车趟次整趟锁定，未来趟次滚动重排）
- 图2 核心机制：取消 c12 释放新能源趟次 → 新增 c99 被同一趟次承接（全文关键结论）
- 图3 当日成本分解与 ΔC 瀑布（结果：成本增量 +47.43 元 / +0.11%）
- 图4 响应时间与扰动（结果：~16ms 实时响应，改动 1 个客户、1% 路径）
- 图5 事件严重度敏感性（鲁棒：成本随强度单调）
- 图6 单事件边际成本与组合协同（机制：I = -8.05 正向协同）

数据来源（只读结果文件，不重跑求解器）：
- results/question3_optimized/question3_optimized_totals.json
- results/question3_optimized/question3_optimized_future_routes.csv
- results/question3/question3_event_set.csv
- results/question3_sensitivity/question3_severity_sensitivity.json
- results/question2_optimized/question2_optimized_routes.csv / *_route_summary.csv（事件前静态）

运行：python tools/plot_question3_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from src.data.loader import load_problem_data
from src.model.domain import DEFAULT_VEHICLE_TYPES, Delivery, Route
from src.model.evaluator import RouteEvaluator
from src.model.policy_q2 import build_q2_policy
from src.model.q3_event import apply_events
from src.solver.q2_initial import load_route_solution
from tools.run_q3_optimized import read_event_sets

# ---------------------------------------------------------------------------
# 配色与常量
# ---------------------------------------------------------------------------
ELECTRIC = "#4C78A8"   # 蓝：新能源
FUEL = "#F58518"       # 橙：燃油
GREEN = "#54A24B"      # 绿：绿色区 / 负向
RED = "#E45756"        # 红：正向 / 静态基准
TEAL = "#72B7B2"
GRAY = "#A8B0BC"
PURPLE = "#7B1FA2"     # 紫：c99 / 组合
TEXT = "#263238"
GRID = "#D9DEE7"

RESTRICT_START = 480.0
RESTRICT_END = 960.0
GREEN_ZONE_RADIUS = 10.0

OUTPUT_DIR = Path("results/figures")
DATA_DIR = Path("data/processed/team_cleaned")
STATIC_ROUTES_CSV = Path("results/question2_optimized/question2_optimized_routes.csv")
STATIC_SUMMARY_CSV = Path("results/question2_optimized/question2_optimized_route_summary.csv")
TOTALS_JSON = Path("results/question3_optimized/question3_optimized_totals.json")
FUTURE_ROUTES_CSV = Path("results/question3_optimized/question3_optimized_future_routes.csv")
EVENT_CSV = Path("results/question3/question3_event_set.csv")
SENS_JSON = Path("results/question3_sensitivity/question3_severity_sensitivity.json")

CANCELLED_CUSTOMER = 12
NEW_CUSTOMER = 99
ADDRESS_CHANGED_CUSTOMER = 82

# 单事件隔离结果（问题三文档 §11.2；结果目录未单独存档，作为论文口径常量）
SINGLE_EVENT_DELTAS = [
    ("取消 c12", -3.19),
    ("新增 c99", +40.51),
    ("变址 c82", +43.16),
    ("改窗 c70", -25.00),
]
COMBINED_DELTA = 47.43
SYNERGY = COMBINED_DELTA - sum(d for _, d in SINGLE_EVENT_DELTAS)  # -8.05

# 计算响应时间（问题三文档 §11.1，N=30 重复实验）
RESPONSE_MEDIAN_MS = 15.7
RESPONSE_P95_MS = 18.9


def configure_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _fmt(minutes: float) -> str:
    minutes = float(minutes)
    hours = int(minutes // 60)
    mins = int(round(minutes % 60))
    if mins >= 60:
        hours += 1
        mins -= 60
    return f"{hours:02d}:{mins:02d}"


def _find_trip_by_customer(routes, customer_id: int) -> Route | None:
    return next(
        (r for r in routes if any(d.customer_id == customer_id for d in r.deliveries)),
        None,
    )


def build_future_routes(problem_after) -> tuple[list[Route], RouteEvaluator]:
    vehicles = {v.name: v for v in DEFAULT_VEHICLE_TYPES}
    deliveries: dict[str, list[Delivery]] = defaultdict(list)
    meta: dict[str, dict] = {}
    order: list[str] = []
    with FUTURE_ROUTES_CSV.open(newline="", encoding="utf-8-sig") as stream:
        rows = sorted(csv.DictReader(stream),
                      key=lambda row: (row["route_id"], int(row["sequence"])))
    for row in rows:
        rid = row["route_id"]
        if rid not in deliveries:
            order.append(rid)
            meta[rid] = {
                "vehicle_name": row["vehicle_type"],
                "vehicle_number": int(row["physical_vehicle_id"].rsplit("-", 1)[1]),
                "trip_number": int(row["trip_number"]),
                "start_minutes": float(row["start_minutes"]),
            }
        deliveries[rid].append(
            Delivery(int(row["customer_id"]), float(row["delivered_weight_kg"]),
                     float(row["delivered_volume_m3"]))
        )
    evaluator = RouteEvaluator(
        problem_after, build_q2_policy(problem_after.green_customer_ids)
    )
    routes = []
    for rid in order:
        info = meta[rid]
        routes.append(Route(
            vehicle_type=vehicles[info["vehicle_name"]],
            vehicle_number=info["vehicle_number"],
            deliveries=deliveries[rid],
            start_minutes=info["start_minutes"],
            trip_number=info["trip_number"],
        ))
    return routes, evaluator


def _vehicle_trips(routes, evaluator, name: str, number: int):
    """返回某物理车辆的全部趟次（含评估出的起止时刻），按发车时刻排序。"""
    trips = [r for r in routes if r.vehicle_type.name == name and r.vehicle_number == number]
    scored = [(r, evaluator.evaluate(r, r.start_minutes)) for r in trips]
    scored.sort(key=lambda item: item[1].start_minutes)
    return scored


# ---------------------------------------------------------------------------
# 图 1  发车级冻结：τ=10:00 前已发车趟次整趟锁定，未来趟次重排
# ---------------------------------------------------------------------------
def plot_fig1_freeze(static_routes, static_eval, output: Path) -> None:
    """把问题二 98 趟静态计划画在时间轴上，按发车级冻结规则切分：
    发车时刻 ≤ τ(=10:00) 的趟次整趟冻结（灰），> τ 的趟次可重排（蓝/橙）。"""
    trigger = 600.0
    scored = [(r, static_eval.evaluate(r, r.start_minutes)) for r in static_routes]
    scored.sort(key=lambda item: item[1].start_minutes)
    n = len(scored)
    frozen = [s for s in scored if s[1].start_minutes <= trigger + 1e-9]
    replannable = [s for s in scored if s[1].start_minutes > trigger + 1e-9]

    fig, ax = plt.subplots(figsize=(12, max(6.5, n * 0.11)), dpi=160)
    ax.axvspan(RESTRICT_START, RESTRICT_END, color=RED, alpha=0.06, zorder=0)

    for i, (route, res) in enumerate(scored):
        y = n - 1 - i  # 最早发车排在最上
        if res.start_minutes <= trigger + 1e-9:
            color = GRAY
        else:
            color = ELECTRIC if route.vehicle_type.propulsion == "electric" else FUEL
        ax.barh(y, res.finish_minutes - res.start_minutes, left=res.start_minutes,
                height=0.75, color=color, alpha=0.9, zorder=2)

    ax.axvline(trigger, color=RED, lw=2.2, ls="--", zorder=4)
    ax.text(trigger, n + 1.2, "事件 τ=10:00\n（发车级冻结边界）", color=RED,
            ha="center", va="bottom", fontsize=9.5)
    ax.text((RESTRICT_START + trigger) / 2, n + 1.2,
            f"已发车冻结\n{len(frozen)} 趟", color=GRAY, ha="center", va="bottom",
            fontsize=10, fontweight="bold")
    ax.text((trigger + 1440) / 2, n + 1.2,
            f"未来可重排\n{len(replannable)} 趟", color=TEXT, ha="center", va="bottom",
            fontsize=10, fontweight="bold")

    ax.set_yticks([])
    ax.set_xlim(RESTRICT_START, 1440)
    ax.set_xticks(range(480, 1441, 120))
    ax.set_xticklabels([_fmt(m) for m in range(480, 1441, 120)], fontsize=8)
    ax.set_xlabel("时刻（每行一个配送趟次，按发车时刻排序）")
    ax.set_ylabel("配送趟次")
    ax.set_title(
        f"图 1  发车级冻结：τ=10:00 前 {len(frozen)} 趟整趟锁定，"
        f"其余 {len(replannable)} 趟进入滚动重优化", fontsize=13, pad=12,
    )
    ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.6, zorder=1)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GRAY, label="已发车冻结趟次"),
        plt.Rectangle((0, 0), 1, 1, color=ELECTRIC, label="新能源可重排趟次"),
        plt.Rectangle((0, 0), 1, 1, color=FUEL, label="燃油可重排趟次"),
        plt.Line2D([], [], color=RED, lw=2, ls="--", label="冻结边界 τ=10:00"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 2  核心机制：c12 取消释放趟次 → c99 承接
# ---------------------------------------------------------------------------
def plot_fig2_mechanism(problem, static_routes, static_eval, future_routes,
                        dynamic_eval, output: Path) -> None:
    static_c12 = _find_trip_by_customer(static_routes, CANCELLED_CUSTOMER)
    dynamic_c99 = _find_trip_by_customer(future_routes, NEW_CUSTOMER)
    name = static_c12.vehicle_type.name
    number = static_c12.vehicle_number
    vehicle_label = f"{name}-{number:03d}"

    static_trips = _vehicle_trips(static_routes, static_eval, name, number)
    dynamic_trips = _vehicle_trips(future_routes, dynamic_eval, name, number)

    fig, (ax_before, ax_after) = plt.subplots(
        2, 1, figsize=(10.5, 5.2), dpi=160, sharex=True, sharey=True,
    )

    def draw_vehicle(ax, trips, highlight_cid, title, color):
        ax.axvspan(RESTRICT_START, RESTRICT_END, color=RED, alpha=0.07, zorder=0)
        for route, res in trips:
            customers = [d.customer_id for d in route.deliveries]
            has_target = highlight_cid in customers
            bar_color = color if has_target else ELECTRIC
            ax.barh(0, res.finish_minutes - res.start_minutes, left=res.start_minutes,
                    height=0.5, color=bar_color, alpha=0.9,
                    edgecolor=TEXT, linewidth=0.5, zorder=2)
            label = f"T{route.trip_number:02d}"
            if has_target:
                label += f"（{'c' + str(highlight_cid)}）"
            ax.text(res.start_minutes + (res.finish_minutes - res.start_minutes) / 2,
                    0.0, label, ha="center", va="center", fontsize=8, color="white")
        ax.set_yticks([0])
        ax.set_yticklabels([title], fontsize=10)
        ax.set_xlim(RESTRICT_START, 1440)
        ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.6, zorder=1)
        ax.set_axisbelow(True)

    draw_vehicle(ax_before, static_trips, CANCELLED_CUSTOMER,
                 f"事件前（静态）", RED)
    draw_vehicle(ax_after, dynamic_trips, NEW_CUSTOMER,
                 f"事件后（动态）", PURPLE)

    ax_before.set_title(
        f"事件前：{vehicle_label} 的 T02 趟次配送 c12（10:00 被取消）", fontsize=11)
    ax_after.set_title(
        f"事件后：c99 插入同一空闲趟次（绿色区，由新能源车承接，无需启用新车）", fontsize=11)
    ticks = list(range(480, 1441, 120))
    ax_after.set_xticks(ticks)
    ax_after.set_xticklabels([_fmt(m) for m in ticks], fontsize=8)
    ax_after.set_xlabel("时刻（08:00-24:00，灰红底为燃油车绿色区限行）")

    fig.suptitle(
        "图 2  核心机制：取消 c12 释放新能源趟次，新增 c99 复用该趟次（零新增车辆）",
        fontsize=13, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 3  当日成本分解与 ΔC 瀑布
# ---------------------------------------------------------------------------
def plot_fig3_cost(totals: dict, output: Path) -> None:
    batch = totals["batches"][0]
    static = float(batch["static_total_cost"])
    executed = float(batch["executed_cost"])
    future_fixed = float(batch["future_fixed_cost"])
    future_oper = float(batch["future_operating_cost"])
    dynamic = float(batch["total_cost"])
    delta = float(batch["delta_cost"])

    fig, (ax_decomp, ax_wf) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=160)

    # (a) 当日总成本分解
    ax_decomp.bar(0, executed, width=0.5, color=ELECTRIC, label="冻结承诺成本")
    ax_decomp.bar(0, future_fixed, bottom=executed, width=0.5, color=FUEL,
                  label="未来固定成本")
    ax_decomp.bar(0, future_oper, bottom=executed + future_fixed, width=0.5,
                  color=GREEN, label="未来运行成本")
    ax_decomp.axhline(static, color=RED, lw=1.2, ls=":")
    ax_decomp.text(0.28, static + 150, f"问题二静态 {static:,.2f}", fontsize=8.5, color=RED)
    ax_decomp.text(0, dynamic + 350, f"动态 {dynamic:,.2f}", ha="center",
                   fontsize=10, fontweight="bold")
    ax_decomp.set_xticks([0])
    ax_decomp.set_xticklabels(["问题三当日总成本"])
    ax_decomp.set_ylabel("成本（元）")
    ax_decomp.set_ylim(0, dynamic * 1.16)
    ax_decomp.set_title("(a) 当日总成本分解（冻结承诺 + 未来计划）", fontsize=11)
    ax_decomp.legend(frameon=False, fontsize=8.5)
    ax_decomp.grid(axis="y", color=GRID, lw=0.7, alpha=0.6, zorder=0)

    # (b) ΔC 瀑布
    step_labels = ["问题二\n静态"] + [n for n, _ in SINGLE_EVENT_DELTAS] \
        + ["交互\n效应", "动态\n当日"]
    deltas = [0.0] + [d for _, d in SINGLE_EVENT_DELTAS] + [SYNERGY, 0.0]
    single = [d for _, d in SINGLE_EVENT_DELTAS]
    colors = [GRAY] + [GREEN if d < 0 else FUEL for d in single] + ["#B9C4D4", TEAL]
    bottoms, heights, running = [0.0], [static], static
    for d in deltas[1:-1]:
        bottoms.append(running)
        heights.append(d)
        running += d
    bottoms.append(0.0)
    heights.append(dynamic)
    x = range(len(step_labels))
    for i in range(len(step_labels)):
        ax_wf.bar(i, heights[i], bottom=bottoms[i], width=0.55, color=colors[i], zorder=3)
        label = f"{heights[i]:,.2f}" if i in (0, len(step_labels) - 1) else f"{heights[i]:+.2f}"
        ax_wf.text(i, bottoms[i] + heights[i], label, ha="center", va="bottom", fontsize=8)
    for i in range(len(step_labels) - 1):
        ax_wf.plot([i + 0.3, i + 1 - 0.3], [bottoms[i + 1], bottoms[i + 1]],
                   color=GRID, lw=1, zorder=2)
    ax_wf.set_xticks(list(x))
    ax_wf.set_xticklabels(step_labels, fontsize=8)
    ax_wf.set_ylabel("成本（元）")
    ax_wf.set_ylim(static - 60, dynamic + 200)
    ax_wf.set_title("(b) ΔC 瀑布（静态 → 四事件 → 交互 → 动态）", fontsize=11)
    ax_wf.grid(axis="y", color=GRID, lw=0.7, alpha=0.6, zorder=0)

    fig.suptitle(
        f"图 3  当日成本与增量：ΔC = {dynamic:,.2f} − {static:,.2f} = {delta:+.2f} 元（+0.11%）",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 4  响应时间与扰动
# ---------------------------------------------------------------------------
def plot_fig4_response_perturbation(totals: dict, output: Path) -> None:
    batch = totals["batches"][0]
    lead = float(batch["lead_time_minutes"])
    changed = int(batch["changed_customer_count"])
    assign = float(batch["assignment_change_ratio"]) * 100.0
    arc = float(batch["arc_change_ratio"]) * 100.0
    kept = int(batch["kept_trip_count"])
    dropped = int(batch["dropped_trip_count"])
    new = int(batch["new_trip_count"])

    fig, (ax_resp, ax_pert) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=160)

    # (a) 响应时间
    ax_resp.bar(["中位数"], [RESPONSE_MEDIAN_MS], width=0.4, color=ELECTRIC)
    ax_resp.errorbar([0], [RESPONSE_MEDIAN_MS], yerr=[[RESPONSE_MEDIAN_MS],
                     [RESPONSE_P95_MS - RESPONSE_MEDIAN_MS]], fmt="none",
                     color=RED, capsize=6, lw=1.6)
    ax_resp.text(0, RESPONSE_P95_MS + 0.5, f"P95 {RESPONSE_P95_MS} ms",
                 ha="center", fontsize=9, color=RED)
    ax_resp.set_ylim(0, RESPONSE_P95_MS * 1.5)
    ax_resp.set_ylabel("计算响应时间（ms）")
    ax_resp.set_title("(a) 计算响应时间（N=30）", fontsize=11)
    ax_resp.grid(axis="y", color=GRID, lw=0.7, alpha=0.6, zorder=0)
    ax_resp.text(0, RESPONSE_P95_MS * 0.2,
                 f"中位 {RESPONSE_MEDIAN_MS} ms\n执行提前量 {lead:.0f} min",
                 ha="center", fontsize=9, color=TEXT)

    # (b) 扰动
    ax_pert.barh([0, 1, 2], [assign, arc, 0], color=[GREEN, PURPLE, "none"])
    ax_pert.set_yticks([0, 1])
    ax_pert.set_yticklabels(["客户重分配率", "路径扰动率"], fontsize=9)
    ax_pert.set_xlim(0, max(assign, arc) * 2.2)
    for i, v in enumerate((assign, arc)):
        ax_pert.text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=9)
    ax_pert.set_xlabel("扰动率（%）")
    ax_pert.set_title("(b) 扰动（改动客户 1 个）", fontsize=11)
    ax_pert.grid(axis="x", color=GRID, lw=0.7, alpha=0.6, zorder=0)
    ax_pert.text(0.02, -0.5, f"趟次：保留 {kept}、删除 {dropped}、新增 {new}",
                 transform=ax_pert.get_xaxis_transform(), fontsize=9, color=TEXT)

    fig.suptitle(
        "图 4  实时响应能力：毫秒级计算、极小扰动（可行性 V=0）", fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 5  事件严重度敏感性
# ---------------------------------------------------------------------------
def plot_fig5_severity(output: Path) -> None:
    data = json.loads(SENS_JSON.read_text(encoding="utf-8"))
    results = data["results"]
    labels = [str(r["severity"]) for r in results]
    deltas = [float(r["delta_cost"]) for r in results]
    arcs = [float(r["arc_change_ratio"]) * 100.0 for r in results]
    assigns = [float(r["assignment_change_ratio"]) * 100.0 for r in results]

    fig, (ax_cost, ax_pert) = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=160)
    color_map = {"low": GREEN, "medium": ELECTRIC, "high": RED}
    bars = ax_cost.bar(labels, deltas, color=[color_map.get(l, GRAY) for l in labels],
                       width=0.55)
    ax_cost.axhline(0.0, color="#555555", lw=0.8)
    ax_cost.set_title("(a) 成本增量随严重度单调上升", fontsize=11)
    ax_cost.set_ylabel("成本增量 ΔC（元）")
    ax_cost.set_xlabel("事件严重度")
    ax_cost.grid(axis="y", color=GRID, lw=0.7, alpha=0.6, zorder=0)
    for bar, v in zip(bars, deltas):
        ax_cost.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:+.2f}",
                     ha="center", fontsize=9)

    ax_pert.plot(labels, arcs, color=RED, marker="o", lw=2, label="路径扰动率")
    ax_pert.plot(labels, assigns, color=GREEN, marker="s", lw=2, label="客户重分配率")
    ax_pert.set_title("(b) 扰动随严重度", fontsize=11)
    ax_pert.set_ylabel("扰动率（%）")
    ax_pert.set_xlabel("事件严重度")
    ax_pert.legend(frameon=False, fontsize=9)
    ax_pert.grid(True, color=GRID, lw=0.6, alpha=0.6)
    ax_pert.set_ylim(bottom=0.0)

    fig.suptitle("图 5  事件严重度敏感性（low/medium/high，取消 c12 为受控变量）",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 6  单事件边际成本与组合协同
# ---------------------------------------------------------------------------
def plot_fig6_event_impact(output: Path) -> None:
    labels = [n for n, _ in SINGLE_EVENT_DELTAS] + ["组合（四事件）"]
    values = [d for _, d in SINGLE_EVENT_DELTAS] + [COMBINED_DELTA]
    colors = [GREEN if v < 0 else FUEL for v in values]
    colors[-1] = PURPLE
    sum_single = sum(d for _, d in SINGLE_EVENT_DELTAS)

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.axhline(0.0, color="#555555", lw=0.9)
    ax.axhline(sum_single, color=GRAY, lw=1.0, ls=":")
    ax.text(3.55, sum_single + 0.6, f"单事件之和 {sum_single:+.2f}",
            fontsize=8.5, color=TEXT, ha="right")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (1.5 if v >= 0 else -3.5),
                f"{v:+.2f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("成本增量 ΔC（元）")
    ax.set_title("图 6  单事件边际成本与组合效应", pad=12, fontsize=13)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6, zorder=0)
    ax.text(0.5, -0.16,
            f"组合 ΔC({COMBINED_DELTA:+.2f}) < 单事件之和({sum_single:+.2f})，"
            f"交互 I = {SYNERGY:+.2f}：取消释放的新能源车承接新增订单",
            transform=ax.transAxes, ha="center", fontsize=9, color=RED)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    problem = load_problem_data(DATA_DIR)
    event_set = read_event_sets(EVENT_CSV)[0]
    problem_after, _ = apply_events(problem, event_set)
    static_routes = load_route_solution(STATIC_ROUTES_CSV, STATIC_SUMMARY_CSV)
    static_eval = RouteEvaluator(problem, build_q2_policy(problem.green_customer_ids))
    future_routes, dynamic_eval = build_future_routes(problem_after)
    totals = json.loads(TOTALS_JSON.read_text(encoding="utf-8"))

    plot_fig1_freeze(static_routes, static_eval,
                     OUTPUT_DIR / "question3_fig1_freeze.png")
    plot_fig2_mechanism(problem, static_routes, static_eval, future_routes,
                        dynamic_eval, OUTPUT_DIR / "question3_fig2_mechanism.png")
    plot_fig3_cost(totals, OUTPUT_DIR / "question3_fig3_cost_waterfall.png")
    plot_fig4_response_perturbation(totals, OUTPUT_DIR / "question3_fig4_response_perturbation.png")
    plot_fig5_severity(OUTPUT_DIR / "question3_fig5_severity.png")
    plot_fig6_event_impact(OUTPUT_DIR / "question3_fig6_event_impact.png")
    print(f"已生成 6 张图到 {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
