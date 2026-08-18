# -*- coding: utf-8 -*-
"""问题三论文作图脚本（2号）：生成 6 张图到 results/figures/。

- 图1 动态调度流程图（事件驱动 + 发车级冻结 + 分层重优化 + 统一评估）
- 图2 动态响应后路线空间分布（全图 + 绿色区局部放大，突出 c99 承接）
- 图3 成本构成与 ΔC 瀑布图（冻结承诺 + 未来固定 + 未来运行）
- 图4 事件严重度敏感性（low/medium/high → ΔC 与扰动）
- 图5 动态响应后车辆甘特图
- 图6 单事件边际成本与组合对比

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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

from src.data.loader import load_problem_data
from src.model.domain import DEFAULT_VEHICLE_TYPES, Delivery, Route
from src.model.evaluator import RouteEvaluator
from src.model.policy_q2 import build_q2_policy
from src.model.q3_event import apply_events
from src.solver.q2_initial import load_route_solution
from tools.run_q3_optimized import read_event_sets

# ---------------------------------------------------------------------------
# 全局配色与字体（与问题二图保持一致）
# ---------------------------------------------------------------------------
ELECTRIC = "#4C78A8"  # 蓝
FUEL = "#F58518"      # 橙
GREEN = "#54A24B"     # 绿
RED = "#E45756"       # 红
TEAL = "#72B7B2"
GRAY = "#A8B0BC"
TEXT = "#263238"
GRID = "#D9DEE7"

RESTRICT_START = 480.0  # 08:00
RESTRICT_END = 960.0    # 16:00
GREEN_ZONE_RADIUS = 10.0
OUTPUT_DIR = Path("results/figures")

DATA_DIR = Path("data/processed/team_cleaned")
ROUTES_CSV = Path("results/question2_optimized/question2_optimized_routes.csv")
SUMMARY_CSV = Path("results/question2_optimized/question2_optimized_route_summary.csv")
TOTALS_JSON = Path("results/question3_optimized/question3_optimized_totals.json")
FUTURE_ROUTES_CSV = Path("results/question3_optimized/question3_optimized_future_routes.csv")
EVENT_CSV = Path("results/question3/question3_event_set.csv")
SENS_JSON = Path("results/question3_sensitivity/question3_severity_sensitivity.json")


def configure_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
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


def read_event_set() -> object:
    return read_event_sets(EVENT_CSV)[0]


def build_future_routes(problem_after) -> tuple[list[Route], RouteEvaluator]:
    """从未来路线 CSV 重建 Route 对象并用事件后评估器评分。"""
    vehicles = {vehicle.name: vehicle for vehicle in DEFAULT_VEHICLE_TYPES}
    deliveries: dict[str, list[Delivery]] = defaultdict(list)
    meta: dict[str, dict] = {}
    order: list[str] = []
    with FUTURE_ROUTES_CSV.open(newline="", encoding="utf-8-sig") as stream:
        rows = sorted(
            csv.DictReader(stream),
            key=lambda row: (row["route_id"], int(row["sequence"])),
        )
    for row in rows:
        route_id = row["route_id"]
        if route_id not in deliveries:
            order.append(route_id)
            meta[route_id] = {
                "vehicle_name": row["vehicle_type"],
                "vehicle_number": int(row["physical_vehicle_id"].rsplit("-", 1)[1]),
                "trip_number": int(row["trip_number"]),
                "start_minutes": float(row["start_minutes"]),
            }
        deliveries[route_id].append(
            Delivery(
                int(row["customer_id"]),
                float(row["delivered_weight_kg"]),
                float(row["delivered_volume_m3"]),
            )
        )
    policy = build_q2_policy(problem_after.green_customer_ids)
    evaluator = RouteEvaluator(problem_after, policy)
    routes = []
    for route_id in order:
        info = meta[route_id]
        routes.append(
            Route(
                vehicle_type=vehicles[info["vehicle_name"]],
                vehicle_number=info["vehicle_number"],
                deliveries=deliveries[route_id],
                start_minutes=info["start_minutes"],
                trip_number=info["trip_number"],
            )
        )
    return routes, evaluator


# ---------------------------------------------------------------------------
# 图 1  动态调度流程图
# ---------------------------------------------------------------------------
def plot_fig1_flow(output: Path) -> None:
    steps = [
        ("事件到达 (τk)", "#E8F1F8", ELECTRIC),
        ("发车级冻结：\n已发车趟次整趟锁定", "#E8F1F8", ELECTRIC),
        ("更新订单状态：\n取消 / 新增 / 变址 / 改窗", "#E8F1F8", ELECTRIC),
        ("提取未来未执行任务\n（剩余需求 + 车辆就绪时刻）", "#E8F1F8", ELECTRIC),
        ("分层重优化 L0→L1→L2→L3\n参数→趟内→趟间→车辆级", "#FDF2E3", FUEL),
        ("统一评估器验收\n硬约束 V=0", "#EAF3E5", GREEN),
        ("输出：成本增量 / 响应时间 / 扰动", "#F2E5F0", TEAL),
    ]
    figure, ax = plt.subplots(figsize=(7.0, 8.6), dpi=160)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.35 + 1.0)
    ax.axis("off")
    box_w, box_h = 7.4, 0.95
    x0, y_gap = 1.3, 1.30
    for index, (text, fc, ec) in enumerate(steps):
        y = (len(steps) - 1 - index) * y_gap + 0.6
        box = FancyBboxPatch(
            (x0, y), box_w, box_h,
            boxstyle="round,pad=0.03", fc=fc, ec=ec, lw=1.4,
        )
        ax.add_patch(box)
        ax.text(
            x0 + box_w / 2, y + box_h / 2, text,
            ha="center", va="center", fontsize=10, color=TEXT,
        )
        if index < len(steps) - 1:
            y_next = (len(steps) - 1 - index - 1) * y_gap + 0.6 + box_h
            ax.add_patch(
                FancyArrowPatch(
                    (x0 + box_w / 2, y), (x0 + box_w / 2, y_next),
                    arrowstyle="-|>", mutation_scale=16, lw=1.4, color=TEXT,
                )
            )
    # 滚动时域反馈箭头
    ax.add_patch(
        FancyArrowPatch(
            (x0 + box_w, 0.9), (x0 + box_w, len(steps) * y_gap - 0.4),
            arrowstyle="-|>", mutation_scale=16, lw=1.1, color=GRAY,
            connectionstyle="arc3,rad=-0.35",
        )
    )
    ax.text(
        x0 + box_w + 0.5, (len(steps) - 1) * y_gap,
        "下一批事件\n（滚动时域）", fontsize=8, color=GRAY, ha="center",
    )
    ax.set_title("图 1  事件驱动的动态调度流程", pad=14, fontsize=13)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# 图 2  动态响应后路线空间分布（全图 + 绿色区局部放大）
# ---------------------------------------------------------------------------
def plot_fig2_route_map(problem, future_routes, evaluator, output: Path) -> None:
    coords = problem.coordinates
    green_ids = set(problem.green_customer_ids)
    future_by_vehicle: dict[tuple[str, int], list[Route]] = defaultdict(list)
    for route in future_routes:
        future_by_vehicle[(route.vehicle_type.name, route.vehicle_number)].append(route)

    c99_trip = next(
        (route for route in future_routes
         if any(item.customer_id == 99 for item in route.deliveries)),
        None,
    )
    c99_vehicle = (
        (c99_trip.vehicle_type.name, c99_trip.vehicle_number) if c99_trip else None
    )
    affected_vehicle_keys = {
        key for key in future_by_vehicle
        if key in {
            ("EV-3000", 9),   # 承接 c99 的车
            ("EV-3000", 4),   # 变址 c82 的车
        }
    }

    def draw_map(ax, xlim, ylim, highlight_affected: bool, title: str) -> None:
        for key, chain in future_by_vehicle.items():
            color = ELECTRIC if key[0].startswith("EV") else FUEL
            lw = 1.6
            if highlight_affected and key in affected_vehicle_keys:
                color = RED
                lw = 2.6
            elif highlight_affected and key == c99_vehicle:
                color = "#7B1FA2"
                lw = 3.0
            for route in chain:
                nodes = [0, *(item.customer_id for item in route.deliveries), 0]
                xs = [coords[node][0] for node in nodes]
                ys = [coords[node][1] for node in nodes]
                ax.plot(xs, ys, color=color, lw=lw, alpha=0.8,
                        solid_capstyle="round", zorder=2)
        circle = Circle((0, 0), GREEN_ZONE_RADIUS, fill=False, color=GREEN,
                        lw=1.4, ls="--", zorder=1)
        ax.add_patch(circle)
        for customer_id, (x, y) in coords.items():
            if customer_id == 0:
                ax.plot(x, y, "k*", ms=11, zorder=5)
                continue
            if customer_id in green_ids:
                ax.plot(x, y, "o", ms=3.4, color=GREEN, zorder=4)
            else:
                ax.plot(x, y, "o", ms=2.4, color=GRAY, zorder=3)
        # 标记 c12（已取消）与 c99（新增承接）
        ax.plot(*coords[12], "x", ms=13, mew=3, color=RED, zorder=6)
        ax.annotate("c12（已取消）", coords[12], xytext=(coords[12][0] - 1.5, coords[12][1] + 1.6),
                    fontsize=9, color=RED)
        if 99 in coords:
            ax.plot(*coords[99], "*", ms=15, color="#7B1FA2", zorder=6)
            ax.annotate("c99（新增）", coords[99], xytext=(coords[99][0] + 0.6, coords[99][1] - 1.8),
                        fontsize=9, color="#7B1FA2")
        ax.plot(0, 0, "k*", ms=13, zorder=7)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("X 坐标（km）")
        ax.set_ylabel("Y 坐标（km）")
        ax.set_title(title, fontsize=11)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.6)

    figure, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160)
    x_max = max(abs(coords[cid][0]) for cid in coords) + 4
    y_max = max(abs(coords[cid][1]) for cid in coords) + 4
    draw_map(ax_full, (-x_max, x_max), (-y_max, y_max), True,
             "动态响应后路线空间分布（全图）")
    draw_map(ax_zoom, (-14, 14), (-14, 14), True, "市中心绿色配送区局部放大")
    ax_zoom.legend(
        handles=[
            plt.Line2D([], [], color=ELECTRIC, lw=2, label="新能源路线"),
            plt.Line2D([], [], color=FUEL, lw=2, label="燃油路线"),
            plt.Line2D([], [], color=RED, lw=2.4, label="受事件影响的车辆链"),
            plt.Line2D([], [], color="#7B1FA2", lw=3, label="承接 c99 的趟次"),
        ],
        loc="lower left", frameon=False, fontsize=8.5,
    )
    figure.suptitle("图 2  动态响应后路线空间分布（取消 c12 释放新能源车承接新增 c99）",
                    fontsize=13, y=0.98)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# 图 3  成本构成与 ΔC 瀑布图
# ---------------------------------------------------------------------------
def plot_fig3_cost_waterfall(output: Path) -> None:
    totals = json.loads(TOTALS_JSON.read_text(encoding="utf-8"))
    batch = totals["batches"][0]
    static_total = float(totals["static_total_cost"])
    executed = float(batch["executed_cost"])
    future_fixed = float(batch["future_fixed_cost"])
    future_oper = float(batch["future_operating_cost"])
    dynamic_total = float(batch["total_cost"])
    delta = float(batch["delta_cost"])

    # 瀑布：静态 → 冻结承诺 → 未来固定 → 未来运行 → 当日总成本
    labels = ["问题二静态\n总成本", "冻结承诺\n成本", "未来\n固定成本", "未来\n运行成本", "动态当日\n总成本"]
    # 数值桥：用累计位置画瀑布
    steps = [
        (static_total, static_total),          # 起点
        (executed, executed),                  # 冻结承诺
        (future_fixed, executed + future_fixed),
        (future_oper, executed + future_fixed + future_oper),
        (dynamic_total, dynamic_total),        # 终点
    ]
    heights = []
    for value, cumulative in steps:
        if cumulative is None:
            heights.append(value)
        else:
            heights.append(None)
    # 手动构建瀑布柱
    figure, ax = plt.subplots(figsize=(8.6, 5.0), dpi=160)
    bar_colors = [GRAY, ELECTRIC, ELECTRIC, ELECTRIC, RED]
    x_positions = range(len(labels))
    bottom = [0.0, 0.0, executed, executed + future_fixed, 0.0]
    for i, (label, value) in enumerate(zip(labels, [static_total, executed, future_fixed, future_oper, dynamic_total])):
        if i == 0:
            bar = ax.bar(i, value, width=0.55, color=bar_colors[i], zorder=3)
            ax.text(i, value * 0.5, f"{value:,.2f}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        elif i == 4:
            ax.bar(i, value, width=0.55, color=bar_colors[i], zorder=3)
            ax.text(i, value + 200, f"{value:,.2f}", ha="center", fontsize=9, color=RED)
        else:
            ax.bar(i, value, bottom=bottom[i], width=0.55, color=bar_colors[i], zorder=3)
            ax.text(i, bottom[i] + value / 2, f"{value:,.2f}",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.axhline(static_total, color=TEXT, lw=1.0, ls=":", zorder=2)
    ax.text(3.6, static_total + 180, f"问题二基准 {static_total:,.2f}",
            fontsize=8.5, color=TEXT)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("成本（元）")
    ax.set_ylim(0, dynamic_total * 1.18)
    ax.set_title("图 3  问题三当日总成本构成（冻结承诺 + 未来计划）", pad=12, fontsize=13)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.7, zorder=0)
    ax.text(0.5, -0.12,
            f"ΔC = {dynamic_total:,.2f} − {static_total:,.2f} = {delta:+.2f} 元（+{float(batch['cost_change_ratio_base']) * 100:.2f}%）",
            transform=ax.transAxes, ha="center", fontsize=9, color=RED)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# 图 4  事件严重度敏感性（中文标签版）
# ---------------------------------------------------------------------------
def plot_fig4_severity(output: Path) -> None:
    data = json.loads(SENS_JSON.read_text(encoding="utf-8"))
    results = data["results"]
    labels = [str(row["severity"]) for row in results]
    deltas = [float(row["delta_cost"]) for row in results]
    arcs = [float(row["arc_change_ratio"]) for row in results]
    assigns = [float(row["assignment_change_ratio"]) for row in results]

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=160)
    colors = [GREEN, ELECTRIC, RED]
    bars = left.bar(labels, deltas, color=colors, width=0.55)
    left.axhline(0.0, color="#555555", linewidth=0.8)
    left.set_title("成本增量随事件严重度", fontsize=12)
    left.set_ylabel("成本增量 ΔC（元）")
    left.set_xlabel("事件严重度")
    for bar, value in zip(bars, deltas):
        left.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:+.2f}",
                  ha="center", fontsize=9, color=TEXT)
    right.plot(labels, arcs, color=RED, marker="o", linewidth=2, label="路径扰动率")
    right.plot(labels, assigns, color=GREEN, marker="s", linewidth=2, label="客户重分配率")
    right.set_title("扰动随事件严重度", fontsize=12)
    right.set_ylabel("扰动率")
    right.set_xlabel("事件严重度")
    right.legend(frameon=False, fontsize=9)
    right.grid(True, color=GRID, lw=0.6, alpha=0.6)
    figure.suptitle("图 4  事件严重度敏感性（low/medium/high，订单取消为受控变量）",
                    fontsize=13, y=0.99)
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# 图 5  动态响应后车辆甘特图
# ---------------------------------------------------------------------------
def plot_fig5_gantt(future_routes, evaluator, output: Path) -> None:
    results = []
    for route in future_routes:
        res = evaluator.evaluate(route, route.start_minutes)
        results.append((route, res))
    results.sort(key=lambda item: (item[0].vehicle_type.name, item[0].vehicle_number,
                                   item[1].start_minutes))

    vehicles = sorted(
        {(r.vehicle_type.name, r.vehicle_number) for r, _ in results},
        key=lambda key: (key[0], key[1]),
    )
    index_by_vehicle = {key: i for i, key in enumerate(vehicles)}

    figure, ax = plt.subplots(figsize=(12, max(7, len(vehicles) * 0.22)), dpi=160)
    ax.axvspan(RESTRICT_START, RESTRICT_END, color=RED, alpha=0.08, zorder=0)
    ax.text((RESTRICT_START + RESTRICT_END) / 2, len(vehicles) + 0.6,
            "绿色区燃油车限行 [08:00, 16:00)", ha="center", fontsize=8.5, color=RED)
    for route, res in results:
        key = (route.vehicle_type.name, route.vehicle_number)
        row = index_by_vehicle[key]
        color = ELECTRIC if route.vehicle_type.propulsion == "electric" else FUEL
        has_c99 = any(item.customer_id == 99 for item in route.deliveries)
        bar = ax.barh(row, res.finish_minutes - res.start_minutes,
                      left=res.start_minutes, height=0.62, color=color, alpha=0.85,
                      edgecolor="#7B1FA2" if has_c99 else "none",
                      linewidth=2.0 if has_c99 else 0.0, zorder=3)
    ax.set_yticks(range(len(vehicles)))
    ax.set_yticklabels([f"{name}-{num:03d}" for name, num in vehicles], fontsize=7)
    ax.set_xlim(480, 1440)
    ax.set_xticks(range(480, 1441, 120))
    ax.set_xticklabels([_fmt(m) for m in range(480, 1441, 120)], fontsize=8)
    ax.set_xlabel("时刻")
    ax.set_ylabel("物理车辆")
    ax.set_title("图 5  动态响应后车辆配送甘特图（紫框为承接 c99 的新增趟次）",
                 pad=12, fontsize=13)
    ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.6, zorder=1)
    ax.invert_yaxis()
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ELECTRIC, label="新能源趟次"),
        plt.Rectangle((0, 0), 1, 1, color=FUEL, label="燃油趟次"),
        plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#7B1FA2", lw=2, label="承接 c99 的趟次"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# 图 6  单事件边际成本对比
# ---------------------------------------------------------------------------
def plot_fig6_event_impact(output: Path) -> None:
    isolation = {
        "取消 c12": -3.19,
        "新增 c99": 40.51,
        "变址 c82": 43.16,
        "改窗 c70": -25.00,
        "组合（四事件）": 47.43,
    }
    labels = list(isolation.keys())
    values = list(isolation.values())
    colors = [RED if value < 0 else ELECTRIC for value in values]
    colors[-1] = "#7B1FA2"
    figure, ax = plt.subplots(figsize=(8.6, 4.6), dpi=160)
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.axhline(0.0, color="#555555", linewidth=0.9)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                value + (1.5 if value >= 0 else -3.5),
                f"{value:+.2f}", ha="center", fontsize=9.5,
                color=TEXT, fontweight="bold")
    ax.set_ylabel("成本增量 ΔC（元）")
    ax.set_title("图 6  各事件边际成本与组合效应（协同 I = −8.05）", pad=12, fontsize=13)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
    ax.text(0.5, -0.14,
            "组合 ΔC(47.43) < 单事件之和(55.48)，事件间存在轻度正向协同（取消释放的新能源车承接新增订单）",
            transform=ax.transAxes, ha="center", fontsize=9, color=RED)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    configure_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    problem = load_problem_data(DATA_DIR)
    event_set = read_event_set()
    problem_after, _ = apply_events(problem, event_set)
    future_routes, evaluator = build_future_routes(problem_after)

    plot_fig1_flow(OUTPUT_DIR / "question3_fig1_flow.png")
    plot_fig2_route_map(problem_after, future_routes, evaluator,
                        OUTPUT_DIR / "question3_fig2_route_map.png")
    plot_fig3_cost_waterfall(OUTPUT_DIR / "question3_fig3_cost_waterfall.png")
    plot_fig4_severity(OUTPUT_DIR / "question3_fig4_severity.png")
    plot_fig5_gantt(future_routes, evaluator, OUTPUT_DIR / "question3_fig5_gantt.png")
    plot_fig6_event_impact(OUTPUT_DIR / "question3_fig6_event_impact.png")
    print(f"已生成 6 张图到 {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
