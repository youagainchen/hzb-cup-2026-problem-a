"""问题二论文作图脚本。

依据 ``paper/问题二模型与算法说明.md`` 第 12 节的作图要点，从已存在的
``results/question2_optimized/`` 与 ``results/tables/`` 正式结果文件直接读取数据，
生成论文 7 张图。本脚本不重跑求解器，只负责把正式结果可视化。

图 1  客户分布和绿色配送区
图 2  问题二路线图（含市中心局部放大）
图 3  成本构成及政策变化瀑布图
图 4  车辆结构对比（物理车辆数 / 执行趟次数分开）
图 5  燃油、电耗和碳排放并列小图
图 6  车辆甘特图
图 7  绿色节点到达时刻合规图
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_problem_data  # noqa: E402

# ---------------------------------------------------------------------------
# 全局配色（与题目说明一致：新能源=蓝，燃油=橙，绿色区=绿，违规区间=红）
# ---------------------------------------------------------------------------
ELECTRIC = "#4C78A8"  # 蓝
FUEL = "#F58518"      # 橙
GREEN = "#54A24B"     # 绿
RED = "#E45756"       # 红
TEAL = "#72B7B2"
GRAY = "#A8B0BC"      # 非绿色客户
TEXT = "#263238"
GRID = "#D9DEE7"

GREEN_ZONE_CENTER = (0.0, 0.0)  # 市中心
GREEN_ZONE_RADIUS = 10.0        # km
RESTRICT_START = 480.0          # 08:00
RESTRICT_END = 960.0            # 16:00


def configure_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_minutes(minutes: float) -> str:
    minutes = float(minutes)
    hours = int(minutes // 60)
    mins = int(round(minutes % 60))
    if mins >= 60:
        hours += 1
        mins -= 60
    return f"{hours:02d}:{mins:02d}"


# ---------------------------------------------------------------------------
# 图 1  客户分布和绿色配送区
# ---------------------------------------------------------------------------
def plot_fig1_green_zone(problem, green_audit: list[dict], output: Path) -> None:
    coords = problem.coordinates
    green_ids = set(problem.green_customer_ids)
    active_ids = set(problem.active_customer_ids)

    inactive_green = {int(r["customer_id"]) for r in green_audit if r["is_active_customer"] == "0"}

    fig, ax = plt.subplots(figsize=(8.2, 8.0))

    # 绿色配送区边界
    ax.add_patch(
        Circle(
            GREEN_ZONE_CENTER,
            GREEN_ZONE_RADIUS,
            fill=False,
            linestyle="--",
            linewidth=1.4,
            edgecolor=GREEN,
            label="绿色配送区（半径 10 km）",
        )
    )

    # 非绿色客户（灰色点）
    nongreen = [cid for cid in coords if cid != 0 and cid not in green_ids]
    ax.scatter(
        [coords[c][0] for c in nongreen],
        [coords[c][1] for c in nongreen],
        s=26,
        color=GRAY,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
        label="非绿色区客户",
    )

    # 绿色区有需求客户（绿色实心）
    green_active = [cid for cid in green_ids if cid in active_ids]
    ax.scatter(
        [coords[c][0] for c in green_active],
        [coords[c][1] for c in green_active],
        s=70,
        color=GREEN,
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
        label="绿色区有需求客户",
    )

    # 绿色区无需求客户（绿色空心）
    if inactive_green:
        ax.scatter(
            [coords[c][0] for c in sorted(inactive_green)],
            [coords[c][1] for c in sorted(inactive_green)],
            s=70,
            facecolor="none",
            edgecolor=GREEN,
            linewidth=1.6,
            zorder=4,
            label="绿色区无需求客户（1、14、15）",
        )

    # 市中心
    ax.scatter(
        [GREEN_ZONE_CENTER[0]],
        [GREEN_ZONE_CENTER[1]],
        marker="+",
        s=140,
        color=TEXT,
        linewidth=1.6,
        zorder=5,
        label="市中心 (0,0)",
    )

    # 配送中心
    depot_x, depot_y = coords[0]
    ax.scatter(
        [depot_x],
        [depot_y],
        marker="*",
        s=320,
        color=RED,
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
        label="配送中心",
    )
    ax.annotate(
        "配送中心",
        (depot_x, depot_y),
        xytext=(9, 9),
        textcoords="offset points",
        fontsize=10,
        color=RED,
    )

    ax.set_xlabel("X 坐标（km）")
    ax.set_ylabel("Y 坐标（km）")
    ax.set_title("图 1  客户分布与绿色配送区", pad=12)
    ax.grid(alpha=0.15)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", frameon=False, fontsize=9)

    note = (
        "注：按市中心 (0,0) 半径 10 km 几何判定，绿色区客户共 15 个"
        "（题面文字所称 30 个与坐标计算不一致）；其中 12 个有当日需求。"
    )
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8.5, color=TEXT)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 2  问题二路线图（含市中心局部放大）
# ---------------------------------------------------------------------------
def plot_fig2_route_map(problem, route_rows: list[dict], output: Path) -> None:
    coords = problem.coordinates
    route_groups: dict[str, list[dict]] = defaultdict(list)
    for row in route_rows:
        route_groups[row["route_id"]].append(row)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(15, 7.6))

    def draw_routes(ax) -> None:
        for rows in route_groups.values():
            ordered = sorted(rows, key=lambda r: int(r["sequence"]))
            nodes = [0] + [int(r["customer_id"]) for r in ordered] + [0]
            xs = [coords[n][0] for n in nodes]
            ys = [coords[n][1] for n in nodes]
            propulsion = ordered[0]["propulsion"]
            color = ELECTRIC if propulsion == "electric" else FUEL
            ax.plot(xs, ys, color=color, linewidth=0.7, alpha=0.30, zorder=1)

    def draw_green_circle(ax) -> None:
        ax.add_patch(
            Circle(
                GREEN_ZONE_CENTER,
                GREEN_ZONE_RADIUS,
                fill=False,
                linestyle="--",
                linewidth=1.4,
                edgecolor=GREEN,
            )
        )

    def draw_depot(ax) -> None:
        depot_x, depot_y = coords[0]
        ax.scatter(
            [depot_x], [depot_y], marker="*", s=260, color=RED,
            edgecolor="white", linewidth=0.8, zorder=5,
        )
        ax.annotate(
            "配送中心", (depot_x, depot_y), xytext=(8, 8),
            textcoords="offset points", fontsize=9, color=RED,
        )

    # 全图
    draw_routes(ax_full)
    draw_green_circle(ax_full)
    draw_depot(ax_full)
    ax_full.set_title("问题二配送路线空间分布（全图）", pad=10)
    ax_full.set_xlabel("X 坐标（km）")
    ax_full.set_ylabel("Y 坐标（km）")
    ax_full.grid(alpha=0.15)
    ax_full.set_aspect("equal", adjustable="datalim")

    # 市中心放大图
    draw_routes(ax_zoom)
    draw_green_circle(ax_zoom)
    ax_zoom.scatter(
        [GREEN_ZONE_CENTER[0]], [GREEN_ZONE_CENTER[1]],
        marker="+", s=160, color=TEXT, linewidth=1.6, zorder=5,
    )
    ax_zoom.set_title("市中心绿色配送区局部放大", pad=10)
    ax_zoom.set_xlabel("X 坐标（km）")
    ax_zoom.set_ylabel("Y 坐标（km）")
    ax_zoom.grid(alpha=0.15)
    ax_zoom.set_aspect("equal", adjustable="datalim")
    # 绿色区客户堆叠在市中心附近，放大到略超半径 10 km
    ax_zoom.set_xlim(-13, 13)
    ax_zoom.set_ylim(-13, 13)

    handles = [
        Line2D([0], [0], color=ELECTRIC, lw=2.6, label="新能源车路线"),
        Line2D([0], [0], color=FUEL, lw=2.6, label="燃油车路线"),
        Line2D([0], [0], color=GREEN, lw=1.6, linestyle="--", label="绿色配送区边界"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("图 2  问题二配送路线图", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 3  成本构成及政策变化瀑布图
# ---------------------------------------------------------------------------
def plot_fig3_cost_waterfall(q1: dict, q2: dict, output: Path) -> None:
    labels = ["固定成本", "能源成本", "碳成本", "等待成本", "迟到成本"]
    keys = ["fixed_cost", "energy_cost", "carbon_cost", "waiting_cost", "late_cost"]
    q1_vals = [float(q1[k]) for k in keys]
    q2_vals = [float(q2[k]) for k in keys]

    fig, (ax_bar, ax_wf) = plt.subplots(1, 2, figsize=(14, 6.2))

    # (a) 分项成本对比
    x = np.arange(len(labels))
    width = 0.38
    ax_bar.bar(x - width / 2, q1_vals, width, color="#B9C4D4", label="问题一")
    ax_bar.bar(x + width / 2, q2_vals, width, color=TEAL, label="问题二")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=9)
    ax_bar.set_ylabel("成本（元）")
    ax_bar.set_title("(a) 分项成本对比", pad=10)
    ax_bar.grid(axis="y", alpha=0.2)
    ax_bar.legend(frameon=False)
    for i, (v1, v2) in enumerate(zip(q1_vals, q2_vals)):
        ax_bar.text(x[i] - width / 2, v1, f"{v1:,.0f}", ha="center", va="bottom", fontsize=7)
        ax_bar.text(x[i] + width / 2, v2, f"{v2:,.0f}", ha="center", va="bottom", fontsize=7)

    # (b) 瀑布图
    q1_total = float(q1["total_cost"])
    q2_total = float(q2["total_cost"])
    deltas = [q2_vals[i] - q1_vals[i] for i in range(len(labels))]

    steps = ["问题一"] + labels + ["问题二"]
    bottoms = [0.0] * (len(steps))
    heights = [0.0] * (len(steps))

    heights[0] = q1_total
    running = q1_total
    for i, delta in enumerate(deltas):
        bottoms[i + 1] = running
        heights[i + 1] = delta
        running += delta
    heights[-1] = q2_total  # 末柱从 0 起
    bottoms[-1] = 0.0

    colors = ["#B9C4D4"] + [
        GREEN if d < 0 else (RED if labels[i] == "迟到成本" else FUEL)
        for i, d in enumerate(deltas)
    ] + [TEAL]

    for i in range(len(steps)):
        ax_wf.bar(i, heights[i], bottom=bottoms[i], color=colors[i], width=0.6, zorder=3)
        top = bottoms[i] + heights[i]
        val_text = f"{heights[i]:+,.0f}" if 0 < i < len(steps) - 1 else f"{heights[i]:,.0f}"
        ax_wf.text(i, top, val_text, ha="center", va="bottom", fontsize=7.5)

    # 连接线
    for i in range(len(steps) - 1):
        y_conn = bottoms[i + 1]
        ax_wf.plot([i + 0.3, i + 1 - 0.3], [y_conn, y_conn], color=GRID, lw=1, zorder=2)

    ax_wf.set_xticks(range(len(steps)))
    ax_wf.set_xticklabels(steps, fontsize=8, rotation=20, ha="right")
    ax_wf.set_ylabel("总成本（元）")
    ax_wf.set_title("(b) 政策影响瀑布图（问题一 → 问题二）", pad=10)
    ax_wf.grid(axis="y", alpha=0.2)
    ax_wf.set_ylim(0, q2_total * 1.18)

    fig.suptitle("图 3  成本构成与政策变化（迟到成本已单独突出）", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 4  车辆结构对比
# ---------------------------------------------------------------------------
def plot_fig4_vehicle_structure(rows: list[dict], output: Path) -> None:
    types = [r["vehicle_type"] for r in rows]
    q1_phys = [int(r["q1_physical_vehicles"]) for r in rows]
    q2_phys = [int(r["q2_physical_vehicles"]) for r in rows]
    q1_trip = [int(r["q1_trips"]) for r in rows]
    q2_trip = [int(r["q2_trips"]) for r in rows]

    fig, (ax_phys, ax_trip) = plt.subplots(1, 2, figsize=(11, 5.4))
    x = np.arange(len(types))
    width = 0.38

    def bar_pair(ax, q1v, q2v, ylabel, title):
        ax.bar(x - width / 2, q1v, width, color="#B9C4D4", label="问题一")
        ax.bar(x + width / 2, q2v, width, color=TEAL, label="问题二")
        ax.set_xticks(x)
        ax.set_xticklabels(types)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
        for i, (v1, v2) in enumerate(zip(q1v, q2v)):
            ax.text(x[i] - width / 2, v1, str(v1), ha="center", va="bottom", fontsize=10)
            ax.text(x[i] + width / 2, v2, str(v2), ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, max(max(q1v), max(q2v)) * 1.18)

    bar_pair(ax_phys, q1_phys, q2_phys, "物理车辆数（辆）", "(a) 物理车辆数对比")
    bar_pair(ax_trip, q1_trip, q2_trip, "执行趟次数（趟）", "(b) 执行趟次数对比")

    fig.suptitle("图 4  车辆结构对比（数量与使用强度分开）", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 5  燃油、电耗和碳排放
# ---------------------------------------------------------------------------
def plot_fig5_energy_emissions(impact_rows: list[dict], output: Path) -> None:
    by_metric = {r["metric"]: r for r in impact_rows}
    panels = [
        ("fuel_liters", "燃油消耗（L）", FUEL),
        ("electricity_kwh", "电耗（kWh）", ELECTRIC),
        ("emissions_kg", "碳排放（kg）", RED),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    x = np.arange(2)
    for ax, (metric, ylabel, color) in zip(axes, panels):
        row = by_metric[metric]
        q1v = float(row["question1"])
        q2v = float(row["question2"])
        ax.bar(x, [q1v, q2v], color=["#B9C4D4", color], width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["问题一", "问题二"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        # 碳排放纵轴不从非零截断，从 0 开始
        ax.set_ylim(0, max(q1v, q2v) * 1.15)
        for i, v in enumerate((q1v, q2v)):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("图 5  燃油、电耗与碳排放对比（并列小图）", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 6  车辆甘特图
# ---------------------------------------------------------------------------
def plot_fig6_gantt(summary_rows: list[dict], route_rows: list[dict], output: Path) -> None:
    # 仅绿色客户停靠需要绿色标记
    green_stops = [r for r in route_rows if r.get("is_green_customer") == "1"]
    # 按物理车辆分组
    vehicles: dict[str, list[dict]] = defaultdict(list)
    for row in summary_rows:
        vehicles[row["physical_vehicle_id"]].append(row)

    # 新能源在前、燃油在后
    def vehicle_sort_key(vid: str):
        propulsion = 0 if vid.startswith("EV") else 1
        number = int(vid.rsplit("-", 1)[1])
        return (propulsion, number)

    vehicle_ids = sorted(vehicles, key=vehicle_sort_key)

    fig, ax = plt.subplots(figsize=(13, max(7, 0.32 * len(vehicle_ids) + 2)))
    n = len(vehicle_ids)
    y_pos = np.arange(n)[::-1]  # 第一辆车画在顶部

    # 08:00-16:00 限行区间背景
    ax.axvspan(RESTRICT_START, RESTRICT_END, color=RED, alpha=0.10, zorder=0)

    for i, vid in enumerate(vehicle_ids):
        y = y_pos[i]
        for row in sorted(vehicles[vid], key=lambda r: float(r["start_minutes_exact"])):
            start = float(row["start_minutes_exact"])
            finish = float(row["finish_minutes_exact"])
            propulsion = row["propulsion"]
            color = ELECTRIC if propulsion == "electric" else FUEL
            ax.barh(y, finish - start, left=start, height=0.62, color=color, alpha=0.85, zorder=2)

    # 绿色客户停靠标记
    for stop in green_stops:
        vid = stop["physical_vehicle_id"]
        if vid in vehicles:
            y = y_pos[vehicle_ids.index(vid)]
            ax.scatter(
                float(stop["arrival_minutes_exact"]),
                y,
                marker="o",
                s=16,
                color=GREEN,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(vehicle_ids, fontsize=6.5)
    ax.set_xlim(RESTRICT_START, 1440)
    ax.set_xlabel("时间")
    ax.set_ylabel("物理车辆")
    ax.set_title("图 6  物理车辆多趟配送甘特图（08:00-16:00 限行区间标灰红）", pad=10)

    # 时间刻度（每 2 小时）
    ticks = list(range(480, 1441, 120))
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt_minutes(t) for t in ticks], fontsize=8)
    ax.grid(axis="x", alpha=0.15)

    handles = [
        Patch(color=ELECTRIC, label="新能源车趟次"),
        Patch(color=FUEL, label="燃油车趟次"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, label="绿色客户停靠"),
        Patch(color=RED, alpha=0.15, label="限行区间 08:00-16:00"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 7  绿色节点到达时刻合规图
# ---------------------------------------------------------------------------
def plot_fig7_green_compliance(checks: list[dict], output: Path) -> None:
    # 纵轴为绿色客户访问记录：按客户编号分组，客户 2..13 排序
    customer_ids = sorted({int(r["customer_id"]) for r in checks})
    fig, ax = plt.subplots(figsize=(11, 5.6))

    ax.axvspan(RESTRICT_START, RESTRICT_END, color=RED, alpha=0.12, zorder=0)

    y_map = {cid: i for i, cid in enumerate(customer_ids)}
    # 轻微抖动避免同客户多次到访重叠
    rng = np.random.default_rng(0)
    for row in checks:
        cid = int(row["customer_id"])
        t = float(row["arrival_minutes_exact"])
        propulsion = row["propulsion"]
        color = ELECTRIC if propulsion == "electric" else FUEL
        y = y_map[cid] + rng.uniform(-0.18, 0.18)
        ax.scatter(t, y, s=42, color=color, edgecolor="white", linewidth=0.5, zorder=3)

    # 16:00 分界线
    ax.axvline(RESTRICT_END, color=RED, linestyle=":", linewidth=1.2)
    ax.text(RESTRICT_END + 6, len(customer_ids) - 0.4, "16:00", color=RED, fontsize=9)

    ax.set_yticks(range(len(customer_ids)))
    ax.set_yticklabels([f"客户 {cid}" for cid in customer_ids], fontsize=8)
    ax.set_xlim(RESTRICT_START - 30, 1440)
    ax.set_xlabel("到达时刻（08:00-24:00）")
    ax.set_ylabel("绿色客户访问记录")
    ax.set_title("图 7  绿色节点到达时刻合规图（红色区间内无燃油车到访）", pad=10)

    ticks = list(range(480, 1441, 120))
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt_minutes(t) for t in ticks], fontsize=8)
    ax.grid(axis="x", alpha=0.15)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ELECTRIC, markersize=8, label="新能源车到达"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FUEL, markersize=8, label="燃油车到达"),
        Patch(color=RED, alpha=0.15, label="燃油车禁入区间 [08:00,16:00)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="生成问题二论文 7 张图")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/team_cleaned"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/question2_optimized"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    configure_font()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    result_dir = args.result_dir if args.result_dir.is_absolute() else ROOT / args.result_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = load_problem_data(ROOT / args.data_dir if not args.data_dir.is_absolute() else args.data_dir)

    # 读取正式结果文件
    green_audit = read_csv(result_dir / "question2_green_customer_audit.csv")
    route_rows = read_csv(result_dir / "question2_optimized_routes.csv")
    summary_rows = read_csv(result_dir / "question2_optimized_route_summary.csv")
    green_checks = read_csv(result_dir / "question2_green_policy_checks.csv")
    vehicle_rows = read_csv(result_dir / "question2_vehicle_structure.csv")
    impact_rows = read_csv(result_dir / "question2_policy_impact.csv")

    q1 = read_json(ROOT / "results" / "tables" / "question1_optimized_totals.json")
    q2 = read_json(result_dir / "question2_optimized_totals.json")

    figure_paths = [
        output_dir / "question2_fig1_customer_green_zone.png",
        output_dir / "question2_fig2_route_map.png",
        output_dir / "question2_fig3_cost_waterfall.png",
        output_dir / "question2_fig4_vehicle_structure.png",
        output_dir / "question2_fig5_energy_emissions.png",
        output_dir / "question2_fig6_vehicle_gantt.png",
        output_dir / "question2_fig7_green_compliance.png",
    ]

    plot_fig1_green_zone(problem, green_audit, figure_paths[0])
    plot_fig2_route_map(problem, route_rows, figure_paths[1])
    plot_fig3_cost_waterfall(q1, q2, figure_paths[2])
    plot_fig4_vehicle_structure(vehicle_rows, figure_paths[3])
    plot_fig5_energy_emissions(impact_rows, figure_paths[4])
    plot_fig6_gantt(summary_rows, route_rows, figure_paths[5])
    plot_fig7_green_compliance(green_checks, figure_paths[6])

    for path in figure_paths:
        print(f"已生成：{path}")


if __name__ == "__main__":
    main()
