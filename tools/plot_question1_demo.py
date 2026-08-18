"""问题一绘图入门：配送路线图 + 重量/体积装载率散点图。"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_problem_data
from src.model.domain import DEFAULT_VEHICLE_TYPES


COLORS = {
    "electric": "#2CA02C",
    "fuel": "#FF7F0E",
    "customer": "#4C78A8",
    "depot": "#E45756",
}


def configure_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_route_map(
    detail_rows: list[dict[str, str]],
    problem,
    output_path: Path,
) -> None:
    route_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in detail_rows:
        route_rows[row["route_id"]].append(row)

    fig, ax = plt.subplots(figsize=(10, 8))
    for rows in route_rows.values():
        ordered = sorted(rows, key=lambda row: int(row["sequence"]))
        customer_ids = [int(row["customer_id"]) for row in ordered]
        nodes = [0, *customer_ids, 0]
        xs = [problem.coordinates[node][0] for node in nodes]
        ys = [problem.coordinates[node][1] for node in nodes]
        propulsion = ordered[0]["propulsion"]
        ax.plot(
            xs,
            ys,
            color=COLORS[propulsion],
            linewidth=0.8,
            alpha=0.24,
            zorder=1,
        )

    active_customers = list(problem.active_customer_ids)
    maximum_demand = max(problem.demands[node][0] for node in active_customers)
    point_sizes = [
        14 + 60 * problem.demands[node][0] / maximum_demand
        for node in active_customers
    ]
    ax.scatter(
        [problem.coordinates[node][0] for node in active_customers],
        [problem.coordinates[node][1] for node in active_customers],
        s=point_sizes,
        color=COLORS["customer"],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
        zorder=3,
    )
    depot_x, depot_y = problem.coordinates[0]
    ax.scatter(
        [depot_x],
        [depot_y],
        marker="*",
        s=260,
        color=COLORS["depot"],
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.annotate(
        "配送中心",
        (depot_x, depot_y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
    )

    ax.legend(
        handles=[
            Line2D([0], [0], color=COLORS["electric"], lw=3, label="新能源车路线"),
            Line2D([0], [0], color=COLORS["fuel"], lw=3, label="燃油车路线"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["customer"], label="客户点"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["depot"], markersize=13, label="配送中心"),
        ],
        loc="best",
        frameon=False,
    )
    ax.set_title("问题一最优配送路线空间分布", pad=14)
    ax.set_xlabel("X 坐标（km）")
    ax.set_ylabel("Y 坐标（km）")
    ax.grid(alpha=0.15)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_load_rate_scatter(
    summary_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    vehicles = {vehicle.name: vehicle for vehicle in DEFAULT_VEHICLE_TYPES}
    points: list[tuple[float, float, str, str]] = []
    for row in summary_rows:
        vehicle = vehicles[row["vehicle_type"]]
        weight_rate = float(row["load_weight_kg"]) / vehicle.capacity_weight
        volume_rate = float(row["load_volume_m3"]) / vehicle.capacity_volume
        points.append((weight_rate, volume_rate, vehicle.propulsion, row["route_id"]))

    fig, ax = plt.subplots(figsize=(8, 7))
    for propulsion, label in (("electric", "新能源车趟次"), ("fuel", "燃油车趟次")):
        selected = [point for point in points if point[2] == propulsion]
        ax.scatter(
            [point[0] for point in selected],
            [point[1] for point in selected],
            s=42,
            color=COLORS[propulsion],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )

    ax.axvline(0.8, color=COLORS["depot"], linestyle="--", linewidth=1.2)
    ax.axhline(0.8, color=COLORS["depot"], linestyle="--", linewidth=1.2, label="80%参考线")
    ax.axvline(1.0, color="#666666", linewidth=0.8)
    ax.axhline(1.0, color="#666666", linewidth=0.8)
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("重量装载率")
    ax.set_ylabel("体积装载率")
    ax.set_title("各配送趟次重量与体积装载率", pad=14)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="lower left")

    low_load_count = sum(max(weight, volume) < 0.8 for weight, volume, _, _ in points)
    ax.text(
        0.98,
        0.04,
        f"低于80%的趟次：{low_load_count}/{len(points)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制问题一路线图和装载率散点图")
    parser.add_argument(
        "--variant",
        choices=("optimized", "balanced_49"),
        default="optimized",
        help="optimized 为38辆最低成本方案；balanced_49 为49辆对照方案",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/demo"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_chinese_font()
    route_path = ROOT / "results" / "routes" / f"question1_{args.variant}_routes.csv"
    summary_path = ROOT / "results" / "tables" / f"question1_{args.variant}_route_summary.csv"
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with route_path.open(encoding="utf-8-sig", newline="") as stream:
        detail_rows = list(csv.DictReader(stream))
    with summary_path.open(encoding="utf-8-sig", newline="") as stream:
        summary_rows = list(csv.DictReader(stream))
    problem = load_problem_data(ROOT / "data" / "processed" / "team_cleaned")

    route_map_path = output_dir / f"question1_{args.variant}_route_map.png"
    load_rate_path = output_dir / f"question1_{args.variant}_load_rates.png"
    plot_route_map(detail_rows, problem, route_map_path)
    plot_load_rate_scatter(summary_rows, load_rate_path)
    print(f"已生成：{route_map_path}")
    print(f"已生成：{load_rate_path}")


if __name__ == "__main__":
    main()
