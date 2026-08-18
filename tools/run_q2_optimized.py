from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_problem_data
from src.model.evaluator import RouteEvaluator, format_clock
from src.model.policy_q2 import build_q2_policy
from src.solver.greedy import validate_solution
from src.solver.q2_initial import load_route_solution
from src.solver.q2_scheduling import search_q2_schedules
from src.solver.scheduling import validate_vehicle_schedule


def _physical_counts(routes) -> dict[str, int]:
    return {
        name: len(
            {
                route.vehicle_number
                for route in routes
                if route.vehicle_type.name == name
            }
        )
        for name in sorted({route.vehicle_type.name for route in routes})
    }


def _write_routes(output_dir: Path, routes, solution) -> None:
    with (output_dir / "question2_optimized_routes.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "route_id",
                "physical_vehicle_id",
                "trip_number",
                "vehicle_type",
                "propulsion",
                "sequence",
                "customer_id",
                "delivered_weight_kg",
                "delivered_volume_m3",
                "leg_distance_km",
                "arrival",
                "arrival_minutes_exact",
                "service_start",
                "departure",
                "waiting_minutes",
                "late_minutes",
                "policy_violation_reason",
            ]
        )
        for route, evaluation in zip(routes, solution.routes, strict=True):
            for stop in evaluation.stops:
                writer.writerow(
                    [
                        route.route_id,
                        f"{route.vehicle_type.name}-{route.vehicle_number:03d}",
                        route.trip_number,
                        route.vehicle_type.name,
                        route.vehicle_type.propulsion,
                        stop.sequence,
                        stop.customer_id,
                        round(stop.delivered_weight, 6),
                        round(stop.delivered_volume, 6),
                        round(stop.leg_distance, 6),
                        format_clock(stop.arrival_minutes),
                        round(stop.arrival_minutes, 9),
                        format_clock(stop.service_start_minutes),
                        format_clock(stop.departure_minutes),
                        round(stop.waiting_minutes, 6),
                        round(stop.late_minutes, 6),
                        stop.policy_violation_reason or "",
                    ]
                )

    with (output_dir / "question2_optimized_route_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "route_id",
                "physical_vehicle_id",
                "trip_number",
                "vehicle_type",
                "propulsion",
                "start",
                "finish",
                "start_minutes_exact",
                "finish_minutes_exact",
                "stops",
                "load_weight_kg",
                "load_volume_m3",
                "distance_km",
                "energy_amount",
                "emissions_kg",
                "fixed_cost",
                "energy_cost",
                "carbon_cost",
                "waiting_cost",
                "late_cost",
                "policy_violation_count",
                "total_cost",
            ]
        )
        for route, evaluation in zip(routes, solution.routes, strict=True):
            writer.writerow(
                [
                    route.route_id,
                    f"{route.vehicle_type.name}-{route.vehicle_number:03d}",
                    route.trip_number,
                    route.vehicle_type.name,
                    route.vehicle_type.propulsion,
                    format_clock(evaluation.start_minutes),
                    format_clock(evaluation.finish_minutes),
                    round(evaluation.start_minutes, 9),
                    round(evaluation.finish_minutes, 9),
                    len(evaluation.stops),
                    round(route.total_weight, 6),
                    round(route.total_volume, 6),
                    round(evaluation.distance_km, 6),
                    round(evaluation.energy_amount, 6),
                    round(evaluation.emissions_kg, 6),
                    round(evaluation.fixed_cost, 6),
                    round(evaluation.energy_cost, 6),
                    round(evaluation.carbon_cost, 6),
                    round(evaluation.waiting_cost, 6),
                    round(evaluation.late_cost, 6),
                    evaluation.policy_violation_count,
                    round(evaluation.total_cost, 6),
                ]
            )


def _delta(before: float, after: float) -> tuple[float, float | None]:
    change = after - before
    percent = change / before * 100.0 if abs(before) > 1e-12 else None
    return change, percent


def _write_comparison(
    output_dir: Path,
    q1: dict[str, object],
    q2: dict[str, object],
) -> list[dict[str, object]]:
    metrics = [
        ("total_cost", "总成本/元", "total_cost", "total_cost"),
        ("physical_vehicles", "物理车辆/辆", "physical_vehicles", "physical_vehicles"),
        ("delivery_trips", "配送趟次/趟", "delivery_trips", "delivery_trips"),
        ("total_distance_km", "总里程/km", "total_distance_km", "total_distance_km"),
        ("emissions_kg", "碳排放/kg", "total_emissions_kg", "emissions_kg"),
        ("fuel_liters", "燃油/L", "fuel_liters", "fuel_liters"),
        ("electricity_kwh", "电耗/kWh", "electricity_kwh", "electricity_kwh"),
        ("fixed_cost", "固定成本/元", "fixed_cost", "fixed_cost"),
        ("energy_cost", "能耗成本/元", "energy_cost", "energy_cost"),
        ("carbon_cost", "碳成本/元", "carbon_cost", "carbon_cost"),
        ("waiting_cost", "等待成本/元", "waiting_cost", "waiting_cost"),
        ("late_cost", "迟到成本/元", "late_cost", "late_cost"),
    ]
    rows: list[dict[str, object]] = []
    for metric, label, q1_key, q2_key in metrics:
        before = float(q1[q1_key])
        after = float(q2[q2_key])
        change, percent = _delta(before, after)
        rows.append(
            {
                "metric": metric,
                "label": label,
                "question1": round(before, 6),
                "question2": round(after, 6),
                "absolute_change": round(change, 6),
                "percent_change": "" if percent is None else round(percent, 6),
            }
        )
    with (output_dir / "question2_policy_impact.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_vehicle_structure(output_dir: Path, q1, routes) -> list[dict[str, object]]:
    q1_trips = Counter({key: int(value) for key, value in q1["trip_usage_by_type"].items()})
    q1_physical = Counter(
        {key: int(value) for key, value in q1["physical_vehicle_usage"].items()}
    )
    q2_trips = Counter(route.vehicle_type.name for route in routes)
    q2_physical = Counter(_physical_counts(routes))
    rows = []
    for vehicle_type in sorted(set(q1_trips) | set(q2_trips)):
        rows.append(
            {
                "vehicle_type": vehicle_type,
                "propulsion": "electric" if vehicle_type.startswith("EV") else "fuel",
                "q1_physical_vehicles": q1_physical[vehicle_type],
                "q2_physical_vehicles": q2_physical[vehicle_type],
                "physical_vehicle_change": q2_physical[vehicle_type]
                - q1_physical[vehicle_type],
                "q1_trips": q1_trips[vehicle_type],
                "q2_trips": q2_trips[vehicle_type],
                "trip_change": q2_trips[vehicle_type] - q1_trips[vehicle_type],
            }
        )
    with (output_dir / "question2_vehicle_structure.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run(data_dir: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    problem = load_problem_data(data_dir)
    evaluator = RouteEvaluator(
        problem,
        policy=build_q2_policy(problem.green_customer_ids),
    )
    q1_routes = load_route_solution(
        Path("results/routes/question1_optimized_routes.csv"),
        Path("results/tables/question1_optimized_route_summary.csv"),
    )
    startup_weights = tuple(range(0, 81, 10)) + tuple(range(81, 101))
    best, runs = search_q2_schedules(
        q1_routes,
        evaluator,
        seeds=(202601, 202602, 202603),
        order_rules=("green_first",),
        startup_cost_weights=startup_weights,
    )
    routes = list(best.routes)
    solution = best.evaluation
    validate_solution(problem, routes)
    validate_vehicle_schedule(routes, evaluator)
    if solution.policy_violation_count != 0:
        raise AssertionError("正式Q2方案仍存在政策违规")

    _write_routes(output_dir, routes, solution)
    with (output_dir / "question2_search_runs.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fieldnames = [
            "seed",
            "order_rule",
            "startup_cost_weight",
            "total_cost",
            "physical_vehicles",
            "delivery_trips",
            "emissions_kg",
            "late_cost",
            "policy_violation_count",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in sorted(runs, key=lambda run: (run.evaluation.total_cost, run.seed)):
            writer.writerow(
                {
                    "seed": item.seed,
                    "order_rule": item.order_rule,
                    "startup_cost_weight": item.startup_cost_weight,
                    "total_cost": round(item.evaluation.total_cost, 6),
                    "physical_vehicles": item.evaluation.vehicle_count,
                    "delivery_trips": item.evaluation.trip_count,
                    "emissions_kg": round(item.evaluation.emissions_kg, 6),
                    "late_cost": round(item.evaluation.late_cost, 6),
                    "policy_violation_count": item.evaluation.policy_violation_count,
                }
            )

    totals: dict[str, object] = {
        "solution_variant": "question2_q1_policy_repair_optimized",
        "algorithm": (
            "Q1路线继承 + 绿色节点顺序修复 + 10分钟发车时刻枚举 + "
            "异构车型选择 + 多趟物理车辆复用 + 固定种子参数搜索"
        ),
        "selected_parameters": {
            "seed": best.seed,
            "order_rule": best.order_rule,
            "startup_cost_weight": best.startup_cost_weight,
            "departure_step_minutes": 10,
        },
        "data_source": problem.data_source,
        "active_customers": len(problem.active_customer_ids),
        "green_customers": len(problem.green_customer_ids),
        "green_active_customers": len(
            set(problem.green_customer_ids) & set(problem.active_customer_ids)
        ),
        "delivery_trips": solution.trip_count,
        "physical_vehicles": solution.vehicle_count,
        "trip_usage_by_type": dict(Counter(route.vehicle_type.name for route in routes)),
        "physical_vehicle_usage": _physical_counts(routes),
        "propulsion_trip_count": dict(
            Counter(route.vehicle_type.propulsion for route in routes)
        ),
        "total_distance_km": solution.total_distance_km,
        "fuel_liters": solution.fuel_liters,
        "electricity_kwh": solution.electricity_kwh,
        "emissions_kg": solution.emissions_kg,
        "fixed_cost": solution.fixed_cost,
        "energy_cost": solution.energy_cost,
        "carbon_cost": solution.carbon_cost,
        "waiting_cost": solution.waiting_cost,
        "late_cost": solution.late_cost,
        "total_cost": solution.total_cost,
        "feasibility": {
            "policy_violation_count": solution.policy_violation_count,
            "unfinished_customer_count": solution.unfinished_customer_count,
            "unfinished_weight_kg": solution.unfinished_weight_kg,
            "unfinished_volume_m3": solution.unfinished_volume_m3,
            "capacity_violation_count": solution.capacity_violation_count,
            "all_routes_return_before_24h": solution.all_routes_return_before_24h,
            "late_stop_count": solution.late_stop_count,
        },
        "assumptions": [
            "绿色区客户集合读取清洗坐标表的是否绿色配送区字段",
            "燃油车到达绿色客户节点的时刻采用[08:00,16:00)判定",
            "新能源车不受绿色区限行影响",
            "同一物理车辆返场后可继续执行下一趟，启动成本每天计一次",
            "所有配送趟次必须在24:00前返回配送中心",
        ],
    }
    q1 = json.loads(
        Path("results/tables/question1_optimized_totals.json").read_text(
            encoding="utf-8"
        )
    )
    impact_rows = _write_comparison(output_dir, q1, totals)
    vehicle_rows = _write_vehicle_structure(output_dir, q1, routes)
    totals["policy_impact"] = {
        row["metric"]: {
            "absolute_change": row["absolute_change"],
            "percent_change": row["percent_change"],
        }
        for row in impact_rows
    }
    (output_dir / "question2_optimized_totals.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    impact = {row["metric"]: row for row in impact_rows}
    markdown = [
        "# 问题二：绿色配送区限行后的优化结果",
        "",
        "政策口径：燃油车在 08:00–16:00 不得到达半径 10 km 绿色配送区客户；新能源车不限行。",
        "",
        "## 核心结果",
        "",
        "| 指标 | 问题一 | 问题二 | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("total_cost", "physical_vehicles", "delivery_trips", "emissions_kg"):
        row = impact[metric]
        markdown.append(
            f"| {row['label']} | {row['question1']:.2f} | {row['question2']:.2f} | "
            f"{row['absolute_change']:+.2f} ({row['percent_change']:+.2f}%) |"
        )
    markdown.extend(
        [
            "",
            "## 车辆结构",
            "",
            "| 车型 | 问题一物理车 | 问题二物理车 | 问题一趟次 | 问题二趟次 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in vehicle_rows:
        markdown.append(
            f"| {row['vehicle_type']} | {row['q1_physical_vehicles']} | "
            f"{row['q2_physical_vehicles']} | {row['q1_trips']} | {row['q2_trips']} |"
        )
    markdown.extend(
        [
            "",
            "## 合规性",
            "",
            f"- 政策违规：{solution.policy_violation_count} 次",
            f"- 未完成客户：{solution.unfinished_customer_count} 个",
            f"- 容量违规：{solution.capacity_violation_count} 趟",
            f"- 全部趟次 24:00 前返场：{'是' if solution.all_routes_return_before_24h else '否'}",
            "",
            "成本增加主要来自为避开限行时段而产生的迟到罚款；同时通过增加电动车执行趟次、提高燃油车多趟复用，物理车辆数和总碳排放均略有下降。",
        ]
    )
    (output_dir / "question2_policy_impact.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题二：基于问题一方案的限行重排与影响分析")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/team_cleaned"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/question2_optimized"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.data_dir, arguments.output_dir)
