from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_problem_data
from src.data.q2_instance import build_q2_instance
from src.model.evaluator import RouteEvaluator, evaluate_solution, format_clock
from src.model.policy_q2 import build_q2_policy
from src.solver.greedy import build_greedy_routes, validate_solution
from src.solver.q2_baseline import build_q2_baseline
from src.solver.scheduling import validate_vehicle_schedule


def _write_instance_outputs(output_dir: Path, customer_frame, summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    customer_frame.to_csv(
        output_dir / "question2_green_customers.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "question2_instance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "question2_instance_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow(
                [
                    key,
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value,
                ]
            )


def _write_route_outputs(output_dir: Path, routes, solution, problem) -> None:
    with (output_dir / "question2_baseline_routes.csv").open(
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
                "arrival",
                "service_start",
                "departure",
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
                        format_clock(stop.arrival_minutes),
                        format_clock(stop.service_start_minutes),
                        format_clock(stop.departure_minutes),
                        round(stop.late_minutes, 3),
                        stop.policy_violation_reason or "",
                    ]
                )

    with (output_dir / "question2_baseline_route_summary.csv").open(
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
                    len(evaluation.stops),
                    round(route.total_weight, 6),
                    round(route.total_volume, 6),
                    round(evaluation.distance_km, 6),
                    round(evaluation.energy_amount, 6),
                    round(evaluation.emissions_kg, 6),
                    round(evaluation.fixed_cost, 2),
                    round(evaluation.energy_cost, 2),
                    round(evaluation.carbon_cost, 2),
                    round(evaluation.waiting_cost, 2),
                    round(evaluation.late_cost, 2),
                    evaluation.policy_violation_count,
                    round(evaluation.total_cost, 2),
                ]
            )

    totals = {
        "solution_variant": "question2_deterministic_baseline",
        "data_source": problem.data_source,
        "active_customers": len(problem.active_customer_ids),
        "green_customers": len(problem.green_customer_ids),
        "green_active_customers": len(
            set(problem.green_customer_ids) & set(problem.active_customer_ids)
        ),
        "delivery_trips": solution.trip_count,
        "physical_vehicles": solution.vehicle_count,
        "route_count_by_type": dict(
            Counter(route.vehicle_type.name for route in routes)
        ),
        "physical_vehicle_count_by_type": {
            vehicle_name: len(
                {
                    route.vehicle_number
                    for route in routes
                    if route.vehicle_type.name == vehicle_name
                }
            )
            for vehicle_name in sorted({route.vehicle_type.name for route in routes})
        },
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
            "绿色区客户集合直接读取清洗坐标表的是否绿色配送区字段",
            "燃油车到达绿色客户节点的时刻采用[08:00,16:00)判定",
            "新能源车不受绿色区限行影响",
            "同一物理车辆返回配送中心后可继续执行下一趟，启动成本每天计一次",
        ],
    }
    (output_dir / "question2_baseline_totals.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_sensitivity(output_dir: Path, problem) -> None:
    base_routes = build_greedy_routes(problem)
    rows = []
    for label, end_inclusive in (
        ("[08:00,16:00)", False),
        ("[08:00,16:00]", True),
    ):
        routes = [
            type(route)(
                route.vehicle_type,
                route.vehicle_number,
                list(route.deliveries),
                route.start_minutes,
                route.trip_number,
            )
            for route in base_routes
        ]
        policy = build_q2_policy(problem.green_customer_ids, end_inclusive=end_inclusive)
        evaluator = RouteEvaluator(problem, policy=policy)
        result = evaluate_solution(routes, evaluator, optimize_departures=True)
        rows.append(
            {
                "endpoint_interpretation": label,
                "policy_violation_count_on_q1_greedy": result.policy_violation_count,
                "total_cost_without_policy_repair": round(result.total_cost, 6),
                "late_stop_count": result.late_stop_count,
                "all_routes_return_before_24h": result.all_routes_return_before_24h,
                "note": "端点差异仅影响恰好在08:00或16:00到达的燃油车绿色节点",
            }
        )
    with (output_dir / "question2_policy_sensitivity.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(data_dir: Path, output_dir: Path) -> dict[str, object]:
    customer_frame, instance_summary = build_q2_instance(data_dir)
    _write_instance_outputs(output_dir, customer_frame, instance_summary)

    problem = load_problem_data(data_dir)
    policy = build_q2_policy(problem.green_customer_ids)
    routes = build_q2_baseline(problem, policy)
    evaluator = RouteEvaluator(problem, policy=policy)
    solution = evaluate_solution(routes, evaluator, optimize_departures=False)
    validate_solution(problem, routes)
    validate_vehicle_schedule(routes, evaluator)
    _write_route_outputs(output_dir, routes, solution, problem)
    _write_sensitivity(output_dir, problem)
    print(json.dumps(instance_summary, ensure_ascii=False, indent=2))
    print(
        f"Q2基线：总成本 {solution.total_cost:.2f} 元，"
        f"{solution.vehicle_count} 辆物理车，{solution.trip_count} 趟，"
        f"政策违规 {solution.policy_violation_count} 次"
    )
    print(f"输出目录：{output_dir.resolve()}")
    return instance_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题二启动：绿色客户、政策模块与确定性基线")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/team_cleaned"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/question2_startup"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.data_dir, arguments.output_dir)
