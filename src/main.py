from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.data.loader import load_problem_data
from src.model.evaluator import RouteEvaluation, RouteEvaluator, format_clock
from src.solver.greedy import build_greedy_routes, validate_solution
from src.solver.local_search import improve_routes_two_opt


def _write_outputs(
    output_dir: Path,
    evaluations: list[RouteEvaluation],
    routes,
    problem,
) -> None:
    route_dir = output_dir / "routes"
    table_dir = output_dir / "tables"
    route_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    detail_path = route_dir / "question1_baseline_routes.csv"
    with detail_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "route_id",
                "vehicle_type",
                "propulsion",
                "sequence",
                "customer_id",
                "delivered_weight_kg",
                "delivered_volume_m3",
                "leg_distance_km",
                "arrival",
                "service_start",
                "departure",
                "waiting_minutes",
                "late_minutes",
            ]
        )
        for route, evaluation in zip(routes, evaluations, strict=True):
            for stop in evaluation.stops:
                writer.writerow(
                    [
                        route.route_id,
                        route.vehicle_type.name,
                        route.vehicle_type.propulsion,
                        stop.sequence,
                        stop.customer_id,
                        round(stop.delivered_weight, 6),
                        round(stop.delivered_volume, 6),
                        round(stop.leg_distance, 6),
                        format_clock(stop.arrival_minutes),
                        format_clock(stop.service_start_minutes),
                        format_clock(stop.departure_minutes),
                        round(stop.waiting_minutes, 3),
                        round(stop.late_minutes, 3),
                    ]
                )

    summary_path = table_dir / "question1_baseline_route_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "route_id",
                "vehicle_type",
                "start",
                "finish",
                "stops",
                "load_weight_kg",
                "load_volume_m3",
                "distance_km",
                "energy_amount",
                "energy_unit",
                "emissions_kg",
                "fixed_cost",
                "energy_cost",
                "carbon_cost",
                "waiting_cost",
                "late_cost",
                "total_cost",
            ]
        )
        for route, result in zip(routes, evaluations, strict=True):
            writer.writerow(
                [
                    route.route_id,
                    route.vehicle_type.name,
                    format_clock(result.start_minutes),
                    format_clock(result.finish_minutes),
                    len(result.stops),
                    round(route.total_weight, 6),
                    round(route.total_volume, 6),
                    round(result.distance_km, 6),
                    round(result.energy_amount, 6),
                    "L" if route.vehicle_type.propulsion == "fuel" else "kWh",
                    round(result.emissions_kg, 6),
                    round(result.fixed_cost, 2),
                    round(result.energy_cost, 2),
                    round(result.carbon_cost, 2),
                    round(result.waiting_cost, 2),
                    round(result.late_cost, 2),
                    round(result.total_cost, 2),
                ]
            )

    totals = {
        "algorithm": "time-window-aware split-delivery greedy + intra-route 2-opt",
        "active_customers": len(problem.active_customer_ids),
        "zero_demand_customers": len(problem.all_customer_ids) - len(problem.active_customer_ids),
        "routes": len(routes),
        "vehicle_usage": dict(Counter(route.vehicle_type.name for route in routes)),
        "imputed_weight_rows": problem.imputed_weight_rows,
        "imputed_volume_rows": problem.imputed_volume_rows,
        "total_distance_km": sum(item.distance_km for item in evaluations),
        "fuel_liters": sum(
            item.energy_amount
            for route, item in zip(routes, evaluations, strict=True)
            if route.vehicle_type.propulsion == "fuel"
        ),
        "electricity_kwh": sum(
            item.energy_amount
            for route, item in zip(routes, evaluations, strict=True)
            if route.vehicle_type.propulsion == "electric"
        ),
        "total_emissions_kg": sum(item.emissions_kg for item in evaluations),
        "fixed_cost": sum(item.fixed_cost for item in evaluations),
        "energy_cost": sum(item.energy_cost for item in evaluations),
        "carbon_cost": sum(item.carbon_cost for item in evaluations),
        "waiting_cost": sum(item.waiting_cost for item in evaluations),
        "late_cost": sum(item.late_cost for item in evaluations),
        "total_cost": sum(item.total_cost for item in evaluations),
        "assumptions": [
            "缺失重量/体积先用同客户中位数填补，再用全局中位数兜底",
            "客户总需求允许按重量与体积同比例拆分给多辆车",
            "车速使用题面正态分布的均值，并按跨时段分段行驶",
            "17:00 后题面未给速度，基线按顺畅时段均值 55.3 km/h",
            "载荷能耗增幅按重量/容积利用率最大值线性插值",
        ],
    }
    (table_dir / "question1_baseline_totals.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(data_dir: Path, output_dir: Path, use_two_opt: bool = True) -> dict[str, float]:
    problem = load_problem_data(data_dir)
    evaluator = RouteEvaluator(problem)
    routes = build_greedy_routes(problem)
    validate_solution(problem, routes)
    if use_two_opt:
        routes = improve_routes_two_opt(routes, evaluator)
    evaluations = [evaluator.best_departure(route) for route in routes]
    validate_solution(problem, routes)
    _write_outputs(output_dir, evaluations, routes, problem)

    result = {
        "routes": float(len(routes)),
        "distance_km": sum(item.distance_km for item in evaluations),
        "emissions_kg": sum(item.emissions_kg for item in evaluations),
        "total_cost": sum(item.total_cost for item in evaluations),
    }
    print(f"路线数: {int(result['routes'])}")
    print(f"总里程: {result['distance_km']:.2f} km")
    print(f"碳排放: {result['emissions_kg']:.2f} kg")
    print(f"总成本: {result['total_cost']:.2f} 元")
    print(f"结果目录: {output_dir.resolve()}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="华中杯 A 题问题一贪心基线")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-two-opt", action="store_true", help="关闭 2-opt 局部优化")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.data_dir, arguments.output_dir, use_two_opt=not arguments.no_two_opt)
