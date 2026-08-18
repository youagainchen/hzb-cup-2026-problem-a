from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from math import ceil
from pathlib import Path

from src.data.loader import load_problem_data
from src.model.domain import DEFAULT_VEHICLE_TYPES
from src.model.evaluator import SolutionEvaluation, RouteEvaluator, evaluate_solution, format_clock
from src.solver.greedy import build_greedy_routes, validate_solution
from src.solver.local_search import (
    eliminate_low_load_routes,
    improve_routes_merge,
    improve_routes_relocate,
    improve_routes_swap,
    improve_routes_two_opt,
    repack_split_low_load_route,
)
from src.solver.savings import build_savings_routes
from src.solver.scheduling import select_and_schedule_multitrip, validate_vehicle_schedule
from src.visualization.plots import plot_solution_figures


def _trip_count_capacity_lower_bound(problem) -> int:
    """分别按重量、体积计算最乐观配送趟次下界，取两者较大值。"""

    total_weight = sum(weight for weight, _ in problem.demands.values())
    total_volume = sum(volume for _, volume in problem.demands.values())

    return max(
        ceil(total_weight / max(vehicle.capacity_weight for vehicle in DEFAULT_VEHICLE_TYPES)),
        ceil(total_volume / max(vehicle.capacity_volume for vehicle in DEFAULT_VEHICLE_TYPES)),
    )


def _clone_routes(routes):
    return [
        type(route)(
            route.vehicle_type,
            route.vehicle_number,
            list(route.deliveries),
            route.start_minutes,
            route.trip_number,
        )
        for route in routes
    ]


def _best_multi_trip_schedule(routes, evaluator: RouteEvaluator):
    candidates = []
    for vehicle_limit in (38, 40, 42, 49, None):
        for order_rule in ("deadline", "long_first", "late_risk"):
            for startup_weight in (0.0, 75.0, 100.0):
                try:
                    scheduled = select_and_schedule_multitrip(
                        _clone_routes(routes),
                        evaluator,
                        max_physical_vehicles=vehicle_limit,
                        startup_cost_weight=startup_weight,
                        order_rule=order_rule,
                    )
                except RuntimeError:
                    continue
                candidates.append(
                    (
                        scheduled,
                        evaluate_solution(
                            scheduled, evaluator, optimize_departures=False
                        ),
                    )
                )
    if not candidates:
        raise RuntimeError("没有找到能在当日24:00前完成的多趟配送排班")
    return min(candidates, key=lambda item: item[1].total_cost)


def _intensive_multi_trip_schedule(routes, evaluator: RouteEvaluator):
    """重点搜索38--42辆及30--36个新能源趟次的成本平衡点。"""

    candidates = []
    best_by_vehicle_count: dict[int, tuple[float, int]] = {}
    best_by_electric_trip_count: dict[int, tuple[float, int]] = {}
    for vehicle_limit in range(38, 43):
        for order_rule in ("deadline", "long_first", "late_risk"):
            for startup_weight in (0.0, 50.0, 100.0):
                for electric_target in (23, 24, 26, 28, 30, 32, 34, 36, None):
                    try:
                        scheduled = select_and_schedule_multitrip(
                            _clone_routes(routes),
                            evaluator,
                            max_physical_vehicles=vehicle_limit,
                            startup_cost_weight=startup_weight,
                            order_rule=order_rule,
                            electric_trip_minimum=electric_target,
                            electric_trip_limit=electric_target,
                        )
                    except RuntimeError:
                        continue
                    result = evaluate_solution(
                        scheduled, evaluator, optimize_departures=False
                    )
                    electric_trips = sum(
                        route.vehicle_type.propulsion == "electric"
                        for route in scheduled
                    )
                    candidates.append((scheduled, result, electric_trips))
                    current = best_by_vehicle_count.get(result.vehicle_count)
                    if current is None or result.total_cost < current[0] - 1e-7:
                        best_by_vehicle_count[result.vehicle_count] = (
                            result.total_cost,
                            electric_trips,
                        )
                    electric_current = best_by_electric_trip_count.get(electric_trips)
                    if (
                        electric_current is None
                        or result.total_cost < electric_current[0] - 1e-7
                    ):
                        best_by_electric_trip_count[electric_trips] = (
                            result.total_cost,
                            result.vehicle_count,
                        )
    if not candidates:
        raise RuntimeError("38--42辆精细搜索没有得到当日可行排班")
    for vehicle_count, (cost, electric_trips) in sorted(best_by_vehicle_count.items()):
        print(
            f"精细排班 {vehicle_count} 辆：成本 {cost:.2f} 元，"
            f"新能源 {electric_trips} 趟"
        )
    for electric_trips in (23, 24, 26, 28, 30, 32, 34, 36):
        if electric_trips not in best_by_electric_trip_count:
            continue
        cost, vehicle_count = best_by_electric_trip_count[electric_trips]
        print(
            f"新能源 {electric_trips} 趟最优：成本 {cost:.2f} 元，"
            f"物理车辆 {vehicle_count} 辆"
        )
    best_routes, best_result, _ = min(candidates, key=lambda item: item[1].total_cost)
    return best_routes, best_result


def _write_outputs(
    output_dir: Path,
    solution: SolutionEvaluation,
    routes,
    problem,
    optimization_trace: list[dict[str, float | int | str]],
    variant: str = "optimized",
) -> None:
    evaluations = solution.routes
    route_dir = output_dir / "routes"
    table_dir = output_dir / "tables"
    route_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    detail_path = route_dir / f"question1_{variant}_routes.csv"
    with detail_path.open("w", newline="", encoding="utf-8-sig") as stream:
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
                        format_clock(stop.service_start_minutes),
                        format_clock(stop.departure_minutes),
                        round(stop.waiting_minutes, 3),
                        round(stop.late_minutes, 3),
                    ]
                )

    summary_path = table_dir / f"question1_{variant}_route_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "route_id",
                "physical_vehicle_id",
                "trip_number",
                "vehicle_type",
                "start",
                "finish",
                "start_minutes_exact",
                "finish_minutes_exact",
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
                    f"{route.vehicle_type.name}-{route.vehicle_number:03d}",
                    route.trip_number,
                    route.vehicle_type.name,
                    format_clock(result.start_minutes),
                    format_clock(result.finish_minutes),
                    round(result.start_minutes, 9),
                    round(result.finish_minutes, 9),
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
        "solution_variant": variant,
        "algorithm": (
            "best of time-window-weighted cost-aware Clarke-Wright and time-window greedy + route elimination "
            "+ 2-opt + relocate + swap + route merge + global vehicle assignment "
            "+ departure-time optimization + multi-trip physical-vehicle scheduling"
        ),
        "data_source": problem.data_source,
        "active_customers": len(problem.active_customer_ids),
        "no_order_customers": len(problem.all_customer_ids) - len(problem.active_customer_ids),
        "delivery_trips": solution.trip_count,
        "physical_vehicles": solution.vehicle_count,
        "trip_count_capacity_lower_bound": _trip_count_capacity_lower_bound(problem),
        "trip_usage_by_type": dict(Counter(route.vehicle_type.name for route in routes)),
        "physical_vehicle_usage": dict(
            Counter(
                vehicle_name
                for vehicle_name, _ in {
                    (route.vehicle_type.name, route.vehicle_number) for route in routes
                }
            )
        ),
        "imputed_weight_rows": problem.imputed_weight_rows,
        "imputed_volume_rows": problem.imputed_volume_rows,
        "total_distance_km": solution.total_distance_km,
        "fuel_liters": solution.fuel_liters,
        "electricity_kwh": solution.electricity_kwh,
        "total_emissions_kg": solution.emissions_kg,
        "fixed_cost": solution.fixed_cost,
        "energy_cost": solution.energy_cost,
        "carbon_cost": solution.carbon_cost,
        "waiting_cost": solution.waiting_cost,
        "late_cost": solution.late_cost,
        "total_cost": solution.total_cost,
        "cost_share": {
            "fixed": solution.fixed_cost / solution.total_cost,
            "energy": solution.energy_cost / solution.total_cost,
            "carbon": solution.carbon_cost / solution.total_cost,
            "waiting": solution.waiting_cost / solution.total_cost,
            "late": solution.late_cost / solution.total_cost,
        },
        "optimization_trace": optimization_trace,
        "assumptions": [
            problem.missing_value_policy,
            "客户总需求允许按重量与体积同比例拆分给多辆车",
            "同一物理车辆可在返回配送中心后执行多趟任务，启动成本每天只计一次",
            "所有配送趟次必须在当日24:00前返回配送中心，不允许跨天配送",
            "车速使用题面正态分布的均值，并按跨时段分段行驶",
            "按清洗说明假设 17:00-19:00 为晚高峰 9.8 km/h，19:00 后为顺畅 55.3 km/h",
            "载荷能耗增幅按重量/容积利用率最大值线性插值",
            "Clarke-Wright 排序同时比较距离节约值与客户时间窗中点差异",
        ],
    }
    (table_dir / f"question1_{variant}_totals.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _choose_initial_routes(problem, evaluator: RouteEvaluator, method: str):
    candidates = []
    if method in ("greedy", "best"):
        greedy = build_greedy_routes(problem)
        validate_solution(problem, greedy)
        greedy, greedy_solution = _best_multi_trip_schedule(greedy, evaluator)
        candidates.append(
            ("greedy", greedy, greedy_solution)
        )
    if method in ("savings", "best"):
        for time_penalty in (0.0, 5.0, 15.0, 30.0):
            savings = build_savings_routes(
                problem,
                time_window_penalty_km_per_hour=time_penalty,
            )
            try:
                savings, savings_solution = _best_multi_trip_schedule(savings, evaluator)
            except RuntimeError:
                continue
            validate_solution(problem, savings)
            candidates.append(
                (
                    f"savings_time_{time_penalty:g}",
                    savings,
                    savings_solution,
                )
            )
    if not candidates:
        raise RuntimeError("没有得到满足车队数量约束的初始解")
    return min(candidates, key=lambda item: item[2].total_cost)


def run(
    data_dir: Path,
    output_dir: Path,
    initial_method: str = "best",
    use_local_search: bool = True,
) -> dict[str, float]:
    problem = load_problem_data(data_dir)
    evaluator = RouteEvaluator(problem)
    initial_name, routes, initial_solution = _choose_initial_routes(
        problem, evaluator, initial_method
    )
    print(
        f"初始解: {initial_name}，车辆 {initial_solution.vehicle_count}，"
        f"成本 {initial_solution.total_cost:.2f} 元"
    )
    optimization_trace: list[dict[str, float | int | str]] = []
    best_routes = _clone_routes(routes)
    best_solution = initial_solution

    def record_stage(stage: str) -> SolutionEvaluation:
        nonlocal routes, best_routes, best_solution
        routes, stage_result = _best_multi_trip_schedule(routes, evaluator)
        if stage_result.total_cost < best_solution.total_cost - 1e-7:
            best_routes = _clone_routes(routes)
            best_solution = stage_result
        optimization_trace.append(
            {
                "stage": stage,
                "vehicles": stage_result.vehicle_count,
                "trips": stage_result.trip_count,
                "distance_km": stage_result.total_distance_km,
                "fixed_cost": stage_result.fixed_cost,
                "energy_cost": stage_result.energy_cost,
                "carbon_cost": stage_result.carbon_cost,
                "waiting_cost": stage_result.waiting_cost,
                "late_cost": stage_result.late_cost,
                "total_cost": stage_result.total_cost,
            }
        )
        print(
            f"阶段 {stage}: 车辆 {stage_result.vehicle_count}，"
            f"趟次 {stage_result.trip_count}，"
            f"成本 {stage_result.total_cost:.2f} 元，"
            f"里程 {stage_result.total_distance_km:.2f} km"
        )
        return stage_result

    record_stage(f"initial_{initial_name}")

    if use_local_search:
        routes = eliminate_low_load_routes(routes, evaluator)
        record_stage("eliminate_low_load")
        routes = repack_split_low_load_route(routes, evaluator)
        record_stage("split_repack_low_load")
        routes = improve_routes_two_opt(routes, evaluator)
        record_stage("two_opt_1")
        routes = improve_routes_relocate(routes, evaluator)
        record_stage("relocate")
        routes = improve_routes_swap(routes, evaluator)
        record_stage("swap")
        routes = improve_routes_merge(routes, evaluator)
        record_stage("route_merge")
        record_stage("select_vehicle")
        routes = eliminate_low_load_routes(routes, evaluator)
        record_stage("eliminate_low_load_2")
        routes = improve_routes_two_opt(routes, evaluator)
        record_stage("two_opt_2")
    else:
        record_stage("select_vehicle")

    routes = best_routes
    solution = best_solution
    refined_routes, refined_solution = _intensive_multi_trip_schedule(routes, evaluator)
    if refined_solution.total_cost < solution.total_cost - 1e-7:
        routes = refined_routes
        solution = refined_solution
    optimization_trace.append(
        {
            "stage": "intensive_38_42_schedule",
            "vehicles": solution.vehicle_count,
            "trips": solution.trip_count,
            "distance_km": solution.total_distance_km,
            "fixed_cost": solution.fixed_cost,
            "energy_cost": solution.energy_cost,
            "carbon_cost": solution.carbon_cost,
            "waiting_cost": solution.waiting_cost,
            "late_cost": solution.late_cost,
            "total_cost": solution.total_cost,
        }
    )

    validate_solution(problem, routes)
    validate_vehicle_schedule(routes, evaluator)
    _write_outputs(output_dir, solution, routes, problem, optimization_trace, "optimized")
    plot_solution_figures(
        routes,
        solution,
        problem,
        optimization_trace,
        output_dir,
        "question1_optimized",
    )

    result = {
        "vehicles": float(solution.vehicle_count),
        "trips": float(solution.trip_count),
        "distance_km": solution.total_distance_km,
        "emissions_kg": solution.emissions_kg,
        "fixed_cost": solution.fixed_cost,
        "energy_cost": solution.energy_cost,
        "carbon_cost": solution.carbon_cost,
        "waiting_cost": solution.waiting_cost,
        "late_cost": solution.late_cost,
        "total_cost": solution.total_cost,
    }
    print(f"总成本：{solution.total_cost:.2f} 元")
    print(f"启动成本：{solution.fixed_cost:.2f} 元")
    print(f"能源成本：{solution.energy_cost:.2f} 元")
    print(f"碳成本：{solution.carbon_cost:.2f} 元")
    print(f"等待成本：{solution.waiting_cost:.2f} 元")
    print(f"迟到成本：{solution.late_cost:.2f} 元")
    print(f"使用车辆：{solution.vehicle_count}")
    print(f"配送趟次：{solution.trip_count}")
    print(f"配送趟次容量下界：{_trip_count_capacity_lower_bound(problem)}")
    print(f"总里程：{solution.total_distance_km:.2f} km")
    print(f"碳排放：{solution.emissions_kg:.2f} kg")
    print(f"结果目录: {output_dir.resolve()}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="华中杯 A 题问题一成本驱动优化")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/team_cleaned"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--initial",
        choices=("best", "greedy", "savings"),
        default="best",
        help="初始解方法；best 会比较贪心与节约算法",
    )
    parser.add_argument("--no-local-search", action="store_true", help="关闭局部搜索")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.data_dir,
        arguments.output_dir,
        initial_method=arguments.initial,
        use_local_search=not arguments.no_local_search,
    )
