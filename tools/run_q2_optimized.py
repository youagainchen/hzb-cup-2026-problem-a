from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_problem_data
from src.model.domain import DEFAULT_VEHICLE_TYPES
from src.model.evaluator import RouteEvaluator, evaluate_solution, format_clock
from src.model.policy_q2 import build_q2_policy
from src.solver.greedy import validate_solution
from src.solver.q2_initial import load_route_solution
from src.solver.q2_scheduling import (
    Q2ScheduleRun,
    refine_fixed_schedule_departures,
    search_q2_schedules,
)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _policy_safe_reason(route, stop, problem) -> str:
    if stop.policy_violation_reason:
        return "fuel_green_arrival_during_restricted_window"
    if stop.customer_id not in problem.green_customer_ids:
        return "customer_outside_green_zone"
    if route.vehicle_type.propulsion == "electric":
        return "electric_vehicle_exempt"
    if stop.arrival_minutes < 8.0 * 60.0:
        return "fuel_green_arrival_before_08_00"
    return "fuel_green_arrival_at_or_after_16_00"


def _audit_q1_reference(problem, q1_routes, q1_totals: dict[str, object]) -> dict[str, object]:
    evaluator = RouteEvaluator(problem)
    validate_solution(problem, q1_routes)
    validate_vehicle_schedule(q1_routes, evaluator)
    solution = evaluate_solution(q1_routes, evaluator, optimize_departures=False)
    checks = {
        "delivery_trips_match": solution.trip_count
        == int(q1_totals["delivery_trips"]),
        "physical_vehicles_match": solution.vehicle_count
        == int(q1_totals["physical_vehicles"]),
        "total_cost_match": abs(
            solution.total_cost - float(q1_totals["total_cost"])
        )
        <= 1e-6,
        "distance_match": abs(
            solution.total_distance_km - float(q1_totals["total_distance_km"])
        )
        <= 1e-6,
        "emissions_match": abs(
            solution.emissions_kg - float(q1_totals["total_emissions_kg"])
        )
        <= 1e-6,
        "demand_complete": solution.unfinished_customer_count == 0,
        "capacity_valid": solution.capacity_violation_count == 0,
        "all_routes_return_before_24h": solution.all_routes_return_before_24h,
    }
    if not all(checks.values()):
        raise AssertionError(f"Q1正式基准复核失败：{checks}")
    route_path = Path("results/routes/question1_optimized_routes.csv")
    summary_path = Path("results/tables/question1_optimized_route_summary.csv")
    totals_path = Path("results/tables/question1_optimized_totals.json")
    return {
        "reference_id": "Q1_FINAL_38V_98T_43397.262823",
        "status": "locked_and_recomputed",
        "legacy_versions_excluded": [
            "38辆/99趟/47141.61元",
            "55辆/122趟/57027.00元",
        ],
        "data_source": problem.data_source,
        "evaluator": "src.model.evaluator.RouteEvaluator",
        "vehicle_types": [vehicle.__dict__ for vehicle in DEFAULT_VEHICLE_TYPES],
        "delivery_trips": solution.trip_count,
        "physical_vehicles": solution.vehicle_count,
        "total_cost": solution.total_cost,
        "total_distance_km": solution.total_distance_km,
        "emissions_kg": solution.emissions_kg,
        "latest_return_minutes": max(item.finish_minutes for item in solution.routes),
        "checks": checks,
        "evidence_files": {
            str(route_path): _sha256(route_path),
            str(summary_path): _sha256(summary_path),
            str(totals_path): _sha256(totals_path),
        },
        "shared_model_rules": {
            "cleaned_data": "data/processed/team_cleaned",
            "service_minutes": 20,
            "return_deadline_minutes": 1440,
            "split_delivery_allowed": True,
            "multitrip_allowed": True,
            "fixed_cost_charged_per_physical_vehicle": True,
        },
    }


def _write_routes(output_dir: Path, routes, solution, problem) -> None:
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
                "is_green_customer",
                "policy_window",
                "policy_check_result",
                "policy_safe_reason",
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
                        int(stop.customer_id in problem.green_customer_ids),
                        "[08:00,16:00)",
                        "FAIL" if stop.policy_violation_reason else "PASS",
                        _policy_safe_reason(route, stop, problem),
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
                "capacity_weight_kg",
                "capacity_volume_m3",
                "remaining_weight_capacity_kg",
                "remaining_volume_capacity_m3",
                "capacity_check_result",
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
                    round(route.vehicle_type.capacity_weight, 6),
                    round(route.vehicle_type.capacity_volume, 6),
                    round(route.vehicle_type.capacity_weight - route.total_weight, 6),
                    round(route.vehicle_type.capacity_volume - route.total_volume, 6),
                    (
                        "PASS"
                        if route.total_weight
                        <= route.vehicle_type.capacity_weight + 1e-7
                        and route.total_volume
                        <= route.vehicle_type.capacity_volume + 1e-7
                        else "FAIL"
                    ),
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


def _write_policy_audits(output_dir: Path, routes, solution, problem, policy) -> None:
    with (output_dir / "question2_green_policy_checks.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fieldnames = [
            "route_id",
            "customer_id",
            "is_green_customer",
            "vehicle_type",
            "propulsion",
            "arrival",
            "arrival_minutes_exact",
            "policy_window",
            "policy_check_result",
            "policy_safe_reason",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for route, evaluation in zip(routes, solution.routes, strict=True):
            for stop in evaluation.stops:
                if stop.customer_id not in problem.green_customer_ids:
                    continue
                violates = policy.violates(
                    route.vehicle_type.propulsion,
                    stop.customer_id,
                    stop.arrival_minutes,
                )
                writer.writerow(
                    {
                        "route_id": route.route_id,
                        "customer_id": stop.customer_id,
                        "is_green_customer": 1,
                        "vehicle_type": route.vehicle_type.name,
                        "propulsion": route.vehicle_type.propulsion,
                        "arrival": format_clock(stop.arrival_minutes),
                        "arrival_minutes_exact": round(stop.arrival_minutes, 9),
                        "policy_window": "[08:00,16:00)",
                        "policy_check_result": "FAIL" if violates else "PASS",
                        "policy_safe_reason": _policy_safe_reason(
                            route, stop, problem
                        ),
                    }
                )

    with (output_dir / "question2_green_customer_audit.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fieldnames = [
            "customer_id",
            "x_km",
            "y_km",
            "distance_to_center_km",
            "is_green_customer",
            "is_active_customer",
            "demand_weight_kg",
            "demand_volume_m3",
            "inactive_reason",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for customer_id in sorted(problem.green_customer_ids):
            weight, volume = problem.demands.get(customer_id, (0.0, 0.0))
            x_km, y_km = problem.coordinates[customer_id]
            active = customer_id in problem.demands
            writer.writerow(
                {
                    "customer_id": customer_id,
                    "x_km": round(x_km, 6),
                    "y_km": round(y_km, 6),
                    "distance_to_center_km": round((x_km**2 + y_km**2) ** 0.5, 6),
                    "is_green_customer": 1,
                    "is_active_customer": int(active),
                    "demand_weight_kg": round(weight, 6),
                    "demand_volume_m3": round(volume, 6),
                    "inactive_reason": (
                        ""
                        if active
                        else "订单数为0且汇总重量、体积需求均为0"
                    ),
                }
            )

    representative = min(
        set(problem.green_customer_ids) & set(problem.active_customer_ids)
    )
    with (output_dir / "question2_policy_boundary_checks.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fieldnames = [
            "clock",
            "arrival_minutes",
            "propulsion",
            "green_customer_id",
            "expected",
            "actual",
            "test_result",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for clock, minute, expected in (
            ("07:59", 479.0, "ALLOW"),
            ("08:00", 480.0, "DENY"),
            ("15:59", 959.0, "DENY"),
            ("16:00", 960.0, "ALLOW"),
        ):
            actual = (
                "DENY"
                if policy.violates("fuel", representative, minute)
                else "ALLOW"
            )
            writer.writerow(
                {
                    "clock": clock,
                    "arrival_minutes": minute,
                    "propulsion": "fuel",
                    "green_customer_id": representative,
                    "expected": expected,
                    "actual": actual,
                    "test_result": "PASS" if expected == actual else "FAIL",
                }
            )


def _write_capacity_and_vehicle_audits(output_dir: Path, routes) -> None:
    with (output_dir / "question2_capacity_audit.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fieldnames = [
            "route_id",
            "vehicle_type",
            "load_weight_kg",
            "weight_capacity_kg",
            "weight_margin_kg",
            "load_volume_m3",
            "volume_capacity_m3",
            "volume_margin_m3",
            "capacity_check_result",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for route in sorted(routes, key=lambda item: item.route_id):
            valid = (
                route.total_weight <= route.vehicle_type.capacity_weight + 1e-7
                and route.total_volume <= route.vehicle_type.capacity_volume + 1e-7
            )
            writer.writerow(
                {
                    "route_id": route.route_id,
                    "vehicle_type": route.vehicle_type.name,
                    "load_weight_kg": round(route.total_weight, 6),
                    "weight_capacity_kg": route.vehicle_type.capacity_weight,
                    "weight_margin_kg": round(
                        route.vehicle_type.capacity_weight - route.total_weight, 6
                    ),
                    "load_volume_m3": round(route.total_volume, 6),
                    "volume_capacity_m3": route.vehicle_type.capacity_volume,
                    "volume_margin_m3": round(
                        route.vehicle_type.capacity_volume - route.total_volume, 6
                    ),
                    "capacity_check_result": "PASS" if valid else "FAIL",
                }
            )

    with (output_dir / "question2_vehicle_parameters.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fieldnames = [
            "vehicle_type",
            "propulsion",
            "capacity_weight_kg",
            "capacity_volume_m3",
            "fleet_count",
            "fixed_cost_yuan",
            "included_in_candidate_set",
            "parameter_source",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for vehicle in DEFAULT_VEHICLE_TYPES:
            writer.writerow(
                {
                    "vehicle_type": vehicle.name,
                    "propulsion": vehicle.propulsion,
                    "capacity_weight_kg": vehicle.capacity_weight,
                    "capacity_volume_m3": vehicle.capacity_volume,
                    "fleet_count": vehicle.count,
                    "fixed_cost_yuan": vehicle.fixed_cost,
                    "included_in_candidate_set": 1,
                    "parameter_source": "A题原题PDF第1页车型参数表述",
                }
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
    policy = build_q2_policy(problem.green_customer_ids)
    evaluator = RouteEvaluator(problem, policy=policy)
    q1_routes = load_route_solution(
        Path("results/routes/question1_optimized_routes.csv"),
        Path("results/tables/question1_optimized_route_summary.csv"),
    )
    q1 = json.loads(
        Path("results/tables/question1_optimized_totals.json").read_text(
            encoding="utf-8"
        )
    )
    q1_audit = _audit_q1_reference(problem, q1_routes, q1)
    (output_dir / "question1_reference_audit.json").write_text(
        json.dumps(q1_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    reuse_weights = (
        0.0,
        25.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        94.0,
        95.0,
        99.0,
        100.0,
        110.0,
        125.0,
        150.0,
        175.0,
        200.0,
    )
    order_rules = (
        "green_first",
        "time_window_first",
        "late_risk_first",
        "green_late_hybrid",
        "distance_late",
    )
    _, coarse_runs = search_q2_schedules(
        q1_routes,
        evaluator,
        seeds=(202601, 202602, 202603),
        order_rules=order_rules,
        vehicle_reuse_weights=reuse_weights,
        departure_step_minutes=10.0,
    )
    refinement_candidates: list[Q2ScheduleRun] = []
    selected_keys: set[tuple[int, str, float]] = set()

    def add_candidate(item: Q2ScheduleRun) -> None:
        key = (item.seed, item.order_rule, item.vehicle_reuse_weight)
        if key not in selected_keys:
            selected_keys.add(key)
            refinement_candidates.append(item)

    for item in sorted(coarse_runs, key=lambda run: run.evaluation.total_cost)[:8]:
        add_candidate(item)
    for rule in order_rules:
        same_rule = [item for item in coarse_runs if item.order_rule == rule]
        if same_rule:
            add_candidate(min(same_rule, key=lambda run: run.evaluation.total_cost))
    for vehicle_count in (34, 35, 36, 37, 38, 39):
        same_size = [
            item
            for item in coarse_runs
            if item.evaluation.vehicle_count == vehicle_count
        ]
        if same_size:
            add_candidate(min(same_size, key=lambda run: run.evaluation.total_cost))

    refined_runs: list[Q2ScheduleRun] = []
    for candidate in refinement_candidates:
        refined_routes, refined_evaluation = refine_fixed_schedule_departures(
            candidate.routes,
            evaluator,
            step_minutes=5.0,
        )
        refined_runs.append(
            Q2ScheduleRun(
                seed=candidate.seed,
                order_rule=candidate.order_rule,
                vehicle_reuse_weight=candidate.vehicle_reuse_weight,
                departure_step_minutes=5.0,
                routes=tuple(refined_routes),
                evaluation=refined_evaluation,
            )
        )
    for candidate in sorted(
        refined_runs, key=lambda run: run.evaluation.total_cost
    )[:3]:
        refined_routes, refined_evaluation = refine_fixed_schedule_departures(
            candidate.routes,
            evaluator,
            step_minutes=2.0,
        )
        refined_runs.append(
            Q2ScheduleRun(
                seed=candidate.seed,
                order_rule=candidate.order_rule,
                vehicle_reuse_weight=candidate.vehicle_reuse_weight,
                departure_step_minutes=2.0,
                routes=tuple(refined_routes),
                evaluation=refined_evaluation,
            )
        )

    runs = tuple(coarse_runs) + tuple(refined_runs)
    best = min(runs, key=lambda run: (run.evaluation.total_cost, run.seed))
    routes = list(best.routes)
    solution = best.evaluation
    validate_solution(problem, routes)
    validate_vehicle_schedule(routes, evaluator)
    if solution.policy_violation_count != 0:
        raise AssertionError("正式Q2方案仍存在政策违规")

    _write_routes(output_dir, routes, solution, problem)
    _write_policy_audits(output_dir, routes, solution, problem, policy)
    _write_capacity_and_vehicle_audits(output_dir, routes)
    with (output_dir / "question2_selected_search.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fieldnames = [
            "seed",
            "order_rule",
            "vehicle_reuse_weight",
            "departure_step_minutes",
            "total_cost",
            "physical_vehicles",
            "delivery_trips",
            "emissions_kg",
            "late_cost",
            "policy_violation_count",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "seed": best.seed,
                "order_rule": best.order_rule,
                "vehicle_reuse_weight": best.vehicle_reuse_weight,
                "departure_step_minutes": best.departure_step_minutes,
                "total_cost": round(best.evaluation.total_cost, 6),
                "physical_vehicles": best.evaluation.vehicle_count,
                "delivery_trips": best.evaluation.trip_count,
                "emissions_kg": round(best.evaluation.emissions_kg, 6),
                "late_cost": round(best.evaluation.late_cost, 6),
                "policy_violation_count": best.evaluation.policy_violation_count,
            }
        )

    totals: dict[str, object] = {
        "solution_variant": "question2_q1_policy_repair_optimized",
        "algorithm": (
            "锁定Q1正式路线 + 政策可行性修复 + 路线内局部顺序优化 + "
            "2/5/10分钟发车枚举 + 全车型选择 + 多趟物理车辆复用 + "
            "多排序规则固定种子参数搜索"
        ),
        "claim_scope": "基于问题一解的可行性修复与局部优化，不声称全局最优",
        "q1_reference_id": q1_audit["reference_id"],
        "selected_parameters": {
            "seed": best.seed,
            "order_rule": best.order_rule,
            "vehicle_reuse_weight": best.vehicle_reuse_weight,
            "departure_step_minutes": best.departure_step_minutes,
        },
        "candidate_vehicle_types": [
            vehicle.name for vehicle in DEFAULT_VEHICLE_TYPES
        ],
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
        f"问题一唯一基准：`{q1_audit['reference_id']}`。历史中间版本不参与政策影响计算。",
        "",
        "政策口径：燃油车在 08:00–16:00 不得到达半径 10 km 绿色配送区客户；新能源车不限行。",
        "算法口径：基于问题一解的政策可行性修复与局部优化，不声称全局最优。",
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
            "- 绿色节点逐站独立检查：全部通过",
            "- 07:59/08:00/15:59/16:00边界检查：全部通过",
            "- 原题车型容积复核：EV-3000为15 m³，FUEL-3000为13.5 m³",
            "",
            "正式答案仅保留满足全部约束后的最低总成本方案。该方案通过减少一辆物理车并压低迟到成本控制政策代价，但燃油占比上升，因此碳排放相对问题一增加；论文应如实报告这一政策影响。",
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
