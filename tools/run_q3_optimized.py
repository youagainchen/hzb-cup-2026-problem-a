from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_problem_data
from src.model.evaluator import RouteEvaluator, evaluate_solution, format_clock
from src.model.q3_event import Q3Event, Q3EventSet, Q3EventType
from src.model.policy_q2 import build_q2_policy
from src.solver.q2_initial import load_route_solution
from src.solver.q3_dynamic import DynamicStep, run_event_sequence


def read_event_sets(path: Path) -> list[Q3EventSet]:
    grouped: dict[float, list[Q3Event]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            def optional_float(name: str) -> float | None:
                value = row.get(name, "")
                return float(value) if value not in (None, "") else None

            grouped[float(row["trigger_time_minutes"])].append(
                Q3Event(
                    event_type=Q3EventType(row["event_type"]),
                    customer_id=int(row["customer_id"]),
                    trigger_time_minutes=float(row["trigger_time_minutes"]),
                    weight_kg=optional_float("weight_kg"),
                    volume_m3=optional_float("volume_m3"),
                    window_start_minutes=optional_float("window_start_minutes"),
                    window_end_minutes=optional_float("window_end_minutes"),
                    new_x_km=optional_float("new_x_km"),
                    new_y_km=optional_float("new_y_km"),
                    severity=row.get("severity", "medium"),
                )
            )
    return [
        Q3EventSet(
            trigger_time_minutes=trigger,
            events=tuple(events),
            description=f"{format_clock(trigger)} 动态事件批次",
        )
        for trigger, events in sorted(grouped.items())
    ]


def _write_routes(path: Path, step: DynamicStep) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "event_trigger",
                "route_id",
                "vehicle_type",
                "propulsion",
                "physical_vehicle_id",
                "trip_number",
                "start_minutes",
                "start",
                "sequence",
                "customer_id",
                "delivered_weight_kg",
                "delivered_volume_m3",
            ]
        )
        for route in step.future_routes:
            for sequence, delivery in enumerate(route.deliveries, start=1):
                writer.writerow(
                    [
                        format_clock(step.event_set.trigger_time_minutes),
                        route.route_id,
                        route.vehicle_type.name,
                        route.vehicle_type.propulsion,
                        f"{route.vehicle_type.name}-{route.vehicle_number:03d}",
                        route.trip_number,
                        route.start_minutes,
                        format_clock(route.start_minutes),
                        sequence,
                        delivery.customer_id,
                        delivery.weight,
                        delivery.volume,
                    ]
                )


def _step_json(step: DynamicStep) -> dict[str, object]:
    evaluation = step.evaluation
    feasibility = evaluation.feasibility
    return {
        "trigger_time_minutes": evaluation.trigger_time_minutes,
        "trigger_clock": format_clock(evaluation.trigger_time_minutes),
        "event_count": len(step.event_set.events),
        "event_types": [event.event_type.value for event in step.event_set.events],
        "response_time_s": step.response_time_s,
        "executed_cost": evaluation.executed_cost,
        "future_fixed_cost": evaluation.future_fixed_cost,
        "future_operating_cost": evaluation.future_operating_cost,
        "future_cost": evaluation.future_cost,
        "total_cost": evaluation.total_cost,
        "static_total_cost": evaluation.static_total_cost,
        "delta_cost": evaluation.delta_cost,
        "delta_cost_base": evaluation.delta_cost_base,
        "cost_change_ratio": evaluation.cost_change_ratio,
        "cost_change_ratio_base": evaluation.cost_change_ratio_base,
        "lead_time_minutes": evaluation.lead_time_minutes,
        "frozen_trip_count": evaluation.frozen_trip_count,
        "replannable_trip_count": evaluation.replannable_trip_count,
        "future_trip_count": evaluation.future_trip_count,
        "dropped_trip_count": evaluation.dropped_trip_count,
        "new_trip_count": evaluation.new_trip_count,
        "kept_trip_count": evaluation.kept_trip_count,
        "changed_customer_count": evaluation.changed_customer_count,
        "assignment_change_ratio": evaluation.assignment_change_ratio,
        "arc_change_ratio": evaluation.arc_change_ratio,
        "feasibility": {
            "passed": feasibility.passed,
            "demand_unfinished_customers": feasibility.demand_unfinished_customers,
            "demand_unfinished_weight_kg": feasibility.demand_unfinished_weight_kg,
            "demand_unfinished_volume_m3": feasibility.demand_unfinished_volume_m3,
            "capacity_violation_count": feasibility.capacity_violation_count,
            "policy_violation_count": feasibility.policy_violation_count,
            "schedule_violation_count": feasibility.schedule_violation_count,
            "all_routes_return_before_24h": feasibility.all_routes_return_before_24h,
            "notes": feasibility.notes,
        },
    }


def _write_event_response(path: Path, steps: tuple[DynamicStep, ...]) -> None:
    fields = [
        "event_set_id",
        "event_id",
        "event_type",
        "customer_id",
        "trigger_clock",
        "static_total_cost",
        "dynamic_total_cost",
        "delta_cost",
        "delta_cost_base",
        "response_time_s",
        "lead_time_minutes",
        "frozen_trip_count",
        "future_trip_count",
        "changed_customer_count",
        "assignment_change_ratio",
        "arc_change_ratio",
        "passed",
        "policy_violation_count",
        "schedule_violation_count",
        "unfinished_customer_count",
        "return_before_24h",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for batch_index, step in enumerate(steps, start=1):
            evaluation = step.evaluation
            feasibility = evaluation.feasibility
            for event_index, event in enumerate(step.event_set.events, start=1):
                writer.writerow(
                    {
                        "event_set_id": f"B{batch_index:02d}",
                        "event_id": f"E{event_index:02d}",
                        "event_type": event.event_type.value,
                        "customer_id": event.customer_id,
                        "trigger_clock": format_clock(evaluation.trigger_time_minutes),
                        "static_total_cost": evaluation.static_total_cost,
                        "dynamic_total_cost": evaluation.total_cost,
                        "delta_cost": evaluation.delta_cost,
                        "delta_cost_base": evaluation.delta_cost_base,
                        "response_time_s": evaluation.response_time_s,
                        "lead_time_minutes": evaluation.lead_time_minutes,
                        "frozen_trip_count": evaluation.frozen_trip_count,
                        "future_trip_count": evaluation.future_trip_count,
                        "changed_customer_count": evaluation.changed_customer_count,
                        "assignment_change_ratio": evaluation.assignment_change_ratio,
                        "arc_change_ratio": evaluation.arc_change_ratio,
                        "passed": feasibility.passed,
                        "policy_violation_count": feasibility.policy_violation_count,
                        "schedule_violation_count": feasibility.schedule_violation_count,
                        "unfinished_customer_count": feasibility.demand_unfinished_customers,
                        "return_before_24h": feasibility.all_routes_return_before_24h,
                    }
                )


def run(
    data_dir: Path,
    event_path: Path,
    route_path: Path,
    summary_path: Path,
    totals_path: Path,
    output_dir: Path,
) -> tuple[DynamicStep, ...]:
    problem = load_problem_data(data_dir)
    static_routes = load_route_solution(route_path, summary_path)
    static_evaluator = RouteEvaluator(
        problem,
        policy=build_q2_policy(problem.green_customer_ids),
    )
    static_result = evaluate_solution(
        static_routes, static_evaluator, optimize_departures=False
    )
    expected_total = json.loads(totals_path.read_text(encoding="utf-8"))["total_cost"]
    if abs(static_result.total_cost - float(expected_total)) > 1e-3:
        raise AssertionError(
            f"Q2静态路线重算不一致: {static_result.total_cost} vs {expected_total}"
        )

    event_sets = read_event_sets(event_path)
    for event_set in event_sets:
        errors = event_set.validate(problem)
        if errors:
            raise ValueError(f"事件实例校验失败: {errors}")
    steps = run_event_sequence(
        static_routes,
        problem,
        event_sets,
        static_total_cost=static_result.total_cost,
    )
    for step in steps:
        if not step.evaluation.feasibility.passed:
            raise AssertionError(
                f"{step.event_set.description}动态方案不可行: "
                f"{step.evaluation.feasibility.notes}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_routes(output_dir / "question3_optimized_future_routes.csv", steps[-1])
    _write_event_response(output_dir / "question3_event_response.csv", steps)
    summaries = [_step_json(step) for step in steps]
    (output_dir / "question3_optimized_totals.json").write_text(
        json.dumps(
            {
                "algorithm": "rolling-horizon event operators: cancel/remove, new-order insertion, address local reorder, time-window reorder, fleet rescheduling",
                "static_total_cost": static_result.total_cost,
                "event_batch_count": len(steps),
                "all_batches_passed": all(
                    item["feasibility"]["passed"] for item in summaries
                ),
                "batches": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Q3动态调度完成：{len(steps)}个事件批次，"
        f"静态成本 {static_result.total_cost:.2f} 元"
    )
    for index, step in enumerate(steps, start=1):
        evaluation = step.evaluation
        print(
            f"批次{index} {format_clock(evaluation.trigger_time_minutes)}："
            f"动态成本 {evaluation.total_cost:.2f} 元，"
            f"ΔC {evaluation.delta_cost:+.2f} 元，"
            f"响应 {step.response_time_s:.3f}s，passed={evaluation.feasibility.passed}"
        )
    print(f"输出目录：{output_dir.resolve()}")
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题三：动态事件滚动调度")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/team_cleaned"))
    parser.add_argument("--event-path", type=Path, default=Path("results/question3/question3_event_set.csv"))
    parser.add_argument("--route-path", type=Path, default=Path("results/question2_optimized/question2_optimized_routes.csv"))
    parser.add_argument("--summary-path", type=Path, default=Path("results/question2_optimized/question2_optimized_route_summary.csv"))
    parser.add_argument("--totals-path", type=Path, default=Path("results/question2_optimized/question2_optimized_totals.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/question3_optimized"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.data_dir,
        arguments.event_path,
        arguments.route_path,
        arguments.summary_path,
        arguments.totals_path,
        arguments.output_dir,
    )
