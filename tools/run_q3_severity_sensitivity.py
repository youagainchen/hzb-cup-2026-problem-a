from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_problem_data
from src.model.evaluator import RouteEvaluator, evaluate_solution
from src.model.policy_q2 import build_q2_policy
from src.model.q3_event import Q3Event, Q3EventSet, Q3EventType, write_event_set_csv
from src.solver.q2_initial import load_route_solution
from src.solver.q3_dynamic import DynamicStep, dispatch_event_set


TRIGGER_TIME = 10.0 * 60.0


def build_severity_scenarios() -> tuple[Q3EventSet, ...]:
    """在同一触发时刻构造低/中/高三档四类事件组合。"""

    configurations = {
        "low": {
            "cancel_customer": 13,
            "new_weight": 300.0,
            "new_volume": 0.9,
            "new_coordinate": (4.0, 5.0),
            "address_coordinate": (16.0, -16.0),
            "window_shift": 15.0,
        },
        "medium": {
            "cancel_customer": 12,
            "new_weight": 600.0,
            "new_volume": 1.8,
            "new_coordinate": (5.0, 6.0),
            "address_coordinate": (20.0, -20.0),
            "window_shift": 30.0,
        },
        "high": {
            "cancel_customer": 12,
            "new_weight": 1000.0,
            "new_volume": 3.0,
            "new_coordinate": (7.0, 7.0),
            "address_coordinate": (23.0, -23.0),
            "window_shift": 45.0,
        },
    }
    scenarios: list[Q3EventSet] = []
    for severity, config in configurations.items():
        shift = float(config["window_shift"])
        scenarios.append(
            Q3EventSet(
                trigger_time_minutes=TRIGGER_TIME,
                description=f"{severity}：取消+新增+变址+改窗四事件组合",
                events=(
                    Q3Event(
                        event_type=Q3EventType.CANCEL,
                        customer_id=int(config["cancel_customer"]),
                        trigger_time_minutes=TRIGGER_TIME,
                        severity=severity,
                    ),
                    Q3Event(
                        event_type=Q3EventType.NEW_ORDER,
                        customer_id=99,
                        trigger_time_minutes=TRIGGER_TIME,
                        weight_kg=float(config["new_weight"]),
                        volume_m3=float(config["new_volume"]),
                        window_start_minutes=13.0 * 60.0,
                        window_end_minutes=15.0 * 60.0,
                        new_x_km=float(config["new_coordinate"][0]),
                        new_y_km=float(config["new_coordinate"][1]),
                        severity=severity,
                    ),
                    Q3Event(
                        event_type=Q3EventType.ADDRESS_CHANGE,
                        customer_id=82,
                        trigger_time_minutes=TRIGGER_TIME,
                        new_x_km=float(config["address_coordinate"][0]),
                        new_y_km=float(config["address_coordinate"][1]),
                        severity=severity,
                    ),
                    Q3Event(
                        event_type=Q3EventType.TIME_WINDOW_CHANGE,
                        customer_id=70,
                        trigger_time_minutes=TRIGGER_TIME,
                        window_start_minutes=11.0 * 60.0 + 36.0 + shift,
                        window_end_minutes=12.0 * 60.0 + 35.0 + shift,
                        severity=severity,
                    ),
                ),
            )
        )
    return tuple(scenarios)


def _row(severity: str, step: DynamicStep) -> dict[str, object]:
    evaluation = step.evaluation
    feasibility = evaluation.feasibility
    return {
        "severity": severity,
        "event_count": len(step.event_set.events),
        "trigger_clock": "10:00",
        "dynamic_total_cost": evaluation.total_cost,
        "delta_cost": evaluation.delta_cost,
        "cost_change_ratio": evaluation.cost_change_ratio,
        "response_time_s": step.response_time_s,
        "lead_time_minutes": evaluation.lead_time_minutes,
        "changed_customer_count": evaluation.changed_customer_count,
        "assignment_change_ratio": evaluation.assignment_change_ratio,
        "arc_change_ratio": evaluation.arc_change_ratio,
        "future_trip_count": evaluation.future_trip_count,
        "policy_violation_count": feasibility.policy_violation_count,
        "schedule_violation_count": feasibility.schedule_violation_count,
        "unfinished_customer_count": feasibility.demand_unfinished_customers,
        "return_before_24h": feasibility.all_routes_return_before_24h,
        "passed": feasibility.passed,
    }


def _plot(rows: list[dict[str, object]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["severity"]) for row in rows]
    deltas = [float(row["delta_cost"]) for row in rows]
    responses_ms = [float(row["response_time_s"]) * 1000.0 for row in rows]
    figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=160)
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    left.bar(labels, deltas, color=colors)
    left.axhline(0.0, color="#555555", linewidth=0.8)
    left.set_title("Cost Increment by Severity")
    left.set_ylabel("Delta Cost (CNY)")
    right.plot(labels, responses_ms, color="#457b9d", marker="o", linewidth=2)
    right.set_title("Measured Response Time")
    right.set_ylabel("Response Time (ms)")
    right.set_ylim(bottom=0.0)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def run(
    data_dir: Path,
    route_path: Path,
    summary_path: Path,
    output_dir: Path,
) -> list[dict[str, object]]:
    problem = load_problem_data(data_dir)
    static_routes = load_route_solution(route_path, summary_path)
    evaluator = RouteEvaluator(
        problem,
        policy=build_q2_policy(problem.green_customer_ids),
    )
    static_total = evaluate_solution(
        static_routes, evaluator, optimize_departures=False
    ).total_cost
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for scenario in build_severity_scenarios():
        errors = scenario.validate(problem)
        if errors:
            raise ValueError(f"{scenario.description} 事件定义无效: {errors}")
        severity = scenario.events[0].severity
        write_event_set_csv(
            scenario,
            output_dir / f"question3_{severity}_event_set.csv",
            problem,
        )
        step = dispatch_event_set(
            static_routes,
            problem,
            scenario,
            static_total_cost=static_total,
        )
        if not step.evaluation.feasibility.passed:
            raise AssertionError(
                f"{severity} 场景不可行: {step.evaluation.feasibility.notes}"
            )
        rows.append(_row(severity, step))

    fields = list(rows[0])
    with (output_dir / "question3_severity_sensitivity.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "question3_severity_sensitivity.json").write_text(
        json.dumps(
            {
                "static_total_cost": static_total,
                "scenario_design": {
                    "low": "取消c13；新增300kg/0.9m³；c82变址至(16,-16)；c70改窗+15min",
                    "medium": "取消c12；新增600kg/1.8m³；c82变址至(20,-20)；c70改窗+30min",
                    "high": "取消c12；新增1000kg/3.0m³；c82变址至(23,-23)；c70改窗+45min",
                },
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot(rows, output_dir / "question3_severity_sensitivity.png")
    for row in rows:
        print(
            f"{row['severity']}: ΔC={float(row['delta_cost']):+.2f} 元，"
            f"响应={float(row['response_time_s']) * 1000.0:.2f}ms，"
            f"passed={row['passed']}"
        )
    print(f"输出目录：{output_dir.resolve()}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题三：事件严重度敏感性实验")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/team_cleaned"))
    parser.add_argument("--route-path", type=Path, default=Path("results/question2_optimized/question2_optimized_routes.csv"))
    parser.add_argument("--summary-path", type=Path, default=Path("results/question2_optimized/question2_optimized_route_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/question3_sensitivity"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.data_dir,
        arguments.route_path,
        arguments.summary_path,
        arguments.output_dir,
    )
