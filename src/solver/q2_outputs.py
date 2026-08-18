from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence

from src.solver.q2_search import Q2ALNSRun, physical_vehicle_count


@dataclass(frozen=True)
class Q2OutputPaths:
    route_csv: Path
    trace_csv: Path
    totals_json: Path


def write_q2_alns_outputs(
    output_dir: Path,
    best: Q2ALNSRun,
    runs: Sequence[Q2ALNSRun],
    prefix: str = "question2_alns",
) -> Q2OutputPaths:
    """写出不依赖2号具体评分类型的路线、收敛日志和汇总。"""

    route_dir = output_dir / "routes"
    table_dir = output_dir / "tables"
    route_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    route_path = route_dir / f"{prefix}_routes.csv"
    trace_path = table_dir / f"{prefix}_trace.csv"
    totals_path = table_dir / f"{prefix}_totals.json"

    with route_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "seed",
                "route_id",
                "vehicle_type",
                "propulsion",
                "physical_vehicle_number",
                "trip_number",
                "start_minutes",
                "sequence",
                "customer_id",
                "delivered_weight_kg",
                "delivered_volume_m3",
            ]
        )
        for route in best.routes:
            for sequence, delivery in enumerate(route.deliveries, start=1):
                writer.writerow(
                    [
                        best.seed,
                        route.route_id,
                        route.vehicle_type.name,
                        route.vehicle_type.propulsion,
                        route.vehicle_number,
                        route.trip_number,
                        route.start_minutes,
                        sequence,
                        delivery.customer_id,
                        delivery.weight,
                        delivery.volume,
                    ]
                )

    with trace_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "seed",
                "iteration",
                "destroy_operator",
                "repair_operator",
                "accepted",
                "current_policy_violations",
                "current_cost",
                "best_policy_violations",
                "best_cost",
                "best_physical_vehicles",
                "temperature",
            ]
        )
        for run in runs:
            for item in run.trace:
                writer.writerow(
                    [
                        run.seed,
                        item.iteration,
                        item.destroy_operator,
                        item.repair_operator,
                        int(item.accepted),
                        item.current_violations,
                        item.current_cost,
                        item.best_violations,
                        item.best_cost,
                        item.physical_vehicles,
                        item.temperature,
                    ]
                )

    totals = {
        "algorithm": "policy-agnostic adaptive large neighborhood search",
        "best_seed": best.seed,
        "policy_violation_count": best.score.policy_violation_count,
        "is_feasible": best.score.is_feasible,
        "total_cost": best.score.total_cost,
        "delivery_trips": len(best.routes),
        "physical_vehicles": physical_vehicle_count(best.routes),
        "seeds": [run.seed for run in runs],
        "runs": [
            {
                "seed": run.seed,
                "policy_violation_count": run.score.policy_violation_count,
                "is_feasible": run.score.is_feasible,
                "total_cost": run.score.total_cost,
                "delivery_trips": len(run.routes),
                "physical_vehicles": physical_vehicle_count(run.routes),
                "iterations": len(run.trace),
                "destroy_weights": run.destroy_weights,
                "repair_weights": run.repair_weights,
            }
            for run in runs
        ],
    }
    compliant_costs = [
        run.score.total_cost
        for run in runs
        if run.score.is_feasible and run.score.policy_violation_count == 0
    ]
    totals["statistics"] = {
        "run_count": len(runs),
        "compliant_run_count": len(compliant_costs),
        "total_cost_mean_compliant": mean(compliant_costs),
        "total_cost_std_compliant": pstdev(compliant_costs),
        "total_cost_min_compliant": min(compliant_costs),
        "total_cost_max_compliant": max(compliant_costs),
    }
    totals_path.write_text(
        json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return Q2OutputPaths(route_path, trace_path, totals_path)
