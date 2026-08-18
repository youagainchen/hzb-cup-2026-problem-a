from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
import unittest

from src.model.domain import Delivery, Route, VehicleType
from src.question2 import run as run_question2
from src.question2_cli import load_context, main as question2_cli_main
from src.solver.q2_search import (
    ALNSConfig,
    DEFAULT_Q2_SEEDS,
    ScorerContractError,
    physical_vehicle_count,
    repair_departure_times,
    repair_vehicle_types,
    run_fixed_seed_alns,
    run_fixed_seed_search,
    search_physical_vehicle_count,
    swap_entire_trips,
)


@dataclass(frozen=True)
class _FakeScore:
    total_cost: float
    policy_violation_count: int
    is_feasible: bool = True


EV = VehicleType("EV", "electric", 100.0, 10.0, 2, fixed_cost=500.0)
FUEL = VehicleType("FUEL", "fuel", 100.0, 10.0, 2, fixed_cost=400.0)


def _fake_q2_scorer(routes: list[Route]) -> _FakeScore:
    """测试专用政策口径，正式求解器中没有这段判定逻辑。"""

    violations = sum(
        route.vehicle_type.propulsion == "fuel"
        and any(item.customer_id == 1 for item in route.deliveries)
        and 8 * 60 <= route.start_minutes < 16 * 60
        for route in routes
    )
    cost = sum(route.vehicle_type.fixed_cost for route in routes)
    cost += sum(abs(route.start_minutes - 8 * 60) * 0.1 for route in routes)
    return _FakeScore(cost, int(violations))


def _fake_schedule_scorer(routes: list[Route]) -> _FakeScore:
    seen_slots: set[tuple[str, int, float]] = set()
    feasible = True
    for route in routes:
        slot = (route.vehicle_type.name, route.vehicle_number, route.start_minutes)
        if slot in seen_slots:
            feasible = False
        seen_slots.add(slot)
    vehicles = {
        (route.vehicle_type.name, route.vehicle_number): route.vehicle_type
        for route in routes
    }
    return _FakeScore(
        total_cost=sum(vehicle.fixed_cost for vehicle in vehicles.values()),
        policy_violation_count=0,
        is_feasible=feasible,
    )


class Question2SearchTests(unittest.TestCase):
    def test_vehicle_repair_uses_injected_policy_score(self) -> None:
        routes = [Route(FUEL, 1, [Delivery(1, 50.0, 1.0)], 8 * 60)]
        repaired, score = repair_vehicle_types(
            routes,
            _fake_q2_scorer,
            Random(1),
            (EV, FUEL),
        )
        self.assertEqual(repaired[0].vehicle_type, EV)
        self.assertEqual(score.policy_violation_count, 0)

    def test_departure_repair_can_remove_violation(self) -> None:
        routes = [Route(FUEL, 1, [Delivery(1, 50.0, 1.0)], 8 * 60)]
        repaired, score = repair_departure_times(
            routes,
            _fake_q2_scorer,
            Random(2),
            (8 * 60, 16 * 60),
        )
        self.assertEqual(repaired[0].start_minutes, 16 * 60)
        self.assertEqual(score.policy_violation_count, 0)

    def test_whole_trip_swap_moves_green_demand_to_ev_slot(self) -> None:
        routes = [
            Route(FUEL, 1, [Delivery(1, 50.0, 1.0)], 8 * 60),
            Route(EV, 1, [Delivery(2, 50.0, 1.0)], 8 * 60),
        ]
        repaired, score = swap_entire_trips(
            routes, _fake_q2_scorer, Random(3)
        )
        self.assertEqual(repaired[1].deliveries[0].customer_id, 1)
        self.assertEqual(score.policy_violation_count, 0)

    def test_fixed_seed_driver_returns_five_reproducible_runs(self) -> None:
        routes = [Route(FUEL, 1, [Delivery(1, 50.0, 1.0)], 8 * 60)]
        best, runs = run_fixed_seed_search(
            routes,
            _fake_q2_scorer,
            vehicle_types=(EV, FUEL),
            departure_candidates=(8 * 60, 16 * 60),
        )
        self.assertEqual(tuple(run.seed for run in runs), DEFAULT_Q2_SEEDS)
        self.assertEqual(best.score.policy_violation_count, 0)
        self.assertEqual(len(runs), 5)

    def test_trip_relocate_can_reduce_physical_vehicle_count(self) -> None:
        routes = [
            Route(FUEL, 1, [Delivery(2, 50.0, 1.0)], 8 * 60),
            Route(FUEL, 2, [Delivery(3, 50.0, 1.0)], 16 * 60),
        ]
        repaired, score = search_physical_vehicle_count(
            routes,
            _fake_schedule_scorer,
            Random(4),
            vehicle_types=(FUEL,),
        )
        self.assertEqual(physical_vehicle_count(repaired), 1)
        self.assertEqual(score.total_cost, FUEL.fixed_cost)
        self.assertEqual({route.trip_number for route in repaired}, {1, 2})

    def test_alns_and_output_pipeline_run_without_real_policy_module(self) -> None:
        routes = [Route(FUEL, 1, [Delivery(1, 50.0, 1.0)], 8 * 60)]
        best, runs = run_fixed_seed_alns(
            routes,
            _fake_q2_scorer,
            vehicle_types=(EV, FUEL),
            departure_candidates=(8 * 60, 16 * 60),
            config=ALNSConfig(iterations=40, max_no_improve=20),
        )
        self.assertEqual(len(runs), 5)
        self.assertEqual(best.score.policy_violation_count, 0)

        with TemporaryDirectory() as temporary_dir:
            result = run_question2(
                routes,
                _fake_q2_scorer,
                output_dir=Path(temporary_dir),
                vehicle_types=(EV, FUEL),
                departure_candidates=(8 * 60, 16 * 60),
                config=ALNSConfig(iterations=40, max_no_improve=20),
            )
            paths = result.outputs
            self.assertTrue(paths.route_csv.exists())
            self.assertTrue(paths.trace_csv.exists())
            totals = json.loads(paths.totals_json.read_text(encoding="utf-8"))
            self.assertEqual(totals["best_seed"], result.best.seed)
            self.assertEqual(totals["policy_violation_count"], 0)
            self.assertEqual(totals["seeds"], list(DEFAULT_Q2_SEEDS))
            self.assertEqual(totals["statistics"]["run_count"], 5)
            self.assertEqual(totals["statistics"]["compliant_run_count"], 5)
            self.assertGreaterEqual(
                totals["statistics"]["total_cost_std_compliant"], 0.0
            )

    def test_mutating_scorer_is_rejected(self) -> None:
        routes = [Route(FUEL, 1, [Delivery(2, 50.0, 1.0)], 8 * 60)]

        def mutating_scorer(candidate: list[Route]) -> _FakeScore:
            candidate[0].start_minutes = 9 * 60
            return _FakeScore(400.0, 0)

        with self.assertRaises(ScorerContractError):
            repair_departure_times(routes, mutating_scorer, Random(5))

    def test_cli_loads_adapter_factory_and_writes_outputs(self) -> None:
        context = load_context("tests.q2_fake_adapter:build_context", Path("unused"))
        self.assertEqual(len(context.baseline_routes), 1)
        with TemporaryDirectory() as temporary_dir, redirect_stdout(StringIO()):
            exit_code = question2_cli_main(
                [
                    "--adapter",
                    "tests.q2_fake_adapter:build_context",
                    "--output-dir",
                    temporary_dir,
                    "--iterations",
                    "20",
                    "--max-no-improve",
                    "10",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(
                (Path(temporary_dir) / "tables" / "question2_alns_totals.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
