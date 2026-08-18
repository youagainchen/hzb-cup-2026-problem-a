from __future__ import annotations

import csv
import unittest
from pathlib import Path

from src.data.loader import load_problem_data
from src.model.evaluator import RouteEvaluator, evaluate_solution
from src.model.policy_q2 import build_q2_policy
from src.model.q3_event import Q3EventType, apply_events
from src.solver.q2_initial import load_route_solution
from src.solver.q3_dynamic import dispatch_event_set
from tools.run_q3_optimized import read_event_sets
from tools.run_q3_severity_sensitivity import build_severity_scenarios


class Question3DynamicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = Path("data/processed/team_cleaned")
        self.problem = load_problem_data(self.data_dir)
        self.route_path = Path("results/question2_optimized/question2_optimized_routes.csv")
        self.summary_path = Path(
            "results/question2_optimized/question2_optimized_route_summary.csv"
        )
        self.routes = load_route_solution(self.route_path, self.summary_path)
        self.event_set = read_event_sets(
            Path("results/question3/question3_event_set.csv")
        )[0]

    def test_four_event_batch_reaches_feasible_dynamic_plan(self) -> None:
        evaluator = RouteEvaluator(
            self.problem,
            policy=build_q2_policy(self.problem.green_customer_ids),
        )
        static_total = evaluate_solution(
            self.routes, evaluator, optimize_departures=False
        ).total_cost
        step = dispatch_event_set(
            self.routes,
            self.problem,
            self.event_set,
            static_total_cost=static_total,
        )
        self.assertTrue(step.evaluation.feasibility.passed)
        self.assertEqual(step.evaluation.feasibility.policy_violation_count, 0)
        self.assertEqual(step.evaluation.feasibility.schedule_violation_count, 0)
        self.assertTrue(step.evaluation.feasibility.all_routes_return_before_24h)
        self.assertEqual(step.evaluation.frozen_trip_count, 15)
        self.assertGreaterEqual(step.response_time_s, 0.0)

    def test_frozen_trips_are_not_replanned(self) -> None:
        evaluator = RouteEvaluator(
            self.problem,
            policy=build_q2_policy(self.problem.green_customer_ids),
        )
        before = {
            route.route_id: (
                route.start_minutes,
                route.vehicle_type.name,
                route.vehicle_number,
                route.trip_number,
                tuple((item.customer_id, item.weight, item.volume) for item in route.deliveries),
            )
            for route in self.routes
            if route.start_minutes < self.event_set.trigger_time_minutes
        }
        static_total = evaluate_solution(
            self.routes, evaluator, optimize_departures=False
        ).total_cost
        step = dispatch_event_set(
            self.routes,
            self.problem,
            self.event_set,
            static_total_cost=static_total,
        )
        self.assertEqual(set(before), set(step.freeze_state.frozen_trip_ids))
        for route in self.routes:
            if route.route_id not in before:
                continue
            self.assertEqual(
                before[route.route_id],
                (
                    route.start_minutes,
                    route.vehicle_type.name,
                    route.vehicle_number,
                    route.trip_number,
                    tuple((item.customer_id, item.weight, item.volume) for item in route.deliveries),
                ),
            )

    def test_event_application_changes_all_four_event_dimensions(self) -> None:
        problem_after, audit = apply_events(self.problem, self.event_set)
        event_types = {event.event_type for event in self.event_set.events}
        self.assertEqual(
            event_types,
            {
                Q3EventType.CANCEL,
                Q3EventType.NEW_ORDER,
                Q3EventType.ADDRESS_CHANGE,
                Q3EventType.TIME_WINDOW_CHANGE,
            },
        )
        self.assertNotIn(12, problem_after.demands)
        self.assertIn(99, problem_after.demands)
        self.assertEqual(problem_after.coordinates[82], (20.0, -20.0))
        self.assertEqual(problem_after.windows[70], (726.0, 785.0))
        self.assertIn(82, audit["changed_node_ids"])
        self.assertIn(99, audit["changed_node_ids"])

    def test_severity_scenarios_remain_feasible(self) -> None:
        evaluator = RouteEvaluator(
            self.problem,
            policy=build_q2_policy(self.problem.green_customer_ids),
        )
        static_total = evaluate_solution(
            self.routes, evaluator, optimize_departures=False
        ).total_cost
        for scenario in build_severity_scenarios():
            step = dispatch_event_set(
                self.routes,
                self.problem,
                scenario,
                static_total_cost=static_total,
            )
            feasibility = step.evaluation.feasibility
            self.assertTrue(feasibility.passed, scenario.description)
            self.assertEqual(feasibility.policy_violation_count, 0)
            self.assertEqual(feasibility.schedule_violation_count, 0)
            self.assertEqual(feasibility.demand_unfinished_customers, 0)


if __name__ == "__main__":
    unittest.main()
