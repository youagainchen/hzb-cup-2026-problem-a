from __future__ import annotations

import unittest

import numpy as np

from src.model.domain import Delivery, ProblemData, Route, VehicleType
from src.model.evaluator import RouteEvaluator, evaluate_solution
from src.model.policy_q2 import build_q2_policy
from src.solver.q2_scheduling import schedule_q2_routes
from src.solver.scheduling import validate_vehicle_schedule


class Question2SchedulingTests(unittest.TestCase):
    def test_scheduler_uses_ev_or_off_peak_for_green_customer(self) -> None:
        problem = ProblemData(
            distance=np.array([[0.0, 5.0], [5.0, 0.0]]),
            demands={1: (80.0, 1.0)},
            windows={1: (480.0, 1200.0)},
            coordinates={0: (0.0, 0.0), 1: (1.0, 0.0)},
            all_customer_ids=(1,),
            green_customer_ids=frozenset({1}),
        )
        fuel = VehicleType("FUEL", "fuel", 100.0, 5.0, 1)
        electric = VehicleType("EV", "electric", 100.0, 5.0, 1)
        source = [Route(fuel, 1, [Delivery(1, 80.0, 1.0)])]
        evaluator = RouteEvaluator(
            problem,
            policy=build_q2_policy(problem.green_customer_ids),
        )
        routes = schedule_q2_routes(
            source,
            evaluator,
            seed=1,
            vehicle_types=(fuel, electric),
        )
        result = evaluate_solution(routes, evaluator, optimize_departures=False)
        self.assertEqual(result.policy_violation_count, 0)
        validate_vehicle_schedule(routes, evaluator)

    def test_fuel_route_serves_non_green_stop_before_restricted_green_stop(self) -> None:
        problem = ProblemData(
            distance=np.array(
                [
                    [0.0, 5.0, 5.0],
                    [5.0, 0.0, 5.0],
                    [5.0, 5.0, 0.0],
                ]
            ),
            demands={1: (40.0, 1.0), 2: (40.0, 1.0)},
            windows={1: (480.0, 1200.0), 2: (480.0, 600.0)},
            coordinates={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (20.0, 0.0)},
            all_customer_ids=(1, 2),
            green_customer_ids=frozenset({1}),
        )
        fuel = VehicleType("FUEL", "fuel", 100.0, 5.0, 1)
        source = [
            Route(
                fuel,
                1,
                [Delivery(1, 40.0, 1.0), Delivery(2, 40.0, 1.0)],
            )
        ]
        evaluator = RouteEvaluator(
            problem,
            policy=build_q2_policy(problem.green_customer_ids),
        )
        routes = schedule_q2_routes(
            source,
            evaluator,
            seed=1,
            vehicle_types=(fuel,),
        )
        self.assertEqual([item.customer_id for item in routes[0].deliveries], [2, 1])
        result = evaluate_solution(routes, evaluator, optimize_departures=False)
        self.assertEqual(result.policy_violation_count, 0)


if __name__ == "__main__":
    unittest.main()
