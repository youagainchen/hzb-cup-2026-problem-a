from __future__ import annotations

import unittest

import numpy as np

from src.model.domain import Delivery, ProblemData, Route, VehicleType
from src.model.evaluator import RouteEvaluator
from src.solver.greedy import build_greedy_routes, validate_solution
from src.solver.local_search import improve_routes_relocate


def _problem() -> ProblemData:
    return ProblemData(
        distance=np.array(
            [
                [0.0, 10.0, 12.0],
                [10.0, 0.0, 3.0],
                [12.0, 3.0, 0.0],
            ]
        ),
        demands={1: (4000.0, 8.0), 2: (1000.0, 2.0)},
        windows={1: (480.0, 900.0), 2: (480.0, 900.0)},
        coordinates={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)},
        all_customer_ids=(1, 2),
    )


class BaselineTests(unittest.TestCase):
    def test_split_delivery_and_capacity(self) -> None:
        problem = _problem()
        fleet = (VehicleType("TEST", "electric", 3000.0, 10.0, 2),)
        routes = build_greedy_routes(problem, fleet)
        self.assertEqual(len(routes), 2)
        self.assertTrue(all(route.total_weight <= 3000.0 + 1e-8 for route in routes))
        delivered = sum(item.weight for route in routes for item in route.deliveries)
        self.assertAlmostEqual(delivered, 5000.0, places=6)

    def test_evaluator_returns_complete_cost(self) -> None:
        problem = _problem()
        fleet = (VehicleType("TEST", "electric", 3000.0, 10.0, 2),)
        routes = build_greedy_routes(problem, fleet)
        result = RouteEvaluator(problem).best_departure(routes[0])
        self.assertGreater(result.distance_km, 0)
        self.assertGreater(result.energy_cost, 0)
        self.assertGreater(result.carbon_cost, 0)
        self.assertGreaterEqual(result.total_cost, result.fixed_cost)

    def test_inter_route_relocate_can_remove_a_vehicle(self) -> None:
        problem = ProblemData(
            distance=np.array(
                [
                    [0.0, 10.0, 11.0],
                    [10.0, 0.0, 1.0],
                    [11.0, 1.0, 0.0],
                ]
            ),
            demands={1: (40.0, 1.0), 2: (40.0, 1.0)},
            windows={1: (480.0, 900.0), 2: (480.0, 900.0)},
            coordinates={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)},
            all_customer_ids=(1, 2),
        )
        vehicle = VehicleType("TEST", "electric", 100.0, 5.0, 2)
        routes = [
            Route(vehicle, 1, [Delivery(1, 40.0, 1.0)]),
            Route(vehicle, 2, [Delivery(2, 40.0, 1.0)]),
        ]
        improved = improve_routes_relocate(routes, RouteEvaluator(problem))
        self.assertEqual(len(improved), 1)
        self.assertAlmostEqual(improved[0].total_weight, 80.0)


if __name__ == "__main__":
    unittest.main()
