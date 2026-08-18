from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.data.loader import load_problem_data
from src.model.evaluator import RouteEvaluator, evaluate_solution
from src.solver.greedy import validate_solution
from src.solver.q2_initial import load_route_solution


class Question2InitialTests(unittest.TestCase):
    def test_loads_frozen_question1_solution_without_losing_demand(self) -> None:
        routes = load_route_solution(
            Path("results/routes/question1_optimized_routes.csv"),
            Path("results/tables/question1_optimized_route_summary.csv"),
        )
        problem = load_problem_data(Path("data/processed/team_cleaned"))
        validate_solution(problem, routes)
        self.assertEqual(len(routes), 98)
        self.assertEqual(
            len({(route.vehicle_type.name, route.vehicle_number) for route in routes}),
            38,
        )
        frozen = json.loads(
            Path("results/tables/question1_optimized_totals.json").read_text(
                encoding="utf-8"
            )
        )
        evaluation = evaluate_solution(
            routes,
            RouteEvaluator(problem),
            optimize_departures=False,
        )
        self.assertAlmostEqual(evaluation.total_cost, frozen["total_cost"], places=6)
        self.assertAlmostEqual(
            evaluation.emissions_kg,
            frozen["total_emissions_kg"],
            places=6,
        )
        self.assertTrue(evaluation.all_routes_return_before_24h)


if __name__ == "__main__":
    unittest.main()
