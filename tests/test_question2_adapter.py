from __future__ import annotations

from pathlib import Path
import unittest

from src.q2_adapter import build_context
from src.solver.q2_search import clone_routes


class Question2AdapterTests(unittest.TestCase):
    def test_real_q2_baseline_is_compliant_and_read_only(self) -> None:
        context = build_context(Path("data/processed/team_cleaned"))
        baseline = clone_routes(context.baseline_routes)
        before = [
            (
                route.route_id,
                route.start_minutes,
                tuple(route.deliveries),
            )
            for route in baseline
        ]
        score = context.scorer(baseline)
        after = [
            (
                route.route_id,
                route.start_minutes,
                tuple(route.deliveries),
            )
            for route in baseline
        ]
        self.assertEqual(score.policy_violation_count, 0)
        self.assertTrue(score.is_feasible)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
