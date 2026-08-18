from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.model.domain import Delivery, Route, VehicleType
from src.question2 import Question2Context


@dataclass(frozen=True)
class FakeScore:
    total_cost: float
    policy_violation_count: int
    is_feasible: bool = True


EV = VehicleType("EV", "electric", 100.0, 10.0, 2, fixed_cost=500.0)
FUEL = VehicleType("FUEL", "fuel", 100.0, 10.0, 2, fixed_cost=400.0)


def _scorer(routes: list[Route]) -> FakeScore:
    violations = sum(
        route.vehicle_type.propulsion == "fuel"
        and any(item.customer_id == 1 for item in route.deliveries)
        and 8 * 60 <= route.start_minutes < 16 * 60
        for route in routes
    )
    cost = sum(route.vehicle_type.fixed_cost for route in routes)
    cost += sum(abs(route.start_minutes - 8 * 60) * 0.1 for route in routes)
    return FakeScore(cost, int(violations))


def build_context(data_dir: Path) -> Question2Context:
    del data_dir
    return Question2Context(
        baseline_routes=(
            Route(FUEL, 1, [Delivery(1, 50.0, 1.0)], 8 * 60),
        ),
        scorer=_scorer,
        vehicle_types=(EV, FUEL),
        departure_candidates=(8 * 60, 16 * 60),
    )
