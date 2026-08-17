from __future__ import annotations

from src.model.domain import Route
from src.model.evaluator import RouteEvaluator


def improve_route_two_opt(
    route: Route,
    evaluator: RouteEvaluator,
    max_passes: int = 3,
) -> Route:
    """以路线总成本为准进行确定性 2-opt；只改变访问顺序。"""

    if len(route.deliveries) < 3:
        return route

    best_cost = evaluator.evaluate(route, route.start_minutes).total_cost
    for _ in range(max_passes):
        improved = False
        size = len(route.deliveries)
        for left in range(size - 1):
            for right in range(left + 2, size + 1):
                candidate_deliveries = (
                    route.deliveries[:left]
                    + list(reversed(route.deliveries[left:right]))
                    + route.deliveries[right:]
                )
                candidate = Route(
                    vehicle_type=route.vehicle_type,
                    vehicle_number=route.vehicle_number,
                    deliveries=candidate_deliveries,
                    start_minutes=route.start_minutes,
                )
                candidate_cost = evaluator.evaluate(candidate, candidate.start_minutes).total_cost
                if candidate_cost < best_cost - 1e-7:
                    route.deliveries = candidate_deliveries
                    best_cost = candidate_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return route


def improve_routes_two_opt(routes: list[Route], evaluator: RouteEvaluator) -> list[Route]:
    return [improve_route_two_opt(route, evaluator) for route in routes]

