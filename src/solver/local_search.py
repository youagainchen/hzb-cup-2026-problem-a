from __future__ import annotations

from dataclasses import replace

from src.model.domain import Delivery, Route
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


def _target_deliveries_after_move(
    target: Route,
    item: Delivery,
    insert_position: int,
) -> list[Delivery]:
    deliveries = list(target.deliveries)
    for index, existing in enumerate(deliveries):
        if existing.customer_id == item.customer_id:
            deliveries[index] = replace(
                existing,
                weight=existing.weight + item.weight,
                volume=existing.volume + item.volume,
            )
            return deliveries
    deliveries.insert(insert_position, item)
    return deliveries


def improve_routes_relocate(
    routes: list[Route],
    evaluator: RouteEvaluator,
    max_moves: int = 100,
    candidate_target_count: int = 20,
    minimum_gain: float = 0.01,
) -> list[Route]:
    """在车辆之间搬移完整配送块；允许合并同一客户并删除空路线。"""

    working = list(routes)
    for route in working:
        evaluator.best_departure(route)

    for _ in range(max_moves):
        route_costs = {
            id(route): evaluator.evaluate(route, route.start_minutes).total_cost
            for route in working
        }
        best_move: tuple[float, Route, Route, list[Delivery], list[Delivery]] | None = None

        for source in working:
            source_old_cost = route_costs[id(source)]
            for item_index, item in enumerate(source.deliveries):
                source_after = source.deliveries[:item_index] + source.deliveries[item_index + 1 :]
                if source_after:
                    source_candidate = Route(
                        vehicle_type=source.vehicle_type,
                        vehicle_number=source.vehicle_number,
                        deliveries=list(source_after),
                        start_minutes=source.start_minutes,
                    )
                    source_new_cost = evaluator.evaluate(
                        source_candidate, source_candidate.start_minutes
                    ).total_cost
                else:
                    source_new_cost = 0.0

                feasible_targets: list[tuple[float, Route]] = []
                for target in working:
                    if target is source:
                        continue
                    if target.total_weight + item.weight > target.vehicle_type.capacity_weight + 1e-7:
                        continue
                    if target.total_volume + item.volume > target.vehicle_type.capacity_volume + 1e-7:
                        continue
                    proximity = min(
                        evaluator.problem.distance[item.customer_id, delivery.customer_id]
                        for delivery in target.deliveries
                    )
                    feasible_targets.append((float(proximity), target))
                feasible_targets.sort(key=lambda pair: pair[0])

                for _, target in feasible_targets[:candidate_target_count]:
                    target_old_cost = route_costs[id(target)]
                    has_same_customer = any(
                        delivery.customer_id == item.customer_id
                        for delivery in target.deliveries
                    )
                    positions = (0,) if has_same_customer else range(len(target.deliveries) + 1)
                    for position in positions:
                        target_after = _target_deliveries_after_move(target, item, position)
                        target_candidate = Route(
                            vehicle_type=target.vehicle_type,
                            vehicle_number=target.vehicle_number,
                            deliveries=target_after,
                            start_minutes=target.start_minutes,
                        )
                        target_new_cost = evaluator.evaluate(
                            target_candidate, target_candidate.start_minutes
                        ).total_cost
                        gain = (
                            source_old_cost
                            + target_old_cost
                            - source_new_cost
                            - target_new_cost
                        )
                        if gain <= minimum_gain:
                            continue
                        if best_move is None or gain > best_move[0]:
                            best_move = (
                                gain,
                                source,
                                target,
                                list(source_after),
                                target_after,
                            )

        if best_move is None:
            break

        _, source, target, source_after, target_after = best_move
        source.deliveries = source_after
        target.deliveries = target_after
        target = improve_route_two_opt(target, evaluator)
        evaluator.best_departure(target)
        if source.deliveries:
            source = improve_route_two_opt(source, evaluator)
            evaluator.best_departure(source)
        else:
            working.remove(source)

    return working
