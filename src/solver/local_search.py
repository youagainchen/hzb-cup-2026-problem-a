from __future__ import annotations

from dataclasses import replace
from itertools import combinations

from src.model.domain import Delivery, Route
from src.model.evaluator import RouteEvaluator
from src.solver.fleet import clone_deliveries, route_load_rate


def improve_route_two_opt(
    route: Route,
    evaluator: RouteEvaluator,
    max_passes: int = 3,
) -> Route:
    """以路线总成本为准进行确定性 2-opt；只改变访问顺序。"""

    if len(route.deliveries) < 3:
        return route

    best_cost = evaluator.best_departure(route).total_cost
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
                candidate_result = evaluator.best_departure(candidate)
                candidate_cost = candidate_result.total_cost
                if candidate_cost < best_cost - 1e-7:
                    route.deliveries = candidate_deliveries
                    route.start_minutes = candidate.start_minutes
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


def eliminate_low_load_routes(
    routes: list[Route],
    evaluator: RouteEvaluator,
    max_removals: int = 30,
    minimum_gain: float = 0.01,
) -> list[Route]:
    """整条删除低装载路线，以最小完整成本增量把配送块插入其他路线。"""

    working = list(routes)
    for route in working:
        evaluator.best_departure(route)

    removed = 0
    while removed < max_removals:
        accepted = False
        for source in sorted(working, key=route_load_rate):
            source_index = working.index(source)
            candidates = [
                Route(
                    route.vehicle_type,
                    route.vehicle_number,
                    clone_deliveries(route.deliveries),
                    route.start_minutes,
                )
                for index, route in enumerate(working)
                if index != source_index
            ]
            old_cost = sum(
                evaluator.evaluate(route, route.start_minutes).total_cost for route in working
            )
            feasible = True
            blocks = sorted(
                source.deliveries,
                key=lambda item: max(
                    item.weight / source.vehicle_type.capacity_weight,
                    item.volume / source.vehicle_type.capacity_volume,
                ),
                reverse=True,
            )

            for item in blocks:
                best_insertion: tuple[float, int, list[Delivery], float] | None = None
                for target_index, target in enumerate(candidates):
                    if target.total_weight + item.weight > target.vehicle_type.capacity_weight + 1e-7:
                        continue
                    if target.total_volume + item.volume > target.vehicle_type.capacity_volume + 1e-7:
                        continue
                    old_target_cost = evaluator.evaluate(
                        target, target.start_minutes
                    ).total_cost
                    same_customer = any(
                        existing.customer_id == item.customer_id
                        for existing in target.deliveries
                    )
                    positions = (0,) if same_customer else range(len(target.deliveries) + 1)
                    for position in positions:
                        deliveries = _target_deliveries_after_move(target, item, position)
                        candidate = Route(
                            target.vehicle_type,
                            target.vehicle_number,
                            deliveries,
                            target.start_minutes,
                        )
                        result = evaluator.best_departure(candidate)
                        delta = result.total_cost - old_target_cost
                        if best_insertion is None or delta < best_insertion[0] - 1e-9:
                            best_insertion = (
                                delta,
                                target_index,
                                deliveries,
                                candidate.start_minutes,
                            )
                if best_insertion is None:
                    feasible = False
                    break
                _, target_index, deliveries, start_minutes = best_insertion
                candidates[target_index].deliveries = deliveries
                candidates[target_index].start_minutes = start_minutes

            if not feasible:
                continue
            new_cost = sum(
                evaluator.evaluate(route, route.start_minutes).total_cost for route in candidates
            )
            if new_cost >= old_cost - minimum_gain:
                continue

            working = candidates
            for route in working:
                improve_route_two_opt(route, evaluator)
                evaluator.best_departure(route)
            removed += 1
            accepted = True
            break

        if not accepted:
            break
    return working


def repack_split_low_load_route(
    routes: list[Route],
    evaluator: RouteEvaluator,
    source_candidate_count: int = 8,
    maximum_local_cost_increase: float = 1000.0,
) -> list[Route]:
    """拆分低装载趟次的配送量，并填入其他路线的零散剩余容量。"""

    working = [
        Route(
            route.vehicle_type,
            route.vehicle_number,
            clone_deliveries(route.deliveries),
            route.start_minutes,
            route.trip_number,
        )
        for route in routes
    ]
    old_cost = sum(evaluator.best_departure(route).total_cost for route in working)
    best_candidate: tuple[float, list[Route]] | None = None

    source_indices = sorted(
        range(len(working)),
        key=lambda index: route_load_rate(working[index]),
    )[:source_candidate_count]
    for source_index in source_indices:
        source = working[source_index]
        candidates = [
            Route(
                route.vehicle_type,
                route.vehicle_number,
                clone_deliveries(route.deliveries),
                route.start_minutes,
                route.trip_number,
            )
            for index, route in enumerate(working)
            if index != source_index
        ]
        feasible = True

        for source_item in sorted(source.deliveries, key=lambda item: item.weight, reverse=True):
            remaining_weight = source_item.weight
            remaining_volume = source_item.volume
            while remaining_weight > 1e-7 or remaining_volume > 1e-7:
                best_insertion: tuple[
                    float,
                    float,
                    int,
                    Route,
                    float,
                    float,
                ] | None = None
                for target_index, target in enumerate(candidates):
                    free_weight = target.vehicle_type.capacity_weight - target.total_weight
                    free_volume = target.vehicle_type.capacity_volume - target.total_volume
                    fractions = [1.0]
                    if remaining_weight > 1e-9:
                        fractions.append(free_weight / remaining_weight)
                    if remaining_volume > 1e-9:
                        fractions.append(free_volume / remaining_volume)
                    fraction = max(0.0, min(fractions))
                    if fraction <= 1e-7:
                        continue
                    chunk = Delivery(
                        source_item.customer_id,
                        remaining_weight * fraction,
                        remaining_volume * fraction,
                    )
                    old_target_cost = evaluator.evaluate(
                        target, target.start_minutes
                    ).total_cost
                    same_customer = any(
                        item.customer_id == chunk.customer_id
                        for item in target.deliveries
                    )
                    positions = (0,) if same_customer else range(len(target.deliveries) + 1)
                    for position in positions:
                        deliveries = _target_deliveries_after_move(target, chunk, position)
                        candidate = Route(
                            target.vehicle_type,
                            target.vehicle_number,
                            deliveries,
                            target.start_minutes,
                            target.trip_number,
                        )
                        result = evaluator.best_departure(candidate)
                        delta = result.total_cost - old_target_cost
                        score = delta / max(fraction, 1e-4)
                        choice = (
                            score,
                            -fraction,
                            target_index,
                            candidate,
                            chunk.weight,
                            chunk.volume,
                        )
                        if best_insertion is None or choice[:2] < best_insertion[:2]:
                            best_insertion = choice

                if best_insertion is None:
                    feasible = False
                    break
                _, _, target_index, candidate, moved_weight, moved_volume = best_insertion
                candidates[target_index] = candidate
                remaining_weight = max(0.0, remaining_weight - moved_weight)
                remaining_volume = max(0.0, remaining_volume - moved_volume)
            if not feasible:
                break

        if not feasible:
            continue
        new_cost = sum(
            evaluator.evaluate(route, route.start_minutes).total_cost
            for route in candidates
        )
        increase = new_cost - old_cost
        if increase > maximum_local_cost_increase:
            continue
        if best_candidate is None or increase < best_candidate[0] - 1e-7:
            best_candidate = (increase, candidates)

    if best_candidate is None:
        return working
    _, result_routes = best_candidate
    return [improve_route_two_opt(route, evaluator) for route in result_routes]


def improve_routes_swap(
    routes: list[Route],
    evaluator: RouteEvaluator,
    max_moves: int = 30,
    candidate_pair_count: int = 2500,
    minimum_gain: float = 0.01,
) -> list[Route]:
    """交换两条路线的配送块，按完整成本下降接受。"""

    working = list(routes)
    for route in working:
        evaluator.best_departure(route)

    for _ in range(max_moves):
        pairs = list(combinations(range(len(working)), 2))
        pairs.sort(
            key=lambda pair: min(
                evaluator.problem.distance[left.customer_id, right.customer_id]
                for left in working[pair[0]].deliveries
                for right in working[pair[1]].deliveries
            )
        )
        best: tuple[float, int, int, Route, Route] | None = None
        for first_index, second_index in pairs[:candidate_pair_count]:
            first = working[first_index]
            second = working[second_index]
            old_cost = (
                evaluator.evaluate(first, first.start_minutes).total_cost
                + evaluator.evaluate(second, second.start_minutes).total_cost
            )
            for left_index, left in enumerate(first.deliveries):
                for right_index, right in enumerate(second.deliveries):
                    if left.customer_id == right.customer_id:
                        continue
                    if any(
                        item.customer_id == right.customer_id
                        for index, item in enumerate(first.deliveries)
                        if index != left_index
                    ):
                        continue
                    if any(
                        item.customer_id == left.customer_id
                        for index, item in enumerate(second.deliveries)
                        if index != right_index
                    ):
                        continue
                    first_weight = first.total_weight - left.weight + right.weight
                    first_volume = first.total_volume - left.volume + right.volume
                    second_weight = second.total_weight - right.weight + left.weight
                    second_volume = second.total_volume - right.volume + left.volume
                    if first_weight > first.vehicle_type.capacity_weight + 1e-7:
                        continue
                    if first_volume > first.vehicle_type.capacity_volume + 1e-7:
                        continue
                    if second_weight > second.vehicle_type.capacity_weight + 1e-7:
                        continue
                    if second_volume > second.vehicle_type.capacity_volume + 1e-7:
                        continue

                    first_deliveries = clone_deliveries(first.deliveries)
                    second_deliveries = clone_deliveries(second.deliveries)
                    first_deliveries[left_index] = Delivery(
                        right.customer_id, right.weight, right.volume
                    )
                    second_deliveries[right_index] = Delivery(
                        left.customer_id, left.weight, left.volume
                    )
                    first_candidate = Route(
                        first.vehicle_type,
                        first.vehicle_number,
                        first_deliveries,
                        first.start_minutes,
                    )
                    second_candidate = Route(
                        second.vehicle_type,
                        second.vehicle_number,
                        second_deliveries,
                        second.start_minutes,
                    )
                    first_result = evaluator.best_departure(first_candidate)
                    second_result = evaluator.best_departure(second_candidate)
                    gain = old_cost - first_result.total_cost - second_result.total_cost
                    if gain <= minimum_gain:
                        continue
                    if best is None or gain > best[0]:
                        best = (
                            gain,
                            first_index,
                            second_index,
                            first_candidate,
                            second_candidate,
                        )
        if best is None:
            break
        _, first_index, second_index, first_candidate, second_candidate = best
        working[first_index] = improve_route_two_opt(first_candidate, evaluator)
        working[second_index] = improve_route_two_opt(second_candidate, evaluator)
    return working


def _coalesce_deliveries(deliveries: list[Delivery]) -> list[Delivery]:
    combined: list[Delivery] = []
    positions: dict[int, int] = {}
    for item in deliveries:
        position = positions.get(item.customer_id)
        if position is None:
            positions[item.customer_id] = len(combined)
            combined.append(Delivery(item.customer_id, item.weight, item.volume))
        else:
            existing = combined[position]
            combined[position] = Delivery(
                existing.customer_id,
                existing.weight + item.weight,
                existing.volume + item.volume,
            )
    return combined


def improve_routes_merge(
    routes: list[Route],
    evaluator: RouteEvaluator,
    max_merges: int = 30,
    candidate_pair_count: int = 3000,
    minimum_gain: float = 0.01,
) -> list[Route]:
    """直接合并两条路线，并以车辆启动费在内的完整成本决定是否接受。"""

    working = list(routes)
    for route in working:
        evaluator.best_departure(route)

    for _ in range(max_merges):
        pairs = list(combinations(range(len(working)), 2))
        pairs.sort(key=lambda pair: route_load_rate(working[pair[0]]) + route_load_rate(working[pair[1]]))
        best: tuple[float, int, int, Route] | None = None
        for first_index, second_index in pairs[:candidate_pair_count]:
            first = working[first_index]
            second = working[second_index]
            old_cost = (
                evaluator.evaluate(first, first.start_minutes).total_cost
                + evaluator.evaluate(second, second.start_minutes).total_cost
            )
            vehicle_options = {first.vehicle_type.name: first.vehicle_type, second.vehicle_type.name: second.vehicle_type}
            orientations = (
                first.deliveries + second.deliveries,
                first.deliveries + list(reversed(second.deliveries)),
                list(reversed(first.deliveries)) + second.deliveries,
                list(reversed(first.deliveries)) + list(reversed(second.deliveries)),
            )
            for raw_deliveries in orientations:
                deliveries = _coalesce_deliveries(raw_deliveries)
                total_weight = sum(item.weight for item in deliveries)
                total_volume = sum(item.volume for item in deliveries)
                for vehicle in vehicle_options.values():
                    if total_weight > vehicle.capacity_weight + 1e-7:
                        continue
                    if total_volume > vehicle.capacity_volume + 1e-7:
                        continue
                    candidate = Route(
                        vehicle,
                        first.vehicle_number if vehicle is first.vehicle_type else second.vehicle_number,
                        clone_deliveries(deliveries),
                    )
                    result = evaluator.best_departure(candidate)
                    gain = old_cost - result.total_cost
                    if gain <= minimum_gain:
                        continue
                    if best is None or gain > best[0]:
                        best = (gain, first_index, second_index, candidate)
        if best is None:
            break
        _, first_index, second_index, candidate = best
        next_routes = [
            route
            for index, route in enumerate(working)
            if index not in (first_index, second_index)
        ]
        next_routes.append(improve_route_two_opt(candidate, evaluator))
        working = next_routes
    return working
