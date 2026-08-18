from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter

from src.model.domain import DEFAULT_VEHICLE_TYPES, Delivery, ProblemData, Route, VehicleType
from src.model.evaluator import RouteEvaluator, evaluate_solution
from src.model.q3_event import Q3EventSet, Q3EventType, apply_events
from src.model.q3_evaluator import (
    DynamicEvaluation,
    FreezeState,
    evaluate_dynamic,
    extract_freeze_state,
)
from src.model.policy_q2 import build_q2_policy


DAY_END_MINUTES = 24.0 * 60.0
EPSILON = 1e-7


@dataclass(frozen=True)
class DynamicStep:
    event_set: Q3EventSet
    problem_after: ProblemData
    freeze_state: FreezeState
    future_routes: tuple[Route, ...]
    evaluation: DynamicEvaluation
    response_time_s: float


def clone_route(route: Route, *, deliveries: list[Delivery] | None = None) -> Route:
    return Route(
        vehicle_type=route.vehicle_type,
        vehicle_number=route.vehicle_number,
        deliveries=list(route.deliveries if deliveries is None else deliveries),
        start_minutes=route.start_minutes,
        trip_number=route.trip_number,
    )


def clone_routes(routes: list[Route] | tuple[Route, ...]) -> list[Route]:
    return [clone_route(route) for route in routes]


def _route_cost(
    route: Route,
    evaluator: RouteEvaluator,
    start_minutes: float | None = None,
) -> tuple[float, object] | None:
    start = route.start_minutes if start_minutes is None else start_minutes
    result = evaluator.evaluate(route, start)
    if result.policy_violation_count != 0:
        return None
    if result.finish_minutes > DAY_END_MINUTES + EPSILON:
        return None
    return result.total_cost, result


def _order_candidates(deliveries: list[Delivery]) -> list[list[Delivery]]:
    candidates: list[list[Delivery]] = [list(deliveries)]
    if len(deliveries) > 1:
        candidates.append(list(reversed(deliveries)))
    for source_index in range(len(deliveries)):
        item = deliveries[source_index]
        remaining = deliveries[:source_index] + deliveries[source_index + 1 :]
        for target_index in range(len(remaining) + 1):
            candidate = remaining[:target_index] + [item] + remaining[target_index:]
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _local_reorder(
    routes: list[Route],
    evaluator: RouteEvaluator,
    affected_customer_ids: set[int],
) -> list[Route]:
    """对受事件影响的路线尝试反转和单点重排，评分完全交给统一评估器。"""

    result_routes = clone_routes(routes)
    for index, route in enumerate(result_routes):
        if not any(item.customer_id in affected_customer_ids for item in route.deliveries):
            continue
        best_route = route
        best_score: tuple[float, float, tuple[int, ...]] | None = None
        for deliveries in _order_candidates(route.deliveries):
            candidate = clone_route(route, deliveries=deliveries)
            scored = _route_cost(candidate, evaluator)
            if scored is None:
                continue
            _, evaluation = scored
            score = (
                float(evaluation.total_cost),
                float(evaluation.finish_minutes),
                tuple(item.customer_id for item in deliveries),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_route = candidate
        result_routes[index] = best_route
    return result_routes


def _remove_cancelled_routes(
    routes: list[Route],
    event_set: Q3EventSet,
) -> list[Route]:
    cancelled = {
        event.customer_id
        for event in event_set.events
        if event.event_type == Q3EventType.CANCEL
    }
    result: list[Route] = []
    for route in routes:
        deliveries = [
            item for item in route.deliveries if item.customer_id not in cancelled
        ]
        if deliveries:
            result.append(clone_route(route, deliveries=deliveries))
    return result


def _insert_one_delivery(
    routes: list[Route],
    delivery: Delivery,
    evaluator: RouteEvaluator,
    trigger_time_minutes: float,
) -> list[Route]:
    best: tuple[tuple[float, float, int, int], int, Route] | None = None
    for route_index, route in enumerate(routes):
        for position in range(len(route.deliveries) + 1):
            deliveries = list(route.deliveries)
            deliveries.insert(position, delivery)
            candidate = clone_route(route, deliveries=deliveries)
            if (
                candidate.total_weight > candidate.vehicle_type.capacity_weight + EPSILON
                or candidate.total_volume > candidate.vehicle_type.capacity_volume + EPSILON
            ):
                continue
            scored = _route_cost(candidate, evaluator, max(route.start_minutes, trigger_time_minutes))
            if scored is None:
                continue
            _, evaluation = scored
            score = (
                float(evaluation.total_cost),
                float(evaluation.finish_minutes),
                route_index,
                position,
            )
            if best is None or score < best[0]:
                best = (score, route_index, candidate)
    if best is not None:
        _, route_index, candidate = best
        result = clone_routes(routes)
        result[route_index] = candidate
        return result

    # If no existing route can absorb the order, open the cheapest feasible vehicle type.
    candidates: list[tuple[tuple[float, float, str], Route]] = []
    for vehicle in DEFAULT_VEHICLE_TYPES:
        if delivery.weight > vehicle.capacity_weight + EPSILON:
            continue
        if delivery.volume > vehicle.capacity_volume + EPSILON:
            continue
        candidate = Route(
            vehicle_type=vehicle,
            vehicle_number=0,
            deliveries=[delivery],
            start_minutes=trigger_time_minutes,
            trip_number=1,
        )
        scored = _route_cost(candidate, evaluator, trigger_time_minutes)
        if scored is None:
            continue
        _, evaluation = scored
        candidates.append(((float(evaluation.total_cost), float(evaluation.finish_minutes), vehicle.name), candidate))
    if not candidates:
        raise RuntimeError("新增订单无法插入任何容量和政策可行路线")
    return routes + [min(candidates, key=lambda item: item[0])[1]]


def _apply_new_orders(
    routes: list[Route],
    event_set: Q3EventSet,
    evaluator: RouteEvaluator,
) -> list[Route]:
    result = clone_routes(routes)
    for event in event_set.events:
        if event.event_type != Q3EventType.NEW_ORDER:
            continue
        if event.weight_kg is None or event.volume_m3 is None:
            raise ValueError("新增订单缺少重量或体积")
        result = _insert_one_delivery(
            result,
            Delivery(event.customer_id, event.weight_kg, event.volume_m3),
            evaluator,
            event.trigger_time_minutes,
        )
    return result


def _schedule_future_routes(
    source_routes: list[Route],
    evaluator: RouteEvaluator,
    freeze_state: FreezeState,
    trigger_time_minutes: float,
) -> list[Route]:
    """按事件后的需求顺序重新分配车型、物理车辆和发车时刻。"""

    ready: dict[tuple[str, int], float] = {
        (vehicle.name, number): max(
            trigger_time_minutes,
            freeze_state.vehicle_ready_minutes.get(
                (vehicle.name, number), trigger_time_minutes
            ),
        )
        for vehicle in DEFAULT_VEHICLE_TYPES
        for number in range(1, vehicle.count + 1)
    }
    trip_counter: dict[tuple[str, int], int] = defaultdict(int)
    used_future_keys: set[tuple[str, int]] = set()
    scheduled: list[Route] = []
    candidate_keys = set(freeze_state.vehicle_ready_minutes)
    for vehicle in DEFAULT_VEHICLE_TYPES:
        used_numbers = [
            number for name, number in candidate_keys if name == vehicle.name
        ]
        next_number = max(used_numbers, default=0) + 1
        if next_number <= vehicle.count:
            candidate_keys.add((vehicle.name, next_number))

    vehicles_by_name = {vehicle.name: vehicle for vehicle in DEFAULT_VEHICLE_TYPES}

    def find_departure(
        source: Route,
        vehicle: VehicleType,
        vehicle_number: int,
        earliest: float,
        trip_number: int,
    ) -> tuple[Route, object] | None:
        if source.total_weight > vehicle.capacity_weight + EPSILON:
            return None
        if source.total_volume > vehicle.capacity_volume + EPSILON:
            return None
        candidate = Route(
            vehicle_type=vehicle,
            vehicle_number=vehicle_number,
            deliveries=[
                Delivery(item.customer_id, item.weight, item.volume)
                for item in source.deliveries
            ],
            start_minutes=earliest,
            trip_number=trip_number,
        )
        best_result = None
        start = earliest
        latest_window = max(
            evaluator.problem.windows[item.customer_id][1]
            for item in candidate.deliveries
        )
        latest_start = min(20.0 * 60.0, latest_window)
        while start <= latest_start + EPSILON:
            evaluation = evaluator.evaluate(candidate, start)
            if (
                evaluation.policy_violation_count == 0
                and evaluation.finish_minutes <= DAY_END_MINUTES + EPSILON
                and (best_result is None or evaluation.total_cost < best_result.total_cost - EPSILON)
            ):
                best_result = evaluation
            start += 10.0
        if best_result is None:
            return None
        candidate.start_minutes = best_result.start_minutes
        return candidate, best_result

    def find_stable_departure(
        source: Route,
        vehicle: VehicleType,
        vehicle_number: int,
        earliest: float,
        trip_number: int,
    ) -> tuple[Route, object] | None:
        """优先保留原计划发车时刻，必要时才从车辆就绪时刻顺延。"""
        if source.total_weight > vehicle.capacity_weight + EPSILON:
            return None
        if source.total_volume > vehicle.capacity_volume + EPSILON:
            return None
        candidate = Route(
            vehicle_type=vehicle,
            vehicle_number=vehicle_number,
            deliveries=[
                Delivery(item.customer_id, item.weight, item.volume)
                for item in source.deliveries
            ],
            start_minutes=max(source.start_minutes, earliest),
            trip_number=trip_number,
        )
        if source.start_minutes + EPSILON >= earliest:
            planned = evaluator.evaluate(candidate, source.start_minutes)
            if (
                planned.policy_violation_count == 0
                and planned.finish_minutes <= DAY_END_MINUTES + EPSILON
            ):
                candidate.start_minutes = source.start_minutes
                return candidate, planned

        latest_start = min(
            20.0 * 60.0,
            max(evaluator.problem.windows[item.customer_id][1] for item in candidate.deliveries),
        )
        start = earliest
        while start <= latest_start + EPSILON:
            evaluation = evaluator.evaluate(candidate, start)
            if (
                evaluation.policy_violation_count == 0
                and evaluation.finish_minutes <= DAY_END_MINUTES + EPSILON
            ):
                candidate.start_minutes = start
                return candidate, evaluation
            start += 5.0
        return None

    # First preserve each original vehicle's future chain, keeping planned departure
    # times whenever possible and shifting only successor trips that would overlap.
    preserved: list[Route] = []
    preserve_ok = True
    grouped: dict[tuple[str, int], list[Route]] = defaultdict(list)
    unassigned: list[Route] = []
    for source in source_routes:
        key = (source.vehicle_type.name, source.vehicle_number)
        if source.vehicle_number > 0 and key in ready:
            grouped[key].append(source)
        else:
            unassigned.append(source)
    for key, chain in grouped.items():
        vehicle = vehicles_by_name[key[0]]
        available = ready[key]
        for source in sorted(chain, key=lambda route: (route.trip_number, route.start_minutes)):
            scheduled_choice = find_stable_departure(
                source,
                vehicle,
                key[1],
                max(trigger_time_minutes, available),
                source.trip_number,
            )
            if scheduled_choice is None:
                preserve_ok = False
                break
            candidate, evaluation = scheduled_choice
            available = evaluation.finish_minutes
            preserved.append(candidate)
        if not preserve_ok:
            break
    if preserve_ok:
        for source in unassigned:
            choices = []
            for vehicle in DEFAULT_VEHICLE_TYPES:
                for name, number in sorted(candidate_keys):
                    if name != vehicle.name:
                        continue
                    key = (name, number)
                    choice = find_stable_departure(
                        source,
                        vehicle,
                        number,
                        max(trigger_time_minutes, ready[key]),
                        1,
                    )
                    if choice is not None:
                        candidate, evaluation = choice
                        choices.append((evaluation.total_cost, candidate, evaluation))
            if not choices:
                preserve_ok = False
                break
            _, candidate, evaluation = min(choices, key=lambda item: (item[0], item[1].route_id))
            preserved.append(candidate)
        if preserve_ok and len(preserved) == len(source_routes):
            return preserved

    def route_priority(route: Route) -> tuple[float, float, int]:
        deadline = min(
            evaluator.problem.windows[item.customer_id][1]
            for item in route.deliveries
        )
        return deadline, route.start_minutes, len(route.deliveries)

    for source in sorted(source_routes, key=route_priority):
        choices: list[tuple[tuple[float, float, int, str, int], Route, object]] = []
        preferred_key = (source.vehicle_type.name, source.vehicle_number)
        for vehicle in DEFAULT_VEHICLE_TYPES:
            if source.total_weight > vehicle.capacity_weight + EPSILON:
                continue
            if source.total_volume > vehicle.capacity_volume + EPSILON:
                continue
            for vehicle_number in sorted(
                number
                for name, number in candidate_keys
                if name == vehicle.name
            ):
                key = (vehicle.name, vehicle_number)
                earliest = max(trigger_time_minutes, ready[key])
                candidate = Route(
                    vehicle_type=vehicle,
                    vehicle_number=vehicle_number,
                    deliveries=[
                        Delivery(item.customer_id, item.weight, item.volume)
                        for item in source.deliveries
                    ],
                    start_minutes=earliest,
                    trip_number=trip_counter[key] + 1,
                )
                scheduled_choice = find_departure(
                    source,
                    vehicle,
                    vehicle_number,
                    earliest,
                    trip_counter[key] + 1,
                )
                if scheduled_choice is None:
                    continue
                candidate, best_result = scheduled_choice
                new_vehicle_penalty = 400.0 if key not in used_future_keys else 0.0
                type_change_penalty = 25.0 if vehicle.name != source.vehicle_type.name else 0.0
                key_preference_penalty = 5.0 if key != preferred_key else 0.0
                score = (
                    best_result.total_cost + new_vehicle_penalty + type_change_penalty + key_preference_penalty,
                    best_result.finish_minutes,
                    int(vehicle.name != source.vehicle_type.name),
                    vehicle.name,
                    vehicle_number,
                )
                choices.append((score, candidate, best_result))
        if not choices:
            raise RuntimeError(
                f"事件后路线 {source.route_id} 无法在车辆、政策和24:00约束下重新排程"
            )
        _, candidate, best_result = min(choices, key=lambda item: item[0])
        candidate.start_minutes = best_result.start_minutes
        key = (candidate.vehicle_type.name, candidate.vehicle_number)
        ready[key] = best_result.finish_minutes
        trip_counter[key] += 1
        used_future_keys.add(key)
        scheduled.append(candidate)
    return scheduled


def dispatch_event_set(
    static_routes: list[Route],
    problem: ProblemData,
    event_set: Q3EventSet,
    static_total_cost: float | None = None,
) -> DynamicStep:
    """执行一次滚动重优化：冻结已发车趟次，只重排未来趟次。"""

    policy = build_q2_policy(problem.green_customer_ids)
    evaluator = RouteEvaluator(problem, policy=policy)
    if static_total_cost is None:
        static_total_cost = evaluate_solution(
            static_routes, evaluator, optimize_departures=False
        ).total_cost

    start_clock = perf_counter()
    problem_after, _ = apply_events(problem, event_set)
    evaluator_after = RouteEvaluator(
        problem_after,
        policy=build_q2_policy(problem_after.green_customer_ids),
    )
    freeze_state = extract_freeze_state(
        static_routes,
        evaluator,
        event_set.trigger_time_minutes,
        problem_after,
    )
    frozen_ids = set(freeze_state.frozen_trip_ids)
    future_source = [
        clone_route(route)
        for route in static_routes
        if route.route_id not in frozen_ids
    ]
    future_source = _remove_cancelled_routes(future_source, event_set)
    future_source = _apply_new_orders(future_source, event_set, evaluator_after)
    affected = {
        event.customer_id
        for event in event_set.events
        if event.event_type in {
            Q3EventType.ADDRESS_CHANGE,
            Q3EventType.TIME_WINDOW_CHANGE,
        }
    }
    future_source = _local_reorder(future_source, evaluator_after, affected)
    future_routes = _schedule_future_routes(
        future_source,
        evaluator_after,
        freeze_state,
        event_set.trigger_time_minutes,
    )
    response_time_s = perf_counter() - start_clock
    evaluation = evaluate_dynamic(
        future_routes,
        freeze_state,
        evaluator_after,
        static_total_cost=float(static_total_cost),
        static_routes=static_routes,
        optimize_departures=False,
        response_time_s=response_time_s,
    )
    return DynamicStep(
        event_set=event_set,
        problem_after=problem_after,
        freeze_state=freeze_state,
        future_routes=tuple(future_routes),
        evaluation=evaluation,
        response_time_s=response_time_s,
    )


def run_event_sequence(
    static_routes: list[Route],
    problem: ProblemData,
    event_sets: list[Q3EventSet],
    static_total_cost: float | None = None,
) -> tuple[DynamicStep, ...]:
    """按触发时刻运行事件序列；同一时刻的事件应组成一个事件集。"""

    current_routes = clone_routes(static_routes)
    current_problem = problem
    current_total = static_total_cost
    steps: list[DynamicStep] = []
    for event_set in sorted(event_sets, key=lambda item: item.trigger_time_minutes):
        step = dispatch_event_set(current_routes, current_problem, event_set, current_total)
        steps.append(step)
        frozen = [
            clone_route(route)
            for route in current_routes
            if route.route_id in set(step.freeze_state.frozen_trip_ids)
        ]
        current_routes = frozen + list(step.future_routes)
        current_problem = step.problem_after
        current_total = step.evaluation.total_cost
    return tuple(steps)
