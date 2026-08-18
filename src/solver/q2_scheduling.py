from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import Sequence

from src.model.domain import DEFAULT_VEHICLE_TYPES, Delivery, Route, VehicleType
from src.model.evaluator import RouteEvaluation, RouteEvaluator, SolutionEvaluation, evaluate_solution
from src.solver.greedy import validate_solution
from src.solver.scheduling import validate_vehicle_schedule


@dataclass(frozen=True)
class Q2ScheduleRun:
    seed: int
    order_rule: str
    vehicle_reuse_weight: float
    departure_step_minutes: float
    routes: tuple[Route, ...]
    evaluation: SolutionEvaluation


def _fits(deliveries: Sequence[Delivery], vehicle: VehicleType) -> bool:
    return (
        sum(item.weight for item in deliveries) <= vehicle.capacity_weight + 1e-7
        and sum(item.volume for item in deliveries) <= vehicle.capacity_volume + 1e-7
    )


def _sequence_variants(
    source: Route,
    vehicle: VehicleType,
    evaluator: RouteEvaluator,
) -> tuple[tuple[Delivery, ...], ...]:
    """生成少量可解释的2-opt/Or-opt近似顺序候选。"""

    original = tuple(source.deliveries)
    if len(original) <= 1:
        return (original,)

    green_ids = evaluator.problem.green_customer_ids
    non_green = tuple(item for item in original if item.customer_id not in green_ids)
    green = tuple(item for item in original if item.customer_id in green_ids)

    def by_deadline(item: Delivery) -> tuple[float, float, int]:
        window = evaluator.problem.windows[item.customer_id]
        return (window[1], window[0], item.customer_id)

    def nearest_neighbor(items: tuple[Delivery, ...]) -> tuple[Delivery, ...]:
        remaining = list(items)
        current = 0
        ordered: list[Delivery] = []
        while remaining:
            best_index = min(
                range(len(remaining)),
                key=lambda index: (
                    float(
                        evaluator.problem.distance[
                            current, remaining[index].customer_id
                        ]
                    ),
                    by_deadline(remaining[index]),
                ),
            )
            chosen = remaining.pop(best_index)
            ordered.append(chosen)
            current = chosen.customer_id
        return tuple(ordered)

    candidates: tuple[tuple[Delivery, ...], ...] = (
        original,
        tuple(reversed(original)),
        tuple(sorted(original, key=by_deadline)),
        nearest_neighbor(original),
    )
    if vehicle.propulsion == "fuel" and green:
        candidates += (
            non_green + green,
            tuple(sorted(non_green, key=by_deadline))
            + tuple(sorted(green, key=by_deadline)),
            nearest_neighbor(non_green) + tuple(sorted(green, key=by_deadline)),
        )
    unique: list[tuple[Delivery, ...]] = []
    seen: set[tuple[tuple[int, float, float], ...]] = set()
    for candidate in candidates:
        key = tuple(
            (item.customer_id, round(item.weight, 8), round(item.volume, 8))
            for item in candidate
        )
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def best_compliant_route_variant(
    source: Route,
    vehicle: VehicleType,
    evaluator: RouteEvaluator,
    *,
    earliest: float = 8.0 * 60.0,
    step_minutes: float = 10.0,
) -> tuple[Route, RouteEvaluation]:
    """在访问顺序与发车时刻上同时选择成本最低的合规趟次。"""

    best: tuple[Route, RouteEvaluation] | None = None
    for deliveries in _sequence_variants(source, vehicle, evaluator):
        candidate = Route(
            vehicle_type=vehicle,
            vehicle_number=0,
            deliveries=[
                Delivery(item.customer_id, item.weight, item.volume)
                for item in deliveries
            ],
        )
        try:
            result = best_compliant_departure(
                candidate,
                evaluator,
                earliest=earliest,
                step_minutes=step_minutes,
            )
        except ValueError:
            continue
        if best is None or (
            result.total_cost,
            result.finish_minutes,
            tuple(item.customer_id for item in candidate.deliveries),
        ) < (
            best[1].total_cost,
            best[1].finish_minutes,
            tuple(item.customer_id for item in best[0].deliveries),
        ):
            best = (candidate, result)
    if best is None:
        raise ValueError("该车型与最早可用时刻下没有政策合规排班")
    return best


def best_compliant_departure(
    route: Route,
    evaluator: RouteEvaluator,
    earliest: float = 8.0 * 60.0,
    latest_start: float = 20.0 * 60.0,
    latest_finish: float = 24.0 * 60.0,
    step_minutes: float = 10.0,
) -> RouteEvaluation:
    """枚举发车时间，只接受政策违规为0且当日返场的路线。"""

    best: RouteEvaluation | None = None
    start = earliest
    final_start = max(earliest, latest_start)
    while start <= final_start + 1e-9:
        result = evaluator.evaluate(route, start)
        if (
            result.policy_violation_count == 0
            and result.finish_minutes <= latest_finish + 1e-9
            and (best is None or result.total_cost < best.total_cost - 1e-9)
        ):
            best = result
        start += step_minutes
    if best is None:
        raise ValueError("该车型与最早可用时刻下没有政策合规排班")
    route.start_minutes = best.start_minutes
    return best


def _route_key(
    route: Route,
    evaluator: RouteEvaluator,
    order_rule: str,
    tie_breaker: float,
) -> tuple[float, ...]:
    windows = [evaluator.problem.windows[item.customer_id] for item in route.deliveries]
    earliest_deadline = min(window[1] for window in windows)
    latest_deadline = max(window[1] for window in windows)
    green_count = sum(
        item.customer_id in evaluator.problem.green_customer_ids
        for item in route.deliveries
    )
    nodes = [0, *(item.customer_id for item in route.deliveries), 0]
    distance = sum(
        float(evaluator.problem.distance[left, right])
        for left, right in zip(nodes, nodes[1:])
    )
    baseline = evaluator.evaluate(route, 8.0 * 60.0)
    late_risk = sum(stop.late_minutes for stop in baseline.stops)
    if order_rule == "green_first":
        return (-green_count, earliest_deadline, -distance, tie_breaker)
    if order_rule in {"time_window_first", "deadline"}:
        return (earliest_deadline, -late_risk, -green_count, -distance, tie_breaker)
    if order_rule == "late_risk_first":
        return (-late_risk, earliest_deadline, -green_count, -distance, tie_breaker)
    if order_rule == "green_late_hybrid":
        return (-green_count, -late_risk, earliest_deadline, -distance, tie_breaker)
    if order_rule == "distance_late":
        return (-distance, -late_risk, earliest_deadline, -green_count, tie_breaker)
    if order_rule == "long_first":
        return (-distance, earliest_deadline, -green_count, tie_breaker)
    if order_rule == "late_first":
        return (-latest_deadline, -green_count, -distance, tie_breaker)
    return (earliest_deadline, -green_count, -distance, tie_breaker)


def schedule_q2_routes(
    source_routes: Sequence[Route],
    evaluator: RouteEvaluator,
    *,
    seed: int,
    order_rule: str = "green_first",
    vehicle_reuse_weight: float = 400.0,
    departure_step_minutes: float = 10.0,
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES,
    turnaround_minutes: float = 0.0,
) -> list[Route]:
    """联合选择车型、物理车复用和政策合规发车时刻。"""

    rng = Random(seed)
    tie_breakers = {id(route): rng.random() for route in source_routes}
    ordered = sorted(
        source_routes,
        key=lambda route: _route_key(
            route,
            evaluator,
            order_rule,
            tie_breakers[id(route)],
        ),
    )
    states_by_type: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    scheduled: list[Route] = []

    for source in ordered:
        best_choice: tuple[
            float,
            float,
            Route,
            str,
            int | None,
        ] | None = None
        candidate_types = list(vehicle_types)
        rng.shuffle(candidate_types)
        for vehicle in candidate_types:
            if not _fits(source.deliveries, vehicle):
                continue
            states = states_by_type[vehicle.name]
            if len(states) < vehicle.count:
                try:
                    candidate, result = best_compliant_route_variant(
                        source,
                        vehicle,
                        evaluator,
                        step_minutes=departure_step_minutes,
                    )
                except ValueError:
                    result = None
                if result is not None:
                    variable_cost = result.total_cost - vehicle.fixed_cost
                    choice = (
                        variable_cost + vehicle_reuse_weight,
                        result.finish_minutes,
                        candidate,
                        vehicle.name,
                        None,
                    )
                    if best_choice is None or choice[:2] < best_choice[:2]:
                        best_choice = choice

            if states:
                state_index = min(
                    range(len(states)),
                    key=lambda index: float(states[index]["available"]),
                )
                state = states[state_index]
                earliest = float(state["available"]) + turnaround_minutes
                try:
                    candidate, result = best_compliant_route_variant(
                        source,
                        vehicle,
                        evaluator,
                        earliest=earliest,
                        step_minutes=departure_step_minutes,
                    )
                except ValueError:
                    result = None
                if result is not None:
                    candidate.vehicle_number = int(state["vehicle_number"])
                    candidate.trip_number = int(state["trip_count"]) + 1
                    variable_cost = result.total_cost - vehicle.fixed_cost
                    choice = (
                        variable_cost,
                        result.finish_minutes,
                        candidate,
                        vehicle.name,
                        state_index,
                    )
                    if best_choice is None or choice[:2] < best_choice[:2]:
                        best_choice = choice

        if best_choice is None:
            raise RuntimeError("现有车队无法为全部Q1趟次生成政策合规排班")
        _, finish, candidate, vehicle_name, state_index = best_choice
        states = states_by_type[vehicle_name]
        if state_index is None:
            candidate.vehicle_number = len(states) + 1
            candidate.trip_number = 1
            states.append(
                {
                    "vehicle_number": candidate.vehicle_number,
                    "available": finish,
                    "trip_count": 1,
                }
            )
        else:
            states[state_index]["available"] = finish
            states[state_index]["trip_count"] = int(states[state_index]["trip_count"]) + 1
        scheduled.append(candidate)
    return scheduled


def refine_fixed_schedule_departures(
    routes: Sequence[Route],
    evaluator: RouteEvaluator,
    *,
    step_minutes: float,
    turnaround_minutes: float = 0.0,
) -> tuple[list[Route], SolutionEvaluation]:
    """保持车型、物理车、趟次链和访问顺序不变，细化发车时刻。"""

    candidates = [
        Route(
            vehicle_type=route.vehicle_type,
            vehicle_number=route.vehicle_number,
            deliveries=[
                Delivery(item.customer_id, item.weight, item.volume)
                for item in route.deliveries
            ],
            start_minutes=route.start_minutes,
            trip_number=route.trip_number,
        )
        for route in routes
    ]
    chains: dict[tuple[str, int], list[Route]] = defaultdict(list)
    for route in candidates:
        chains[(route.vehicle_type.name, route.vehicle_number)].append(route)
    for chain in chains.values():
        earliest = 8.0 * 60.0
        for route in sorted(chain, key=lambda item: item.trip_number):
            result = best_compliant_departure(
                route,
                evaluator,
                earliest=earliest,
                step_minutes=step_minutes,
            )
            earliest = result.finish_minutes + turnaround_minutes
    validate_vehicle_schedule(candidates, evaluator)
    refined = evaluate_solution(candidates, evaluator, optimize_departures=False)

    original = [
        Route(
            vehicle_type=route.vehicle_type,
            vehicle_number=route.vehicle_number,
            deliveries=[
                Delivery(item.customer_id, item.weight, item.volume)
                for item in route.deliveries
            ],
            start_minutes=route.start_minutes,
            trip_number=route.trip_number,
        )
        for route in routes
    ]
    original_evaluation = evaluate_solution(
        original,
        evaluator,
        optimize_departures=False,
    )
    if refined.total_cost < original_evaluation.total_cost - 1e-9:
        return candidates, refined
    return original, original_evaluation


def search_q2_schedules(
    source_routes: Sequence[Route],
    evaluator: RouteEvaluator,
    *,
    seeds: Sequence[int],
    order_rules: Sequence[str] = (
        "green_first",
        "time_window_first",
        "late_risk_first",
        "green_late_hybrid",
        "distance_late",
    ),
    vehicle_reuse_weights: Sequence[float] = (300.0, 400.0, 500.0),
    departure_step_minutes: float = 10.0,
) -> tuple[Q2ScheduleRun, tuple[Q2ScheduleRun, ...]]:
    runs: list[Q2ScheduleRun] = []
    for seed in seeds:
        for order_rule in order_rules:
            for reuse_weight in vehicle_reuse_weights:
                try:
                    routes = schedule_q2_routes(
                        source_routes,
                        evaluator,
                        seed=int(seed),
                        order_rule=order_rule,
                        vehicle_reuse_weight=float(reuse_weight),
                        departure_step_minutes=float(departure_step_minutes),
                    )
                    validate_solution(evaluator.problem, routes)
                    validate_vehicle_schedule(routes, evaluator)
                    result = evaluate_solution(
                        routes,
                        evaluator,
                        optimize_departures=False,
                    )
                except (AssertionError, RuntimeError, ValueError):
                    continue
                if (
                    result.policy_violation_count != 0
                    or result.unfinished_customer_count != 0
                    or result.capacity_violation_count != 0
                    or not result.all_routes_return_before_24h
                ):
                    continue
                runs.append(
                    Q2ScheduleRun(
                        seed=int(seed),
                        order_rule=order_rule,
                        vehicle_reuse_weight=float(reuse_weight),
                        departure_step_minutes=float(departure_step_minutes),
                        routes=tuple(routes),
                        evaluation=result,
                    )
                )
    if not runs:
        raise RuntimeError("没有找到基于Q1路线的政策合规排班")
    best = min(runs, key=lambda run: (run.evaluation.total_cost, run.seed))
    return best, tuple(runs)
