from __future__ import annotations

import heapq
from collections import defaultdict

from src.model.domain import DEFAULT_VEHICLE_TYPES, Delivery, Route, VehicleType
from src.model.evaluator import RouteEvaluator


CAPACITY_TOLERANCE = 1e-7


def _fits(deliveries: list[Delivery], vehicle: VehicleType) -> bool:
    return (
        sum(item.weight for item in deliveries)
        <= vehicle.capacity_weight + CAPACITY_TOLERANCE
        and sum(item.volume for item in deliveries)
        <= vehicle.capacity_volume + CAPACITY_TOLERANCE
    )


def _clone_for_vehicle(route: Route, vehicle: VehicleType) -> Route:
    return Route(
        vehicle_type=vehicle,
        vehicle_number=0,
        deliveries=[
            Delivery(item.customer_id, item.weight, item.volume)
            for item in route.deliveries
        ],
    )


def select_and_schedule_multitrip(
    routes: list[Route],
    evaluator: RouteEvaluator,
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES,
    max_physical_vehicles: int | None = None,
    startup_cost_weight: float = 400.0,
    order_rule: str = "deadline",
    turnaround_minutes: float = 0.0,
) -> list[Route]:
    """联合选择车型、物理车辆和多趟时序。

    `vehicle.count` 仅限制该车型可启用的物理车辆数，不限制全天趟次数。
    同车型复用时，只需考察最早可用车辆；它不会劣于更晚可用的车辆。
    """

    if not routes:
        return []

    def route_key(route: Route) -> tuple[float, ...]:
        windows = [evaluator.problem.windows[item.customer_id] for item in route.deliveries]
        earliest_deadline = min(window[1] for window in windows)
        latest_deadline = max(window[1] for window in windows)
        tightness = min(window[1] - window[0] for window in windows)
        depot_distance = max(
            float(evaluator.problem.distance[0, item.customer_id])
            for item in route.deliveries
        )
        if order_rule == "long_first":
            return (-depot_distance, earliest_deadline, latest_deadline)
        if order_rule == "tight_first":
            return (tightness, earliest_deadline, -depot_distance)
        return (earliest_deadline, latest_deadline, -depot_distance)

    states_by_type: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    selected: list[Route] = []

    for source_route in sorted(routes, key=route_key):
        best_choice: tuple[
            float,
            float,
            Route,
            float,
            str,
            int | None,
        ] | None = None
        total_started = sum(len(states) for states in states_by_type.values())

        for vehicle in vehicle_types:
            if not _fits(source_route.deliveries, vehicle):
                continue
            states = states_by_type[vehicle.name]

            if (
                len(states) < vehicle.count
                and (
                    max_physical_vehicles is None
                    or total_started < max_physical_vehicles
                )
            ):
                candidate = _clone_for_vehicle(source_route, vehicle)
                result = evaluator.best_departure(candidate)
                variable_cost = result.total_cost - vehicle.fixed_cost
                choice = (
                    variable_cost + startup_cost_weight,
                    result.finish_minutes,
                    candidate,
                    result.finish_minutes,
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
                candidate = _clone_for_vehicle(source_route, vehicle)
                candidate.vehicle_number = int(state["vehicle_number"])
                candidate.trip_number = int(state["trip_count"]) + 1
                result = evaluator.best_departure(candidate, earliest=earliest)
                variable_cost = result.total_cost - vehicle.fixed_cost
                choice = (
                    variable_cost,
                    result.finish_minutes,
                    candidate,
                    result.finish_minutes,
                    vehicle.name,
                    state_index,
                )
                if best_choice is None or choice[:2] < best_choice[:2]:
                    best_choice = choice

        if best_choice is None:
            limit = "无限制" if max_physical_vehicles is None else str(max_physical_vehicles)
            raise RuntimeError(f"在物理车辆上限 {limit} 下无法安排全部配送趟次")

        _, _, candidate, finish, vehicle_name, state_index = best_choice
        states = states_by_type[vehicle_name]
        if state_index is None:
            vehicle_number = len(states) + 1
            candidate.vehicle_number = vehicle_number
            candidate.trip_number = 1
            states.append(
                {
                    "vehicle_number": vehicle_number,
                    "available": finish,
                    "trip_count": 1,
                }
            )
        else:
            state = states[state_index]
            state["available"] = finish
            state["trip_count"] = int(state["trip_count"]) + 1
        selected.append(candidate)

    return selected


def assign_physical_vehicles(
    routes: list[Route],
    evaluator: RouteEvaluator,
    turnaround_minutes: float = 0.0,
) -> list[Route]:
    """把同车型、时间不重叠的配送趟次分配给同一辆物理车辆。"""

    routes_by_type: dict[str, list[tuple[float, float, Route]]] = defaultdict(list)
    for route in routes:
        result = evaluator.best_departure(route)
        routes_by_type[route.vehicle_type.name].append(
            (result.start_minutes, result.finish_minutes, route)
        )

    for jobs in routes_by_type.values():
        available: list[tuple[float, int]] = []
        next_vehicle_number = 1
        assignments: list[tuple[float, Route]] = []
        for start, finish, route in sorted(jobs, key=lambda item: (item[0], item[1])):
            if available and available[0][0] + turnaround_minutes <= start + 1e-9:
                _, vehicle_number = heapq.heappop(available)
            else:
                vehicle_number = next_vehicle_number
                next_vehicle_number += 1
            route.vehicle_number = vehicle_number
            heapq.heappush(available, (finish, vehicle_number))
            assignments.append((start, route))

        physical_vehicle_count = next_vehicle_number - 1
        if physical_vehicle_count > jobs[0][2].vehicle_type.count:
            raise RuntimeError(
                f"车型 {jobs[0][2].vehicle_type.name} 同时在途车辆数超过车队上限"
            )

        trip_counter: dict[int, int] = defaultdict(int)
        for _, route in sorted(assignments, key=lambda item: (item[1].vehicle_number, item[0])):
            trip_counter[route.vehicle_number] += 1
            route.trip_number = trip_counter[route.vehicle_number]
    return routes


def optimize_multi_trip_schedule(
    routes: list[Route],
    evaluator: RouteEvaluator,
    turnaround_minutes: float = 0.0,
    startup_cost_weight: float = 400.0,
) -> list[Route]:
    """联合权衡400元启动费与时间成本，贪心安排一车多趟。"""

    routes_by_type: dict[str, list[tuple[float, float, Route]]] = defaultdict(list)
    for route in routes:
        result = evaluator.best_departure(route)
        latest_window = max(
            evaluator.problem.windows[item.customer_id][1]
            for item in route.deliveries
        )
        routes_by_type[route.vehicle_type.name].append(
            (result.start_minutes, latest_window, route)
        )

    for jobs in routes_by_type.values():
        vehicle_states: list[dict[str, float | int]] = []
        for _, _, route in sorted(jobs, key=lambda item: (item[0], item[1])):
            best_choice: tuple[float, Route, float, int | None] | None = None

            if len(vehicle_states) < route.vehicle_type.count:
                candidate = Route(
                    route.vehicle_type,
                    0,
                    list(route.deliveries),
                )
                result = evaluator.best_departure(candidate)
                variable_cost = result.total_cost - route.vehicle_type.fixed_cost
                best_choice = (
                    variable_cost + startup_cost_weight,
                    candidate,
                    result.finish_minutes,
                    None,
                )

            for state_index, state in enumerate(vehicle_states):
                earliest = float(state["available"]) + turnaround_minutes
                candidate = Route(
                    route.vehicle_type,
                    int(state["vehicle_number"]),
                    list(route.deliveries),
                    earliest,
                    int(state["trip_count"]) + 1,
                )
                result = evaluator.best_departure(candidate, earliest=earliest)
                incremental_cost = result.total_cost - route.vehicle_type.fixed_cost
                choice = (incremental_cost, candidate, result.finish_minutes, state_index)
                if best_choice is None or choice[0] < best_choice[0] - 1e-9:
                    best_choice = choice

            if best_choice is None:
                raise RuntimeError(f"车型 {route.vehicle_type.name} 没有可用物理车辆")
            _, candidate, finish_minutes, state_index = best_choice
            if state_index is None:
                vehicle_number = len(vehicle_states) + 1
                candidate.vehicle_number = vehicle_number
                candidate.trip_number = 1
                vehicle_states.append(
                    {
                        "vehicle_number": vehicle_number,
                        "available": finish_minutes,
                        "trip_count": 1,
                    }
                )
            else:
                state = vehicle_states[state_index]
                state["available"] = finish_minutes
                state["trip_count"] = int(state["trip_count"]) + 1

            route.vehicle_number = candidate.vehicle_number
            route.trip_number = candidate.trip_number
            route.start_minutes = candidate.start_minutes
    return routes


def validate_vehicle_schedule(
    routes: list[Route],
    evaluator: RouteEvaluator,
    turnaround_minutes: float = 0.0,
) -> None:
    """检查同一物理车辆的各趟次编号连续且时间不重叠。"""

    jobs_by_vehicle: dict[tuple[str, int], list[tuple[float, float, int]]] = defaultdict(list)
    vehicle_types = {}
    for route in routes:
        result = evaluator.evaluate(route, route.start_minutes)
        key = (route.vehicle_type.name, route.vehicle_number)
        vehicle_types[route.vehicle_type.name] = route.vehicle_type
        jobs_by_vehicle[key].append(
            (result.start_minutes, result.finish_minutes, route.trip_number)
        )

    used_by_type: dict[str, set[int]] = defaultdict(set)
    for (vehicle_name, vehicle_number), jobs in jobs_by_vehicle.items():
        used_by_type[vehicle_name].add(vehicle_number)
        ordered = sorted(jobs)
        trip_numbers = sorted(job[2] for job in jobs)
        if trip_numbers != list(range(1, len(jobs) + 1)):
            raise AssertionError(f"车辆 {vehicle_name}-{vehicle_number} 趟次编号不连续")
        for previous, current in zip(ordered, ordered[1:]):
            if previous[1] + turnaround_minutes > current[0] + 1e-7:
                raise AssertionError(f"车辆 {vehicle_name}-{vehicle_number} 的配送趟次时间重叠")

    for vehicle_name, vehicle_numbers in used_by_type.items():
        if len(vehicle_numbers) > vehicle_types[vehicle_name].count:
            raise AssertionError(f"车型 {vehicle_name} 使用物理车辆数超限")
