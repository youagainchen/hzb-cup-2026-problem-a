from __future__ import annotations

from collections import Counter

from src.model.domain import DEFAULT_VEHICLE_TYPES, Delivery, ProblemData, Route, VehicleType
from src.model.evaluator import RouteEvaluator, SERVICE_MINUTES


EPSILON = 1e-7


def _take_proportional_delivery(
    demand_weight: float,
    demand_volume: float,
    capacity_weight: float,
    capacity_volume: float,
) -> tuple[float, float]:
    fractions = [1.0]
    if demand_weight > EPSILON:
        fractions.append(capacity_weight / demand_weight)
    if demand_volume > EPSILON:
        fractions.append(capacity_volume / demand_volume)
    fraction = max(0.0, min(fractions))
    return demand_weight * fraction, demand_volume * fraction


def build_greedy_routes(
    problem: ProblemData,
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES,
) -> list[Route]:
    """距离+时间窗贪心，允许把一个客户的需求拆到多辆车。"""

    remaining = {
        customer_id: [float(weight), float(volume)]
        for customer_id, (weight, volume) in problem.demands.items()
    }
    evaluator = RouteEvaluator(problem)
    routes: list[Route] = []

    for vehicle_type in vehicle_types:
        for vehicle_number in range(1, vehicle_type.count + 1):
            if not any(weight > EPSILON or volume > EPSILON for weight, volume in remaining.values()):
                return routes

            route = Route(vehicle_type=vehicle_type, vehicle_number=vehicle_number)
            free_weight = vehicle_type.capacity_weight
            free_volume = vehicle_type.capacity_volume
            current_node = 0
            clock = 480.0
            visited: set[int] = set()

            while free_weight > EPSILON or free_volume > EPSILON:
                candidates: list[tuple[float, int, float, float, float]] = []
                for customer_id, (demand_weight, demand_volume) in remaining.items():
                    if customer_id in visited:
                        continue
                    if demand_weight <= EPSILON and demand_volume <= EPSILON:
                        continue
                    delivered_weight, delivered_volume = _take_proportional_delivery(
                        demand_weight,
                        demand_volume,
                        free_weight,
                        free_volume,
                    )
                    if delivered_weight <= EPSILON and delivered_volume <= EPSILON:
                        continue

                    distance = float(problem.distance[current_node, customer_id])
                    arrival, _ = evaluator.travel_leg(
                        distance,
                        clock,
                        vehicle_type.propulsion,
                        1.0,
                    )
                    window_start, window_end = problem.windows[customer_id]
                    early = max(0.0, window_start - arrival)
                    late = max(0.0, arrival - window_end)
                    score = distance + (20.0 / 60.0) * early + (50.0 / 60.0) * late
                    candidates.append(
                        (score, customer_id, delivered_weight, delivered_volume, arrival)
                    )

                if not candidates:
                    break

                _, customer_id, delivered_weight, delivered_volume, arrival = min(candidates)
                route.deliveries.append(
                    Delivery(
                        customer_id=customer_id,
                        weight=delivered_weight,
                        volume=delivered_volume,
                    )
                )
                remaining[customer_id][0] = max(0.0, remaining[customer_id][0] - delivered_weight)
                remaining[customer_id][1] = max(0.0, remaining[customer_id][1] - delivered_volume)
                free_weight = max(0.0, free_weight - delivered_weight)
                free_volume = max(0.0, free_volume - delivered_volume)
                window_start, _ = problem.windows[customer_id]
                clock = max(arrival, window_start) + SERVICE_MINUTES
                current_node = customer_id
                visited.add(customer_id)

            if route.deliveries:
                routes.append(route)

    unfinished = {
        customer_id: (weight, volume)
        for customer_id, (weight, volume) in remaining.items()
        if weight > EPSILON or volume > EPSILON
    }
    if not unfinished:
        return routes
    raise RuntimeError(f"现有车队无法装完全部需求，剩余客户数: {len(unfinished)}")


def validate_solution(problem: ProblemData, routes: list[Route]) -> None:
    delivered: dict[int, list[float]] = {
        customer_id: [0.0, 0.0] for customer_id in problem.demands
    }
    used_vehicle_keys = {
        (route.vehicle_type.name, route.vehicle_number) for route in routes
    }
    used = Counter(vehicle_name for vehicle_name, _ in used_vehicle_keys)
    fleet = {vehicle.name: vehicle.count for vehicle in DEFAULT_VEHICLE_TYPES}

    for route in routes:
        if route.total_weight > route.vehicle_type.capacity_weight + 1e-5:
            raise AssertionError(f"{route.route_id} 载重超限")
        if route.total_volume > route.vehicle_type.capacity_volume + 1e-5:
            raise AssertionError(f"{route.route_id} 容积超限")
        for item in route.deliveries:
            if item.customer_id not in delivered:
                raise AssertionError(f"出现无需求客户 {item.customer_id}")
            delivered[item.customer_id][0] += item.weight
            delivered[item.customer_id][1] += item.volume

    for vehicle_name, count in used.items():
        if count > fleet.get(vehicle_name, 0):
            raise AssertionError(f"车型 {vehicle_name} 使用数量超限")

    for customer_id, (expected_weight, expected_volume) in problem.demands.items():
        actual_weight, actual_volume = delivered[customer_id]
        if abs(actual_weight - expected_weight) > 1e-4:
            raise AssertionError(f"客户 {customer_id} 重量未配送完整")
        if abs(actual_volume - expected_volume) > 1e-5:
            raise AssertionError(f"客户 {customer_id} 体积未配送完整")
