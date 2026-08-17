from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import inf

from src.model.domain import DEFAULT_VEHICLE_TYPES, Delivery, Route, VehicleType
from src.model.evaluator import RouteEvaluation, RouteEvaluator


CAPACITY_TOLERANCE = 1e-7


def clone_deliveries(deliveries: list[Delivery]) -> list[Delivery]:
    return [Delivery(item.customer_id, item.weight, item.volume) for item in deliveries]


def route_load_rate(route: Route) -> float:
    return max(
        route.total_weight / route.vehicle_type.capacity_weight,
        route.total_volume / route.vehicle_type.capacity_volume,
    )


def _fits(deliveries: list[Delivery], vehicle: VehicleType) -> bool:
    total_weight = sum(item.weight for item in deliveries)
    total_volume = sum(item.volume for item in deliveries)
    return (
        total_weight <= vehicle.capacity_weight + CAPACITY_TOLERANCE
        and total_volume <= vehicle.capacity_volume + CAPACITY_TOLERANCE
    )


def best_vehicle_for_deliveries(
    deliveries: list[Delivery],
    evaluator: RouteEvaluator,
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES,
) -> tuple[Route, RouteEvaluation]:
    """忽略车队数量，为一条路线选择容量可行且完整成本最低的车型。"""

    best: tuple[Route, RouteEvaluation] | None = None
    for vehicle in vehicle_types:
        if not _fits(deliveries, vehicle):
            continue
        candidate = Route(
            vehicle_type=vehicle,
            vehicle_number=0,
            deliveries=clone_deliveries(deliveries),
        )
        result = evaluator.best_departure(candidate)
        if best is None or result.total_cost < best[1].total_cost - 1e-9:
            best = (candidate, result)
    if best is None:
        raise ValueError("没有车型能够承载该路线")
    return best


@dataclass
class _Edge:
    to: int
    reverse: int
    capacity: int
    cost: float


def _add_edge(graph: list[list[_Edge]], source: int, target: int, capacity: int, cost: float) -> int:
    forward_index = len(graph[source])
    graph[source].append(_Edge(target, len(graph[target]), capacity, cost))
    graph[target].append(_Edge(source, forward_index, 0, -cost))
    return forward_index


def select_vehicles(
    routes: list[Route],
    evaluator: RouteEvaluator,
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES,
) -> list[Route]:
    """以最小费用流全局选择车型，同时严格满足各车型数量限制。"""

    if not routes:
        return []

    route_count = len(routes)
    type_count = len(vehicle_types)
    source = 0
    route_offset = 1
    type_offset = route_offset + route_count
    sink = type_offset + type_count
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]
    candidate_routes: dict[tuple[int, int], Route] = {}
    assignment_edges: dict[tuple[int, int], tuple[int, int]] = {}

    for route_index, route in enumerate(routes):
        route_node = route_offset + route_index
        _add_edge(graph, source, route_node, 1, 0.0)
        for type_index, vehicle in enumerate(vehicle_types):
            if not _fits(route.deliveries, vehicle):
                continue
            candidate = Route(
                vehicle_type=vehicle,
                vehicle_number=0,
                deliveries=clone_deliveries(route.deliveries),
                start_minutes=route.start_minutes,
            )
            result = evaluator.best_departure(candidate)
            edge_index = _add_edge(
                graph,
                route_node,
                type_offset + type_index,
                1,
                result.total_cost,
            )
            candidate_routes[(route_index, type_index)] = candidate
            assignment_edges[(route_index, type_index)] = (route_node, edge_index)

    for type_index, vehicle in enumerate(vehicle_types):
        _add_edge(graph, type_offset + type_index, sink, vehicle.count, 0.0)

    flow = 0
    while flow < route_count:
        distance = [inf] * len(graph)
        previous_node = [-1] * len(graph)
        previous_edge = [-1] * len(graph)
        in_queue = [False] * len(graph)
        distance[source] = 0.0
        queue: deque[int] = deque([source])
        in_queue[source] = True

        while queue:
            node = queue.popleft()
            in_queue[node] = False
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                candidate_distance = distance[node] + edge.cost
                if candidate_distance >= distance[edge.to] - 1e-12:
                    continue
                distance[edge.to] = candidate_distance
                previous_node[edge.to] = node
                previous_edge[edge.to] = edge_index
                if not in_queue[edge.to]:
                    queue.append(edge.to)
                    in_queue[edge.to] = True

        if previous_node[sink] < 0:
            raise RuntimeError("现有车型数量无法覆盖所有路线，请先调整路线划分")

        node = sink
        while node != source:
            parent = previous_node[node]
            edge_index = previous_edge[node]
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent
        flow += 1

    selected: list[Route] = []
    for route_index in range(route_count):
        assigned: Route | None = None
        for type_index in range(type_count):
            location = assignment_edges.get((route_index, type_index))
            if location is None:
                continue
            node, edge_index = location
            if graph[node][edge_index].capacity == 0:
                assigned = candidate_routes[(route_index, type_index)]
                break
        if assigned is None:
            raise AssertionError(f"路线 {route_index} 未获得车型")
        selected.append(assigned)

    counters: dict[str, int] = {}
    for route in selected:
        counters[route.vehicle_type.name] = counters.get(route.vehicle_type.name, 0) + 1
        route.vehicle_number = counters[route.vehicle_type.name]
    return selected
