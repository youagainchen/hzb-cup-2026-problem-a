from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from src.model.domain import DEFAULT_VEHICLE_TYPES, Delivery, ProblemData, Route, VehicleType
from src.model.evaluator import RouteEvaluator
from src.solver.fleet import best_vehicle_for_deliveries


EPSILON = 1e-9


@dataclass(frozen=True)
class _Atom:
    atom_id: int
    delivery: Delivery


def _split_demands(
    problem: ProblemData,
    vehicle_types: tuple[VehicleType, ...],
) -> list[_Atom]:
    max_weight = max(vehicle.capacity_weight for vehicle in vehicle_types)
    max_volume = max(vehicle.capacity_volume for vehicle in vehicle_types)
    atoms: list[_Atom] = []
    atom_id = 0
    for customer_id, (demand_weight, demand_volume) in sorted(problem.demands.items()):
        remaining_weight = float(demand_weight)
        remaining_volume = float(demand_volume)
        while remaining_weight > EPSILON or remaining_volume > EPSILON:
            fractions = [1.0]
            if remaining_weight > EPSILON:
                fractions.append(max_weight / remaining_weight)
            if remaining_volume > EPSILON:
                fractions.append(max_volume / remaining_volume)
            fraction = min(fractions)
            delivered_weight = remaining_weight * fraction
            delivered_volume = remaining_volume * fraction
            atoms.append(
                _Atom(
                    atom_id,
                    Delivery(customer_id, delivered_weight, delivered_volume),
                )
            )
            atom_id += 1
            remaining_weight = max(0.0, remaining_weight - delivered_weight)
            remaining_volume = max(0.0, remaining_volume - delivered_volume)
    return atoms


def _coalesce(atoms: list[_Atom]) -> list[Delivery]:
    deliveries: list[Delivery] = []
    position_by_customer: dict[int, int] = {}
    for atom in atoms:
        item = atom.delivery
        position = position_by_customer.get(item.customer_id)
        if position is None:
            position_by_customer[item.customer_id] = len(deliveries)
            deliveries.append(Delivery(item.customer_id, item.weight, item.volume))
        else:
            current = deliveries[position]
            deliveries[position] = Delivery(
                current.customer_id,
                current.weight + item.weight,
                current.volume + item.volume,
            )
    return deliveries


def _build_parallel_savings_routes(
    problem: ProblemData,
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES,
    minimum_gain: float = 0.01,
    time_window_penalty_km_per_hour: float = 0.0,
) -> list[Route]:
    """Clarke-Wright 初始解；合并时以完整成本而非仅距离决定是否接受。"""

    evaluator = RouteEvaluator(problem)
    atoms = _split_demands(problem, vehicle_types)
    states: dict[int, list[_Atom]] = {atom.atom_id: [atom] for atom in atoms}
    atom_to_route = {atom.atom_id: atom.atom_id for atom in atoms}
    route_costs: dict[int, float] = {}
    route_models: dict[int, Route] = {}
    planning_vehicle = max(
        vehicle_types,
        key=lambda vehicle: (vehicle.capacity_weight, vehicle.capacity_volume),
    )
    # 趟次构造不能受“拥有多少辆车”限制。车辆数量只约束同一时刻
    # 的物理车辆数；同一辆车返回配送中心后可以继续执行下一趟。
    singleton_routes = [Route(planning_vehicle, 0, [atom.delivery]) for atom in atoms]
    for atom, route in zip(atoms, singleton_routes, strict=True):
        result = evaluator.evaluate(route, route.start_minutes)
        route_costs[atom.atom_id] = result.total_cost
        route_models[atom.atom_id] = route

    savings = []
    for left, right in combinations(atoms, 2):
        left_customer = left.delivery.customer_id
        right_customer = right.delivery.customer_id
        distance_saving = (
            problem.distance[0, left_customer]
            + problem.distance[0, right_customer]
            - problem.distance[left_customer, right_customer]
        )
        left_window = problem.windows[left_customer]
        right_window = problem.windows[right_customer]
        left_midpoint = (left_window[0] + left_window[1]) / 2.0
        right_midpoint = (right_window[0] + right_window[1]) / 2.0
        midpoint_gap_hours = abs(left_midpoint - right_midpoint) / 60.0
        combined_score = (
            float(distance_saving)
            - time_window_penalty_km_per_hour * midpoint_gap_hours
        )
        savings.append((combined_score, left.atom_id, right.atom_id))
    savings.sort(reverse=True)

    for _, left_atom_id, right_atom_id in savings:
        left_route_id = atom_to_route[left_atom_id]
        right_route_id = atom_to_route[right_atom_id]
        if left_route_id == right_route_id:
            continue
        left_atoms = states[left_route_id]
        right_atoms = states[right_route_id]
        left_ids = (left_atoms[0].atom_id, left_atoms[-1].atom_id)
        right_ids = (right_atoms[0].atom_id, right_atoms[-1].atom_id)
        if left_atom_id not in left_ids or right_atom_id not in right_ids:
            continue

        oriented_left = (
            left_atoms if left_atoms[-1].atom_id == left_atom_id else list(reversed(left_atoms))
        )
        oriented_right = (
            right_atoms if right_atoms[0].atom_id == right_atom_id else list(reversed(right_atoms))
        )
        merged_atoms = list(oriented_left) + list(oriented_right)
        deliveries = _coalesce(merged_atoms)
        retained_types = tuple(
            {
                route_models[left_route_id].vehicle_type.name: route_models[left_route_id].vehicle_type,
                route_models[right_route_id].vehicle_type.name: route_models[right_route_id].vehicle_type,
            }.values()
        )
        try:
            merged_route, merged_result = best_vehicle_for_deliveries(
                deliveries, evaluator, retained_types
            )
        except ValueError:
            continue
        old_cost = route_costs[left_route_id] + route_costs[right_route_id]
        if merged_result.total_cost >= old_cost - minimum_gain:
            continue

        states[left_route_id] = merged_atoms
        route_costs[left_route_id] = merged_result.total_cost
        route_models[left_route_id] = merged_route
        del states[right_route_id]
        del route_costs[right_route_id]
        del route_models[right_route_id]
        for atom in merged_atoms:
            atom_to_route[atom.atom_id] = left_route_id

    return [route_models[route_id] for route_id in states]


def _take_delivery(
    demand_weight: float,
    demand_volume: float,
    free_weight: float,
    free_volume: float,
) -> tuple[float, float]:
    fractions = [1.0]
    if demand_weight > EPSILON:
        fractions.append(free_weight / demand_weight)
    if demand_volume > EPSILON:
        fractions.append(free_volume / demand_volume)
    fraction = max(0.0, min(fractions))
    return demand_weight * fraction, demand_volume * fraction


def _build_fleet_aware_savings_routes(
    problem: ProblemData,
    vehicle_types: tuple[VehicleType, ...],
) -> list[Route]:
    """车队受限时的成本节约插入法，始终保持车型数量和容量可行。"""

    evaluator = RouteEvaluator(problem)
    remaining = {
        customer_id: [float(weight), float(volume)]
        for customer_id, (weight, volume) in problem.demands.items()
    }
    routes: list[Route] = []

    for vehicle in vehicle_types:
        for vehicle_number in range(1, vehicle.count + 1):
            if not any(weight > EPSILON or volume > EPSILON for weight, volume in remaining.values()):
                return routes
            route = Route(vehicle, vehicle_number)
            free_weight = vehicle.capacity_weight
            free_volume = vehicle.capacity_volume
            visited: set[int] = set()

            while free_weight > EPSILON or free_volume > EPSILON:
                current_cost = (
                    evaluator.evaluate(route, route.start_minutes).total_cost
                    if route.deliveries
                    else 0.0
                )
                best: tuple[float, float, int, float, float, Route] | None = None
                for customer_id, (demand_weight, demand_volume) in remaining.items():
                    if customer_id in visited:
                        continue
                    if demand_weight <= EPSILON and demand_volume <= EPSILON:
                        continue
                    delivered_weight, delivered_volume = _take_delivery(
                        demand_weight,
                        demand_volume,
                        free_weight,
                        free_volume,
                    )
                    if delivered_weight <= EPSILON and delivered_volume <= EPSILON:
                        continue
                    item = Delivery(customer_id, delivered_weight, delivered_volume)
                    singleton = Route(vehicle, vehicle_number, [item])
                    singleton_cost = evaluator.best_departure(singleton).total_cost
                    positions = range(len(route.deliveries) + 1)
                    for position in positions:
                        deliveries = list(route.deliveries)
                        deliveries.insert(position, item)
                        candidate = Route(
                            vehicle,
                            vehicle_number,
                            deliveries,
                            route.start_minutes,
                        )
                        result = evaluator.best_departure(candidate)
                        cost_saving = current_cost + singleton_cost - result.total_cost
                        score = (cost_saving, singleton_cost)
                        if best is None or score > (best[0], best[1]):
                            best = (
                                cost_saving,
                                singleton_cost,
                                customer_id,
                                delivered_weight,
                                delivered_volume,
                                candidate,
                            )
                if best is None:
                    break
                _, _, customer_id, delivered_weight, delivered_volume, candidate = best
                route = candidate
                remaining[customer_id][0] = max(
                    0.0, remaining[customer_id][0] - delivered_weight
                )
                remaining[customer_id][1] = max(
                    0.0, remaining[customer_id][1] - delivered_volume
                )
                free_weight = max(0.0, free_weight - delivered_weight)
                free_volume = max(0.0, free_volume - delivered_volume)
                visited.add(customer_id)
            if route.deliveries:
                routes.append(route)

    unfinished = {
        customer_id: values
        for customer_id, values in remaining.items()
        if values[0] > EPSILON or values[1] > EPSILON
    }
    if unfinished:
        raise RuntimeError(f"现有车队无法装完全部需求，剩余客户数: {len(unfinished)}")
    return routes


def build_savings_routes(
    problem: ProblemData,
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES,
    minimum_gain: float = 0.01,
    time_window_penalty_km_per_hour: float = 0.0,
) -> list[Route]:
    """构造不受全天车辆趟次数限制的并行 Clarke-Wright 初始解。"""

    return _build_parallel_savings_routes(
        problem,
        vehicle_types,
        minimum_gain,
        time_window_penalty_km_per_hour,
    )
