from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf, isinf

from src.model.domain import ProblemData, Route


SERVICE_MINUTES = 20.0
WAITING_COST_PER_HOUR = 20.0
LATE_COST_PER_HOUR = 50.0
CARBON_COST_PER_KG = 0.65


@dataclass(frozen=True)
class StopResult:
    sequence: int
    customer_id: int
    delivered_weight: float
    delivered_volume: float
    leg_distance: float
    arrival_minutes: float
    service_start_minutes: float
    departure_minutes: float
    waiting_minutes: float
    late_minutes: float


@dataclass(frozen=True)
class RouteEvaluation:
    route_id: str
    start_minutes: float
    finish_minutes: float
    distance_km: float
    energy_amount: float
    emissions_kg: float
    fixed_cost: float
    energy_cost: float
    carbon_cost: float
    waiting_cost: float
    late_cost: float
    total_cost: float
    stops: tuple[StopResult, ...]


@dataclass(frozen=True)
class SolutionEvaluation:
    vehicle_count: int
    trip_count: int
    total_distance_km: float
    fuel_liters: float
    electricity_kwh: float
    emissions_kg: float
    fixed_cost: float
    energy_cost: float
    carbon_cost: float
    waiting_cost: float
    late_cost: float
    total_cost: float
    routes: tuple[RouteEvaluation, ...]


class RouteEvaluator:
    """用期望车速评估路线；软时间窗允许迟到但计罚款。"""

    _BOUNDARIES = (480.0, 540.0, 600.0, 690.0, 780.0, 900.0, 1020.0, 1140.0, 1440.0)

    def __init__(self, problem: ProblemData):
        self.problem = problem

    @staticmethod
    def speed_kmh(clock_minutes: float) -> float:
        minute = clock_minutes % (24.0 * 60.0)
        if 480.0 <= minute < 540.0:
            return 9.8
        if 540.0 <= minute < 600.0:
            return 55.3
        if 600.0 <= minute < 690.0:
            return 35.4
        if 690.0 <= minute < 780.0:
            return 9.8
        if 780.0 <= minute < 900.0:
            return 55.3
        if 900.0 <= minute < 1020.0:
            return 35.4
        if 1020.0 <= minute < 1140.0:
            return 9.8
        return 55.3

    @classmethod
    def _next_boundary(cls, clock_minutes: float) -> float:
        minute = clock_minutes % (24.0 * 60.0)
        day_base = clock_minutes - minute
        for boundary in cls._BOUNDARIES:
            if boundary > minute + 1e-9:
                return day_base + boundary
        return day_base + 24.0 * 60.0

    @staticmethod
    def _energy_per_100km(propulsion: str, speed: float) -> float:
        if propulsion == "fuel":
            return 0.0025 * speed**2 - 0.2554 * speed + 31.75
        return 0.0014 * speed**2 - 0.12 * speed + 36.19

    def travel_leg(
        self,
        distance_km: float,
        depart_minutes: float,
        propulsion: str,
        load_ratio: float,
    ) -> tuple[float, float]:
        remaining = max(0.0, distance_km)
        clock = depart_minutes
        energy = 0.0
        full_load_increase = 0.40 if propulsion == "fuel" else 0.35
        load_factor = 1.0 + full_load_increase * min(1.0, max(0.0, load_ratio))

        while remaining > 1e-9:
            speed = self.speed_kmh(clock)
            boundary = self._next_boundary(clock)
            possible_distance = inf if isinf(boundary) else speed * (boundary - clock) / 60.0
            segment_distance = min(remaining, possible_distance)
            segment_minutes = segment_distance / speed * 60.0
            energy += (
                segment_distance
                / 100.0
                * self._energy_per_100km(propulsion, speed)
                * load_factor
            )
            clock += segment_minutes
            remaining -= segment_distance
            if segment_distance <= 1e-12:
                clock = boundary
        return clock, energy

    def evaluate(self, route: Route, start_minutes: float | None = None) -> RouteEvaluation:
        vehicle = route.vehicle_type
        if route.total_weight > vehicle.capacity_weight + 1e-6:
            raise ValueError(f"{route.route_id} 超过载重容量")
        if route.total_volume > vehicle.capacity_volume + 1e-6:
            raise ValueError(f"{route.route_id} 超过容积容量")

        clock = route.start_minutes if start_minutes is None else start_minutes
        start = clock
        current_node = 0
        current_weight = route.total_weight
        current_volume = route.total_volume
        distance_total = 0.0
        energy_total = 0.0
        waiting_minutes_total = 0.0
        late_minutes_total = 0.0
        stops: list[StopResult] = []

        for sequence, delivery in enumerate(route.deliveries, start=1):
            leg_distance = float(self.problem.distance[current_node, delivery.customer_id])
            load_ratio = max(
                current_weight / vehicle.capacity_weight,
                current_volume / vehicle.capacity_volume,
            )
            arrival, energy = self.travel_leg(
                leg_distance,
                clock,
                vehicle.propulsion,
                load_ratio,
            )
            window_start, window_end = self.problem.windows[delivery.customer_id]
            waiting = max(0.0, window_start - arrival)
            late = max(0.0, arrival - window_end)
            service_start = max(arrival, window_start)
            departure = service_start + SERVICE_MINUTES
            stops.append(
                StopResult(
                    sequence=sequence,
                    customer_id=delivery.customer_id,
                    delivered_weight=delivery.weight,
                    delivered_volume=delivery.volume,
                    leg_distance=leg_distance,
                    arrival_minutes=arrival,
                    service_start_minutes=service_start,
                    departure_minutes=departure,
                    waiting_minutes=waiting,
                    late_minutes=late,
                )
            )
            distance_total += leg_distance
            energy_total += energy
            waiting_minutes_total += waiting
            late_minutes_total += late
            current_weight -= delivery.weight
            current_volume -= delivery.volume
            current_node = delivery.customer_id
            clock = departure

        if route.deliveries:
            return_distance = float(self.problem.distance[current_node, 0])
            clock, energy = self.travel_leg(
                return_distance,
                clock,
                vehicle.propulsion,
                0.0,
            )
            distance_total += return_distance
            energy_total += energy

        if vehicle.propulsion == "fuel":
            energy_cost = energy_total * 7.61
            emissions = energy_total * 2.547
        else:
            energy_cost = energy_total * 1.64
            emissions = energy_total * 0.501
        carbon_cost = emissions * CARBON_COST_PER_KG
        waiting_cost = waiting_minutes_total / 60.0 * WAITING_COST_PER_HOUR
        late_cost = late_minutes_total / 60.0 * LATE_COST_PER_HOUR
        total = vehicle.fixed_cost + energy_cost + carbon_cost + waiting_cost + late_cost

        return RouteEvaluation(
            route_id=route.route_id,
            start_minutes=start,
            finish_minutes=clock,
            distance_km=distance_total,
            energy_amount=energy_total,
            emissions_kg=emissions,
            fixed_cost=vehicle.fixed_cost,
            energy_cost=energy_cost,
            carbon_cost=carbon_cost,
            waiting_cost=waiting_cost,
            late_cost=late_cost,
            total_cost=total,
            stops=tuple(stops),
        )

    def best_departure(
        self,
        route: Route,
        earliest: float = 480.0,
        step_minutes: float = 10.0,
        latest_finish: float | None = None,
    ) -> RouteEvaluation:
        if not route.deliveries:
            result = self.evaluate(route, earliest)
            if latest_finish is not None and result.finish_minutes > latest_finish + 1e-9:
                raise ValueError("路线无法在当日截止时间前完成")
            return result
        latest_window = max(self.problem.windows[item.customer_id][1] for item in route.deliveries)
        latest_start = max(earliest, min(latest_window, 20.0 * 60.0))
        best: RouteEvaluation | None = None
        candidate = earliest
        while candidate <= latest_start + 1e-9:
            result = self.evaluate(route, candidate)
            if latest_finish is not None and result.finish_minutes > latest_finish + 1e-9:
                candidate += step_minutes
                continue
            if best is None or result.total_cost < best.total_cost - 1e-9:
                best = result
            candidate += step_minutes
        if best is None:
            raise ValueError("路线无法在当日截止时间前完成")
        route.start_minutes = best.start_minutes
        return best


def evaluate_solution(
    routes: list[Route],
    evaluator: RouteEvaluator,
    optimize_departures: bool = True,
) -> SolutionEvaluation:
    """用同一评估器汇总完整成本，保证所有搜索算子采用相同口径。"""

    raw_results = tuple(
        evaluator.best_departure(route)
        if optimize_departures
        else evaluator.evaluate(route, route.start_minutes)
        for route in routes
    )
    all_vehicles: set[tuple[str, int]] = set()
    charged_vehicles: set[tuple[str, int]] = set()
    adjusted_results: list[RouteEvaluation] = []
    for route, result in zip(routes, raw_results, strict=True):
        vehicle_key = (route.vehicle_type.name, route.vehicle_number)
        all_vehicles.add(vehicle_key)
        fixed_cost = 0.0
        if route.trip_number == 1 and vehicle_key not in charged_vehicles:
            fixed_cost = route.vehicle_type.fixed_cost
            charged_vehicles.add(vehicle_key)
        adjusted_results.append(
            replace(
                result,
                fixed_cost=fixed_cost,
                total_cost=result.total_cost - result.fixed_cost + fixed_cost,
            )
        )
    route_results = tuple(adjusted_results)
    fuel_liters = sum(
        result.energy_amount
        for route, result in zip(routes, route_results, strict=True)
        if route.vehicle_type.propulsion == "fuel"
    )
    electricity_kwh = sum(
        result.energy_amount
        for route, result in zip(routes, route_results, strict=True)
        if route.vehicle_type.propulsion == "electric"
    )
    fixed_cost = sum(result.fixed_cost for result in route_results)
    energy_cost = sum(result.energy_cost for result in route_results)
    carbon_cost = sum(result.carbon_cost for result in route_results)
    waiting_cost = sum(result.waiting_cost for result in route_results)
    late_cost = sum(result.late_cost for result in route_results)
    total_cost = sum(result.total_cost for result in route_results)
    component_total = fixed_cost + energy_cost + carbon_cost + waiting_cost + late_cost
    if abs(total_cost - component_total) > 1e-6:
        raise AssertionError("总成本与分项成本之和不一致")

    return SolutionEvaluation(
        vehicle_count=len(all_vehicles),
        trip_count=len(routes),
        total_distance_km=sum(result.distance_km for result in route_results),
        fuel_liters=fuel_liters,
        electricity_kwh=electricity_kwh,
        emissions_kg=sum(result.emissions_kg for result in route_results),
        fixed_cost=fixed_cost,
        energy_cost=energy_cost,
        carbon_cost=carbon_cost,
        waiting_cost=waiting_cost,
        late_cost=late_cost,
        total_cost=total_cost,
        routes=route_results,
    )


def format_clock(minutes: float) -> str:
    whole = int(round(minutes))
    day, minute_of_day = divmod(whole, 24 * 60)
    hours, mins = divmod(minute_of_day, 60)
    prefix = f"D+{day} " if day else ""
    return f"{prefix}{hours:02d}:{mins:02d}"
