# -*- coding: utf-8 -*-
"""问题三动态评估器：发车级冻结点口径下的冻结态提取与动态重计费。

由 2 号维护。1 号动态调度器把候选未来计划（list[Route]，start_minutes 已定）
交给 evaluate_dynamic 统一评分；本模块不包含任何重优化算子。

成本口径（沉没 + 增量）：
- 事件触发时刻 T 前已发车的趟次整趟冻结，其全部运行成本与 400 元固定成本沉没；
- 未来计划只对「不在已发车集合中的物理车辆」计 400 元固定成本；
- 当日总成本 = 已执行成本(沉没) + 未来计划成本；ΔC = 总成本 − Q2 静态计划总成本。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.model.domain import ProblemData, Route
from src.model.evaluator import RouteEvaluator, evaluate_solution


FIXED_STARTUP_COST = 400.0


@dataclass(frozen=True)
class FreezeState:
    """T 时刻发车级冻结后的静态计划切分。"""

    trigger_time_minutes: float
    frozen_trip_ids: tuple[str, ...]
    replannable_trip_ids: tuple[str, ...]
    sunk_vehicle_keys: frozenset[tuple[str, int]]
    sunk_fixed_cost: float
    sunk_operating_cost: float
    executed_cost: float
    remaining_demand: dict[int, tuple[float, float]]
    vehicle_ready_minutes: dict[tuple[str, int], float]


def _route_signature(route: Route) -> tuple[tuple[int, float, float], ...]:
    return tuple(
        (item.customer_id, round(item.weight, 6), round(item.volume, 6))
        for item in route.deliveries
    )


def extract_freeze_state(
    static_routes: list[Route],
    evaluator: RouteEvaluator,
    trigger_time_minutes: float,
    problem_after: ProblemData | None = None,
) -> FreezeState:
    """按发车级冻结把静态计划切分为已执行与可重排两部分。

    - 已执行：start_minutes < T 的趟次整趟冻结，全部成本沉没；
    - 可重排：start_minutes >= T 的趟次，其客户进入剩余需求；
    - remaining_demand = 事件后需求 − 冻结趟次已承诺配送量（可因已发车而部分浪费）。
    """
    problem = problem_after if problem_after is not None else evaluator.problem
    evaluations = {
        route.route_id: evaluator.evaluate(route, route.start_minutes)
        for route in static_routes
    }
    frozen = [route for route in static_routes if route.start_minutes < trigger_time_minutes - 1e-9]
    replannable = [route for route in static_routes if route.start_minutes >= trigger_time_minutes - 1e-9]

    sunk_vehicles = {(route.vehicle_type.name, route.vehicle_number) for route in frozen}
    sunk_operating = sum(
        evaluations[route.route_id].total_cost - evaluations[route.route_id].fixed_cost
        for route in frozen
    )
    sunk_fixed = FIXED_STARTUP_COST * len(sunk_vehicles)

    remaining: dict[int, list[float]] = {
        customer_id: [float(weight), float(volume)]
        for customer_id, (weight, volume) in problem.demands.items()
    }
    for route in frozen:
        for item in route.deliveries:
            if item.customer_id in remaining:
                remaining[item.customer_id][0] -= item.weight
                remaining[item.customer_id][1] -= item.volume
    remaining_demand = {
        customer_id: (max(0.0, weight), max(0.0, volume))
        for customer_id, (weight, volume) in remaining.items()
        if weight > 1e-6 or volume > 1e-6
    }

    finish_by_vehicle: dict[tuple[str, int], float] = {}
    for route in frozen:
        key = (route.vehicle_type.name, route.vehicle_number)
        finish = evaluations[route.route_id].finish_minutes
        finish_by_vehicle[key] = max(finish_by_vehicle.get(key, 0.0), finish)
    all_vehicle_keys = {(route.vehicle_type.name, route.vehicle_number) for route in static_routes}
    vehicle_ready = {
        key: finish_by_vehicle.get(key, trigger_time_minutes) for key in all_vehicle_keys
    }

    return FreezeState(
        trigger_time_minutes=trigger_time_minutes,
        frozen_trip_ids=tuple(route.route_id for route in frozen),
        replannable_trip_ids=tuple(route.route_id for route in replannable),
        sunk_vehicle_keys=frozenset(sunk_vehicles),
        sunk_fixed_cost=sunk_fixed,
        sunk_operating_cost=sunk_operating,
        executed_cost=sunk_fixed + sunk_operating,
        remaining_demand=remaining_demand,
        vehicle_ready_minutes=vehicle_ready,
    )


@dataclass(frozen=True)
class FutureFeasibility:
    demand_unfinished_customers: int
    demand_unfinished_weight_kg: float
    demand_unfinished_volume_m3: float
    capacity_violation_count: int
    policy_violation_count: int
    schedule_violation_count: int
    all_routes_return_before_24h: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return (
            self.demand_unfinished_customers == 0
            and self.capacity_violation_count == 0
            and self.policy_violation_count == 0
            and self.schedule_violation_count == 0
            and self.all_routes_return_before_24h
        )


@dataclass(frozen=True)
class DynamicEvaluation:
    trigger_time_minutes: float
    executed_cost: float
    future_fixed_cost: float
    future_operating_cost: float
    future_cost: float
    total_cost: float
    static_total_cost: float
    delta_cost: float
    delta_cost_base: float
    cost_change_ratio: float
    cost_change_ratio_base: float
    lead_time_minutes: float | None
    sunk_vehicle_count: int
    future_vehicle_count: int
    frozen_trip_count: int
    replannable_trip_count: int
    future_trip_count: int
    dropped_trip_count: int
    new_trip_count: int
    kept_trip_count: int
    changed_customer_count: int
    assignment_change_ratio: float
    arc_change_ratio: float
    response_time_s: float | None
    feasibility: FutureFeasibility


def _check_remaining_demand(future_routes, remaining_demand: dict[int, tuple[float, float]]):
    delivered: dict[int, list[float]] = {}
    for route in future_routes:
        for item in route.deliveries:
            entry = delivered.setdefault(item.customer_id, [0.0, 0.0])
            entry[0] += item.weight
            entry[1] += item.volume
    unfinished_customers = 0
    unfinished_weight = 0.0
    unfinished_volume = 0.0
    notes = []
    for customer_id, (expected_w, expected_v) in remaining_demand.items():
        actual_w, actual_v = delivered.get(customer_id, (0.0, 0.0))
        if abs(actual_w - expected_w) > 1e-4 or abs(actual_v - expected_v) > 1e-5:
            unfinished_customers += 1
            unfinished_weight += max(0.0, expected_w - actual_w)
            unfinished_volume += max(0.0, expected_v - actual_v)
            notes.append(f"客户 {customer_id} 剩余需求未完整配送")
    extra = set(delivered) - set(remaining_demand)
    for customer_id in sorted(extra):
        notes.append(f"客户 {customer_id} 未来计划配送了已取消/已冻结覆盖的需求")
    return unfinished_customers, unfinished_weight, unfinished_volume, tuple(notes)


def _check_vehicle_schedule(future_routes, route_results, freeze_state: FreezeState):
    jobs_by_vehicle: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for route, result in zip(future_routes, route_results, strict=True):
        key = (route.vehicle_type.name, route.vehicle_number)
        jobs_by_vehicle.setdefault(key, []).append(
            (result.start_minutes, result.finish_minutes)
        )
    violations = 0
    notes = []
    # 容差 1e-6 分钟：吸收 CSV 往返取整的 ~1e-9 级噪声，远低于分钟级真实重叠
    schedule_tolerance = 1e-6
    for key, jobs in jobs_by_vehicle.items():
        ready = freeze_state.vehicle_ready_minutes.get(key, freeze_state.trigger_time_minutes)
        ordered = sorted(jobs)
        if ordered[0][0] + schedule_tolerance < ready:
            violations += 1
            notes.append(
                f"车辆 {key[0]}-{key[1]:03d} 首趟发车早于就绪时刻 {ready:.1f}"
            )
        for previous, current in zip(ordered, ordered[1:]):
            if previous[1] > current[0] + schedule_tolerance:
                violations += 1
                notes.append(f"车辆 {key[0]}-{key[1]:03d} 趟次时间重叠")
    return violations, tuple(notes)


def _serving_signature_map(routes, replannable_ids, customer_ids):
    """客户 → 其所在趟次的停靠结构签名集合（静态可重排计划与未来计划比较扰动）。"""
    result: dict[int, set] = {customer_id: set() for customer_id in customer_ids}
    for route in routes:
        if route.route_id not in replannable_ids:
            continue
        signature = _route_signature(route)
        for item in route.deliveries:
            if item.customer_id in result:
                result[item.customer_id].add(signature)
    return result


def _edge_set(routes, replannable_ids) -> set[tuple[int, int]]:
    """方案的有向边集合（0 为配送中心），含返程边。"""
    edges: set[tuple[int, int]] = set()
    for route in routes:
        if route.route_id not in replannable_ids:
            continue
        nodes = [0, *(item.customer_id for item in route.deliveries), 0]
        for left, right in zip(nodes, nodes[1:]):
            edges.add((route.route_id, left, right))
    return edges


def count_disturbance(
    static_routes: list[Route],
    future_routes: list[Route],
    freeze_state: FreezeState,
) -> dict[str, float | int]:
    """按停靠结构比较可重排趟次与客户，返回扰动指标。

    - kept/dropped/new：趟次级别，按停靠签名匹配；被重构的趟次计 1 删 1 增。
    - changed_customer_count / assignment_change_ratio：只统计事件前后共同存在
      的原有客户 U_common = U_before ∩ U_after；取消/新增订单本身不视为扰动。
    - arc_change_ratio：带趟次标签的有向边 (route_id, i, j) 集合对称差度量。
    """
    replannable_ids = set(freeze_state.replannable_trip_ids)
    static_sigs = Counter(
        _route_signature(route)
        for route in static_routes
        if route.route_id in replannable_ids
    )
    future_sigs = Counter(_route_signature(route) for route in future_routes)
    kept = sum((static_sigs & future_sigs).values())
    dropped = sum((static_sigs - future_sigs).values())
    new_trips = sum((future_sigs - static_sigs).values())

    u_before = {
        item.customer_id
        for route in static_routes
        if route.route_id in replannable_ids
        for item in route.deliveries
    }
    u_after = set(freeze_state.remaining_demand)
    common_customers = u_before & u_after
    future_ids = {route.route_id for route in future_routes}
    static_serving = _serving_signature_map(
        static_routes, replannable_ids, common_customers
    )
    future_serving = _serving_signature_map(future_routes, future_ids, common_customers)
    changed_customers = sum(
        1 for customer_id in common_customers
        if static_serving[customer_id] != future_serving[customer_id]
    )
    assignment_change_ratio = (
        changed_customers / len(common_customers) if common_customers else 0.0
    )

    static_edges = _edge_set(static_routes, replannable_ids)
    future_edges = _edge_set(future_routes, future_ids)
    union_size = len(static_edges) + len(future_edges)
    arc_change_ratio = (
        1.0 - 2.0 * len(static_edges & future_edges) / union_size if union_size else 0.0
    )

    return {
        "kept": kept,
        "dropped": dropped,
        "new": new_trips,
        "changed_customers": changed_customers,
        "assignment_change_ratio": assignment_change_ratio,
        "arc_change_ratio": arc_change_ratio,
    }


def evaluate_dynamic(
    future_routes: list[Route],
    freeze_state: FreezeState,
    evaluator_after: RouteEvaluator,
    static_total_cost: float,
    static_routes: list[Route],
    optimize_departures: bool = False,
    response_time_s: float | None = None,
    base_total_cost: float | None = None,
) -> DynamicEvaluation:
    """对候选未来计划统一评分并返回动态口径的完整账目与可行性。

    - `static_total_cost` 为上一批次总成本 C_{k-1}，`delta_cost` 即事件边际增量 ΔC_step；
    - `base_total_cost` 为问题二静态基准 C₀（未传则与 static_total_cost 相同），
      `delta_cost_base` 为相对问题二基准的累计增量 ΔC_base。
    未来计划必须覆盖 freeze_state.remaining_demand，且每辆车的趟次不得早于
    其就绪时刻、不得重叠；这些约束由本评估器校验并返回可行性标志。
    """
    if optimize_departures:
        solution = evaluate_solution(future_routes, evaluator_after, optimize_departures=True)
    else:
        solution = evaluate_solution(future_routes, evaluator_after, optimize_departures=False)

    future_vehicle_keys = {
        (route.vehicle_type.name, route.vehicle_number) for route in future_routes
    }
    sunk_overlap = future_vehicle_keys & set(freeze_state.sunk_vehicle_keys)
    future_fixed = FIXED_STARTUP_COST * (len(future_vehicle_keys) - len(sunk_overlap))
    # 运行成本 = 总分 − 评估器计得的固定成本（后者与 vehicle_number 无关，运行成本不受沉没影响）
    future_operating = solution.total_cost - solution.fixed_cost
    future_cost = future_fixed + future_operating
    total_cost = freeze_state.executed_cost + future_cost
    delta_cost = total_cost - static_total_cost

    unfinished_customers, unfinished_weight, unfinished_volume, demand_notes = (
        _check_remaining_demand(future_routes, freeze_state.remaining_demand)
    )
    schedule_violations, schedule_notes = _check_vehicle_schedule(
        future_routes, solution.routes, freeze_state
    )
    capacity_violations = solution.capacity_violation_count
    policy_violations = solution.policy_violation_count
    before_24h = solution.all_routes_return_before_24h
    notes = (*demand_notes, *schedule_notes)

    feasibility = FutureFeasibility(
        demand_unfinished_customers=unfinished_customers,
        demand_unfinished_weight_kg=unfinished_weight,
        demand_unfinished_volume_m3=unfinished_volume,
        capacity_violation_count=capacity_violations,
        policy_violation_count=policy_violations,
        schedule_violation_count=schedule_violations,
        all_routes_return_before_24h=before_24h,
        notes=notes,
    )

    disturbance = count_disturbance(static_routes, future_routes, freeze_state)
    delta_cost = total_cost - static_total_cost
    if base_total_cost is None:
        base_total_cost = static_total_cost
    delta_cost_base = total_cost - base_total_cost
    lead_time_minutes = (
        min(route.start_minutes for route in future_routes) - freeze_state.trigger_time_minutes
        if future_routes
        else None
    )

    return DynamicEvaluation(
        trigger_time_minutes=freeze_state.trigger_time_minutes,
        executed_cost=freeze_state.executed_cost,
        future_fixed_cost=future_fixed,
        future_operating_cost=future_operating,
        future_cost=future_cost,
        total_cost=total_cost,
        static_total_cost=static_total_cost,
        delta_cost=delta_cost,
        delta_cost_base=delta_cost_base,
        cost_change_ratio=delta_cost / static_total_cost if static_total_cost else 0.0,
        cost_change_ratio_base=delta_cost_base / base_total_cost if base_total_cost else 0.0,
        lead_time_minutes=lead_time_minutes,
        sunk_vehicle_count=len(freeze_state.sunk_vehicle_keys),
        future_vehicle_count=len(future_vehicle_keys),
        frozen_trip_count=len(freeze_state.frozen_trip_ids),
        replannable_trip_count=len(freeze_state.replannable_trip_ids),
        future_trip_count=len(future_routes),
        dropped_trip_count=int(disturbance["dropped"]),
        new_trip_count=int(disturbance["new"]),
        kept_trip_count=int(disturbance["kept"]),
        changed_customer_count=int(disturbance["changed_customers"]),
        assignment_change_ratio=float(disturbance["assignment_change_ratio"]),
        arc_change_ratio=float(disturbance["arc_change_ratio"]),
        response_time_s=response_time_s,
        feasibility=feasibility,
    )
