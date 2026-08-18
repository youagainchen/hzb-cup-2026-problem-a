from __future__ import annotations

from dataclasses import replace

from src.model.domain import DEFAULT_VEHICLE_TYPES, ProblemData, Route, VehicleType
from src.model.evaluator import RouteEvaluator, SolutionEvaluation, evaluate_solution
from src.model.policy_q2 import Q2Policy
from src.solver.greedy import build_greedy_routes, validate_solution
from src.solver.scheduling import select_and_schedule_multitrip, validate_vehicle_schedule


def _subset_problem(problem: ProblemData, customer_ids: set[int]) -> ProblemData:
    return replace(
        problem,
        demands={
            customer_id: problem.demands[customer_id]
            for customer_id in sorted(customer_ids)
            if customer_id in problem.demands
        },
        all_customer_ids=tuple(sorted(customer_ids)),
    )


def _vehicle_types(propulsion: str) -> tuple[VehicleType, ...]:
    return tuple(
        vehicle
        for vehicle in DEFAULT_VEHICLE_TYPES
        if vehicle.propulsion == propulsion
    )


def build_q2_baseline(
    problem: ProblemData,
    policy: Q2Policy,
) -> list[Route]:
    """生成确定性 Q2 合规基线。

    绿色客户全部由新能源车服务，非绿色客户先由燃油车服务；两组路线
    分别排班但共享车型数量口径，从而把政策修复步骤与后续降本搜索解耦。
    """

    green_ids = set(problem.green_customer_ids) & set(problem.demands)
    non_green_ids = set(problem.demands) - green_ids
    green_problem = _subset_problem(problem, green_ids)
    non_green_problem = _subset_problem(problem, non_green_ids)

    green_routes = build_greedy_routes(
        green_problem,
        _vehicle_types("electric"),
    )
    non_green_routes = build_greedy_routes(
        non_green_problem,
        _vehicle_types("fuel"),
    )

    evaluator = RouteEvaluator(problem, policy=policy)
    scheduled_green = select_and_schedule_multitrip(
        green_routes,
        evaluator,
        vehicle_types=_vehicle_types("electric"),
        startup_cost_weight=0.0,
        order_rule="long_first",
    )
    used_by_type = {
        vehicle.name: len(
            {
                route.vehicle_number
                for route in scheduled_green
                if route.vehicle_type.name == vehicle.name
            }
        )
        for vehicle in _vehicle_types("electric")
    }
    remaining_vehicle_types = tuple(
        replace(
            vehicle,
            count=vehicle.count - used_by_type.get(vehicle.name, 0),
        )
        for vehicle in DEFAULT_VEHICLE_TYPES
        if vehicle.count - used_by_type.get(vehicle.name, 0) > 0
    )
    scheduled_non_green = select_and_schedule_multitrip(
        non_green_routes,
        evaluator,
        vehicle_types=remaining_vehicle_types,
        startup_cost_weight=0.0,
        order_rule="long_first",
    )
    for route in scheduled_non_green:
        route.vehicle_number += used_by_type.get(route.vehicle_type.name, 0)
    routes = scheduled_green + scheduled_non_green
    validate_solution(problem, routes)
    validate_vehicle_schedule(routes, evaluator)
    evaluation = evaluate_solution(routes, evaluator, optimize_departures=False)
    if evaluation.policy_violation_count != 0:
        raise AssertionError("Q2 确定性基线存在绿色区限行违规")
    if evaluation.unfinished_customer_count != 0:
        raise AssertionError("Q2 确定性基线存在漏单或需求不平衡")
    if not evaluation.all_routes_return_before_24h:
        raise AssertionError("Q2 确定性基线存在 24:00 后返场路线")
    return routes


def evaluate_q2_baseline(
    problem: ProblemData,
    policy: Q2Policy,
) -> tuple[list[Route], SolutionEvaluation]:
    routes = build_q2_baseline(problem, policy)
    evaluation = evaluate_solution(
        routes,
        RouteEvaluator(problem, policy=policy),
        optimize_departures=False,
    )
    return routes, evaluation
