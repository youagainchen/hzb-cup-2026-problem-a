from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data.loader import load_problem_data
from src.model.domain import Route
from src.model.evaluator import RouteEvaluator, SolutionEvaluation, evaluate_solution
from src.model.policy_q2 import build_q2_policy
from src.question2 import Question2Context
from src.solver.q2_baseline import build_q2_baseline
from src.solver.q2_search import clone_routes
from src.solver.scheduling import validate_vehicle_schedule


@dataclass(frozen=True)
class Q2AdapterScore:
    total_cost: float
    policy_violation_count: int
    is_feasible: bool
    feasibility: dict[str, bool]
    raw: SolutionEvaluation


def build_context(data_dir: Path) -> Question2Context:
    """把2号实例、政策和评估器接入1号搜索器的稳定上下文。"""

    problem = load_problem_data(data_dir)
    policy = build_q2_policy(problem.green_customer_ids)
    evaluator = RouteEvaluator(problem, policy=policy)
    baseline_routes = build_q2_baseline(problem, policy)

    def score(routes: list[Route]) -> Q2AdapterScore:
        candidates = clone_routes(routes)
        evaluation = evaluate_solution(
            candidates,
            evaluator,
            optimize_departures=False,
        )
        schedule_valid = True
        try:
            validate_vehicle_schedule(candidates, evaluator)
        except AssertionError:
            schedule_valid = False
        feasibility = {
            "demand_complete": evaluation.unfinished_customer_count == 0,
            "capacity_valid": evaluation.capacity_violation_count == 0,
            "return_before_24h": evaluation.all_routes_return_before_24h,
            "vehicle_schedule_valid": schedule_valid,
        }
        return Q2AdapterScore(
            total_cost=evaluation.total_cost,
            policy_violation_count=evaluation.policy_violation_count,
            is_feasible=all(feasibility.values()),
            feasibility=feasibility,
            raw=evaluation,
        )

    return Question2Context(
        baseline_routes=tuple(clone_routes(baseline_routes)),
        scorer=score,
    )
