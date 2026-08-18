from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import exp, isfinite
from random import Random
from typing import Callable, Protocol, Sequence

from src.model.domain import DEFAULT_VEHICLE_TYPES, Delivery, Route, VehicleType


DEFAULT_Q2_SEEDS: tuple[int, ...] = (202601, 202602, 202603, 202604, 202605)
DEFAULT_DEPARTURE_CANDIDATES: tuple[float, ...] = tuple(
    float(minute) for minute in range(8 * 60, 20 * 60 + 1, 30)
)


class ScoreLike(Protocol):
    """2号统一评估器返回值的最小接口。"""

    total_cost: float
    policy_violation_count: int


SolutionScorer = Callable[[list[Route]], ScoreLike]


class ScorerContractError(TypeError):
    """统一评分器未遵守只读、有限数值返回约定。"""


@dataclass(frozen=True)
class NormalizedQ2Score:
    """求解器内部只依赖的三个评分字段。"""

    total_cost: float
    policy_violation_count: int
    is_feasible: bool = True
    raw: object = field(default=None, compare=False, repr=False)

    @property
    def objective_key(self) -> tuple[int, int, float]:
        return (
            0 if self.is_feasible else 1,
            self.policy_violation_count,
            self.total_cost,
        )


@dataclass(frozen=True)
class SearchTraceEntry:
    pass_index: int
    operator: str
    violations_before: int
    violations_after: int
    cost_before: float
    cost_after: float


@dataclass(frozen=True)
class Q2SearchRun:
    seed: int
    routes: tuple[Route, ...]
    score: NormalizedQ2Score
    trace: tuple[SearchTraceEntry, ...]


@dataclass(frozen=True)
class ALNSConfig:
    iterations: int = 200
    destroy_fraction: float = 0.10
    initial_temperature: float = 100.0
    cooling_rate: float = 0.98
    reaction_factor: float = 0.20
    max_no_improve: int = 80

    def validate(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations 至少为1")
        if not 0.0 < self.destroy_fraction <= 1.0:
            raise ValueError("destroy_fraction 必须位于(0, 1]")
        if self.initial_temperature <= 0.0:
            raise ValueError("initial_temperature 必须为正数")
        if not 0.0 < self.cooling_rate <= 1.0:
            raise ValueError("cooling_rate 必须位于(0, 1]")
        if not 0.0 < self.reaction_factor <= 1.0:
            raise ValueError("reaction_factor 必须位于(0, 1]")
        if self.max_no_improve < 1:
            raise ValueError("max_no_improve 至少为1")


@dataclass(frozen=True)
class ALNSTraceEntry:
    iteration: int
    destroy_operator: str
    repair_operator: str
    accepted: bool
    current_violations: int
    current_cost: float
    best_violations: int
    best_cost: float
    physical_vehicles: int
    temperature: float


@dataclass(frozen=True)
class Q2ALNSRun:
    seed: int
    routes: tuple[Route, ...]
    score: NormalizedQ2Score
    trace: tuple[ALNSTraceEntry, ...]
    destroy_weights: dict[str, float]
    repair_weights: dict[str, float]


def clone_routes(routes: Sequence[Route]) -> list[Route]:
    return [
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


def normalize_score(score: ScoreLike) -> NormalizedQ2Score:
    """兼容2号后续可能采用的 `feasible` 或 `is_feasible` 字段。"""

    try:
        total_cost = float(score.total_cost)
        raw_violations = score.policy_violation_count
        violations = int(raw_violations)
    except (AttributeError, TypeError, ValueError) as error:
        raise ScorerContractError(
            "评分结果必须包含数值型 total_cost 和 policy_violation_count"
        ) from error
    if not isfinite(total_cost) or total_cost < 0.0:
        raise ScorerContractError("total_cost 必须是有限的非负数")
    if float(raw_violations) != float(violations):
        raise ScorerContractError("policy_violation_count 必须是整数")
    if violations < 0:
        raise ScorerContractError("policy_violation_count 不能为负数")

    feasible = bool(
        getattr(score, "is_feasible", getattr(score, "feasible", True))
    )
    feasibility = getattr(score, "feasibility", None)
    if isinstance(feasibility, dict):
        feasible = feasible and all(bool(value) for value in feasibility.values())
    return NormalizedQ2Score(total_cost, violations, feasible, score)


def _route_fingerprint(routes: Sequence[Route]) -> tuple[object, ...]:
    return tuple(
        (
            route.vehicle_type,
            route.vehicle_number,
            route.start_minutes,
            route.trip_number,
            tuple(route.deliveries),
        )
        for route in routes
    )


def _score_or_none(
    routes: list[Route], scorer: SolutionScorer
) -> NormalizedQ2Score | None:
    before = _route_fingerprint(routes)
    try:
        raw_score = scorer(routes)
    except (AssertionError, RuntimeError, ValueError):
        if _route_fingerprint(routes) != before:
            raise ScorerContractError("统一评分器异常退出前修改了候选路线")
        return None
    if _route_fingerprint(routes) != before:
        raise ScorerContractError(
            "统一评分器必须只读评分；请在2号适配器内部复制路线"
        )
    return normalize_score(raw_score)


def _capacity_fits(deliveries: Sequence[Delivery], vehicle: VehicleType) -> bool:
    return (
        sum(item.weight for item in deliveries) <= vehicle.capacity_weight + 1e-7
        and sum(item.volume for item in deliveries) <= vehicle.capacity_volume + 1e-7
    )


def _ordered_indices(length: int, rng: Random) -> list[int]:
    indices = list(range(length))
    rng.shuffle(indices)
    return indices


def physical_vehicle_count(routes: Sequence[Route]) -> int:
    return len(
        {
            (route.vehicle_type.name, route.vehicle_number)
            for route in routes
        }
    )


def _renumber_trips(routes: list[Route]) -> None:
    grouped: dict[tuple[str, int], list[Route]] = defaultdict(list)
    for route in routes:
        grouped[(route.vehicle_type.name, route.vehicle_number)].append(route)
    for jobs in grouped.values():
        for trip_number, route in enumerate(
            sorted(jobs, key=lambda item: (item.start_minutes, item.trip_number)),
            start=1,
        ):
            route.trip_number = trip_number


def _candidate_vehicle_numbers(
    routes: Sequence[Route], route_index: int, vehicle: VehicleType
) -> tuple[int, ...]:
    used = sorted(
        {
            route.vehicle_number
            for index, route in enumerate(routes)
            if index != route_index and route.vehicle_type.name == vehicle.name
        }
    )
    candidates = list(used)
    first_unused = next(
        (number for number in range(1, vehicle.count + 1) if number not in used),
        None,
    )
    if first_unused is not None:
        candidates.append(first_unused)
    return tuple(dict.fromkeys(candidates))


def repair_vehicle_types(
    routes: Sequence[Route],
    scorer: SolutionScorer,
    rng: Random,
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES,
) -> tuple[list[Route], NormalizedQ2Score]:
    """逐趟尝试车型重选；是否合规完全由注入的统一评分器判断。"""

    current = clone_routes(routes)
    current_score = _score_or_none(current, scorer)
    if current_score is None:
        raise ValueError("输入基线无法被统一评估器评分")

    for route_index in _ordered_indices(len(current), rng):
        best_routes = current
        best_score = current_score
        candidates = list(vehicle_types)
        rng.shuffle(candidates)
        for vehicle in candidates:
            source = current[route_index]
            if vehicle == source.vehicle_type or not _capacity_fits(
                source.deliveries, vehicle
            ):
                continue
            candidate = clone_routes(current)
            candidate[route_index].vehicle_type = vehicle
            candidate_score = _score_or_none(candidate, scorer)
            if (
                candidate_score is not None
                and candidate_score.objective_key < best_score.objective_key
            ):
                best_routes = candidate
                best_score = candidate_score
        current = best_routes
        current_score = best_score
    return current, current_score


def repair_departure_times(
    routes: Sequence[Route],
    scorer: SolutionScorer,
    rng: Random,
    departure_candidates: Sequence[float] = DEFAULT_DEPARTURE_CANDIDATES,
) -> tuple[list[Route], NormalizedQ2Score]:
    """逐趟枚举发车时刻；限行端点解释仍由2号评分器唯一维护。"""

    current = clone_routes(routes)
    current_score = _score_or_none(current, scorer)
    if current_score is None:
        raise ValueError("输入基线无法被统一评估器评分")

    for route_index in _ordered_indices(len(current), rng):
        best_routes = current
        best_score = current_score
        candidates = list(dict.fromkeys(float(value) for value in departure_candidates))
        rng.shuffle(candidates)
        for start_minutes in candidates:
            if abs(start_minutes - current[route_index].start_minutes) <= 1e-9:
                continue
            candidate = clone_routes(current)
            candidate[route_index].start_minutes = start_minutes
            candidate_score = _score_or_none(candidate, scorer)
            if (
                candidate_score is not None
                and candidate_score.objective_key < best_score.objective_key
            ):
                best_routes = candidate
                best_score = candidate_score
        current = best_routes
        current_score = best_score
    return current, current_score


def swap_entire_trips(
    routes: Sequence[Route],
    scorer: SolutionScorer,
    rng: Random,
) -> tuple[list[Route], NormalizedQ2Score]:
    """交换两条趟次的完整配送序列，保留各自车型、车号和发车槽位。"""

    current = clone_routes(routes)
    current_score = _score_or_none(current, scorer)
    if current_score is None:
        raise ValueError("输入基线无法被统一评估器评分")

    pairs = [
        (left, right)
        for left in range(len(current))
        for right in range(left + 1, len(current))
    ]
    rng.shuffle(pairs)
    for left, right in pairs:
        candidate = clone_routes(current)
        candidate[left].deliveries, candidate[right].deliveries = (
            candidate[right].deliveries,
            candidate[left].deliveries,
        )
        if not _capacity_fits(
            candidate[left].deliveries, candidate[left].vehicle_type
        ) or not _capacity_fits(
            candidate[right].deliveries, candidate[right].vehicle_type
        ):
            continue
        candidate_score = _score_or_none(candidate, scorer)
        if (
            candidate_score is not None
            and candidate_score.objective_key < current_score.objective_key
        ):
            current = candidate
            current_score = candidate_score
    return current, current_score


def relocate_entire_trips(
    routes: Sequence[Route],
    scorer: SolutionScorer,
    rng: Random,
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES,
) -> tuple[list[Route], NormalizedQ2Score]:
    """把完整趟次迁移到其他车型/物理车，时序与政策均交给评分器验收。"""

    current = clone_routes(routes)
    current_score = _score_or_none(current, scorer)
    if current_score is None:
        raise ValueError("输入基线无法被统一评估器评分")

    for route_index in _ordered_indices(len(current), rng):
        best_routes = current
        best_score = current_score
        best_count = physical_vehicle_count(current)
        candidate_types = list(vehicle_types)
        rng.shuffle(candidate_types)
        for vehicle in candidate_types:
            source = current[route_index]
            if not _capacity_fits(source.deliveries, vehicle):
                continue
            numbers = list(_candidate_vehicle_numbers(current, route_index, vehicle))
            rng.shuffle(numbers)
            for vehicle_number in numbers:
                if (
                    vehicle.name == source.vehicle_type.name
                    and vehicle_number == source.vehicle_number
                ):
                    continue
                candidate = clone_routes(current)
                candidate[route_index].vehicle_type = vehicle
                candidate[route_index].vehicle_number = vehicle_number
                candidate[route_index].trip_number = 1
                _renumber_trips(candidate)
                candidate_score = _score_or_none(candidate, scorer)
                if candidate_score is None:
                    continue
                candidate_count = physical_vehicle_count(candidate)
                if (
                    candidate_score.objective_key,
                    candidate_count,
                ) < (best_score.objective_key, best_count):
                    best_routes = candidate
                    best_score = candidate_score
                    best_count = candidate_count
        current = best_routes
        current_score = best_score
    return current, current_score


def search_physical_vehicle_count(
    routes: Sequence[Route],
    scorer: SolutionScorer,
    rng: Random,
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES,
    max_passes: int = 3,
) -> tuple[list[Route], NormalizedQ2Score]:
    """反复执行整趟迁移，压缩物理车辆数但不牺牲统一目标值。"""

    if max_passes < 1:
        raise ValueError("max_passes 至少为1")
    current = clone_routes(routes)
    current_score = _score_or_none(current, scorer)
    if current_score is None:
        raise ValueError("输入基线无法被统一评估器评分")
    for _ in range(max_passes):
        before_key = (current_score.objective_key, physical_vehicle_count(current))
        candidate, candidate_score = relocate_entire_trips(
            current, scorer, rng, vehicle_types
        )
        after_key = (candidate_score.objective_key, physical_vehicle_count(candidate))
        if after_key >= before_key:
            break
        current = candidate
        current_score = candidate_score
    return current, current_score


def run_q2_search(
    baseline_routes: Sequence[Route],
    scorer: SolutionScorer,
    seed: int,
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES,
    departure_candidates: Sequence[float] = DEFAULT_DEPARTURE_CANDIDATES,
    max_passes: int = 5,
) -> Q2SearchRun:
    """运行依赖安全的第一阶段搜索骨架，后续可直接作为ALNS修复层。"""

    if max_passes < 1:
        raise ValueError("max_passes 至少为1")
    rng = Random(seed)
    current = clone_routes(baseline_routes)
    current_score = _score_or_none(current, scorer)
    if current_score is None:
        raise ValueError("Q2基线无法被统一评估器评分")
    trace: list[SearchTraceEntry] = []

    for pass_index in range(1, max_passes + 1):
        improved = False
        operators = [
            "vehicle_reselect",
            "departure_repair",
            "whole_trip_swap",
            "whole_trip_relocate",
        ]
        rng.shuffle(operators)
        for operator in operators:
            before = current_score
            if operator == "vehicle_reselect":
                candidate, candidate_score = repair_vehicle_types(
                    current, scorer, rng, vehicle_types
                )
            elif operator == "departure_repair":
                candidate, candidate_score = repair_departure_times(
                    current, scorer, rng, departure_candidates
                )
            elif operator == "whole_trip_swap":
                candidate, candidate_score = swap_entire_trips(current, scorer, rng)
            else:
                candidate, candidate_score = relocate_entire_trips(
                    current, scorer, rng, vehicle_types
                )

            if candidate_score.objective_key < current_score.objective_key:
                current = candidate
                current_score = candidate_score
                improved = True
                trace.append(
                    SearchTraceEntry(
                        pass_index=pass_index,
                        operator=operator,
                        violations_before=before.policy_violation_count,
                        violations_after=current_score.policy_violation_count,
                        cost_before=before.total_cost,
                        cost_after=current_score.total_cost,
                    )
                )
        if not improved:
            break

    return Q2SearchRun(seed, tuple(clone_routes(current)), current_score, tuple(trace))


def run_fixed_seed_search(
    baseline_routes: Sequence[Route],
    scorer: SolutionScorer,
    seeds: Sequence[int] = DEFAULT_Q2_SEEDS,
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES,
    departure_candidates: Sequence[float] = DEFAULT_DEPARTURE_CANDIDATES,
    max_passes: int = 5,
) -> tuple[Q2SearchRun, tuple[Q2SearchRun, ...]]:
    """运行固定种子并返回最优合规结果与全部日志对象。"""

    if not seeds:
        raise ValueError("至少需要一个随机种子")
    runs = tuple(
        run_q2_search(
            baseline_routes,
            scorer,
            int(seed),
            vehicle_types,
            departure_candidates,
            max_passes,
        )
        for seed in seeds
    )
    compliant = [
        run
        for run in runs
        if run.score.is_feasible and run.score.policy_violation_count == 0
    ]
    if not compliant:
        raise RuntimeError("固定种子搜索尚未得到违规为0的Q2解")
    best = min(compliant, key=lambda run: (run.score.total_cost, run.seed))
    return best, runs


def _objective_value(score: NormalizedQ2Score) -> float:
    hard_penalty = 1.0e15 if not score.is_feasible else 0.0
    policy_penalty = float(score.policy_violation_count) * 1.0e12
    return hard_penalty + policy_penalty + score.total_cost


def _weighted_choice(weights: dict[str, float], rng: Random) -> str:
    names = list(weights)
    return rng.choices(names, weights=[weights[name] for name in names], k=1)[0]


def _destroy_solution(
    routes: Sequence[Route],
    operator: str,
    rng: Random,
    fraction: float,
    vehicle_types: tuple[VehicleType, ...],
    departure_candidates: Sequence[float],
) -> list[Route]:
    candidate = clone_routes(routes)
    if not candidate:
        return candidate
    destroy_count = max(1, round(len(candidate) * fraction))
    indices = rng.sample(range(len(candidate)), min(destroy_count, len(candidate)))

    if operator == "vehicle_change":
        for index in indices:
            feasible_types = [
                vehicle
                for vehicle in vehicle_types
                if _capacity_fits(candidate[index].deliveries, vehicle)
            ]
            if feasible_types:
                vehicle = rng.choice(feasible_types)
                candidate[index].vehicle_type = vehicle
                candidate[index].vehicle_number = min(
                    max(1, candidate[index].vehicle_number), vehicle.count
                )
        _renumber_trips(candidate)
    elif operator == "departure_shift":
        starts = tuple(float(value) for value in departure_candidates)
        for index in indices:
            if starts:
                candidate[index].start_minutes = rng.choice(starts)
        _renumber_trips(candidate)
    elif operator == "trip_swap" and len(candidate) >= 2:
        for left in indices:
            right = rng.randrange(len(candidate) - 1)
            if right >= left:
                right += 1
            if _capacity_fits(
                candidate[right].deliveries, candidate[left].vehicle_type
            ) and _capacity_fits(
                candidate[left].deliveries, candidate[right].vehicle_type
            ):
                candidate[left].deliveries, candidate[right].deliveries = (
                    candidate[right].deliveries,
                    candidate[left].deliveries,
                )
    else:
        for index in indices:
            vehicle = rng.choice(vehicle_types)
            if not _capacity_fits(candidate[index].deliveries, vehicle):
                continue
            numbers = _candidate_vehicle_numbers(candidate, index, vehicle)
            if not numbers:
                continue
            candidate[index].vehicle_type = vehicle
            candidate[index].vehicle_number = rng.choice(numbers)
        _renumber_trips(candidate)
    return candidate


def _repair_solution(
    routes: Sequence[Route],
    operator: str,
    scorer: SolutionScorer,
    rng: Random,
    vehicle_types: tuple[VehicleType, ...],
    departure_candidates: Sequence[float],
) -> tuple[list[Route], NormalizedQ2Score]:
    if operator == "vehicle_reselect":
        return repair_vehicle_types(routes, scorer, rng, vehicle_types)
    if operator == "departure_repair":
        return repair_departure_times(routes, scorer, rng, departure_candidates)
    if operator == "whole_trip_swap":
        return swap_entire_trips(routes, scorer, rng)
    return relocate_entire_trips(routes, scorer, rng, vehicle_types)


def _update_weight(
    weights: dict[str, float], operator: str, reward: float, reaction: float
) -> None:
    weights[operator] = max(
        0.05,
        (1.0 - reaction) * weights[operator] + reaction * reward,
    )


def run_q2_alns(
    baseline_routes: Sequence[Route],
    scorer: SolutionScorer,
    seed: int,
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES,
    departure_candidates: Sequence[float] = DEFAULT_DEPARTURE_CANDIDATES,
    config: ALNSConfig = ALNSConfig(),
) -> Q2ALNSRun:
    """运行政策无关的ALNS；所有候选解仍由2号统一评分器最终裁决。"""

    config.validate()
    rng = Random(seed)
    current = clone_routes(baseline_routes)
    current_score = _score_or_none(current, scorer)
    if current_score is None:
        raise ValueError("Q2基线无法被统一评估器评分")
    best = clone_routes(current)
    best_score = current_score
    destroy_weights = {
        "vehicle_change": 1.0,
        "departure_shift": 1.0,
        "trip_swap": 1.0,
        "trip_relocate": 1.0,
    }
    repair_weights = {
        "vehicle_reselect": 1.0,
        "departure_repair": 1.0,
        "whole_trip_swap": 1.0,
        "whole_trip_relocate": 1.0,
    }
    temperature = config.initial_temperature
    no_improve = 0
    trace: list[ALNSTraceEntry] = []

    for iteration in range(1, config.iterations + 1):
        destroy_operator = _weighted_choice(destroy_weights, rng)
        repair_operator = _weighted_choice(repair_weights, rng)
        destroyed = _destroy_solution(
            current,
            destroy_operator,
            rng,
            config.destroy_fraction,
            vehicle_types,
            departure_candidates,
        )
        destroyed_score = _score_or_none(destroyed, scorer)
        candidate: list[Route] | None = None
        candidate_score: NormalizedQ2Score | None = None
        if destroyed_score is not None:
            candidate, candidate_score = _repair_solution(
                destroyed,
                repair_operator,
                scorer,
                rng,
                vehicle_types,
                departure_candidates,
            )

        accepted = False
        reward = 0.05
        if candidate is not None and candidate_score is not None:
            delta = _objective_value(candidate_score) - _objective_value(current_score)
            accepted = delta <= 0.0 or rng.random() < exp(
                -min(delta, 700.0 * temperature) / temperature
            )
            if accepted:
                current = candidate
                current_score = candidate_score
                reward = 1.0
            if candidate_score.objective_key < best_score.objective_key:
                best = clone_routes(candidate)
                best_score = candidate_score
                no_improve = 0
                reward = 6.0
            else:
                no_improve += 1
        else:
            no_improve += 1

        _update_weight(
            destroy_weights,
            destroy_operator,
            reward,
            config.reaction_factor,
        )
        _update_weight(
            repair_weights,
            repair_operator,
            reward,
            config.reaction_factor,
        )
        trace.append(
            ALNSTraceEntry(
                iteration=iteration,
                destroy_operator=destroy_operator,
                repair_operator=repair_operator,
                accepted=accepted,
                current_violations=current_score.policy_violation_count,
                current_cost=current_score.total_cost,
                best_violations=best_score.policy_violation_count,
                best_cost=best_score.total_cost,
                physical_vehicles=physical_vehicle_count(best),
                temperature=temperature,
            )
        )
        temperature = max(1.0e-6, temperature * config.cooling_rate)
        if no_improve >= config.max_no_improve:
            break

    return Q2ALNSRun(
        seed=seed,
        routes=tuple(clone_routes(best)),
        score=best_score,
        trace=tuple(trace),
        destroy_weights=dict(destroy_weights),
        repair_weights=dict(repair_weights),
    )


def run_fixed_seed_alns(
    baseline_routes: Sequence[Route],
    scorer: SolutionScorer,
    seeds: Sequence[int] = DEFAULT_Q2_SEEDS,
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES,
    departure_candidates: Sequence[float] = DEFAULT_DEPARTURE_CANDIDATES,
    config: ALNSConfig = ALNSConfig(),
) -> tuple[Q2ALNSRun, tuple[Q2ALNSRun, ...]]:
    if not seeds:
        raise ValueError("至少需要一个随机种子")
    runs = tuple(
        run_q2_alns(
            baseline_routes,
            scorer,
            int(seed),
            vehicle_types,
            departure_candidates,
            config,
        )
        for seed in seeds
    )
    compliant = [
        run
        for run in runs
        if run.score.is_feasible and run.score.policy_violation_count == 0
    ]
    if not compliant:
        raise RuntimeError("5种子ALNS尚未得到违规为0的Q2解")
    best = min(compliant, key=lambda run: (run.score.total_cost, run.seed))
    return best, runs


def infeasible_score() -> NormalizedQ2Score:
    """供外部适配器在无法评分时显式返回，不应作为2号正式接口。"""

    return NormalizedQ2Score(0.0, 0, False)
