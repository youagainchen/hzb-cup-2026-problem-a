from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.model.domain import DEFAULT_VEHICLE_TYPES, Route, VehicleType
from src.solver.q2_outputs import Q2OutputPaths, write_q2_alns_outputs
from src.solver.q2_search import (
    ALNSConfig,
    DEFAULT_DEPARTURE_CANDIDATES,
    DEFAULT_Q2_SEEDS,
    Q2ALNSRun,
    SolutionScorer,
    run_fixed_seed_alns,
)


@dataclass(frozen=True)
class Question2RunResult:
    best: Q2ALNSRun
    runs: tuple[Q2ALNSRun, ...]
    outputs: Q2OutputPaths


@dataclass(frozen=True)
class Question2Context:
    """2号适配器工厂需要返回的稳定交接对象。"""

    baseline_routes: Sequence[Route]
    scorer: SolutionScorer
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES
    departure_candidates: Sequence[float] = DEFAULT_DEPARTURE_CANDIDATES


def run(
    baseline_routes: Sequence[Route],
    scorer: SolutionScorer,
    output_dir: Path = Path("results"),
    seeds: Sequence[int] = DEFAULT_Q2_SEEDS,
    vehicle_types: tuple[VehicleType, ...] = DEFAULT_VEHICLE_TYPES,
    departure_candidates: Sequence[float] = DEFAULT_DEPARTURE_CANDIDATES,
    config: ALNSConfig = ALNSConfig(),
) -> Question2RunResult:
    """问题二运行入口；真实实例和评分器由2号交接后从外部注入。"""

    best, runs = run_fixed_seed_alns(
        baseline_routes=baseline_routes,
        scorer=scorer,
        seeds=seeds,
        vehicle_types=vehicle_types,
        departure_candidates=departure_candidates,
        config=config,
    )
    outputs = write_q2_alns_outputs(output_dir, best, runs)
    return Question2RunResult(best=best, runs=runs, outputs=outputs)


def run_context(
    context: Question2Context,
    output_dir: Path = Path("results"),
    seeds: Sequence[int] = DEFAULT_Q2_SEEDS,
    config: ALNSConfig = ALNSConfig(),
) -> Question2RunResult:
    return run(
        baseline_routes=context.baseline_routes,
        scorer=context.scorer,
        output_dir=output_dir,
        seeds=seeds,
        vehicle_types=context.vehicle_types,
        departure_candidates=context.departure_candidates,
        config=config,
    )
