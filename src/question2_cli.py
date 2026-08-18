from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Callable, Sequence

from src.question2 import Question2Context, run_context
from src.solver.q2_search import ALNSConfig, DEFAULT_Q2_SEEDS


ContextFactory = Callable[[Path], Question2Context]


def load_context(adapter: str, data_dir: Path) -> Question2Context:
    """从 `模块:工厂函数` 加载2号交接适配器。"""

    module_name, separator, factory_name = adapter.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("--adapter 必须使用 模块:工厂函数 格式")
    module = import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise TypeError(f"{adapter} 不是可调用的上下文工厂")
    context = factory(data_dir)
    if not isinstance(context, Question2Context):
        raise TypeError("适配器工厂必须返回 Question2Context")
    return context


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("至少需要一个随机种子")
    return seeds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行问题二5种子ALNS")
    parser.add_argument(
        "--adapter",
        required=True,
        help="2号适配器，格式为 模块:build_context",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/team_cleaned"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=DEFAULT_Q2_SEEDS,
        help="逗号分隔随机种子",
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--destroy-fraction", type=float, default=0.10)
    parser.add_argument("--initial-temperature", type=float, default=100.0)
    parser.add_argument("--cooling-rate", type=float, default=0.98)
    parser.add_argument("--reaction-factor", type=float, default=0.20)
    parser.add_argument("--max-no-improve", type=int, default=80)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = load_context(args.adapter, args.data_dir)
    config = ALNSConfig(
        iterations=args.iterations,
        destroy_fraction=args.destroy_fraction,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        reaction_factor=args.reaction_factor,
        max_no_improve=args.max_no_improve,
    )
    result = run_context(
        context,
        output_dir=args.output_dir,
        seeds=args.seeds,
        config=config,
    )
    print(
        f"Q2完成：seed={result.best.seed}，"
        f"违规={result.best.score.policy_violation_count}，"
        f"成本={result.best.score.total_cost:.2f}"
    )
    print(f"路线：{result.outputs.route_csv}")
    print(f"收敛：{result.outputs.trace_csv}")
    print(f"汇总：{result.outputs.totals_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
