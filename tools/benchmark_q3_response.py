# -*- coding: utf-8 -*-
"""问题三响应时间重复实验：同一事件场景独立运行 N 次，报告中位数与 P95。

用法：python tools/benchmark_q3_response.py [--n 30]
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_problem_data
from src.model.evaluator import RouteEvaluator, evaluate_solution
from src.model.policy_q2 import build_q2_policy
from src.solver.q2_initial import load_route_solution
from src.solver.q3_dynamic import dispatch_event_set
from tools.run_q3_optimized import read_event_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题三响应时间重复实验")
    parser.add_argument("--n", type=int, default=30, help="独立重复次数（默认30）")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/team_cleaned"))
    parser.add_argument("--event-path", type=Path, default=Path("results/question3/question3_event_set.csv"))
    parser.add_argument("--route-path", type=Path, default=Path("results/question2_optimized/question2_optimized_routes.csv"))
    parser.add_argument("--summary-path", type=Path, default=Path("results/question2_optimized/question2_optimized_route_summary.csv"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    problem = load_problem_data(arguments.data_dir)
    evaluator = RouteEvaluator(problem, build_q2_policy(problem.green_customer_ids))
    static_routes = load_route_solution(arguments.route_path, arguments.summary_path)
    static_total = evaluate_solution(
        static_routes, evaluator, optimize_departures=False
    ).total_cost
    event_set = read_event_sets(arguments.event_path)[0]

    # 预热一次，排除首次导入/加载影响
    dispatch_event_set(static_routes, problem, event_set, static_total_cost=static_total)

    samples_s: list[float] = []
    for _ in range(arguments.n):
        step = dispatch_event_set(
            static_routes, problem, event_set, static_total_cost=static_total
        )
        if not step.evaluation.feasibility.passed:
            raise AssertionError("重复实验中出现了不可行方案")
        samples_s.append(step.response_time_s)

    samples_ms = [value * 1000.0 for value in samples_s]
    ordered = sorted(samples_ms)
    median = statistics.median(samples_ms)
    p95 = ordered[int(len(ordered) * 0.95) - 1]
    p_min, p_max = ordered[0], ordered[-1]

    print("=== 响应时间重复实验（同一四事件场景） ===")
    print(f"重复次数 N = {arguments.n}")
    print(f"中位数 T_median = {median:.1f} ms")
    print(f"95%分位  T_P95   = {p95:.1f} ms")
    print(f"范围             = [{p_min:.1f}, {p_max:.1f}] ms")
    print()
    print("实验环境：")
    print(f"  OS      : {platform.system()} {platform.release()} ({platform.platform()})")
    print(f"  Python  : {platform.python_version()}")
    print(f"  CPU     : {platform.processor() or '未知'}")
    try:
        import psutil

        print(f"  内存    : {psutil.virtual_memory().total / 2**30:.1f} GB")
    except ImportError:
        print("  内存    : （安装 psutil 可显示）")


if __name__ == "__main__":
    main()
