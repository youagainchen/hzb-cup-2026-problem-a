# -*- coding: utf-8 -*-
"""问题三场景运行器：冻结态 + 四类事件 + 动态评估（2号交付的接口演示）。

本脚本不包含重优化算子。它把 Q2 正式解（44,861.52 元）按发车级冻结在 T=10:00
切分，叠加四类事件，构造一个「只处理取消、其余趟次保持静态」的最小未来计划，
交给 evaluate_dynamic 统一评分并输出完整账目与可行性。

1 号动态调度器应生成满足全部可行性标志（passed=True）的未来计划。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_problem_data
from src.model.domain import Delivery, Route
from src.model.evaluator import RouteEvaluator, evaluate_solution, format_clock
from src.model.policy_q2 import build_q2_policy
from src.model.q3_event import (
    EUCLID_DETOUR_FACTOR,
    Q3Event,
    Q3EventSet,
    Q3EventType,
    apply_events,
    estimate_detour_factor,
    write_event_set_csv,
)
from src.model.q3_evaluator import evaluate_dynamic, extract_freeze_state
from src.solver.q2_initial import load_route_solution


DATA_DIR = Path("data/processed/team_cleaned")
ROUTES_CSV = Path("results/question2_optimized/question2_optimized_routes.csv")
SUMMARY_CSV = Path("results/question2_optimized/question2_optimized_route_summary.csv")
TOTALS_JSON = Path("results/question2_optimized/question2_optimized_totals.json")
OUTPUT_DIR = Path("results/question3")


def build_scenario(problem) -> Q3EventSet:
    trigger = 10.0 * 60.0
    return Q3EventSet(
        trigger_time_minutes=trigger,
        events=(
            Q3Event(
                event_type=Q3EventType.CANCEL,
                customer_id=12,
                trigger_time_minutes=trigger,
                severity="medium",
            ),
            Q3Event(
                event_type=Q3EventType.NEW_ORDER,
                customer_id=99,
                trigger_time_minutes=trigger,
                weight_kg=600.0,
                volume_m3=1.8,
                window_start_minutes=13.0 * 60.0,
                window_end_minutes=15.0 * 60.0,
                new_x_km=5.0,
                new_y_km=6.0,
                severity="medium",
            ),
            Q3Event(
                event_type=Q3EventType.ADDRESS_CHANGE,
                customer_id=82,
                trigger_time_minutes=trigger,
                new_x_km=20.0,
                new_y_km=-20.0,
                severity="medium",
            ),
            Q3Event(
                event_type=Q3EventType.TIME_WINDOW_CHANGE,
                customer_id=70,
                trigger_time_minutes=trigger,
                window_start_minutes=11.0 * 60.0 + 36.0 + 30.0,
                window_end_minutes=12.0 * 60.0 + 35.0 + 30.0,
                severity="medium",
            ),
        ),
        description="T=10:00 单时刻四类事件（均中度）：取消c12、新增c99(绿色新节点)、变址c82、改窗c70",
    )


def build_minimal_future_plan(static_routes, event_set: Q3EventSet, trigger: float) -> list[Route]:
    """最小未来计划：只响应取消（删除对应配送），其余保持静态趟次。

    新增订单、地址变更、时间窗调整均未处理，用于演示评估器如何报告缺口。
    """
    cancelled = {
        event.customer_id
        for event in event_set.events
        if event.event_type == Q3EventType.CANCEL
    }
    future = []
    for route in static_routes:
        if route.start_minutes < trigger - 1e-9:
            continue  # 已发车趟次冻结，不进未来计划
        deliveries = [
            item
            for item in route.deliveries
            if item.customer_id not in cancelled
        ]
        if not deliveries:
            continue
        future.append(
            Route(
                vehicle_type=route.vehicle_type,
                vehicle_number=route.vehicle_number,
                deliveries=list(deliveries),
                start_minutes=route.start_minutes,
                trip_number=route.trip_number,
            )
        )
    return future


def main() -> None:
    totals = json.loads(TOTALS_JSON.read_text(encoding="utf-8"))
    static_total = float(totals["total_cost"])

    problem = load_problem_data(DATA_DIR)
    policy = build_q2_policy(problem.green_customer_ids)
    evaluator = RouteEvaluator(problem, policy)

    static_routes = load_route_solution(ROUTES_CSV, SUMMARY_CSV)

    # 1) 静态计划可复现性校验
    solution = evaluate_solution(static_routes, evaluator, optimize_departures=False)
    assert abs(solution.total_cost - static_total) <= 1e-4, (
        f"静态计划重算不一致: {solution.total_cost} vs {static_total}"
    )
    print(f"[1] Q2 静态计划重算一致: 总成本 {solution.total_cost:.2f} 元 "
          f"(车辆 {solution.vehicle_count}, 趟次 {solution.trip_count})")

    # 2) 事件场景与应用
    event_set = build_scenario(problem)
    errors = event_set.validate(problem)
    if errors:
        raise AssertionError(f"事件场景校验失败: {errors}")
    detour = estimate_detour_factor(problem)
    print(f"[2] 路网/欧氏距离比值中位数 α = {detour:.4f}（口径采用 {EUCLID_DETOUR_FACTOR}）")
    problem_after, audit = apply_events(
        problem, event_set, detour_factor=EUCLID_DETOUR_FACTOR
    )
    evaluator_after = RouteEvaluator(problem_after, policy)

    # 3) 冻结态
    freeze = extract_freeze_state(
        static_routes, evaluator, event_set.trigger_time_minutes, problem_after
    )
    print(f"[3] 冻结态 @ {format_clock(freeze.trigger_time_minutes)}")
    print(f"    已发车趟次(冻结): {len(freeze.frozen_trip_ids)}，物理车辆: {len(freeze.sunk_vehicle_keys)}")
    print(f"    可重排趟次: {len(freeze.replannable_trip_ids)}")
    print(f"    沉没固定成本: {freeze.sunk_fixed_cost:.0f} 元，沉没运行成本: {freeze.sunk_operating_cost:.2f} 元")
    print(f"    已执行成本(沉没): {freeze.executed_cost:.2f} 元")
    print(f"    剩余需求客户数: {len(freeze.remaining_demand)}")

    # 4) 最小未来计划 + 动态评估
    future = build_minimal_future_plan(static_routes, event_set, freeze.trigger_time_minutes)
    evaluation = evaluate_dynamic(
        future,
        freeze,
        evaluator_after,
        static_total_cost=static_total,
        static_routes=static_routes,
    )
    print(f"[4] 最小未来计划（只响应取消，其余保持静态）")
    print(f"    未来趟次 {evaluation.future_trip_count}（保留 {evaluation.kept_trip_count}，"
          f"删除 {evaluation.dropped_trip_count}，新增 {evaluation.new_trip_count}），"
          f"未来车辆 {evaluation.future_vehicle_count}")
    print(f"    未来固定成本: {evaluation.future_fixed_cost:.2f} 元，"
          f"未来运行成本: {evaluation.future_operating_cost:.2f} 元")
    print(f"    当日总成本: {evaluation.executed_cost:.2f} + {evaluation.future_cost:.2f} = "
          f"{evaluation.total_cost:.2f} 元")
    print(f"    ΔC = {evaluation.total_cost:.2f} − {evaluation.static_total_cost:.2f} = "
          f"{evaluation.delta_cost:+.2f} 元（相对变化 {evaluation.cost_change_ratio:+.2%}）")
    print(f"    扰动: 改动客户 {evaluation.changed_customer_count}，"
          f"客户重分配率 {evaluation.assignment_change_ratio:.2%}，"
          f"路径扰动率 {evaluation.arc_change_ratio:.2%}；"
          f"执行提前量 {evaluation.lead_time_minutes:.0f} 分钟")
    f = evaluation.feasibility
    print(f"    可行性: passed={f.passed} | 未完成客户 {f.demand_unfinished_customers} "
          f"(重 {f.demand_unfinished_weight_kg:.1f}kg 容 {f.demand_unfinished_volume_m3:.2f}m³) | "
          f"容量违规 {f.capacity_violation_count} | 政策违规 {f.policy_violation_count} | "
          f"排程违规 {f.schedule_violation_count} | 24:00前返场 {f.all_routes_return_before_24h}")
    for note in f.notes[:12]:
        print(f"      - {note}")

    # 5) 输出归档
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_event_set_csv(event_set, OUTPUT_DIR / "question3_event_set.csv", problem)
    freeze_summary = {
        "trigger_time_minutes": freeze.trigger_time_minutes,
        "trigger_clock": format_clock(freeze.trigger_time_minutes),
        "frozen_trip_ids": freeze.frozen_trip_ids,
        "replannable_trip_ids": freeze.replannable_trip_ids,
        "sunk_vehicle_keys": [
            f"{name}-{number:03d}" for name, number in sorted(freeze.sunk_vehicle_keys)
        ],
        "sunk_fixed_cost": freeze.sunk_fixed_cost,
        "sunk_operating_cost": freeze.sunk_operating_cost,
        "executed_cost": freeze.executed_cost,
        "remaining_demand": {
            str(cid): [round(w, 6), round(v, 6)]
            for cid, (w, v) in freeze.remaining_demand.items()
        },
        "vehicle_ready_minutes": {
            f"{name}-{number:03d}": ready
            for (name, number), ready in sorted(freeze.vehicle_ready_minutes.items())
        },
        "event_audit": audit,
        "detour_factor": detour,
    }
    (OUTPUT_DIR / "question3_freeze_state.json").write_text(
        json.dumps(freeze_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "question3_dynamic_evaluation.json").write_text(
        json.dumps(
            {
                "executed_cost": evaluation.executed_cost,
                "future_fixed_cost": evaluation.future_fixed_cost,
                "future_operating_cost": evaluation.future_operating_cost,
                "future_cost": evaluation.future_cost,
                "total_cost": evaluation.total_cost,
                "static_total_cost": evaluation.static_total_cost,
                "delta_cost": evaluation.delta_cost,
                "cost_change_ratio": evaluation.cost_change_ratio,
                "lead_time_minutes": evaluation.lead_time_minutes,
                "sunk_vehicle_count": evaluation.sunk_vehicle_count,
                "future_vehicle_count": evaluation.future_vehicle_count,
                "frozen_trip_count": evaluation.frozen_trip_count,
                "replannable_trip_count": evaluation.replannable_trip_count,
                "future_trip_count": evaluation.future_trip_count,
                "dropped_trip_count": evaluation.dropped_trip_count,
                "new_trip_count": evaluation.new_trip_count,
                "kept_trip_count": evaluation.kept_trip_count,
                "changed_customer_count": evaluation.changed_customer_count,
                "assignment_change_ratio": evaluation.assignment_change_ratio,
                "arc_change_ratio": evaluation.arc_change_ratio,
                "feasibility": {
                    "passed": f.passed,
                    "demand_unfinished_customers": f.demand_unfinished_customers,
                    "demand_unfinished_weight_kg": f.demand_unfinished_weight_kg,
                    "demand_unfinished_volume_m3": f.demand_unfinished_volume_m3,
                    "capacity_violation_count": f.capacity_violation_count,
                    "policy_violation_count": f.policy_violation_count,
                    "schedule_violation_count": f.schedule_violation_count,
                    "all_routes_return_before_24h": f.all_routes_return_before_24h,
                    "notes": f.notes,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[5] 输出已写入 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
