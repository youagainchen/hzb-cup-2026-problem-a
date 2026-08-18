# -*- coding: utf-8 -*-
"""问题三动态事件：事件类型、事件实例表与应用到问题数据。

由 2 号维护。只定义事件口径与把事件应用到 ProblemData 的结果，
不包含任何重优化算子（重优化算子属于 1 号动态调度器）。

新增/变更地址的节点距离采用「欧氏距离 × 路网曲折系数 α」近似，
α 取既有距离矩阵路网距离/欧氏距离比值的中位数，见 model_spec Q3 章节。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import hypot
from pathlib import Path

import numpy as np

from src.model.domain import ProblemData
from src.model.evaluator import format_clock


class Q3EventType(str, Enum):
    CANCEL = "cancel"
    NEW_ORDER = "new_order"
    ADDRESS_CHANGE = "address_change"
    TIME_WINDOW_CHANGE = "time_window_change"

    @property
    def label(self) -> str:
        return {
            "cancel": "订单取消",
            "new_order": "新增订单",
            "address_change": "配送地址变更",
            "time_window_change": "时间窗调整",
        }[self.value]


# 既有距离矩阵路网/欧氏距离比值中位数（含配送中心），由 estimate_detour_factor 计算，
# 作为新增/变更地址路段的距离折算系数。见 model_spec Q3 章节与问题三文档。
EUCLID_DETOUR_FACTOR = 1.434


@dataclass(frozen=True)
class Q3Event:
    """单一动态事件。

    各事件类型使用字段：
    - CANCEL：customer_id。
    - NEW_ORDER：customer_id；若该客户已存在则 weight_kg/volume_m3 为新增量，
      否则为新节点，还需 new_x_km/new_y_km 与 window_start/end_minutes。
    - ADDRESS_CHANGE：customer_id、new_x_km、new_y_km。
    - TIME_WINDOW_CHANGE：customer_id、window_start_minutes、window_end_minutes。
    """

    event_type: Q3EventType
    customer_id: int
    trigger_time_minutes: float
    weight_kg: float | None = None
    volume_m3: float | None = None
    window_start_minutes: float | None = None
    window_end_minutes: float | None = None
    new_x_km: float | None = None
    new_y_km: float | None = None
    severity: str = "medium"  # low | medium | high，用于事件强度分级

    def validate(self, problem: ProblemData) -> str | None:
        if self.severity not in ("low", "medium", "high"):
            return f"事件 {self.customer_id} 严重度必须为 low/medium/high"
        if self.event_type == Q3EventType.CANCEL:
            if self.customer_id not in problem.demands:
                return f"取消客户 {self.customer_id} 不存在或已无需求"
            return None
        if self.event_type == Q3EventType.NEW_ORDER:
            if self.weight_kg is None or self.volume_m3 is None:
                return f"新增订单客户 {self.customer_id} 缺少重量/体积"
            if self.weight_kg <= 0 or self.volume_m3 <= 0:
                return f"新增订单客户 {self.customer_id} 重量/体积必须为正"
            if self.customer_id not in problem.all_customer_ids:
                if (
                    self.new_x_km is None
                    or self.new_y_km is None
                    or self.window_start_minutes is None
                    or self.window_end_minutes is None
                ):
                    return f"新增订单客户 {self.customer_id} 为新节点，缺少坐标或时间窗"
                if self.window_end_minutes < self.window_start_minutes:
                    return f"新增订单客户 {self.customer_id} 时间窗颠倒"
            return None
        if self.event_type == Q3EventType.ADDRESS_CHANGE:
            if self.new_x_km is None or self.new_y_km is None:
                return f"地址变更客户 {self.customer_id} 缺少新坐标"
            if self.customer_id not in problem.all_customer_ids:
                return f"地址变更客户 {self.customer_id} 不在已知客户集合"
            return None
        if self.event_type == Q3EventType.TIME_WINDOW_CHANGE:
            if self.window_start_minutes is None or self.window_end_minutes is None:
                return f"时间窗调整客户 {self.customer_id} 缺少新窗口"
            if self.window_end_minutes < self.window_start_minutes:
                return f"时间窗调整客户 {self.customer_id} 新窗口颠倒"
            if self.customer_id not in problem.all_customer_ids:
                return f"时间窗调整客户 {self.customer_id} 不在已知客户集合"
            return None
        return None

    def describe(self, problem: ProblemData | None = None) -> str:
        clock = format_clock(self.trigger_time_minutes)
        suffix = f"（{self.severity}）"
        if self.event_type == Q3EventType.CANCEL:
            return f"{clock} 取消客户 {self.customer_id} 的全部当日订单{suffix}"
        if self.event_type == Q3EventType.NEW_ORDER:
            if problem is None or self.customer_id not in problem.all_customer_ids:
                node = f"客户 {self.customer_id}（新节点）"
            else:
                node = f"客户 {self.customer_id}（已有节点）"
            extra = ""
            if self.window_start_minutes is not None:
                extra = f"，窗[{format_clock(self.window_start_minutes)},{format_clock(self.window_end_minutes)}]"
            return (
                f"{clock} 新增订单：{node} {self.weight_kg:.1f}kg/{self.volume_m3:.2f}m³{extra}{suffix}"
            )
        if self.event_type == Q3EventType.ADDRESS_CHANGE:
            return (
                f"{clock} 客户 {self.customer_id} 配送地址变更为 "
                f"({self.new_x_km:.2f},{self.new_y_km:.2f}){suffix}"
            )
        return (
            f"{clock} 客户 {self.customer_id} 时间窗调整为 "
            f"[{format_clock(self.window_start_minutes)},{format_clock(self.window_end_minutes)}]{suffix}"
        )

    def old_value(self, problem: ProblemData | None = None) -> str:
        if problem is None:
            return "-"
        if self.event_type == Q3EventType.CANCEL:
            if self.customer_id in problem.demands:
                weight, volume = problem.demands[self.customer_id]
                return f"active {weight:.1f}kg/{volume:.2f}m³"
            return "active"
        if self.event_type == Q3EventType.NEW_ORDER:
            if self.customer_id in problem.demands:
                weight, volume = problem.demands[self.customer_id]
                return f"{weight:.1f}kg/{volume:.2f}m³"
            return "—（新客户）"
        if self.event_type == Q3EventType.ADDRESS_CHANGE:
            if self.customer_id in problem.coordinates:
                x, y = problem.coordinates[self.customer_id]
                return f"({x:.2f},{y:.2f})"
            return "-"
        if self.event_type == Q3EventType.TIME_WINDOW_CHANGE:
            if self.customer_id in problem.windows:
                start, end = problem.windows[self.customer_id]
                return f"[{format_clock(start)},{format_clock(end)}]"
            return "-"
        return "-"

    def new_value(self) -> str:
        if self.event_type == Q3EventType.CANCEL:
            return "0（取消）"
        if self.event_type == Q3EventType.NEW_ORDER:
            extra = ""
            if self.window_start_minutes is not None:
                extra = f"，窗[{format_clock(self.window_start_minutes)},{format_clock(self.window_end_minutes)}]"
            coord = (
                f"，坐标({self.new_x_km:.2f},{self.new_y_km:.2f})"
                if self.new_x_km is not None
                else ""
            )
            return f"{self.weight_kg:.1f}kg/{self.volume_m3:.2f}m³{extra}{coord}"
        if self.event_type == Q3EventType.ADDRESS_CHANGE:
            return f"({self.new_x_km:.2f},{self.new_y_km:.2f})"
        return (
            f"[{format_clock(self.window_start_minutes)},{format_clock(self.window_end_minutes)}]"
        )


@dataclass(frozen=True)
class Q3EventSet:
    """同一触发时刻到达的一组动态事件（单时刻多事件口径）。"""

    trigger_time_minutes: float
    events: tuple[Q3Event, ...]
    description: str = ""

    def validate(self, problem: ProblemData) -> list[str]:
        errors = []
        for event in self.events:
            if abs(event.trigger_time_minutes - self.trigger_time_minutes) > 1e-9:
                errors.append(
                    f"事件 {event.describe(problem)} 触发时刻与事件集不一致"
                )
            error = event.validate(problem)
            if error:
                errors.append(error)
        return errors


def estimate_detour_factor(problem: ProblemData) -> float:
    """既有距离矩阵（含配送中心）节点对的路网/欧氏距离比值中位数。"""
    matrix_node_ids = sorted(problem.coordinates)  # 含配送中心 ID 0，与距离矩阵行序一致
    coords = problem.coordinates
    xs = np.array([coords[cid][0] for cid in matrix_node_ids])
    ys = np.array([coords[cid][1] for cid in matrix_node_ids])
    euclid = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
    road = problem.distance
    mask = ~np.eye(len(matrix_node_ids), dtype=bool)
    ratio = road[mask] / euclid[mask]
    return float(np.median(ratio))


def _scaled_distance(alpha: float, x1: float, y1: float, x2: float, y2: float) -> float:
    return hypot(x2 - x1, y2 - y1) * alpha


def apply_events(
    problem: ProblemData,
    events: Q3EventSet | tuple[Q3Event, ...],
    detour_factor: float = EUCLID_DETOUR_FACTOR,
) -> tuple[ProblemData, dict[str, object]]:
    """把事件应用到问题数据，返回事件后 ProblemData 与变更审计信息。

    距离口径：新增/变更地址节点到其他所有节点的路段距离用欧氏距离 × α 近似；
    未变节点之间的路段距离保持原距离矩阵。绿色区归属按新坐标到市中心的欧氏
    距离重判（半径 10 km），未变节点沿用原标记。
    """
    events_seq = events.events if isinstance(events, Q3EventSet) else tuple(events)
    if not events_seq:
        return problem, {"changed_node_ids": [], "removed_customer_ids": []}

    coordinates = dict(problem.coordinates)
    demands = {cid: (float(w), float(v)) for cid, (w, v) in problem.demands.items()}
    windows = dict(problem.windows)
    green = set(problem.green_customer_ids)
    radius = problem.green_zone_radius_km

    original_ids = list(problem.all_customer_ids)
    changed_nodes: set[int] = set()
    removed_customers: set[int] = set()

    for event in events_seq:
        error = event.validate(problem)
        if error:
            raise ValueError(error)
        if event.event_type == Q3EventType.CANCEL:
            demands.pop(event.customer_id, None)
            removed_customers.add(event.customer_id)
        elif event.event_type == Q3EventType.NEW_ORDER:
            assert event.weight_kg is not None and event.volume_m3 is not None
            if event.customer_id in demands:
                old_w, old_v = demands[event.customer_id]
                demands[event.customer_id] = (
                    old_w + event.weight_kg,
                    old_v + event.volume_m3,
                )
            else:
                if event.customer_id in coordinates:
                    # 已有坐标/窗口的无需求客户（如绿色区客户1/14/15），只补需求
                    demands[event.customer_id] = (event.weight_kg, event.volume_m3)
                else:
                    assert event.new_x_km is not None and event.new_y_km is not None
                    coordinates[event.customer_id] = (event.new_x_km, event.new_y_km)
                    windows[event.customer_id] = (
                        event.window_start_minutes,
                        event.window_end_minutes,
                    )
                    demands[event.customer_id] = (event.weight_kg, event.volume_m3)
                    changed_nodes.add(event.customer_id)
                if (
                    event.customer_id not in green
                    and hypot(event.new_x_km if event.new_x_km is not None else coordinates[event.customer_id][0],
                              event.new_y_km if event.new_y_km is not None else coordinates[event.customer_id][1])
                    <= radius + 1e-9
                ):
                    green.add(event.customer_id)
        elif event.event_type == Q3EventType.ADDRESS_CHANGE:
            assert event.new_x_km is not None and event.new_y_km is not None
            coordinates[event.customer_id] = (event.new_x_km, event.new_y_km)
            changed_nodes.add(event.customer_id)
            new_d = hypot(event.new_x_km, event.new_y_km)
            if new_d <= radius + 1e-9:
                green.add(event.customer_id)
            else:
                green.discard(event.customer_id)
        elif event.event_type == Q3EventType.TIME_WINDOW_CHANGE:
            assert event.window_start_minutes is not None and event.window_end_minutes is not None
            windows[event.customer_id] = (
                event.window_start_minutes,
                event.window_end_minutes,
            )

    # 重建距离矩阵：按完整节点ID（含配送中心0）索引，评估器用原始ID直接下标
    original_node_ids = tuple(sorted(problem.coordinates))
    node_ids = tuple(sorted(coordinates))  # 含配送中心0与新增节点
    original_index = {cid: index for index, cid in enumerate(original_node_ids)}
    n = len(node_ids)
    new_distance = np.zeros((n, n), dtype=float)
    for i, a in enumerate(node_ids):
        for j, b in enumerate(node_ids):
            if a in changed_nodes or b in changed_nodes:
                ax, ay = coordinates[a]
                bx, by = coordinates[b]
                new_distance[i, j] = _scaled_distance(detour_factor, ax, ay, bx, by)
            else:
                new_distance[i, j] = float(
                    problem.distance[original_index[a], original_index[b]]
                )

    all_customer_ids = tuple(cid for cid in node_ids if cid != 0)
    green_ids = frozenset(
        cid for cid in all_customer_ids
        if cid in green or hypot(*coordinates[cid]) <= radius + 1e-9
    )

    problem_after = ProblemData(
        distance=new_distance,
        demands=demands,
        windows=windows,
        coordinates=coordinates,
        all_customer_ids=all_customer_ids,
        green_customer_ids=green_ids,
        green_zone_radius_km=radius,
        imputed_weight_rows=problem.imputed_weight_rows,
        imputed_volume_rows=problem.imputed_volume_rows,
        data_source=f"{problem.data_source} | Q3动态事件后",
        missing_value_policy=problem.missing_value_policy,
    )
    audit: dict[str, object] = {
        "changed_node_ids": sorted(changed_nodes),
        "removed_customer_ids": sorted(removed_customers),
        "detour_factor": detour_factor,
        "green_customer_ids_after": sorted(green_ids),
    }
    return problem_after, audit


def write_event_set_csv(
    eventset: Q3EventSet,
    path: str | Path,
    problem: ProblemData | None = None,
) -> None:
    import pandas as pd

    rows = []
    for index, event in enumerate(eventset.events, start=1):
        rows.append(
            {
                "event_id": f"E{index:02d}",
                "event_type": event.event_type.value,
                "event_type_label": event.event_type.label,
                "customer_id": event.customer_id,
                "trigger_time_minutes": event.trigger_time_minutes,
                "trigger_clock": format_clock(event.trigger_time_minutes),
                "severity": event.severity,
                "old_value": event.old_value(problem),
                "new_value": event.new_value(),
                "weight_kg": event.weight_kg,
                "volume_m3": event.volume_m3,
                "window_start_minutes": event.window_start_minutes,
                "window_end_minutes": event.window_end_minutes,
                "new_x_km": event.new_x_km,
                "new_y_km": event.new_y_km,
                "description": event.describe(problem),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
