from __future__ import annotations

from math import ceil
from pathlib import Path

import pandas as pd

from src.data.loader import load_problem_data


def build_q2_instance(data_dir: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    data_path = Path(data_dir)
    problem = load_problem_data(data_path)
    cleaned_coordinates = data_path / "客户坐标信息_清洗后.xlsx"
    cleaned_demands = data_path / "客户需求汇总_清洗后.xlsx"
    cleaned_orders = data_path / "订单信息_清洗后.xlsx"
    cleaned_windows = data_path / "时间窗_清洗后.xlsx"

    if cleaned_coordinates.exists():
        coordinate_frame = pd.read_excel(cleaned_coordinates)
        coordinate_frame = coordinate_frame.rename(
            columns={"X_km": "x_km", "Y_km": "y_km"}
        )
    else:
        coordinate_frame = pd.read_excel(data_path / "客户坐标信息.xlsx")
        coordinate_frame = coordinate_frame.rename(
            columns={"X (km)": "x_km", "Y (km)": "y_km"}
        )

    if "是否绿色配送区" in coordinate_frame:
        coordinate_frame["is_green_zone"] = coordinate_frame["是否绿色配送区"].astype(int)
    else:
        coordinate_frame["is_green_zone"] = (
            (coordinate_frame["x_km"] ** 2 + coordinate_frame["y_km"] ** 2) ** 0.5
            <= 10.0 + 1e-9
        ).astype(int)
    coordinate_frame["distance_to_center_km"] = (
        coordinate_frame["x_km"] ** 2 + coordinate_frame["y_km"] ** 2
    ) ** 0.5

    if cleaned_demands.exists():
        demand_frame = pd.read_excel(cleaned_demands)
        demand_frame = demand_frame.rename(
            columns={
                "客户编号": "customer_id",
                "订单数": "order_count",
                "总重量_kg": "total_weight_kg",
                "总体积_m3": "total_volume_m3",
                "是否待服务": "needs_service",
                "需车次下界": "trip_lower_bound",
                "开始时间_分钟": "window_start_minutes",
                "结束时间_分钟": "window_end_minutes",
            }
        )
    else:
        orders = pd.read_excel(data_path / "订单信息.xlsx")
        demand_frame = (
            orders.groupby("目标客户编号", as_index=False)
            .agg(
                order_count=("订单编号", "count"),
                total_weight_kg=("重量", "sum"),
                total_volume_m3=("体积", "sum"),
            )
            .rename(columns={"目标客户编号": "customer_id"})
        )
        demand_frame["needs_service"] = 1
        demand_frame["trip_lower_bound"] = 0

    if cleaned_windows.exists():
        window_frame = pd.read_excel(cleaned_windows).rename(
            columns={
                "客户编号": "customer_id",
                "开始时间_分钟": "window_start_minutes",
                "结束时间_分钟": "window_end_minutes",
                "时间窗宽度_分钟": "window_width_minutes",
            }
        )
    else:
        window_frame = pd.read_excel(data_path / "时间窗.xlsx").rename(
            columns={"客户编号": "customer_id"}
        )

    if cleaned_orders.exists():
        total_orders = int(len(pd.read_excel(cleaned_orders)))
    else:
        total_orders = int(demand_frame["order_count"].sum())

    demand_columns = [
        "customer_id",
        "order_count",
        "total_weight_kg",
        "total_volume_m3",
        "needs_service",
        "trip_lower_bound",
    ]
    demand_view = demand_frame[[column for column in demand_columns if column in demand_frame]]
    window_columns = [
        "customer_id",
        "window_start_minutes",
        "window_end_minutes",
        "window_width_minutes",
    ]
    window_view = window_frame[[column for column in window_columns if column in window_frame]]
    coordinate_view = coordinate_frame[
        [
            "ID",
            "x_km",
            "y_km",
            "distance_to_center_km",
            "is_green_zone",
        ]
    ].rename(columns={"ID": "customer_id"})

    customer_frame = coordinate_view.merge(demand_view, on="customer_id", how="left")
    customer_frame = customer_frame.merge(window_view, on="customer_id", how="left")
    customer_frame["has_order"] = customer_frame["order_count"].fillna(0).gt(0).astype(int)
    customer_frame["needs_service"] = customer_frame["needs_service"].fillna(0).astype(int)
    customer_frame["is_active_green_customer"] = (
        (customer_frame["is_green_zone"] == 1)
        & (customer_frame["needs_service"] == 1)
    ).astype(int)
    customer_frame = customer_frame[customer_frame["customer_id"] != 0].copy()
    customer_frame = customer_frame.sort_values("customer_id").reset_index(drop=True)

    active = customer_frame[customer_frame["needs_service"] == 1]
    green = customer_frame[customer_frame["is_green_zone"] == 1]
    active_green = customer_frame[customer_frame["is_active_green_customer"] == 1]
    max_ev_weight = 3000.0
    max_ev_volume = 15.0
    summary: dict[str, object] = {
        "data_source": str(data_path.resolve()),
        "customer_count": int(len(customer_frame)),
        "active_customer_count": int(len(active)),
        "total_order_count": total_orders,
        "total_weight_kg": float(active["total_weight_kg"].fillna(0).sum()),
        "total_volume_m3": float(active["total_volume_m3"].fillna(0).sum()),
        "green_customer_count": int(len(green)),
        "active_green_customer_count": int(len(active_green)),
        "green_customer_ids": [int(x) for x in green["customer_id"]],
        "active_green_customer_ids": [int(x) for x in active_green["customer_id"]],
        "green_order_count": int(green["order_count"].fillna(0).sum()),
        "green_weight_kg": float(green["total_weight_kg"].fillna(0).sum()),
        "green_volume_m3": float(green["total_volume_m3"].fillna(0).sum()),
        "restricted_period": "08:00-16:00",
        "policy_primary_interpretation": "燃油车在绿色客户节点的到达时刻采用[08:00,16:00)判定",
        "green_trip_capacity_lower_bound": max(
            ceil(float(active_green["total_weight_kg"].fillna(0).sum()) / max_ev_weight),
            ceil(float(active_green["total_volume_m3"].fillna(0).sum()) / max_ev_volume),
        ),
        "statement_green_customer_count": 30,
        "green_customer_count_audit": (
            "清洗坐标标记与题面一致"
            if len(green) == 30
            else "清洗坐标标记与题面30个不一致，以清洗文件标记为模型输入"
        ),
    }
    return customer_frame, summary
