from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from math import hypot

import numpy as np
import pandas as pd

from src.model.domain import ProblemData


def _time_to_minutes(value: object) -> float:
    if isinstance(value, str):
        hours, minutes = value.strip().split(":")[:2]
        return int(hours) * 60.0 + int(minutes)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.hour * 60.0 + value.minute + value.second / 60.0
    if isinstance(value, time):
        return value.hour * 60.0 + value.minute + value.second / 60.0
    if isinstance(value, (float, np.floating)) and 0.0 <= value < 1.0:
        return float(value) * 24.0 * 60.0
    raise ValueError(f"无法识别时间值: {value!r}")


def _impute_orders(orders: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    result = orders.copy()
    result["目标客户编号"] = pd.to_numeric(result["目标客户编号"], errors="raise").astype(int)

    missing_counts: dict[str, int] = {}
    for column in ("重量", "体积"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        missing_counts[column] = int(result[column].isna().sum())
        by_customer = result.groupby("目标客户编号")[column].transform("median")
        result[column] = result[column].fillna(by_customer)
        result[column] = result[column].fillna(result[column].median())
        if result[column].isna().any():
            raise ValueError(f"{column} 缺失值无法完成中位数填补")
        if (result[column] < 0).any():
            raise ValueError(f"{column} 中存在负数")

    return result, missing_counts["重量"], missing_counts["体积"]


def load_problem_data(data_dir: str | Path) -> ProblemData:
    data_path = Path(data_dir)
    cleaned_files = {
        "orders": data_path / "订单信息_清洗后.xlsx",
        "demand_summary": data_path / "客户需求汇总_清洗后.xlsx",
        "distance": data_path / "距离矩阵_清洗后.xlsx",
        "windows": data_path / "时间窗_清洗后.xlsx",
        "coordinates": data_path / "客户坐标信息_清洗后.xlsx",
    }
    use_cleaned = all(path.exists() for path in cleaned_files.values())
    required = cleaned_files if use_cleaned else {
        "orders": data_path / "订单信息.xlsx",
        "distance": data_path / "距离矩阵.xlsx",
        "windows": data_path / "时间窗.xlsx",
        "coordinates": data_path / "客户坐标信息.xlsx",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少数据文件: {missing}")

    if use_cleaned:
        orders = pd.read_excel(required["orders"])
        missing_weight = int(orders["重量_是否填充"].sum())
        missing_volume = int(orders["体积_是否填充"].sum())
        demand_frame = pd.read_excel(required["demand_summary"])
        demands = {
            int(row["客户编号"]): (
                float(row["总重量_kg"]),
                float(row["总体积_m3"]),
            )
            for _, row in demand_frame.iterrows()
            if int(row["是否待服务"]) == 1
        }
        window_frame = pd.read_excel(required["windows"])
        windows = {
            int(row["客户编号"]): (
                float(row["开始时间_分钟"]),
                float(row["结束时间_分钟"]),
            )
            for _, row in window_frame.iterrows()
        }
        coordinate_frame = pd.read_excel(required["coordinates"])
        coordinates = {
            int(row["ID"]): (float(row["X_km"]), float(row["Y_km"]))
            for _, row in coordinate_frame.iterrows()
        }
        green_customer_ids = frozenset(
            int(row["ID"])
            for _, row in coordinate_frame.iterrows()
            if int(row.get("是否绿色配送区", 0)) == 1
        )
        data_source = str(data_path.resolve())
        missing_value_policy = "队友清洗数据：同客户均值填补"
    else:
        raw_orders = pd.read_excel(required["orders"])
        orders, missing_weight, missing_volume = _impute_orders(raw_orders)
        demand_frame = orders.groupby("目标客户编号", as_index=True)[["重量", "体积"]].sum()
        demands = {
            int(customer_id): (float(row["重量"]), float(row["体积"]))
            for customer_id, row in demand_frame.iterrows()
            if float(row["重量"]) > 1e-9 or float(row["体积"]) > 1e-9
        }
        window_frame = pd.read_excel(required["windows"])
        windows = {
            int(row["客户编号"]): (
                _time_to_minutes(row["开始时间"]),
                _time_to_minutes(row["结束时间"]),
            )
            for _, row in window_frame.iterrows()
        }
        coordinate_frame = pd.read_excel(required["coordinates"])
        coordinates = {
            int(row["ID"]): (float(row["X (km)"]), float(row["Y (km)"]))
            for _, row in coordinate_frame.iterrows()
        }
        green_customer_ids = frozenset(
            customer_id
            for customer_id, (x_km, y_km) in coordinates.items()
            if customer_id != 0 and hypot(x_km, y_km) <= 10.0 + 1e-9
        )
        data_source = str(data_path.resolve())
        missing_value_policy = "运行时同客户中位数填补"

    if any(start > end for start, end in windows.values()):
        raise ValueError("存在开始时间晚于结束时间的时间窗")

    distance_frame = pd.read_excel(required["distance"], index_col=0)
    distance_frame.index = distance_frame.index.astype(int)
    distance_frame.columns = [int(column) for column in distance_frame.columns]
    node_ids = tuple(sorted(coordinates))
    if set(distance_frame.index) != set(node_ids) or set(distance_frame.columns) != set(node_ids):
        raise ValueError("距离矩阵节点与坐标文件节点不一致")
    distance = distance_frame.loc[node_ids, node_ids].to_numpy(dtype=float)
    if not np.isfinite(distance).all() or (distance < 0).any():
        raise ValueError("距离矩阵包含缺失值、无穷值或负值")
    if not np.allclose(distance, distance.T, atol=1e-8):
        raise ValueError("当前基线求解器要求对称距离矩阵")
    if not np.allclose(np.diag(distance), 0.0, atol=1e-8):
        raise ValueError("距离矩阵对角线必须为 0")

    all_customer_ids = tuple(sorted(customer_id for customer_id in coordinates if customer_id != 0))
    unknown = set(demands) - set(all_customer_ids)
    if unknown:
        raise ValueError(f"订单中出现未知客户: {sorted(unknown)}")
    missing_windows = set(all_customer_ids) - set(windows)
    if missing_windows:
        raise ValueError(f"客户缺少时间窗: {sorted(missing_windows)}")

    return ProblemData(
        distance=distance,
        demands=demands,
        windows=windows,
        coordinates=coordinates,
        all_customer_ids=all_customer_ids,
        green_customer_ids=green_customer_ids,
        green_zone_radius_km=10.0,
        imputed_weight_rows=missing_weight,
        imputed_volume_rows=missing_volume,
        data_source=data_source,
        missing_value_policy=missing_value_policy,
    )
