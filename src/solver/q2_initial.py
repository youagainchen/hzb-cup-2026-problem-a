from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from src.model.domain import DEFAULT_VEHICLE_TYPES, Delivery, Route, VehicleType


def _clock_to_minutes(value: str) -> float:
    hours, minutes = value.split(":", maxsplit=1)
    return float(int(hours) * 60 + int(minutes))


def load_route_solution(
    route_csv: Path,
    summary_csv: Path,
    vehicle_types: Sequence[VehicleType] = DEFAULT_VEHICLE_TYPES,
) -> list[Route]:
    """从正式逐站表和趟次汇总表重建可评分的路线对象。"""

    vehicles = {vehicle.name: vehicle for vehicle in vehicle_types}
    with summary_csv.open(newline="", encoding="utf-8-sig") as stream:
        summaries = {
            row["route_id"]: row
            for row in csv.DictReader(stream)
        }

    deliveries_by_route: dict[str, list[Delivery]] = defaultdict(list)
    route_order: list[str] = []
    with route_csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = sorted(
            csv.DictReader(stream),
            key=lambda row: (row["route_id"], int(row["sequence"])),
        )
    for row in rows:
        route_id = row["route_id"]
        if route_id not in deliveries_by_route:
            route_order.append(route_id)
        deliveries_by_route[route_id].append(
            Delivery(
                customer_id=int(row["customer_id"]),
                weight=float(row["delivered_weight_kg"]),
                volume=float(row["delivered_volume_m3"]),
            )
        )

    routes: list[Route] = []
    for route_id in route_order:
        summary = summaries[route_id]
        vehicle_name = summary["vehicle_type"]
        if vehicle_name not in vehicles:
            raise ValueError(f"未知车型：{vehicle_name}")
        vehicle_id = summary["physical_vehicle_id"]
        vehicle_number = int(vehicle_id.removeprefix(f"{vehicle_name}-"))
        start = (
            float(summary["start_minutes_exact"])
            if summary.get("start_minutes_exact")
            else _clock_to_minutes(summary["start"])
        )
        routes.append(
            Route(
                vehicle_type=vehicles[vehicle_name],
                vehicle_number=vehicle_number,
                deliveries=deliveries_by_route[route_id],
                start_minutes=start,
                trip_number=int(summary["trip_number"]),
            )
        )
    return routes
