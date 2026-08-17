from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


Propulsion = Literal["fuel", "electric"]


@dataclass(frozen=True)
class VehicleType:
    name: str
    propulsion: Propulsion
    capacity_weight: float
    capacity_volume: float
    count: int
    fixed_cost: float = 400.0


DEFAULT_VEHICLE_TYPES: tuple[VehicleType, ...] = (
    VehicleType("EV-3000", "electric", 3000.0, 15.0, 10),
    VehicleType("FUEL-3000", "fuel", 3000.0, 13.5, 60),
    VehicleType("FUEL-1500", "fuel", 1500.0, 10.8, 50),
    VehicleType("EV-1250", "electric", 1250.0, 8.5, 15),
    VehicleType("FUEL-1250", "fuel", 1250.0, 6.5, 50),
)


@dataclass(frozen=True)
class Delivery:
    customer_id: int
    weight: float
    volume: float


@dataclass
class Route:
    vehicle_type: VehicleType
    vehicle_number: int
    deliveries: list[Delivery] = field(default_factory=list)
    start_minutes: float = 8.0 * 60.0

    @property
    def route_id(self) -> str:
        return f"{self.vehicle_type.name}-{self.vehicle_number:03d}"

    @property
    def total_weight(self) -> float:
        return sum(item.weight for item in self.deliveries)

    @property
    def total_volume(self) -> float:
        return sum(item.volume for item in self.deliveries)


@dataclass(frozen=True)
class ProblemData:
    distance: np.ndarray
    demands: dict[int, tuple[float, float]]
    windows: dict[int, tuple[float, float]]
    coordinates: dict[int, tuple[float, float]]
    all_customer_ids: tuple[int, ...]
    imputed_weight_rows: int = 0
    imputed_volume_rows: int = 0

    @property
    def active_customer_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.demands))

