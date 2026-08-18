from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Q2Policy:
    """问题二绿色配送区限行政策。

    主口径为半开区间 [08:00, 16:00)：燃油车到达绿色客户节点即违规，
    新能源车不受该政策限制。距离矩阵没有道路级轨迹，因此政策落点按客户
    节点的到达时刻判定。
    """

    green_customer_ids: frozenset[int]
    restricted_start_minutes: float = 8.0 * 60.0
    restricted_end_minutes: float = 16.0 * 60.0
    end_inclusive: bool = False

    def is_restricted_clock(self, arrival_minutes: float) -> bool:
        if self.end_inclusive:
            return (
                self.restricted_start_minutes <= arrival_minutes
                <= self.restricted_end_minutes
            )
        return (
            self.restricted_start_minutes <= arrival_minutes
            < self.restricted_end_minutes
        )

    def violation_reason(
        self,
        propulsion: str,
        customer_id: int,
        arrival_minutes: float,
    ) -> str | None:
        if propulsion != "fuel":
            return None
        if customer_id not in self.green_customer_ids:
            return None
        if not self.is_restricted_clock(arrival_minutes):
            return None
        return "fuel_vehicle_in_green_zone_during_restricted_period"

    def violates(
        self,
        propulsion: str,
        customer_id: int,
        arrival_minutes: float,
    ) -> bool:
        return self.violation_reason(propulsion, customer_id, arrival_minutes) is not None


def build_q2_policy(
    green_customer_ids: frozenset[int] | set[int] | tuple[int, ...],
    *,
    end_inclusive: bool = False,
) -> Q2Policy:
    return Q2Policy(
        green_customer_ids=frozenset(green_customer_ids),
        end_inclusive=end_inclusive,
    )
