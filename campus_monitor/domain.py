from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class MonitorEvent:
    event_type: str
    source: str
    detail: str
    severity: str = "info"
    value: str = "-"
    status: str = "待确认"
    occurred_at: str = ""
    snapshot: str = ""

    def __post_init__(self) -> None:
        if not self.occurred_at:
            self.occurred_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class SpeedMeasurementService:
    def __init__(self, distance_m: float = 12.0, threshold_kmh: float = 15.0) -> None:
        self.distance_m = distance_m
        self.threshold_kmh = threshold_kmh

    def calculate_kmh(self, crossed_a_at: float, crossed_b_at: float) -> float:
        elapsed = crossed_b_at - crossed_a_at
        if elapsed <= 0:
            raise ValueError("通过虚拟线 B 的时刻必须晚于虚拟线 A")
        if self.distance_m <= 0:
            raise ValueError("双线实际距离必须大于 0")
        return self.distance_m / elapsed * 3.6

    def is_violation(self, speed_kmh: float) -> bool:
        return speed_kmh > self.threshold_kmh


class ParkingRuleService:
    """Rule-based parking checks used before the Anomalib adapter is connected."""

    def __init__(self, zone: tuple[float, float, float, float] = (0.16, 0.18, 0.82, 0.78)) -> None:
        self.zone = zone

    def classify(
        self,
        rect: tuple[float, float, float, float],
        angle_deg: float = 0.0,
        nearby_count: int = 1,
        anomaly_score: float = 0.0,
    ) -> str | None:
        if abs(angle_deg) >= 45 or anomaly_score >= 0.82:
            return "倒地异常"
        if nearby_count >= 3 or anomaly_score >= 0.68:
            return "异常堆放"
        if self._inside_ratio(rect) < 0.9:
            return "越界停放"
        return None

    def _inside_ratio(self, rect: tuple[float, float, float, float]) -> float:
        x, y, width, height = rect
        left, top, right, bottom = self.zone
        intersection_w = max(0.0, min(x + width, right) - max(x, left))
        intersection_h = max(0.0, min(y + height, bottom) - max(y, top))
        area = width * height
        return intersection_w * intersection_h / area if area > 0 else 0.0


class EventStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: MonitorEvent) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def load(self, limit: int = 100) -> list[MonitorEvent]:
        if not self.path.exists():
            return []
        rows: list[MonitorEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(MonitorEvent(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
        return rows[-limit:]


def average(values: Iterable[float]) -> float:
    items = list(values)
    return math.fsum(items) / len(items) if items else 0.0

