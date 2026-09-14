from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_PARKING_ZONE = (0.16, 0.18, 0.82, 0.78)
DEFAULT_PARKING_POLYGON = (
    (0.16, 0.18),
    (0.82, 0.18),
    (0.82, 0.78),
    (0.16, 0.78),
)
DEFAULT_SPEED_LINES = (
    (0.30, 0.05, 0.30, 0.95),
    (0.68, 0.05, 0.68, 0.95),
)

Point = tuple[float, float]
LineSegment = tuple[float, float, float, float]


def validate_parking_zone(
    zone: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if len(zone) != 4:
        raise ValueError("停车区域必须包含 left、top、right、bottom 四个坐标")
    left, top, right, bottom = (float(value) for value in zone)
    if not all(0.0 <= value <= 1.0 for value in (left, top, right, bottom)):
        raise ValueError("停车区域坐标必须在 0 到 1 之间")
    if right - left < 0.02 or bottom - top < 0.02:
        raise ValueError("停车区域过小，请重新拖拽一个更大的区域")
    return left, top, right, bottom


def validate_line_segment(line: LineSegment) -> LineSegment:
    if len(line) != 4:
        raise ValueError("线段必须包含 x1、y1、x2、y2 四个坐标")
    x1, y1, x2, y2 = (float(value) for value in line)
    if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        raise ValueError("线段坐标必须在 0 到 1 之间")
    if math.hypot(x2 - x1, y2 - y1) < 0.03:
        raise ValueError("线段过短，请重新绘制")
    return x1, y1, x2, y2


def validate_speed_lines(
    lines: tuple[LineSegment, LineSegment],
) -> tuple[LineSegment, LineSegment]:
    if len(lines) != 2:
        raise ValueError("测速需要虚拟线 A、B 两条线")
    line_a, line_b = (validate_line_segment(line) for line in lines)
    return line_a, line_b


def validate_parking_polygon(points: Iterable[Point]) -> tuple[Point, ...]:
    polygon = tuple((float(x), float(y)) for x, y in points)
    if not 3 <= len(polygon) <= 12:
        raise ValueError("停车边界需要 3 到 12 个顶点")
    if not all(0.0 <= value <= 1.0 for point in polygon for value in point):
        raise ValueError("停车边界坐标必须在 0 到 1 之间")
    area = abs(
        math.fsum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1], strict=True)
        )
    ) / 2
    if area < 0.005:
        raise ValueError("停车边界面积过小，请重新绘制")
    return polygon


def segments_intersect(start: Point, end: Point, line: LineSegment) -> bool:
    """Return whether a tracked movement segment crosses a configured line segment."""

    line_start = (line[0], line[1])
    line_end = (line[2], line[3])

    def orientation(a: Point, b: Point, c: Point) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: Point, b: Point, point: Point) -> bool:
        epsilon = 1e-9
        return (
            min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
            and min(a[1], b[1]) - epsilon <= point[1] <= max(a[1], b[1]) + epsilon
        )

    o1 = orientation(start, end, line_start)
    o2 = orientation(start, end, line_end)
    o3 = orientation(line_start, line_end, start)
    o4 = orientation(line_start, line_end, end)
    epsilon = 1e-9

    if ((o1 > epsilon and o2 < -epsilon) or (o1 < -epsilon and o2 > epsilon)) and (
        (o3 > epsilon and o4 < -epsilon) or (o3 < -epsilon and o4 > epsilon)
    ):
        return True
    return (
        (abs(o1) <= epsilon and on_segment(start, end, line_start))
        or (abs(o2) <= epsilon and on_segment(start, end, line_end))
        or (abs(o3) <= epsilon and on_segment(line_start, line_end, start))
        or (abs(o4) <= epsilon and on_segment(line_start, line_end, end))
    )


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        cross = (y1 > y) != (y2 > y)
        if cross and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        previous = current
    return inside


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
    def __init__(
        self,
        distance_m: float = 12.0,
        threshold_kmh: float = 15.0,
        lines: tuple[LineSegment, LineSegment] = DEFAULT_SPEED_LINES,
    ) -> None:
        self.distance_m = distance_m
        self.threshold_kmh = threshold_kmh
        self.lines = validate_speed_lines(lines)

    def set_lines(self, lines: tuple[LineSegment, LineSegment]) -> None:
        self.lines = validate_speed_lines(lines)

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

    def __init__(
        self,
        zone: tuple[float, float, float, float] = DEFAULT_PARKING_ZONE,
    ) -> None:
        self.set_zone(zone)

    def set_zone(self, zone: tuple[float, float, float, float]) -> None:
        left, top, right, bottom = validate_parking_zone(zone)
        self.polygon = validate_parking_polygon(
            ((left, top), (right, top), (right, bottom), (left, bottom))
        )

    def set_polygon(self, points: Iterable[Point]) -> None:
        self.polygon = validate_parking_polygon(points)

    @property
    def zone(self) -> tuple[float, float, float, float]:
        xs = [point[0] for point in self.polygon]
        ys = [point[1] for point in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)

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
        if width <= 0 or height <= 0:
            return 0.0
        samples_per_axis = 12
        inside = 0
        for row in range(samples_per_axis):
            sample_y = y + height * (row + 0.5) / samples_per_axis
            for column in range(samples_per_axis):
                sample_x = x + width * (column + 0.5) / samples_per_axis
                if point_in_polygon((sample_x, sample_y), self.polygon):
                    inside += 1
        return inside / (samples_per_axis * samples_per_axis)


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
