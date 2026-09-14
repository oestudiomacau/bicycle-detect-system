from __future__ import annotations

import random
from dataclasses import dataclass

from PySide6.QtCore import QObject, QTimer, Signal

from .domain import MonitorEvent, ParkingRuleService, SpeedMeasurementService


@dataclass(slots=True)
class RoadTrack:
    track_id: int
    x: float
    y: float
    target_speed_kmh: float
    crossed_a_at: float | None = None
    measured_speed_kmh: float | None = None
    emitted: bool = False


@dataclass(slots=True)
class ParkedVehicle:
    track_id: int
    rect: tuple[float, float, float, float]
    angle_deg: float = 0.0
    nearby_count: int = 1
    anomaly_score: float = 0.0
    violation: str | None = None


class SimulationEngine(QObject):
    frame_changed = Signal()
    event_raised = Signal(object)
    metrics_changed = Signal(dict)

    LINE_A = 0.30
    LINE_B = 0.68

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.mode = "road"
        self.running = True
        self.elapsed_s = 0.0
        self.speed_service = SpeedMeasurementService()
        self.parking_service = ParkingRuleService()
        self.road_tracks = [
            RoadTrack(101, -0.10, 0.37, 10.5),
            RoadTrack(102, 0.02, 0.56, 17.8),
            RoadTrack(103, 0.18, 0.72, 22.4),
        ]
        self.parked_vehicles = self._build_parking_scene()
        self._parking_events_emitted = False
        self._last_tick_ms = 33
        self._timer = QTimer(self)
        self._timer.setInterval(self._last_tick_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_mode(self, mode: str) -> None:
        if mode not in {"road", "parking"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        if mode == "parking":
            self._parking_events_emitted = False
        self.frame_changed.emit()
        self._emit_metrics()

    def set_running(self, running: bool) -> None:
        self.running = running

    def set_speed_config(self, distance_m: float, threshold_kmh: float) -> None:
        self.speed_service.distance_m = distance_m
        self.speed_service.threshold_kmh = threshold_kmh

    def reset_scene(self) -> None:
        self.elapsed_s = 0.0
        self.road_tracks = [
            RoadTrack(101, -0.10, 0.37, 10.5),
            RoadTrack(102, 0.02, 0.56, 17.8),
            RoadTrack(103, 0.18, 0.72, 22.4),
        ]
        self.parked_vehicles = self._build_parking_scene()
        self._parking_events_emitted = False
        self.frame_changed.emit()

    def _tick(self) -> None:
        if not self.running:
            return
        dt = self._last_tick_ms / 1000.0
        self.elapsed_s += dt
        if self.mode == "road":
            self._update_road(dt)
        else:
            self._update_parking()
        self.frame_changed.emit()
        self._emit_metrics()

    def _update_road(self, dt: float) -> None:
        line_gap = self.LINE_B - self.LINE_A
        for track in self.road_tracks:
            travel_time = self.speed_service.distance_m / (track.target_speed_kmh / 3.6)
            previous_x = track.x
            track.x += line_gap / travel_time * dt

            if previous_x < self.LINE_A <= track.x:
                track.crossed_a_at = self.elapsed_s
                track.measured_speed_kmh = None
                track.emitted = False

            if (
                previous_x < self.LINE_B <= track.x
                and track.crossed_a_at is not None
                and not track.emitted
            ):
                speed = self.speed_service.calculate_kmh(track.crossed_a_at, self.elapsed_s)
                track.measured_speed_kmh = speed
                track.emitted = True
                violation = self.speed_service.is_violation(speed)
                self.event_raised.emit(
                    MonitorEvent(
                        event_type="超速告警" if violation else "速度记录",
                        source="道路观察位",
                        detail=f"目标 #{track.track_id} 通过双线测速区",
                        severity="warning" if violation else "info",
                        value=f"{speed:.1f} km/h",
                        status="待确认" if violation else "已记录",
                    )
                )

            if track.x > 1.12:
                track.x = random.uniform(-0.24, -0.08)
                track.target_speed_kmh = random.choice([9.8, 12.6, 16.4, 20.2, 23.5])
                track.crossed_a_at = None
                track.measured_speed_kmh = None
                track.emitted = False

    def _update_parking(self) -> None:
        if self._parking_events_emitted or self.elapsed_s < 0.6:
            return
        for vehicle in self.parked_vehicles:
            if not vehicle.violation:
                continue
            self.event_raised.emit(
                MonitorEvent(
                    event_type=vehicle.violation,
                    source="停车观察位",
                    detail=f"停车目标 #{vehicle.track_id} 状态异常",
                    severity="warning",
                    value=f"异常分数 {vehicle.anomaly_score:.2f}" if vehicle.anomaly_score else "规则判断",
                )
            )
        self._parking_events_emitted = True

    def _build_parking_scene(self) -> list[ParkedVehicle]:
        raw = [
            ParkedVehicle(201, (0.24, 0.28, 0.14, 0.10)),
            ParkedVehicle(202, (0.44, 0.46, 0.14, 0.10)),
            ParkedVehicle(203, (0.76, 0.60, 0.16, 0.11)),
            ParkedVehicle(204, (0.52, 0.67, 0.18, 0.08), angle_deg=58, anomaly_score=0.91),
            ParkedVehicle(205, (0.28, 0.58, 0.15, 0.10), nearby_count=3, anomaly_score=0.73),
        ]
        for vehicle in raw:
            vehicle.violation = self.parking_service.classify(
                vehicle.rect,
                vehicle.angle_deg,
                vehicle.nearby_count,
                vehicle.anomaly_score,
            )
        return raw

    def _emit_metrics(self) -> None:
        measured = [
            track.measured_speed_kmh
            for track in self.road_tracks
            if track.measured_speed_kmh is not None
        ]
        parking_violations = sum(1 for vehicle in self.parked_vehicles if vehicle.violation)
        self.metrics_changed.emit(
            {
                "active_tracks": len(self.road_tracks) if self.mode == "road" else len(self.parked_vehicles),
                "current_speed": max(measured, default=0.0),
                "parking_violations": parking_violations,
            }
        )

