from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage

from .domain import MonitorEvent, ParkingRuleService, SpeedMeasurementService


@dataclass(slots=True)
class Detection:
    rect: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x, y, width, height = self.rect
        return x + width / 2, y + height / 2


@dataclass(slots=True)
class TrackedDetection:
    track_id: int
    rect: tuple[float, float, float, float]
    confidence: float
    speed_kmh: float | None = None
    violation: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        x, y, width, height = self.rect
        return x + width / 2, y + height / 2


class MobileNetBicycleDetector:
    BICYCLE_CLASS_ID = 2

    def __init__(self, prototxt: Path, weights: Path, confidence: float = 0.35) -> None:
        if not prototxt.exists() or not weights.exists():
            raise FileNotFoundError("MobileNet-SSD 模型文件不完整")
        # OpenCV's Windows file loader cannot reliably open paths containing Chinese text.
        self.net = cv2.dnn.readNetFromCaffe(
            np.frombuffer(prototxt.read_bytes(), dtype=np.uint8),
            np.frombuffer(weights.read_bytes(), dtype=np.uint8),
        )
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5,
        )
        self.net.setInput(blob)
        output = self.net.forward()
        detections: list[Detection] = []
        for index in range(output.shape[2]):
            confidence = float(output[0, 0, index, 2])
            class_id = int(output[0, 0, index, 1])
            if class_id != self.BICYCLE_CLASS_ID or confidence < self.confidence:
                continue
            x1, y1, x2, y2 = output[0, 0, index, 3:7]
            x1 = float(np.clip(x1, 0.0, 1.0))
            y1 = float(np.clip(y1, 0.0, 1.0))
            x2 = float(np.clip(x2, 0.0, 1.0))
            y2 = float(np.clip(y2, 0.0, 1.0))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(Detection((x1, y1, x2 - x1, y2 - y1), confidence))
        return detections


class CentroidTracker:
    def __init__(self, max_distance: float = 0.30, max_misses: int = 8) -> None:
        self.max_distance = max_distance
        self.max_misses = max_misses
        self._next_id = 1
        self._tracks: dict[int, TrackedDetection] = {}
        self._misses: dict[int, int] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()
        self._misses.clear()

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        unmatched_tracks = set(self._tracks)
        unmatched_detections = set(range(len(detections)))
        candidates: list[tuple[float, int, int]] = []

        for track_id, track in self._tracks.items():
            tx, ty = track.center
            for index, detection in enumerate(detections):
                dx, dy = detection.center
                distance = math.hypot(tx - dx, ty - dy)
                if distance <= self.max_distance:
                    candidates.append((distance, track_id, index))

        for _, track_id, detection_index in sorted(candidates):
            if track_id not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            detection = detections[detection_index]
            previous = self._tracks[track_id]
            self._tracks[track_id] = TrackedDetection(
                track_id,
                detection.rect,
                detection.confidence,
                previous.speed_kmh,
                previous.violation,
            )
            self._misses[track_id] = 0
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(detection_index)

        for track_id in unmatched_tracks:
            self._misses[track_id] = self._misses.get(track_id, 0) + 1
            if self._misses[track_id] > self.max_misses:
                self._tracks.pop(track_id, None)
                self._misses.pop(track_id, None)

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = TrackedDetection(track_id, detection.rect, detection.confidence)
            self._misses[track_id] = 0

        return [self._tracks[track_id] for track_id in sorted(self._tracks)]


class VideoAnalysisController(QObject):
    frame_ready = Signal(object, object, str)
    event_raised = Signal(object)
    metrics_changed = Signal(dict)
    status_changed = Signal(str)

    LINE_A = 0.30
    LINE_B = 0.68

    def __init__(self, prototxt: Path, weights: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.detector = MobileNetBicycleDetector(prototxt, weights)
        self.tracker = CentroidTracker()
        self.speed_service = SpeedMeasurementService()
        self.parking_service = ParkingRuleService()
        self.capture: cv2.VideoCapture | None = None
        self.path: Path | None = None
        self.mode = "road"
        self.running = False
        self.source_name = ""
        self.current_frame: QImage | None = None
        self.current_tracks: list[TrackedDetection] = []
        self._frame_index = 0
        self._detect_every = 2
        self._previous_centers: dict[int, float] = {}
        self._crossing_state: dict[int, tuple[str, float]] = {}
        self._completed_speed_ids: set[int] = set()
        self._parking_event_ids: set[int] = set()
        self._last_speed = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def open(self, path: Path) -> None:
        self.close()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频：{path}")
        self.capture = capture
        self.path = path
        self.source_name = path.name
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        self._timer.setInterval(max(20, min(100, round(1000 / fps))))
        self._reset_analysis_state()
        self.running = True
        self._timer.start()
        self.status_changed.emit(f"本地视频：{path.name}")

    def close(self) -> None:
        self._timer.stop()
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.running = False

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._reset_analysis_state(keep_frame=True)

    def set_running(self, running: bool) -> None:
        self.running = running
        if running and self.capture is not None:
            self._timer.start()
        else:
            self._timer.stop()

    def restart(self) -> None:
        if self.capture is None:
            return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._reset_analysis_state()
        self.set_running(True)

    def set_speed_config(self, distance_m: float, threshold_kmh: float) -> None:
        self.speed_service.distance_m = distance_m
        self.speed_service.threshold_kmh = threshold_kmh

    def _tick(self) -> None:
        if not self.running or self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._reset_analysis_state()
            ok, frame = self.capture.read()
            if not ok:
                self.set_running(False)
                self.status_changed.emit("视频读取结束")
                return

        self._frame_index += 1
        if self._frame_index % self._detect_every == 1 or not self.current_tracks:
            detections = self.detector.detect(frame)
            self.current_tracks = self.tracker.update(detections)

        timestamp_s = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if self.mode == "road":
            self._evaluate_speed(timestamp_s)
        else:
            self._evaluate_parking()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        self.current_frame = image
        self.frame_ready.emit(image, list(self.current_tracks), self.source_name)
        self.metrics_changed.emit(
            {
                "active_tracks": len(self.current_tracks),
                "current_speed": self._last_speed,
                "parking_violations": sum(1 for item in self.current_tracks if item.violation),
            }
        )

    def _evaluate_speed(self, timestamp_s: float) -> None:
        active_ids = {item.track_id for item in self.current_tracks}
        self._drop_missing_state(active_ids)
        for track in self.current_tracks:
            center_x, _ = track.center
            previous_x = self._previous_centers.get(track.track_id)
            self._previous_centers[track.track_id] = center_x
            if previous_x is None or track.track_id in self._completed_speed_ids:
                continue
            crossed_a = self._crossed(previous_x, center_x, self.LINE_A)
            crossed_b = self._crossed(previous_x, center_x, self.LINE_B)
            state = self._crossing_state.get(track.track_id)
            if state is None:
                if crossed_a:
                    self._crossing_state[track.track_id] = ("A", timestamp_s)
                elif crossed_b:
                    self._crossing_state[track.track_id] = ("B", timestamp_s)
                continue

            first_line, first_time = state
            completed = (first_line == "A" and crossed_b) or (first_line == "B" and crossed_a)
            if not completed or timestamp_s <= first_time:
                continue
            if timestamp_s - first_time < 0.5:
                # A large one-frame jump is usually an ID switch, not a bicycle crossing the zone.
                self._crossing_state[track.track_id] = (
                    "A" if crossed_a else "B",
                    timestamp_s,
                )
                continue
            speed = self.speed_service.calculate_kmh(first_time, timestamp_s)
            track.speed_kmh = speed
            self._last_speed = speed
            self._completed_speed_ids.add(track.track_id)
            violation = self.speed_service.is_violation(speed)
            self.event_raised.emit(
                MonitorEvent(
                    event_type="超速告警" if violation else "速度记录",
                    source="本地视频 · 道路观察位",
                    detail=f"自行车目标 #{track.track_id} 通过双线测速区",
                    severity="warning" if violation else "info",
                    value=f"{speed:.1f} km/h",
                    status="待确认" if violation else "已记录",
                )
            )

    def _evaluate_parking(self) -> None:
        active_ids = {item.track_id for item in self.current_tracks}
        self._parking_event_ids.intersection_update(active_ids)
        for track in self.current_tracks:
            violation = self.parking_service.classify(track.rect)
            track.violation = violation
            if not violation or track.track_id in self._parking_event_ids:
                continue
            self._parking_event_ids.add(track.track_id)
            self.event_raised.emit(
                MonitorEvent(
                    event_type=violation,
                    source="本地视频 · 停车观察位",
                    detail=f"自行车目标 #{track.track_id} 超出配置停车区域",
                    severity="warning",
                    value=f"置信度 {track.confidence:.2f}",
                )
            )

    def _drop_missing_state(self, active_ids: set[int]) -> None:
        for mapping in (self._previous_centers, self._crossing_state):
            for track_id in list(mapping):
                if track_id not in active_ids:
                    mapping.pop(track_id, None)
        self._completed_speed_ids.intersection_update(active_ids)

    def _reset_analysis_state(self, keep_frame: bool = False) -> None:
        self.tracker.reset()
        self.current_tracks = []
        self._frame_index = 0
        self._previous_centers.clear()
        self._crossing_state.clear()
        self._completed_speed_ids.clear()
        self._parking_event_ids.clear()
        self._last_speed = 0.0
        if not keep_frame:
            self.current_frame = None

    @staticmethod
    def _crossed(previous: float, current: float, line: float) -> bool:
        return (previous < line <= current) or (previous > line >= current)
