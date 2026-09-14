from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage

from .domain import (
    LineSegment,
    MonitorEvent,
    ParkingRuleService,
    Point,
    SpeedMeasurementService,
    segments_intersect,
)


@dataclass(slots=True)
class Detection:
    rect: tuple[float, float, float, float]
    confidence: float
    label: str = "自行车"

    @property
    def center(self) -> tuple[float, float]:
        x, y, width, height = self.rect
        return x + width / 2, y + height / 2


@dataclass(slots=True)
class TrackedDetection:
    track_id: int
    rect: tuple[float, float, float, float]
    confidence: float
    label: str = "自行车"
    speed_kmh: float | None = None
    violation: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        x, y, width, height = self.rect
        return x + width / 2, y + height / 2


class MobileNetBicycleDetector:
    CLASS_LABELS = {
        2: "自行车",
        14: "电动车/摩托车",
    }

    def __init__(self, prototxt: Path, weights: Path, confidence: float = 0.25) -> None:
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
            if class_id not in self.CLASS_LABELS or confidence < self.confidence:
                continue
            x1, y1, x2, y2 = output[0, 0, index, 3:7]
            x1 = float(np.clip(x1, 0.0, 1.0))
            y1 = float(np.clip(y1, 0.0, 1.0))
            x2 = float(np.clip(x2, 0.0, 1.0))
            y2 = float(np.clip(y2, 0.0, 1.0))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    (x1, y1, x2 - x1, y2 - y1),
                    confidence,
                    self.CLASS_LABELS[class_id],
                )
            )
        return detections


class YoloXTwoWheelerDetector:
    INPUT_SIZE = 416
    CLASS_LABELS = {
        1: "自行车",
        3: "电动车/摩托车",
    }

    def __init__(self, weights: Path, confidence: float = 0.12, nms_threshold: float = 0.45) -> None:
        if not weights.exists():
            raise FileNotFoundError("YOLOX-Tiny 模型文件不存在")
        self.net = cv2.dnn.readNetFromONNX(
            np.frombuffer(weights.read_bytes(), dtype=np.uint8)
        )
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self._grid, self._expanded_strides = self._build_decoder_grid()

    @classmethod
    def _build_decoder_grid(cls) -> tuple[np.ndarray, np.ndarray]:
        grids: list[np.ndarray] = []
        expanded_strides: list[np.ndarray] = []
        for stride in (8, 16, 32):
            grid_size = cls.INPUT_SIZE // stride
            y_grid, x_grid = np.meshgrid(
                np.arange(grid_size),
                np.arange(grid_size),
                indexing="ij",
            )
            grid = np.stack((x_grid, y_grid), axis=2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
        return np.concatenate(grids, axis=1), np.concatenate(expanded_strides, axis=1)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        ratio = min(self.INPUT_SIZE / height, self.INPUT_SIZE / width)
        resized_width = max(1, round(width * ratio))
        resized_height = max(1, round(height * ratio))
        padded = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[:resized_height, :resized_width] = cv2.resize(
            frame,
            (resized_width, resized_height),
        )
        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1.0,
            size=(self.INPUT_SIZE, self.INPUT_SIZE),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward().copy()
        output[..., :2] = (output[..., :2] + self._grid) * self._expanded_strides
        output[..., 2:4] = np.exp(np.clip(output[..., 2:4], -20, 20)) * self._expanded_strides

        predictions = output[0]
        class_scores = predictions[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = predictions[:, 4] * class_scores[np.arange(len(predictions)), class_ids]
        accepted = np.isin(class_ids, tuple(self.CLASS_LABELS)) & (scores >= self.confidence)

        boxes: list[list[int]] = []
        candidates: list[Detection] = []
        for prediction, score, class_id in zip(
            predictions[accepted],
            scores[accepted],
            class_ids[accepted],
            strict=True,
        ):
            center_x, center_y, box_width, box_height = prediction[:4] / ratio
            x1 = float(np.clip(center_x - box_width / 2, 0, width - 1))
            y1 = float(np.clip(center_y - box_height / 2, 0, height - 1))
            x2 = float(np.clip(center_x + box_width / 2, x1 + 1, width))
            y2 = float(np.clip(center_y + box_height / 2, y1 + 1, height))
            pixel_width = x2 - x1
            pixel_height = y2 - y1
            boxes.append([round(x1), round(y1), round(pixel_width), round(pixel_height)])
            candidates.append(
                Detection(
                    (x1 / width, y1 / height, pixel_width / width, pixel_height / height),
                    float(score),
                    self.CLASS_LABELS[int(class_id)],
                )
            )

        if not boxes:
            return []
        kept = cv2.dnn.NMSBoxes(
            boxes,
            [item.confidence for item in candidates],
            self.confidence,
            self.nms_threshold,
        )
        indexes = np.asarray(kept).reshape(-1).tolist() if len(kept) else []
        return [candidates[index] for index in indexes]


class RTDetrWorkerDetector:
    REQUIRED_MODEL_FILES = (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    )

    def __init__(
        self,
        environment_python: Path,
        worker_script: Path,
        model_dir: Path,
        runtime_dir: Path,
        confidence: float = 0.30,
    ) -> None:
        self.environment_python = environment_python
        self.worker_script = worker_script
        self.model_dir = model_dir
        self.runtime_dir = runtime_dir
        self.confidence = confidence
        self.process: subprocess.Popen[bytes] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[list[Detection]] | None = None
        self._last_result: list[Detection] | None = None
        self._io_lock = threading.Lock()
        self._log_handle = None
        self._failed = False
        self._ready_file = runtime_dir / f"rtdetr_worker_{os.getpid()}_{uuid.uuid4().hex}.json"

    @property
    def available(self) -> bool:
        return (
            self.environment_python.is_file()
            and self.worker_script.is_file()
            and all((self.model_dir / name).is_file() for name in self.REQUIRED_MODEL_FILES)
        )

    @property
    def ready(self) -> bool:
        status = self._read_status()
        return (
            not self._failed
            and self.process is not None
            and self.process.poll() is None
            and status.get("status") == "ready"
        )

    @property
    def status_text(self) -> str:
        if not self.available:
            return "RT-DETR 未安装"
        if self._failed or (self.process is not None and self.process.poll() is not None):
            status = self._read_status()
            return str(status.get("message") or "RT-DETR 启动失败")
        if self.ready:
            device = self._read_status().get("device", "CUDA")
            return f"Hugging Face RT-DETR R50 · {str(device).upper()}"
        if self.process is not None:
            return "RT-DETR R50 正在加载"
        return "RT-DETR R50 已安装"

    def start(self) -> None:
        if not self.available or self._failed:
            return
        if self.process is not None and self.process.poll() is None:
            return
        if self.process is not None:
            self._failed = True
            return
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._ready_file.unlink(missing_ok=True)
        self._log_handle = (self.runtime_dir / "rtdetr_worker.log").open("ab")
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        try:
            self.process = subprocess.Popen(
                [
                    str(self.environment_python),
                    "-u",
                    str(self.worker_script),
                    "--model",
                    str(self.model_dir),
                    "--ready-file",
                    str(self._ready_file),
                    "--confidence",
                    str(self.confidence),
                    "--device",
                    "cuda",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log_handle,
                env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            self._failed = True
            self._close_log()
            return
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtdetr")

    def detect(self, frame: np.ndarray) -> list[Detection] | None:
        if not self.ready or self._executor is None:
            return None
        if self._future is None:
            self._future = self._executor.submit(self._request, frame.copy())
            return None
        if not self._future.done():
            return self._last_result
        try:
            self._last_result = self._future.result()
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            self._failed = True
            self._last_result = None
            return None
        self._future = self._executor.submit(self._request, frame.copy())
        return self._last_result

    def shutdown(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            if self._io_lock.acquire(timeout=0.5):
                try:
                    if process.stdin is not None:
                        process.stdin.write(struct.pack("<I", 0))
                        process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                finally:
                    self._io_lock.release()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
        self._executor = None
        self._future = None
        self._last_result = None
        self.process = None
        self._ready_file.unlink(missing_ok=True)
        self._close_log()

    def _request(self, frame: np.ndarray) -> list[Detection]:
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("RT-DETR 推理进程不可用")
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        if not encoded_ok:
            raise ValueError("视频帧编码失败")
        request = encoded.tobytes()
        with self._io_lock:
            process.stdin.write(struct.pack("<I", len(request)))
            process.stdin.write(request)
            process.stdin.flush()
            response_size = struct.unpack("<I", self._read_exact(process.stdout, 4))[0]
            if response_size > 10_000_000:
                raise ValueError("RT-DETR 返回数据异常")
            payload = json.loads(self._read_exact(process.stdout, response_size).decode("utf-8"))
        return [
            Detection(
                tuple(float(value) for value in item["rect"]),
                float(item["confidence"]),
                str(item["label"]),
            )
            for item in payload
        ]

    @staticmethod
    def _read_exact(stream, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                raise RuntimeError("RT-DETR 推理进程已断开")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_status(self) -> dict:
        if not self._ready_file.is_file():
            return {}
        try:
            payload = json.loads(self._ready_file.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


class TwoWheelerDetector:
    def __init__(
        self,
        prototxt: Path,
        caffe_weights: Path,
        yolox_weights: Path | None,
        rtdetr: RTDetrWorkerDetector | None = None,
    ) -> None:
        self.fallback = MobileNetBicycleDetector(prototxt, caffe_weights)
        self.primary: YoloXTwoWheelerDetector | None = None
        self.rtdetr = rtdetr
        if yolox_weights is not None and yolox_weights.exists():
            try:
                self.primary = YoloXTwoWheelerDetector(yolox_weights)
            except (cv2.error, OSError, ValueError):
                self.primary = None
    @property
    def name(self) -> str:
        if self.rtdetr is not None and self.rtdetr.ready:
            return self.rtdetr.status_text
        fallback = (
            "YOLOX-Tiny 两轮车检测"
            if self.primary is not None
            else "MobileNet-SSD 两轮车检测"
        )
        if (
            self.rtdetr is not None
            and self.rtdetr.available
            and self.rtdetr.process is not None
            and self.rtdetr.process.poll() is None
        ):
            return f"{fallback} · RT-DETR 加载中"
        return fallback

    @property
    def status_text(self) -> str:
        if self.rtdetr is None:
            return self.name
        return self.rtdetr.status_text

    def start(self) -> None:
        if self.rtdetr is not None:
            self.rtdetr.start()

    def shutdown(self) -> None:
        if self.rtdetr is not None:
            self.rtdetr.shutdown()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.rtdetr is not None:
            detections = self.rtdetr.detect(frame)
            if detections is not None:
                return detections
        if self.primary is not None:
            detections = self.primary.detect(frame)
            if detections:
                return detections
        return self.fallback.detect(frame)


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
                detection.label,
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
            self._tracks[track_id] = TrackedDetection(
                track_id,
                detection.rect,
                detection.confidence,
                detection.label,
            )
            self._misses[track_id] = 0

        return [self._tracks[track_id] for track_id in sorted(self._tracks)]


class VideoAnalysisController(QObject):
    frame_ready = Signal(object, object, str)
    event_raised = Signal(object)
    metrics_changed = Signal(dict)
    status_changed = Signal(str)
    detector_status_changed = Signal(str)

    def __init__(
        self,
        prototxt: Path,
        weights: Path,
        yolox_weights: Path | None = None,
        rtdetr: RTDetrWorkerDetector | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.detector = TwoWheelerDetector(prototxt, weights, yolox_weights, rtdetr)
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
        self._previous_centers: dict[int, Point] = {}
        self._crossing_state: dict[int, tuple[str, float]] = {}
        self._completed_speed_ids: set[int] = set()
        self._parking_event_ids: set[int] = set()
        self._last_speed = 0.0
        self._detector_name = self.detector.name
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def open(self, path: Path) -> None:
        self.close()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频：{path}")
        self.capture = capture
        self.detector.start()
        self.path = path
        self.source_name = path.name
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        self._timer.setInterval(max(20, min(100, round(1000 / fps))))
        self._reset_analysis_state()
        self.running = True
        self._timer.start()
        self.status_changed.emit(f"本地视频：{path.name} · {self.detector.name}")
        self.detector_status_changed.emit(self.detector.status_text)

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

    def set_speed_lines(self, lines: tuple[LineSegment, LineSegment]) -> None:
        self.speed_service.set_lines(lines)
        self._previous_centers.clear()
        self._crossing_state.clear()
        self._completed_speed_ids.clear()
        self._last_speed = 0.0

    def set_parking_zone(self, zone: tuple[float, float, float, float]) -> None:
        self.parking_service.set_zone(zone)
        self._parking_event_ids.clear()
        if self.mode == "parking":
            self._evaluate_parking()

    def set_parking_polygon(self, polygon: tuple[Point, ...]) -> None:
        self.parking_service.set_polygon(polygon)
        self._parking_event_ids.clear()
        if self.mode == "parking":
            self._evaluate_parking()

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
            detector_name = self.detector.name
            if detector_name != self._detector_name:
                self._detector_name = detector_name
                self.status_changed.emit(f"本地视频：{self.source_name} · {detector_name}")
                self.detector_status_changed.emit(self.detector.status_text)

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
        line_a, line_b = self.speed_service.lines
        for track in self.current_tracks:
            center = track.center
            previous = self._previous_centers.get(track.track_id)
            self._previous_centers[track.track_id] = center
            if previous is None or track.track_id in self._completed_speed_ids:
                continue
            crossed_a = segments_intersect(previous, center, line_a)
            crossed_b = segments_intersect(previous, center, line_b)
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
                    detail=f"{track.label}目标 #{track.track_id} 通过双线测速区",
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
                    detail=f"{track.label}目标 #{track.track_id} 超出配置停车区域",
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

    def shutdown(self) -> None:
        self.close()
        self.detector.shutdown()
