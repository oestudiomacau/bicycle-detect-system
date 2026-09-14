from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .simulation import ParkedVehicle, RoadTrack, SimulationEngine
from .video_analysis import TrackedDetection


class MetricCard(QFrame):
    def __init__(self, title: str, value: str, suffix: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 13, 16, 13)
        layout.setSpacing(4)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        value_row = QHBoxLayout()
        value_row.setSpacing(5)
        value_row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        suffix_label = QLabel(suffix)
        suffix_label.setObjectName("metricSuffix")
        value_row.addWidget(self.value_label)
        value_row.addWidget(suffix_label)
        value_row.addStretch()
        layout.addWidget(title_label)
        layout.addLayout(value_row)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class VideoCanvas(QWidget):
    def __init__(self, engine: SimulationEngine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.video_frame: QImage | None = None
        self.real_detections: list[TrackedDetection] = []
        self.source_name = "模拟视频源"
        self.real_video_running = False
        self.setObjectName("videoCanvas")
        self.setMinimumSize(650, 370)
        self.engine.frame_changed.connect(self.update)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#172026"))
        if self.video_frame is not None:
            video_rect = self._paint_video_frame(painter)
            self._paint_real_overlays(painter, video_rect)
        elif self.engine.mode == "road":
            self._paint_road(painter)
        else:
            self._paint_parking(painter)
        self._paint_hud(painter)

    def set_video_frame(
        self,
        image: QImage,
        detections: list[TrackedDetection],
        source_name: str,
    ) -> None:
        self.video_frame = image
        self.real_detections = detections
        self.source_name = source_name
        self.real_video_running = True
        self.update()

    def set_video_running(self, running: bool) -> None:
        self.real_video_running = running
        self.update()

    def clear_video(self) -> None:
        self.video_frame = None
        self.real_detections = []
        self.source_name = "模拟视频源"
        self.real_video_running = False
        self.update()

    def _paint_video_frame(self, painter: QPainter) -> QRectF:
        assert self.video_frame is not None
        image_size = self.video_frame.size()
        scale = min(self.width() / image_size.width(), self.height() / image_size.height())
        draw_width = image_size.width() * scale
        draw_height = image_size.height() * scale
        target = QRectF(
            (self.width() - draw_width) / 2,
            (self.height() - draw_height) / 2,
            draw_width,
            draw_height,
        )
        painter.drawImage(target, self.video_frame)
        return target

    def _paint_real_overlays(self, painter: QPainter, video_rect: QRectF) -> None:
        if self.engine.mode == "road":
            line_a = video_rect.left() + video_rect.width() * self.engine.LINE_A
            line_b = video_rect.left() + video_rect.width() * self.engine.LINE_B
            painter.setPen(QPen(QColor("#f0b542"), 3))
            painter.drawLine(QPointF(line_a, video_rect.top()), QPointF(line_a, video_rect.bottom()))
            painter.setPen(QPen(QColor("#31b7c2"), 3))
            painter.drawLine(QPointF(line_b, video_rect.top()), QPointF(line_b, video_rect.bottom()))
            self._tag(painter, QRectF(line_a + 6, video_rect.top() + 52, 76, 23), "虚拟线 A", QColor("#f0b542"))
            self._tag(painter, QRectF(line_b + 6, video_rect.top() + 52, 76, 23), "虚拟线 B", QColor("#31b7c2"))
        else:
            left, top, right, bottom = self.engine.parking_service.zone
            zone = QRectF(
                video_rect.left() + left * video_rect.width(),
                video_rect.top() + top * video_rect.height(),
                (right - left) * video_rect.width(),
                (bottom - top) * video_rect.height(),
            )
            painter.setPen(QPen(QColor("#46c3a2"), 3, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(70, 195, 162, 20))
            painter.drawRoundedRect(zone, 4, 4)
            self._tag(painter, QRectF(zone.left() + 8, zone.top() + 8, 105, 24), "规定停车区域", QColor("#46c3a2"))

        for detection in self.real_detections:
            self._paint_real_detection(painter, video_rect, detection)

    def _paint_real_detection(
        self,
        painter: QPainter,
        video_rect: QRectF,
        detection: TrackedDetection,
    ) -> None:
        x, y, width, height = detection.rect
        box = QRectF(
            video_rect.left() + x * video_rect.width(),
            video_rect.top() + y * video_rect.height(),
            width * video_rect.width(),
            height * video_rect.height(),
        )
        warning = bool(detection.violation) or (
            detection.speed_kmh is not None
            and detection.speed_kmh > self.engine.speed_service.threshold_kmh
        )
        color = QColor("#ef6a62") if warning else QColor("#68d3a5")
        painter.setPen(QPen(color, 2))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 24))
        painter.drawRoundedRect(box, 3, 3)
        if detection.violation:
            detail = detection.violation
        elif detection.speed_kmh is not None:
            detail = f"{detection.speed_kmh:.1f} km/h"
        else:
            detail = f"自行车 {detection.confidence:.0%}"
        label_width = min(150.0, max(95.0, box.width() + 32))
        self._tag(
            painter,
            QRectF(box.left(), max(video_rect.top() + 4, box.top() - 24), label_width, 22),
            f"#{detection.track_id} {detail}",
            color,
        )

    def _paint_road(self, painter: QPainter) -> None:
        width, height = self.width(), self.height()
        painter.fillRect(QRectF(0, 0, width, height * 0.22), QColor("#738a70"))
        painter.fillRect(QRectF(0, height * 0.22, width, height * 0.78), QColor("#41494d"))

        painter.setPen(QPen(QColor("#687176"), 1))
        for index in range(1, 5):
            y = height * (0.22 + index * 0.15)
            painter.drawLine(0, int(y), width, int(y))

        painter.setPen(QPen(QColor("#d4d7d8"), 2, Qt.PenStyle.DashLine))
        painter.drawLine(0, int(height * 0.48), width, int(height * 0.48))
        painter.drawLine(0, int(height * 0.70), width, int(height * 0.70))

        line_a_x = int(width * self.engine.LINE_A)
        line_b_x = int(width * self.engine.LINE_B)
        painter.setPen(QPen(QColor("#f0b542"), 3))
        painter.drawLine(line_a_x, int(height * 0.22), line_a_x, height)
        painter.setPen(QPen(QColor("#31b7c2"), 3))
        painter.drawLine(line_b_x, int(height * 0.22), line_b_x, height)

        self._tag(painter, QRectF(line_a_x + 8, height * 0.25, 78, 25), "虚拟线 A", QColor("#f0b542"))
        self._tag(painter, QRectF(line_b_x + 8, height * 0.25, 78, 25), "虚拟线 B", QColor("#31b7c2"))
        self._tag(
            painter,
            QRectF((line_a_x + line_b_x) / 2 - 56, height * 0.89, 112, 26),
            f"实际距离 {self.engine.speed_service.distance_m:.1f} m",
            QColor("#d9dee0"),
        )

        for track in self.engine.road_tracks:
            self._paint_road_track(painter, track)

    def _paint_road_track(self, painter: QPainter, track: RoadTrack) -> None:
        width, height = self.width(), self.height()
        x = width * track.x
        y = height * track.y
        box = QRectF(x - 30, y - 19, 60, 38)
        speed = track.measured_speed_kmh or track.target_speed_kmh
        warning = speed > self.engine.speed_service.threshold_kmh
        color = QColor("#ef6a62") if warning else QColor("#68d3a5")
        painter.setPen(QPen(color, 2))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 32))
        painter.drawRoundedRect(box, 3, 3)

        painter.setPen(QPen(QColor("#dfe6e8"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(x - 13, y + 7), 6, 6)
        painter.drawEllipse(QPointF(x + 13, y + 7), 6, 6)
        painter.drawLine(QPointF(x - 13, y + 7), QPointF(x, y - 5))
        painter.drawLine(QPointF(x, y - 5), QPointF(x + 13, y + 7))
        painter.drawEllipse(QPointF(x + 1, y - 11), 4, 4)

        label = f"#{track.track_id}  {speed:.1f} km/h"
        self._tag(painter, QRectF(x - 30, y - 44, 118, 22), label, color)

    def _paint_parking(self, painter: QPainter) -> None:
        width, height = self.width(), self.height()
        painter.fillRect(self.rect(), QColor("#3e474c"))
        painter.setPen(QPen(QColor("#687278"), 1))
        for x in range(0, width, 45):
            painter.drawLine(x, 0, x, height)
        for y in range(0, height, 45):
            painter.drawLine(0, y, width, y)

        left, top, right, bottom = self.engine.parking_service.zone
        zone = QRectF(left * width, top * height, (right - left) * width, (bottom - top) * height)
        painter.setPen(QPen(QColor("#46c3a2"), 3, Qt.PenStyle.DashLine))
        painter.setBrush(QColor(70, 195, 162, 20))
        painter.drawRoundedRect(zone, 4, 4)
        self._tag(painter, QRectF(zone.left() + 10, zone.top() + 10, 105, 25), "规定停车区域", QColor("#46c3a2"))

        for vehicle in self.engine.parked_vehicles:
            self._paint_parked_vehicle(painter, vehicle)

    def _paint_parked_vehicle(self, painter: QPainter, vehicle: ParkedVehicle) -> None:
        width, height = self.width(), self.height()
        x, y, w, h = vehicle.rect
        rect = QRectF(x * width, y * height, w * width, h * height)
        color = QColor("#ef6a62") if vehicle.violation else QColor("#68d3a5")

        painter.save()
        painter.translate(rect.center())
        painter.rotate(vehicle.angle_deg)
        local = QRectF(-rect.width() / 2, -rect.height() / 2, rect.width(), rect.height())
        painter.setPen(QPen(color, 2))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 35))
        painter.drawRoundedRect(local, 3, 3)
        painter.setPen(QPen(QColor("#dfe6e8"), 2))
        painter.drawEllipse(QPointF(local.left() + 12, 0), 7, 7)
        painter.drawEllipse(QPointF(local.right() - 12, 0), 7, 7)
        painter.drawLine(QPointF(local.left() + 12, 0), QPointF(0, local.top() + 6))
        painter.drawLine(QPointF(0, local.top() + 6), QPointF(local.right() - 12, 0))
        painter.restore()

        if vehicle.violation:
            painter.setBrush(QColor(239, 106, 98, 30))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(rect.adjusted(-14, -14, 14, 14))
        label = vehicle.violation or "停放正常"
        self._tag(painter, QRectF(rect.left(), rect.top() - 25, 108, 22), f"#{vehicle.track_id} {label}", color)

    def _paint_hud(self, painter: QPainter) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(8, 13, 16, 190))
        painter.drawRoundedRect(QRectF(14, 13, 310, 34), 4, 4)
        painter.setPen(QColor("#e7ecee"))
        painter.setFont(QFont("Noto Sans SC", 9))
        mode = "道路观察位 · 双线测速" if self.engine.mode == "road" else "停车观察位 · 违规停放"
        state = "分析中" if self.engine.running else "已暂停"
        source = self.source_name if self.video_frame is not None else "模拟视频源"
        if self.video_frame is not None:
            state = "真实检测中" if self.real_video_running else "视频已暂停"
        painter.drawText(QRectF(26, 13, 286, 34), Qt.AlignmentFlag.AlignVCenter, f"{source}  |  {mode}  |  {state}")

        painter.setBrush(QColor(8, 13, 16, 165))
        painter.drawRoundedRect(QRectF(self.width() - 180, self.height() - 39, 166, 26), 4, 4)
        painter.setPen(QColor("#cfd7da"))
        painter.setFont(QFont("Noto Sans SC", 8))
        painter.drawText(
            QRectF(self.width() - 170, self.height() - 39, 148, 26),
            Qt.AlignmentFlag.AlignCenter,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    @staticmethod
    def _tag(painter: QPainter, rect: QRectF, text: str, color: QColor) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(12, 18, 21, 215))
        painter.drawRoundedRect(rect, 3, 3)
        painter.setPen(color)
        painter.setFont(QFont("Noto Sans SC", 8))
        painter.drawText(rect.adjusted(7, 0, -5, 0), Qt.AlignmentFlag.AlignVCenter, text)
