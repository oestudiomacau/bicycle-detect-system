from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .domain import LineSegment, Point
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
    zone_changed = Signal(object)
    speed_lines_changed = Signal(object)
    parking_polygon_changed = Signal(object)

    def __init__(self, engine: SimulationEngine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.video_frame: QImage | None = None
        self.real_detections: list[TrackedDetection] = []
        self.source_name = "模拟视频源"
        self.real_video_running = False
        self.calibration_mode: str | None = None
        self._calibration_start: QPointF | None = None
        self._calibration_end: QPointF | None = None
        self._draft_zone: tuple[float, float, float, float] | None = None
        self._draft_lines: list[LineSegment] = []
        self._draft_polygon: list[Point] = []
        self.setObjectName("videoCanvas")
        self.setMinimumSize(650, 370)
        self.setMouseTracking(True)
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
        self._paint_calibration(painter)
        self._paint_hud(painter)

    @property
    def draft_zone(self) -> tuple[float, float, float, float] | None:
        return self._draft_zone

    @property
    def calibration_active(self) -> bool:
        return self.calibration_mode is not None

    def begin_zone_calibration(self) -> None:
        self._begin_calibration("parking_rectangle")

    def begin_speed_line_calibration(self) -> None:
        self._begin_calibration("speed_lines")

    def begin_parking_boundary_calibration(self) -> None:
        self._begin_calibration("parking_polygon")

    def _begin_calibration(self, mode: str) -> None:
        self.calibration_mode = mode
        self._calibration_start = None
        self._calibration_end = None
        self._draft_zone = None
        self._draft_lines = []
        self._draft_polygon = []
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def confirm_zone_calibration(self) -> bool:
        if self.calibration_mode != "parking_rectangle" or self._draft_zone is None:
            return False
        zone = self._draft_zone
        self._finish_calibration()
        self.zone_changed.emit(zone)
        return True

    def confirm_speed_line_calibration(self) -> bool:
        if self.calibration_mode != "speed_lines" or len(self._draft_lines) != 2:
            return False
        lines = tuple(self._draft_lines)
        self._finish_calibration()
        self.speed_lines_changed.emit(lines)
        return True

    def confirm_parking_boundary_calibration(self) -> bool:
        if self.calibration_mode != "parking_polygon" or len(self._draft_polygon) < 3:
            return False
        polygon = tuple(self._draft_polygon)
        self._finish_calibration()
        self.parking_polygon_changed.emit(polygon)
        return True

    def cancel_zone_calibration(self) -> None:
        self._finish_calibration()

    def cancel_calibration(self) -> None:
        self._finish_calibration()

    def _finish_calibration(self) -> None:
        self.calibration_mode = None
        self._calibration_start = None
        self._calibration_end = None
        self._draft_zone = None
        self._draft_lines = []
        self._draft_polygon = []
        self.unsetCursor()
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (
            not self.calibration_active
            or not self._content_rect().contains(event.position())
        ):
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.RightButton:
            if self.calibration_mode == "speed_lines" and self._draft_lines:
                self._draft_lines.pop()
            elif self.calibration_mode == "parking_polygon" and self._draft_polygon:
                self._draft_polygon.pop()
            self._calibration_start = None
            event.accept()
            self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        if self.calibration_mode == "parking_polygon":
            if len(self._draft_polygon) < 12:
                point = self._point_to_normalized(self._clamp_to_content(event.position()))
                if not self._draft_polygon or self._point_distance(point, self._draft_polygon[-1]) >= 0.01:
                    self._draft_polygon.append(point)
            self._calibration_end = self._clamp_to_content(event.position())
            event.accept()
            self.update()
            return
        if self.calibration_mode == "speed_lines" and len(self._draft_lines) >= 2:
            event.accept()
            return
        self._calibration_start = self._clamp_to_content(event.position())
        self._calibration_end = self._calibration_start
        if self.calibration_mode == "parking_rectangle":
            self._draft_zone = None
        event.accept()
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self.calibration_active:
            super().mouseMoveEvent(event)
            return
        self._calibration_end = self._clamp_to_content(event.position())
        if self.calibration_mode == "parking_polygon":
            event.accept()
            self.update()
            return
        if self._calibration_start is None:
            super().mouseMoveEvent(event)
            return
        if self.calibration_mode == "parking_rectangle":
            self._draft_zone = self._zone_from_points(self._calibration_start, self._calibration_end)
        event.accept()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if (
            not self.calibration_active
            or self._calibration_start is None
            or event.button() != Qt.MouseButton.LeftButton
        ):
            super().mouseReleaseEvent(event)
            return
        self._calibration_end = self._clamp_to_content(event.position())
        if self.calibration_mode == "parking_rectangle":
            self._draft_zone = self._zone_from_points(self._calibration_start, self._calibration_end)
        elif self.calibration_mode == "speed_lines":
            line = self._line_from_points(self._calibration_start, self._calibration_end)
            if line is not None:
                self._draft_lines.append(line)
            self._calibration_start = None
            self._calibration_end = None
        event.accept()
        self.update()

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

    def _content_rect(self) -> QRectF:
        if self.video_frame is None:
            return QRectF(self.rect())
        image_size = self.video_frame.size()
        scale = min(self.width() / image_size.width(), self.height() / image_size.height())
        draw_width = image_size.width() * scale
        draw_height = image_size.height() * scale
        return QRectF(
            (self.width() - draw_width) / 2,
            (self.height() - draw_height) / 2,
            draw_width,
            draw_height,
        )

    def _clamp_to_content(self, point: QPointF) -> QPointF:
        content = self._content_rect()
        return QPointF(
            min(max(point.x(), content.left()), content.right()),
            min(max(point.y(), content.top()), content.bottom()),
        )

    def _point_to_normalized(self, point: QPointF) -> Point:
        content = self._content_rect()
        return (
            (point.x() - content.left()) / content.width(),
            (point.y() - content.top()) / content.height(),
        )

    @staticmethod
    def _point_distance(first: Point, second: Point) -> float:
        return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5

    def _point_from_normalized(self, point: Point, content: QRectF | None = None) -> QPointF:
        target = content or self._content_rect()
        return QPointF(
            target.left() + point[0] * target.width(),
            target.top() + point[1] * target.height(),
        )

    def _line_from_points(self, start: QPointF, end: QPointF) -> LineSegment | None:
        start_normalized = self._point_to_normalized(start)
        end_normalized = self._point_to_normalized(end)
        if self._point_distance(start_normalized, end_normalized) < 0.03:
            return None
        return (*start_normalized, *end_normalized)

    def _zone_from_points(
        self,
        start: QPointF,
        end: QPointF,
    ) -> tuple[float, float, float, float] | None:
        content = self._content_rect()
        if content.width() <= 0 or content.height() <= 0:
            return None
        left = (min(start.x(), end.x()) - content.left()) / content.width()
        top = (min(start.y(), end.y()) - content.top()) / content.height()
        right = (max(start.x(), end.x()) - content.left()) / content.width()
        bottom = (max(start.y(), end.y()) - content.top()) / content.height()
        zone = (left, top, right, bottom)
        if right - left < 0.02 or bottom - top < 0.02:
            return None
        return zone

    def _paint_calibration(self, painter: QPainter) -> None:
        if not self.calibration_active:
            return
        content = self._content_rect()
        painter.save()
        hint_text = ""
        if self.calibration_mode == "parking_rectangle" and self._draft_zone is not None:
            painter.setPen(QPen(QColor("#f0b542"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(240, 181, 66, 30))
            left, top, right, bottom = self._draft_zone
            draft = QRectF(
                content.left() + left * content.width(),
                content.top() + top * content.height(),
                (right - left) * content.width(),
                (bottom - top) * content.height(),
            )
            painter.drawRect(draft)
            hint_text = "按住鼠标左键拖拽矩形停车区域"
        elif self.calibration_mode == "parking_rectangle":
            hint_text = "按住鼠标左键拖拽矩形停车区域"
        elif self.calibration_mode == "speed_lines":
            colors = (QColor("#f0b542"), QColor("#31b7c2"))
            for index, line in enumerate(self._draft_lines):
                self._paint_line_segment(painter, content, line, colors[index], 3)
            if self._calibration_start is not None and self._calibration_end is not None:
                painter.setPen(QPen(colors[len(self._draft_lines)], 3, Qt.PenStyle.DashLine))
                painter.drawLine(self._calibration_start, self._calibration_end)
            next_name = "A" if not self._draft_lines else "B"
            hint_text = (
                "两条测速线已画好，点击右侧确认"
                if len(self._draft_lines) == 2
                else f"拖拽绘制测速线 {next_name}；右键撤销上一条"
            )
        elif self.calibration_mode == "parking_polygon":
            points = [self._point_from_normalized(point, content) for point in self._draft_polygon]
            if len(points) >= 3:
                painter.setPen(QPen(QColor("#46c3a2"), 2, Qt.PenStyle.DashLine))
                painter.setBrush(QColor(70, 195, 162, 30))
                painter.drawPolygon(QPolygonF(points))
            elif len(points) >= 2:
                painter.setPen(QPen(QColor("#46c3a2"), 3))
                painter.drawPolyline(QPolygonF(points))
            if points and self._calibration_end is not None:
                painter.setPen(QPen(QColor("#f0b542"), 2, Qt.PenStyle.DashLine))
                painter.drawLine(points[-1], self._calibration_end)
            painter.setBrush(QColor("#f7d58b"))
            painter.setPen(Qt.PenStyle.NoPen)
            for point in points:
                painter.drawEllipse(point, 4, 4)
            hint_text = f"依次点击边界顶点（已画 {len(points)} 个）；右键撤销，确认后自动闭合"
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(8, 13, 16, 220))
        hint = QRectF(content.left() + 14, content.bottom() - 48, min(430, content.width() - 28), 32)
        painter.drawRoundedRect(hint, 4, 4)
        painter.setPen(QColor("#f7d58b"))
        painter.setFont(QFont("Microsoft YaHei UI", 9))
        painter.drawText(hint.adjusted(10, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter, hint_text)
        painter.restore()

    @staticmethod
    def _paint_line_segment(
        painter: QPainter,
        content: QRectF,
        line: LineSegment,
        color: QColor,
        width: int = 3,
    ) -> tuple[QPointF, QPointF]:
        start = QPointF(
            content.left() + line[0] * content.width(),
            content.top() + line[1] * content.height(),
        )
        end = QPointF(
            content.left() + line[2] * content.width(),
            content.top() + line[3] * content.height(),
        )
        painter.setPen(QPen(color, width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(start, end)
        return start, end

    def _paint_parking_boundary(self, painter: QPainter, content: QRectF) -> None:
        points = [
            self._point_from_normalized(point, content)
            for point in self.engine.parking_service.polygon
        ]
        polygon = QPolygonF(points)
        painter.setPen(QPen(QColor("#46c3a2"), 3, Qt.PenStyle.DashLine))
        painter.setBrush(QColor(70, 195, 162, 20))
        painter.drawPolygon(polygon)
        if points:
            first = points[0]
            self._tag(
                painter,
                QRectF(first.x() + 8, first.y() + 8, 105, 24),
                "规定停车区域",
                QColor("#46c3a2"),
            )

    def _paint_real_overlays(self, painter: QPainter, video_rect: QRectF) -> None:
        if self.engine.mode == "road":
            line_a, line_b = self.engine.speed_service.lines
            start_a, _ = self._paint_line_segment(painter, video_rect, line_a, QColor("#f0b542"))
            start_b, _ = self._paint_line_segment(painter, video_rect, line_b, QColor("#31b7c2"))
            self._tag(
                painter,
                QRectF(start_a.x() + 6, start_a.y() + 6, 76, 23),
                "虚拟线 A",
                QColor("#f0b542"),
            )
            self._tag(
                painter,
                QRectF(start_b.x() + 6, start_b.y() + 6, 76, 23),
                "虚拟线 B",
                QColor("#31b7c2"),
            )
        else:
            self._paint_parking_boundary(painter, video_rect)

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
            detail = f"{detection.label} {detection.confidence:.0%}"
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

        content = QRectF(self.rect())
        line_a, line_b = self.engine.speed_service.lines
        start_a, _ = self._paint_line_segment(painter, content, line_a, QColor("#f0b542"))
        start_b, _ = self._paint_line_segment(painter, content, line_b, QColor("#31b7c2"))
        self._tag(painter, QRectF(start_a.x() + 8, start_a.y() + 8, 78, 25), "虚拟线 A", QColor("#f0b542"))
        self._tag(painter, QRectF(start_b.x() + 8, start_b.y() + 8, 78, 25), "虚拟线 B", QColor("#31b7c2"))
        self._tag(
            painter,
            QRectF(width / 2 - 56, height * 0.89, 112, 26),
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

        self._paint_parking_boundary(painter, QRectF(self.rect()))

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
        painter.setFont(QFont("Microsoft YaHei UI", 9))
        mode = "道路观察位 · 双线测速" if self.engine.mode == "road" else "停车观察位 · 违规停放"
        state = "分析中" if self.engine.running else "已暂停"
        source = self.source_name if self.video_frame is not None else "模拟视频源"
        if self.video_frame is not None:
            state = "真实检测中" if self.real_video_running else "视频已暂停"
        painter.drawText(QRectF(26, 13, 286, 34), Qt.AlignmentFlag.AlignVCenter, f"{source}  |  {mode}  |  {state}")

        painter.setBrush(QColor(8, 13, 16, 165))
        painter.drawRoundedRect(QRectF(self.width() - 180, self.height() - 39, 166, 26), 4, 4)
        painter.setPen(QColor("#cfd7da"))
        painter.setFont(QFont("Microsoft YaHei UI", 8))
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
        painter.setFont(QFont("Microsoft YaHei UI", 8))
        painter.drawText(rect.adjusted(7, 0, -5, 0), Qt.AlignmentFlag.AlignVCenter, text)
