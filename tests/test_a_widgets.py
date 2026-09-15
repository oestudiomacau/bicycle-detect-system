import os
import tempfile
import unittest
from pathlib import Path

import cv2

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QImage, QPalette
from PySide6.QtWidgets import QApplication

from campus_monitor.domain import EventStore, MonitorEvent
from campus_monitor.main_window import MainWindow
from campus_monitor.settings import SettingsStore
from campus_monitor.simulation import SimulationEngine
from campus_monitor.widgets import VideoCanvas


class VideoCanvasCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_maps_drag_to_letterboxed_video_coordinates(self) -> None:
        engine = SimulationEngine()
        engine._timer.stop()
        canvas = VideoCanvas(engine)
        canvas.resize(1000, 600)
        canvas.set_video_frame(QImage(800, 400, QImage.Format.Format_RGB888), [], "test.mp4")

        zone = canvas._zone_from_points(QPointF(100, 100), QPointF(900, 500))

        self.assertIsNotNone(zone)
        assert zone is not None
        for actual, expected in zip(zone, (0.1, 0.1, 0.9, 0.9), strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_maps_drawn_line_to_video_coordinates(self) -> None:
        engine = SimulationEngine()
        engine._timer.stop()
        canvas = VideoCanvas(engine)
        canvas.resize(1000, 600)
        canvas.set_video_frame(QImage(800, 400, QImage.Format.Format_RGB888), [], "test.mp4")

        line = canvas._line_from_points(QPointF(200, 150), QPointF(800, 450))

        self.assertIsNotNone(line)
        assert line is not None
        for actual, expected in zip(line, (0.2, 0.2, 0.8, 0.8), strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_training_path_input_uses_dark_text(self) -> None:
        window = MainWindow()
        window.engine._timer.stop()
        window.video_controller._timer.stop()
        line_edit = window.training_page.normal_picker.line_edit
        line_edit.ensurePolished()

        self.assertEqual(
            line_edit.palette().color(QPalette.ColorRole.Text),
            QColor("#20292e"),
        )
        window.close()

    def test_training_text_uses_chinese_font_and_paths_start_at_beginning(self) -> None:
        window = MainWindow()
        window.engine._timer.stop()
        window.video_controller._timer.stop()
        picker = window.training_page.normal_picker

        self.assertEqual(picker.line_edit.cursorPosition(), 0)
        self.assertEqual(picker.line_edit.toolTip(), picker.line_edit.text())
        self.assertEqual(window.training_page.log.font().family(), "Microsoft YaHei UI")
        window.close()

    def test_loads_configured_startup_video_and_mode(self) -> None:
        window = MainWindow()
        window.engine._timer.stop()
        window.video_controller.detector.rtdetr = None
        with tempfile.TemporaryDirectory() as folder:
            store = SettingsStore(Path(folder) / "settings.json")
            video = Path(__file__).resolve().parents[1] / "sample_videos" / "parking_dense.mp4"
            store.save_startup_video(video, "parking")
            window.settings_store = store

            loaded = window.load_startup_video()

            self.assertTrue(loaded)
            self.assertEqual(window.active_source, "video")
            self.assertEqual(window.engine.mode, "parking")
            self.assertEqual(window.video_controller.path, video)
        window.close()

    def test_warning_event_does_not_create_automatic_snapshot(self) -> None:
        window = MainWindow()
        window.engine._timer.stop()
        with tempfile.TemporaryDirectory() as folder:
            runtime_dir = Path(folder)
            window.event_store = EventStore(runtime_dir / "events.jsonl")
            window.snapshot_dir = runtime_dir / "snapshots"

            event = MonitorEvent("测试告警", "测试", "不应自动截图", severity="warning")
            window._handle_event(event)

            self.assertEqual(event.snapshot, "")
            self.assertFalse(window.snapshot_dir.exists())
            self.assertTrue(window.event_store.path.exists())
        window.close()

    def test_header_model_status_is_separate_from_source(self) -> None:
        window = MainWindow()
        window.engine._timer.stop()

        window._update_source_status("本地图片：demo.jpg")
        window._update_detector_status("Hugging Face RT-DETR R50 · CUDA")

        self.assertEqual(window.source_status.text(), "本地图片：demo.jpg")
        self.assertEqual(window.header_model_status.text(), "RT-DETR R50 · CUDA")
        window.close()

    def test_loads_image_into_detection_canvas(self) -> None:
        window = MainWindow()
        window.engine._timer.stop()
        window.video_controller.detector.rtdetr = None
        sample = Path(__file__).resolve().parents[1] / "sample_videos" / "parking_dense.mp4"
        capture = cv2.VideoCapture(str(sample))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 90)
        ok, frame = capture.read()
        capture.release()
        self.assertTrue(ok)

        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "bicycles.jpg"
            self.assertTrue(cv2.imwrite(str(image_path), frame))
            window.event_store = EventStore(Path(folder) / "events.jsonl")

            loaded = window.load_image(image_path, "parking")

            self.assertTrue(loaded)
            self.assertEqual(window.active_source, "image")
            self.assertEqual(window.canvas.source_name, image_path.name)
            self.assertIsNotNone(window.canvas.video_frame)
            self.assertGreaterEqual(len(window.video_controller.current_tracks), 5)
            self.assertEqual(window.run_button.text(), "重新检测")
        window.close()

    def test_analysis_panel_does_not_clip_detector_status(self) -> None:
        window = MainWindow()
        window.engine._timer.stop()
        window.resize(1440, 900)
        window.show()
        self.app.processEvents()

        self.assertGreaterEqual(
            window.detector_value.height(),
            window.detector_value.minimumSizeHint().height(),
        )
        window.close()

    def test_events_navigation_opens_history_and_snapshot_page(self) -> None:
        window = MainWindow()
        window.engine._timer.stop()
        with tempfile.TemporaryDirectory() as folder:
            runtime_dir = Path(folder)
            window.event_store = EventStore(runtime_dir / "events.jsonl")
            window.snapshot_dir = runtime_dir / "snapshots"
            window.snapshot_dir.mkdir()
            snapshot = QImage(320, 180, QImage.Format.Format_RGB888)
            snapshot.fill(QColor("#267e6b"))
            self.assertTrue(snapshot.save(str(window.snapshot_dir / "manual_test.png")))
            window._handle_event(
                MonitorEvent("测试告警", "测试观察位", "事件页面应显示此记录", severity="warning")
            )

            window.events_nav.click()
            self.app.processEvents()

            self.assertIs(window.pages.currentWidget(), window.events_page)
            self.assertEqual(window.event_history_table.rowCount(), 1)
            self.assertEqual(window.snapshot_table.rowCount(), 1)
            window.snapshot_table.selectRow(0)
            self.app.processEvents()
            self.assertIsNotNone(window.snapshot_preview.pixmap())
        window.close()


if __name__ == "__main__":
    unittest.main()
