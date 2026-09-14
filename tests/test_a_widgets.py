import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QImage, QPalette
from PySide6.QtWidgets import QApplication

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


if __name__ == "__main__":
    unittest.main()
