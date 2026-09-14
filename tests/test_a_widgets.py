import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

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


if __name__ == "__main__":
    unittest.main()
