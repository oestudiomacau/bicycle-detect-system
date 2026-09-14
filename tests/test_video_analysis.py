import unittest
from pathlib import Path

import cv2

from campus_monitor.video_analysis import CentroidTracker, Detection, MobileNetBicycleDetector


ROOT = Path(__file__).resolve().parents[1]


class CentroidTrackerTest(unittest.TestCase):
    def test_keeps_track_id_for_nearby_detection(self) -> None:
        tracker = CentroidTracker()
        first = tracker.update([Detection((0.2, 0.2, 0.2, 0.2), 0.8)])
        second = tracker.update([Detection((0.24, 0.2, 0.2, 0.2), 0.85)])
        self.assertEqual(first[0].track_id, second[0].track_id)


class SampleVideoDetectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = MobileNetBicycleDetector(
            ROOT / "models" / "mobilenet_ssd.prototxt",
            ROOT / "models" / "mobilenet_ssd.caffemodel",
            confidence=0.25,
        )

    def test_road_sample_contains_detectable_bicycle(self) -> None:
        frame = self._read_frame(ROOT / "sample_videos" / "road_cyclists.mp4", 66)
        self.assertGreaterEqual(len(self.detector.detect(frame)), 1)

    def test_parking_sample_contains_detectable_bicycle(self) -> None:
        frame = self._read_frame(ROOT / "sample_videos" / "parking_dense.mp4", 90)
        self.assertGreaterEqual(len(self.detector.detect(frame)), 1)

    @staticmethod
    def _read_frame(path: Path, frame_index: int):
        capture = cv2.VideoCapture(str(path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise AssertionError(f"无法读取示例视频帧：{path}")
        return frame


if __name__ == "__main__":
    unittest.main()

