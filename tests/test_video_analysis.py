import unittest
import time
from pathlib import Path

import cv2

from campus_monitor.video_analysis import (
    CentroidTracker,
    Detection,
    MobileNetBicycleDetector,
    RTDetrWorkerDetector,
    TwoWheelerDetector,
    analyze_image,
    YoloXTwoWheelerDetector,
)


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
        cls.yolox_detector = YoloXTwoWheelerDetector(
            ROOT / "models" / "yolox_tiny.onnx"
        )

    def test_road_sample_contains_detectable_bicycle(self) -> None:
        frame = self._read_frame(ROOT / "sample_videos" / "road_cyclists.mp4", 66)
        self.assertGreaterEqual(len(self.detector.detect(frame)), 1)

    def test_parking_sample_contains_detectable_bicycle(self) -> None:
        frame = self._read_frame(ROOT / "sample_videos" / "parking_dense.mp4", 90)
        self.assertGreaterEqual(len(self.detector.detect(frame)), 1)

    def test_parking_sample_accepts_motorbike_as_two_wheeler(self) -> None:
        frame = self._read_frame(ROOT / "sample_videos" / "parking_removal.mp4", 376)

        detections = self.detector.detect(frame)

        self.assertTrue(any(item.label == "电动车/摩托车" for item in detections))

    def test_yolox_detects_dense_parked_two_wheelers(self) -> None:
        frame = self._read_frame(ROOT / "sample_videos" / "parking_dense.mp4", 90)

        detections = self.yolox_detector.detect(frame)

        self.assertGreaterEqual(len(detections), 5)

    def test_analyzes_image_with_current_detector(self) -> None:
        frame = self._read_frame(ROOT / "sample_videos" / "parking_dense.mp4", 90)
        image_path = ROOT / "runtime" / "test_image_detection.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), frame)
        detector = TwoWheelerDetector(
            ROOT / "models" / "mobilenet_ssd.prototxt",
            ROOT / "models" / "mobilenet_ssd.caffemodel",
            ROOT / "models" / "yolox_tiny.onnx",
        )
        try:
            image, detections = analyze_image(image_path, detector)

            self.assertEqual((image.width(), image.height()), (frame.shape[1], frame.shape[0]))
            self.assertGreaterEqual(len(detections), 5)
        finally:
            image_path.unlink(missing_ok=True)

    @staticmethod
    def _read_frame(path: Path, frame_index: int):
        capture = cv2.VideoCapture(str(path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise AssertionError(f"无法读取示例视频帧：{path}")
        return frame


class RTDetrWorkerTest(unittest.TestCase):
    MODEL_DIR = ROOT / "models" / "rtdetr_r50vd_coco_o365"
    ENVIRONMENT_PYTHON = ROOT / ".venv-anomalib" / "Scripts" / "python.exe"

    @unittest.skipUnless(
        ENVIRONMENT_PYTHON.is_file() and (MODEL_DIR / "model.safetensors").is_file(),
        "本机未安装可选 RT-DETR 模型",
    )
    def test_detects_parked_bicycles_through_worker(self) -> None:
        detector = RTDetrWorkerDetector(
            self.ENVIRONMENT_PYTHON,
            ROOT / "scripts" / "rtdetr_worker.py",
            self.MODEL_DIR,
            ROOT / "runtime",
        )
        try:
            detector.start()
            ready_deadline = time.monotonic() + 20
            while not detector.ready and time.monotonic() < ready_deadline:
                time.sleep(0.1)
            self.assertTrue(detector.ready, detector.status_text)
            frame = SampleVideoDetectionTest._read_frame(
                ROOT / "sample_videos" / "parking_dense.mp4",
                90,
            )
            detections = None
            result_deadline = time.monotonic() + 5
            while detections is None and time.monotonic() < result_deadline:
                detections = detector.detect(frame)
                time.sleep(0.05)
            self.assertIsNotNone(detections)
            self.assertGreaterEqual(len(detections or []), 5)
        finally:
            detector.shutdown()


if __name__ == "__main__":
    unittest.main()
