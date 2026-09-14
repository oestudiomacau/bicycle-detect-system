import tempfile
import unittest
from pathlib import Path

from campus_monitor.anomalib_training import dataset_summary, validate_dataset


class AnomalibDatasetValidationTest(unittest.TestCase):
    def test_counts_supported_images_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "nested").mkdir()
            for index in range(4):
                (root / f"normal_{index}.jpg").touch()
            (root / "nested" / "normal.png").touch()
            (root / "notes.txt").touch()
            self.assertEqual(dataset_summary(root), {"normal": 5, "abnormal": 0})

    def test_requires_minimum_normal_images(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "one.jpg").touch()
            with self.assertRaises(ValueError):
                validate_dataset(root)


if __name__ == "__main__":
    unittest.main()
