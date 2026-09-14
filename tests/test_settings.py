import json
import tempfile
import unittest
from pathlib import Path

from campus_monitor.domain import DEFAULT_PARKING_ZONE, DEFAULT_SPEED_LINES
from campus_monitor.settings import SettingsStore


class SettingsStoreTest(unittest.TestCase):
    def test_round_trips_parking_zone(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = SettingsStore(Path(folder) / "settings.json")
            zone = (0.1, 0.2, 0.8, 0.9)
            store.save_parking_zone(zone)
            self.assertEqual(store.load_parking_zone(), zone)

    def test_invalid_file_falls_back_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps({"parking_zone": [2, 0, 3, 1]}), encoding="utf-8")
            self.assertEqual(SettingsStore(path).load_parking_zone(), DEFAULT_PARKING_ZONE)

    def test_round_trips_drawn_lines_and_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = SettingsStore(Path(folder) / "settings.json")
            lines = ((0.2, 0.1, 0.3, 0.9), (0.7, 0.1, 0.8, 0.9))
            polygon = ((0.1, 0.2), (0.8, 0.1), (0.9, 0.8), (0.2, 0.9))
            store.save_speed_lines(lines)
            store.save_parking_polygon(polygon)
            self.assertEqual(store.load_speed_lines(), lines)
            self.assertEqual(store.load_parking_polygon(), polygon)

    def test_invalid_speed_lines_fall_back_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps({"speed_lines": [[0, 0, 0, 0]]}), encoding="utf-8")
            self.assertEqual(SettingsStore(path).load_speed_lines(), DEFAULT_SPEED_LINES)

    def test_round_trips_startup_video_and_mode(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "demo.mp4"
            video.touch()
            store = SettingsStore(Path(folder) / "settings.json")

            store.save_startup_video(video, "parking")

            self.assertEqual(store.load_startup_video(), (video.resolve(), "parking"))

    def test_missing_startup_video_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = SettingsStore(Path(folder) / "settings.json")
            missing = Path(folder) / "missing.mp4"

            store.save_startup_video(missing, "road", require_exists=False)

            self.assertIsNone(store.load_startup_video())

    def test_clears_startup_video_without_changing_other_settings(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "demo.mp4"
            video.touch()
            store = SettingsStore(Path(folder) / "settings.json")
            store.save_speed_lines(DEFAULT_SPEED_LINES)
            store.save_startup_video(video, "road")

            store.clear_startup_video()

            self.assertIsNone(store.load_startup_video())
            self.assertFalse(store.startup_video_enabled())
            self.assertEqual(store.load_speed_lines(), DEFAULT_SPEED_LINES)

    def test_startup_video_is_enabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = SettingsStore(Path(folder) / "settings.json")

            self.assertTrue(store.startup_video_enabled())


if __name__ == "__main__":
    unittest.main()
