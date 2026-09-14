import json
import tempfile
import unittest
from pathlib import Path

from campus_monitor.domain import DEFAULT_PARKING_ZONE
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


if __name__ == "__main__":
    unittest.main()
