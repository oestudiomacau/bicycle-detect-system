import unittest

from PySide6.QtCore import QCoreApplication

from campus_monitor.simulation import SimulationEngine


class SimulationEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self) -> None:
        self.engine = SimulationEngine()
        self.engine._timer.stop()
        self.events = []
        self.engine.event_raised.connect(self.events.append)

    def test_road_pipeline_emits_speed_records_and_warnings(self) -> None:
        for _ in range(400):
            self.engine._tick()
        types = {event.event_type for event in self.events}
        self.assertIn("速度记录", types)
        self.assertIn("超速告警", types)

    def test_parking_pipeline_emits_three_violation_types(self) -> None:
        self.engine.set_mode("parking")
        for _ in range(30):
            self.engine._tick()
        types = {event.event_type for event in self.events}
        self.assertEqual(types, {"越界停放", "倒地异常", "异常堆放"})


if __name__ == "__main__":
    unittest.main()

