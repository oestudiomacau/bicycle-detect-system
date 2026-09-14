import unittest

from campus_monitor.domain import ParkingRuleService, SpeedMeasurementService, validate_parking_zone


class SpeedMeasurementServiceTest(unittest.TestCase):
    def test_calculates_interval_speed(self) -> None:
        service = SpeedMeasurementService(distance_m=12.0, threshold_kmh=15.0)
        self.assertAlmostEqual(service.calculate_kmh(10.0, 13.0), 14.4)

    def test_marks_speed_above_threshold(self) -> None:
        service = SpeedMeasurementService(distance_m=12.0, threshold_kmh=15.0)
        self.assertTrue(service.is_violation(15.1))
        self.assertFalse(service.is_violation(15.0))


class ParkingRuleServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ParkingRuleService(zone=(0.1, 0.1, 0.9, 0.9))

    def test_detects_outside_parking(self) -> None:
        self.assertEqual(self.service.classify((0.82, 0.4, 0.2, 0.1)), "越界停放")

    def test_detects_fallen_vehicle(self) -> None:
        self.assertEqual(self.service.classify((0.2, 0.2, 0.2, 0.1), angle_deg=60), "倒地异常")

    def test_detects_stacked_vehicles(self) -> None:
        self.assertEqual(self.service.classify((0.2, 0.2, 0.2, 0.1), nearby_count=3), "异常堆放")

    def test_accepts_normal_parking(self) -> None:
        self.assertIsNone(self.service.classify((0.2, 0.2, 0.2, 0.1)))

    def test_updates_manually_calibrated_zone(self) -> None:
        self.service.set_zone((0.2, 0.2, 0.8, 0.8))
        self.assertEqual(self.service.zone, (0.2, 0.2, 0.8, 0.8))

    def test_rejects_tiny_or_out_of_range_zone(self) -> None:
        with self.assertRaises(ValueError):
            validate_parking_zone((0.2, 0.2, 0.21, 0.8))
        with self.assertRaises(ValueError):
            validate_parking_zone((-0.1, 0.2, 0.8, 0.8))


if __name__ == "__main__":
    unittest.main()
