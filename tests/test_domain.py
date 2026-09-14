import unittest

from campus_monitor.domain import (
    ParkingRuleService,
    SpeedMeasurementService,
    segments_intersect,
    validate_parking_zone,
)


class SpeedMeasurementServiceTest(unittest.TestCase):
    def test_calculates_interval_speed(self) -> None:
        service = SpeedMeasurementService(distance_m=12.0, threshold_kmh=15.0)
        self.assertAlmostEqual(service.calculate_kmh(10.0, 13.0), 14.4)

    def test_marks_speed_above_threshold(self) -> None:
        service = SpeedMeasurementService(distance_m=12.0, threshold_kmh=15.0)
        self.assertTrue(service.is_violation(15.1))
        self.assertFalse(service.is_violation(15.0))

    def test_arbitrary_line_segment_crossing(self) -> None:
        diagonal_line = (0.5, 0.2, 0.7, 0.8)
        self.assertTrue(segments_intersect((0.4, 0.5), (0.8, 0.5), diagonal_line))
        self.assertFalse(segments_intersect((0.4, 0.05), (0.8, 0.05), diagonal_line))


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

    def test_uses_irregular_polygon_boundary(self) -> None:
        self.service.set_polygon(((0.1, 0.1), (0.9, 0.2), (0.7, 0.9), (0.2, 0.8)))
        self.assertIsNone(self.service.classify((0.35, 0.35, 0.15, 0.12)))
        self.assertEqual(self.service.classify((0.78, 0.7, 0.15, 0.12)), "越界停放")


if __name__ == "__main__":
    unittest.main()
