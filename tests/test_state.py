import unittest

from gbdt_vision.state import Detection, Zone, assign_occupancy


class OccupancyTests(unittest.TestCase):
    def setUp(self):
        self.zones = [Zone("left", (0, 0, 0.5, 1)), Zone("right", (0.5, 0, 1, 1))]

    def test_assigns_by_bottom_center(self):
        counts, unassigned = assign_occupancy(
            [Detection((10, 10, 30, 80), 0.9), Detection((60, 10, 90, 80), 0.8)],
            self.zones, 100, 100,
        )
        self.assertEqual(counts, {"left": 1, "right": 1})
        self.assertEqual(unassigned, 0)

    def test_shared_boundary_counts_once(self):
        counts, _ = assign_occupancy([Detection((40, 0, 60, 100), 0.9)], self.zones, 100, 100)
        self.assertEqual(counts, {"left": 0, "right": 1})

    def test_overlapping_zones_use_first_match(self):
        zones = [Zone("first", (0, 0, 1, 1)), Zone("second", (0, 0, 1, 1))]
        counts, _ = assign_occupancy([Detection((10, 0, 30, 80), 0.9)], zones, 100, 100)
        self.assertEqual(counts, {"first": 1, "second": 0})

    def test_unassigned_detection(self):
        counts, unassigned = assign_occupancy(
            [Detection((60, 0, 80, 80), 0.9)], self.zones[:1], 100, 100,
        )
        self.assertEqual(counts, {"left": 0})
        self.assertEqual(unassigned, 1)

    def test_empty_frame_retains_zero_counts(self):
        self.assertEqual(assign_occupancy([], self.zones, 100, 100), ({"left": 0, "right": 0}, 0))

    def test_invalid_configuration(self):
        for bounds in ((0, 0, 2, 1), (0.5, 0, 0.2, 1), (0, 0, float("nan"), 1)):
            with self.assertRaises(ValueError):
                Zone("room", bounds)
        with self.assertRaises(ValueError):
            assign_occupancy([], self.zones * 2, 100, 100)
        with self.assertRaises(ValueError):
            assign_occupancy([], self.zones, 0, 100)


if __name__ == "__main__":
    unittest.main()