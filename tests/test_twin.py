from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gbdt_vision.twin import build_snapshot, read_observations


class TwinTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
        self.floor = {"name": "Test", "rooms": [{"id": "workspace", "camera_id": "camera-1", "vision_room_id": "room-1"}]}

    def observation(self, count=2, age=0, camera="camera-1"):
        return {"schema_version": "1.0", "status": "observed", "camera_id": camera,
                "source_type": "camera", "observed_at": (self.now - timedelta(seconds=age)).isoformat(),
                "rooms": [{"room_id": "room-1", "occupancy_count": count}]}

    def snapshot(self, records):
        return build_snapshot(self.floor, records, self.now)

    def test_fresh_occupied_and_empty(self):
        for count, status in ((2, "occupied"), (0, "empty")):
            room = self.snapshot([self.observation(count)])["rooms"][0]
            self.assertEqual(room["status"], status)
            self.assertEqual(room["occupancy_count"], count)

    def test_stale_zero_is_not_empty(self):
        result = self.snapshot([self.observation(0, 16)])
        self.assertEqual(result["rooms"][0]["status"], "stale")
        self.assertIsNone(result["rooms"][0]["occupancy_count"])
        self.assertEqual(result["summary"]["reporting_rooms"], 0)

    def test_unknown_without_matching_camera(self):
        self.assertEqual(self.snapshot([self.observation(camera="other")])["rooms"][0]["status"], "unknown")

    def test_latest_timestamp_wins_not_file_order(self):
        result = self.snapshot([self.observation(3), self.observation(1, 5)])
        self.assertEqual(result["summary"]["visible_people"], 3)

    def test_invalid_records_and_future_times_are_rejected(self):
        result = self.snapshot([self.observation(-1), self.observation(True), self.observation(age=-30), {}])
        self.assertEqual(result["rejected_records"], 4)
        self.assertEqual(result["rooms"][0]["status"], "unknown")

    def test_stale_brightness_is_hidden(self):
        observation = self.observation(age=20)
        observation["rooms"][0]["lighting"] = {"mean_brightness": 0.7}
        self.assertIsNone(self.snapshot([observation])["rooms"][0]["brightness"])

    def test_partial_jsonl_and_malformed_lines(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "observations.jsonl"
            path.write_bytes(b'{"status":"observed"}\ninvalid\n{"partial":')
            records, invalid = read_observations(path)
            self.assertEqual(records, [{"status": "observed"}])
            self.assertEqual(invalid, 1)
            self.assertEqual(read_observations(Path(directory) / "missing"), ([], 0))


if __name__ == "__main__":
    unittest.main()