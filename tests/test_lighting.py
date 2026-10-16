from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from gbdt_vision.lighting import classify_brightness, evaluate_lighting
from gbdt_vision.twin_server import create_app, load_floor


class BrightnessClassificationTests(unittest.TestCase):
    def test_three_scenarios_and_exact_boundaries(self):
        for brightness, status in ((0, "low"), (0.29, "low"), (0.30, "normal"),
                                   (0.5, "normal"), (0.75, "normal"), (0.76, "excessive"), (1, "excessive")):
            self.assertEqual(classify_brightness(brightness)["status"], status)

    def test_missing_and_invalid_brightness_are_unknown(self):
        for brightness in (None, True, -1, 2, float("nan"), float("inf"), "bright"):
            self.assertEqual(classify_brightness(brightness)["status"], "unknown")

    def test_custom_thresholds(self):
        self.assertEqual(classify_brightness(0.4, 0.45, 0.8)["status"], "low")
        with self.assertRaises(ValueError):
            classify_brightness(0.4, 0.8, 0.3)


class LightingTests(unittest.TestCase):
    def test_classification_does_not_depend_on_occupancy_or_sensor(self):
        now = datetime.now(timezone.utc)
        for occupancy in ("occupied", "empty"):
            room = {"status": occupancy, "age_seconds": 0, "brightness": 0.1, "source_type": "camera"}
            result = evaluate_lighting(room, {"state": "on"}, now)
            self.assertEqual(result["status"], "low")
            self.assertEqual(result["message"], "The scene is too dark.")

    def test_stale_and_missing_brightness_are_unknown(self):
        for status, age, brightness in (("stale", 20, 0.8), ("occupied", 20, 0.8), ("empty", 0, None)):
            room = {"status": status, "age_seconds": age, "brightness": brightness}
            self.assertEqual(evaluate_lighting(room, None, datetime.now(timezone.utc))["status"], "unknown")

    def test_demo_has_all_three_scenarios_without_live_file_changes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vision.jsonl"
            client = TestClient(create_app(observations_path=path))
            result = client.get("/api/twin?mode=demo").json()
            self.assertEqual([room["lighting_analysis"]["status"] for room in result["rooms"]], ["low", "excessive", "normal"])
            self.assertEqual(result["lighting_summary"], {"low_rooms": 1, "normal_rooms": 1, "excessive_rooms": 1, "unknown_rooms": 0})
            self.assertFalse(path.exists())
            self.assertEqual(client.get("/api/twin").json()["lighting_summary"]["unknown_rooms"], 3)

    def test_live_frame_brightness_updates_the_scenario(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vision.jsonl"
            client = TestClient(create_app(observations_path=path))
            for brightness, expected, age in ((0.1, "low", 0), (0.5, "normal", 0), (0.9, "excessive", 0), (0.5, "unknown", 20)):
                record = {"schema_version": "1.0", "status": "observed", "source_type": "camera", "camera_id": "camera-1",
                          "observed_at": (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat(),
                          "rooms": [{"room_id": "room-1", "occupancy_count": 1,
                                     "lighting": {"mean_brightness": brightness, "scope": "full_frame"}}]}
                path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                result = client.get("/api/twin").json()["rooms"][0]["lighting_analysis"]
                self.assertEqual(result["status"], expected)

    def test_invalid_threshold_configuration_is_rejected(self):
        floor = json.loads(Path("config/floor-plan.json").read_text())
        floor["rooms"][0]["lighting"] = {"low_brightness_threshold": 0.8, "high_brightness_threshold": 0.3}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "floor.json"
            path.write_text(json.dumps(floor))
            with self.assertRaises(ValueError):
                load_floor(path)


if __name__ == "__main__":
    unittest.main()