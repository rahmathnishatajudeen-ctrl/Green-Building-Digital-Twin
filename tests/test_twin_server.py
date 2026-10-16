from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from gbdt_vision.twin_server import create_app, load_floor


class TwinServerTests(unittest.TestCase):
    def test_dashboard_assets_are_served(self):
        client = TestClient(create_app())
        self.assertIn("Floor Twin", client.get("/").text)
        for asset in ("app.js", "style.css", "dashboard.css"):
            self.assertEqual(client.get(f"/static/{asset}").status_code, 200)

    def test_live_missing_file_is_unknown_and_demo_is_separate(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "observations.jsonl"
            client = TestClient(create_app(observations_path=path))
            live = client.get("/api/twin")
            self.assertEqual(live.status_code, 200)
            self.assertEqual(live.headers["cache-control"], "no-store")
            self.assertTrue(all(room["status"] == "unknown" for room in live.json()["rooms"]))
            demo = client.get("/api/twin?mode=demo").json()
            self.assertEqual(demo["mode"], "demo")
            self.assertTrue(all(room["source_type"] == "demo" for room in demo["rooms"]))
            self.assertFalse(path.exists())
            self.assertEqual(client.get("/api/twin?mode=invalid").status_code, 422)

    def test_new_observations_are_read_on_next_request(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "observations.jsonl"
            client = TestClient(create_app(observations_path=path))
            record = {"schema_version": "1.0", "status": "observed", "source_type": "camera",
                      "camera_id": "camera-1", "observed_at": datetime.now(timezone.utc).isoformat(),
                      "rooms": [{"room_id": "room-1", "occupancy_count": 4}]}
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            result = client.get("/api/twin").json()
            self.assertEqual(result["rooms"][0]["occupancy_count"], 4)
            self.assertEqual(result["summary"]["reporting_rooms"], 1)
            record["rooms"][0]["occupancy_count"] = 1
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(client.get("/api/twin").json()["summary"]["visible_people"], 1)

    def test_invalid_bindings_are_rejected(self):
        floor = json.loads(Path("config/floor-plan.json").read_text())
        floor["rooms"][1]["camera_id"] = floor["rooms"][0]["camera_id"]
        floor["rooms"][1]["vision_room_id"] = floor["rooms"][0]["vision_room_id"]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "floor.json"
            path.write_text(json.dumps(floor))
            with self.assertRaises(ValueError):
                load_floor(path)


if __name__ == "__main__":
    unittest.main()