import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from gbdt_vision.cli import analyze_frame, build_parser, load_zones, run
from gbdt_vision.state import Zone


def fake_detector(boxes):
    coordinates = Mock()
    coordinates.cpu.return_value.tolist.return_value = boxes
    confidence = Mock()
    confidence.cpu.return_value.tolist.return_value = [0.9] * len(boxes)
    detector = Mock()
    detector.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(xyxy=coordinates, conf=confidence))]
    return detector


class PipelineTests(unittest.TestCase):
    def test_observation_and_brightness(self):
        detector = fake_detector([[10, 10, 30, 80]])
        zones = [Zone("room", (0, 0, 1, 1), (0, 0, 1, 1))]
        observation, _ = analyze_frame(np.full((100, 100, 3), 255, np.uint8), detector, zones, 0.4, 640, "cpu")
        self.assertEqual(observation["rooms"][0]["occupancy_count"], 1)
        self.assertEqual(observation["rooms"][0]["lighting"]["mean_brightness"], 1)
        self.assertEqual(observation["status"], "observed")
        self.assertTrue(observation["observed_at"].endswith("+00:00"))
        self.assertNotIn("detections", observation)
        self.assertEqual(detector.predict.call_args.kwargs["classes"], [0])
        json.dumps(observation, allow_nan=False)

    def test_full_frame_brightness_is_measured_without_configuration(self):
        observation, _ = analyze_frame(np.zeros((100, 100, 3), np.uint8), fake_detector([]),
                                       [Zone("room", (0, 0, 1, 1))], 0.4, 640, "cpu")
        self.assertEqual(observation["total_visible_people"], 0)
        self.assertEqual(observation["rooms"][0]["lighting"]["status"], "observed")
        self.assertEqual(observation["rooms"][0]["lighting"]["mean_brightness"], 0)
        self.assertEqual(observation["rooms"][0]["lighting"]["scope"], "full_frame")

    def test_brightness_changes_with_frame_and_region(self):
        for level in (20, 128, 240):
            frame = np.full((100, 100, 3), level, np.uint8)
            observation, _ = analyze_frame(frame, fake_detector([]), [Zone("room", (0, 0, 1, 1))], 0.4, 640, "cpu")
            self.assertAlmostEqual(observation["rooms"][0]["lighting"]["mean_brightness"], level / 255, places=3)
        frame = np.zeros((100, 100, 3), np.uint8)
        frame[:50] = 255
        observation, _ = analyze_frame(frame, fake_detector([]),
                                       [Zone("room", (0, 0, 1, 1), (0, 0, 1, 0.5))], 0.4, 640, "cpu")
        self.assertEqual(observation["rooms"][0]["lighting"]["mean_brightness"], 1)
        self.assertEqual(observation["rooms"][0]["lighting"]["scope"], "configured_region")

    def test_default_and_example_zones(self):
        self.assertEqual(load_zones(None)[0].room_id, "room-1")
        self.assertEqual(len(load_zones(Path("config/zones.example.json"))), 2)

    def test_invalid_configuration(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "zones.json"
            path.write_text('{"zones": []}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_zones(path)

    def test_video_sampling_and_jsonl(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "test.avi"
            output = Path(directory) / "observations.jsonl"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 100))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(6):
                    writer.write(np.zeros((100, 100, 3), np.uint8))
            finally:
                writer.release()
            args = build_parser().parse_args(["--source", str(source), "--every", "2", "--output", str(output)])
            with patch("gbdt_vision.cli.YOLO", return_value=fake_detector([])):
                run(args)
            observations = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([entry["frame_index"] for entry in observations], [0, 2, 4])
            self.assertEqual([entry["video_offset_seconds"] for entry in observations], [0, 0.2, 0.4])
            self.assertEqual(observations[0]["source_type"], "video")

    def test_capture_failure_is_not_zero_occupancy(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "observations.jsonl"
            args = build_parser().parse_args(["--output", str(output)])
            with patch("gbdt_vision.cli.cv2.VideoCapture", return_value=capture), patch("gbdt_vision.cli.YOLO"):
                with self.assertRaises(RuntimeError):
                    run(args)
            self.assertEqual(output.read_text(), "")
            capture.release.assert_called_once()

    def test_invalid_arguments(self):
        for arguments in (["--confidence", "0"], ["--every", "0"], ["--max-frames", "0"]):
            with self.assertRaises(ValueError):
                run(build_parser().parse_args(arguments))


if __name__ == "__main__":
    unittest.main()