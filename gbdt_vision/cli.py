"""Video ingestion, YOLO inference, and digital-twin observation output."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from gbdt_vision.state import Detection, Zone, assign_occupancy


def load_zones(path: Path | None) -> list[Zone]:
    if path is None:
        return [Zone("room-1", (0, 0, 1, 1))]
    data = json.loads(path.read_text(encoding="utf-8"))
    zones = [
        Zone(
            entry["room_id"], tuple(entry["bounds"]),
            tuple(entry["lighting_bounds"]) if entry.get("lighting_bounds") is not None else None,
        )
        for entry in data["zones"]
    ]
    if not zones:
        raise ValueError("Configure at least one room zone")
    assign_occupancy([], zones, 1, 1)
    return zones


def analyze_frame(frame, detector, zones, confidence, image_size, device):
    height, width = frame.shape[:2]
    result = detector.predict(
        frame, classes=[0], conf=confidence, imgsz=image_size, device=device, verbose=False,
    )[0]
    detections = []
    if result.boxes is not None:
        for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            detections.append(Detection(tuple(box), float(score)))
    counts, unassigned = assign_occupancy(detections, zones, width, height)
    rooms = []
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_brightness = float(np.mean(grayscale)) / 255
    for zone in zones:
        mean = frame_brightness
        scope = "full_frame"
        if zone.lighting_bounds is not None:
            left, top, right, bottom = zone.lighting_bounds
            crop = grayscale[int(top * height):math.ceil(bottom * height), int(left * width):math.ceil(right * width)]
            mean = float(np.mean(crop)) / 255
            scope = "configured_region"
        lighting = {
            "status": "observed", "mean_brightness": round(mean, 4), "scope": scope,
            "measurement": "relative_camera_brightness_not_lux_or_light_switch_state",
        }
        rooms.append({"room_id": zone.room_id, "occupancy_count": counts[zone.room_id], "lighting": lighting})
    observation = {
        "schema_version": "1.0", "status": "observed",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "frame_size": {"width": width, "height": height},
        "total_visible_people": len(detections), "unassigned_people": unassigned,
        "rooms": rooms,
    }
    return observation, detections


def draw_preview(frame, zones, observation, detections):
    preview = frame.copy()
    height, width = preview.shape[:2]
    for zone, room in zip(zones, observation["rooms"]):
        left, top, right, bottom = zone.bounds
        start = (int(left * width), int(top * height))
        end = (min(width - 1, int(right * width)), min(height - 1, int(bottom * height)))
        cv2.rectangle(preview, start, end, (80, 200, 100), 2)
        label = f"{zone.room_id}: {room['occupancy_count']}"
        cv2.putText(preview, label, (start[0] + 6, min(height - 8, start[1] + 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 100), 2)
    for detection in detections:
        left, top, right, bottom = map(int, detection.box)
        cv2.rectangle(preview, (left, top), (right, bottom), (240, 180, 60), 2)
    return preview


def build_parser():
    parser = argparse.ArgumentParser(description="Green Building Digital Twin: computer vision layer")
    parser.add_argument("--source", default="0", help="Webcam index or local video path; default: 0")
    parser.add_argument("--camera-id", default="camera-1", help="Non-identifying source label in observations")
    parser.add_argument("--zones", type=Path, help="JSON room-zone configuration")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO checkpoint; downloads on first run")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu, or CUDA device index such as 0")
    parser.add_argument("--every", type=int, default=5, help="Analyze every Nth frame (first always analyzed)")
    parser.add_argument("--max-frames", type=int, help="Stop after N analyzed frames")
    parser.add_argument("--output", type=Path, default=Path("data/observations.jsonl"))
    parser.add_argument("--preview", action="store_true", help="Show local video with counts; Q exits")
    return parser


def run(args):
    if not 0 < args.confidence <= 1:
        raise ValueError("confidence must be greater than 0 and at most 1")
    if args.every < 1 or args.image_size < 32 or (args.max_frames is not None and args.max_frames < 1):
        raise ValueError("every/max-frames must be positive and image-size must be at least 32")
    zones = load_zones(args.zones)
    is_camera = args.source.isdecimal()
    source = int(args.source) if is_camera else str(Path(args.source))
    if not is_camera and not Path(source).is_file():
        raise ValueError("source must be a webcam index or an existing local video file")
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise RuntimeError("Cannot open video source. Check the path, permissions, or camera index.")
        detector = YOLO(args.model)
        fps = capture.get(cv2.CAP_PROP_FPS)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame_index = 0
        analyzed = 0
        print(f"Writing observations to {args.output}; no images are saved.", file=sys.stderr)
        with args.output.open("a", encoding="utf-8") as output:
            while True:
                success, frame = capture.read()
                if not success:
                    if is_camera or analyzed == 0:
                        raise RuntimeError("Video capture failed; no zero-occupancy observation was emitted.")
                    break
                current_index = frame_index
                frame_index += 1
                if current_index % args.every:
                    continue
                observation, detections = analyze_frame(
                    frame, detector, zones, args.confidence, args.image_size, args.device,
                )
                observation.update({
                    "camera_id": args.camera_id, "source_type": "camera" if is_camera else "video",
                    "frame_index": current_index,
                    "video_offset_seconds": round(current_index / fps, 3)
                    if not is_camera and math.isfinite(fps) and fps > 0 else None,
                    "model": Path(args.model).name, "confidence_threshold": args.confidence,
                })
                output.write(json.dumps(observation, allow_nan=False) + "\n")
                output.flush()
                print(f"frame={current_index} people={observation['total_visible_people']} "
                      f"rooms={[(room['room_id'], room['occupancy_count']) for room in observation['rooms']]}",
                      file=sys.stderr)
                analyzed += 1
                if args.preview:
                    cv2.imshow("Green Building Vision | Q to exit", draw_preview(frame, zones, observation, detections))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if args.max_frames is not None and analyzed >= args.max_frames:
                    break
    finally:
        capture.release()
        if args.preview:
            cv2.destroyAllWindows()


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Vision error: {error}\n")
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()