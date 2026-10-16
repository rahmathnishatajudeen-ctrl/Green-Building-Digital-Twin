"""Map camera observations to freshness-aware floor-plan room states."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path


def read_observations(path: Path, max_bytes: int = 2_000_000):
    if not path.exists():
        return [], 0
    with path.open("rb") as source:
        size = source.seek(0, 2)
        offset = max(0, size - max_bytes)
        source.seek(offset)
        if offset:
            source.readline()
        lines = source.read(max_bytes).splitlines(keepends=True)
    records = []
    invalid = 0
    for line in lines:
        if not line.endswith(b"\n"):
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Observation must be an object")
            records.append(record)
        except (ValueError, UnicodeDecodeError):
            invalid += 1
    return records, invalid


def build_snapshot(floor, observations, now=None, stale_seconds=15):
    now = now or datetime.now(timezone.utc)
    latest = {}
    histories = {}
    rejected = 0
    for observation in observations:
        try:
            if observation.get("status") != "observed" or observation.get("schema_version") != "1.0":
                raise ValueError("Unsupported observation")
            timestamp = datetime.fromisoformat(observation["observed_at"])
            if timestamp.tzinfo is None or (timestamp - now).total_seconds() > 5:
                raise ValueError("Invalid observation time")
            camera_id = observation["camera_id"]
            if not isinstance(camera_id, str):
                raise ValueError("Invalid camera ID")
            rooms = observation["rooms"]
            if not isinstance(rooms, list):
                raise ValueError("Invalid rooms")
            updates = []
            seen = set()
            for room in rooms:
                room_id = room["room_id"]
                count = room["occupancy_count"]
                if not isinstance(room_id, str) or room_id in seen or type(count) is not int or count < 0:
                    raise ValueError("Invalid room count")
                seen.add(room_id)
                updates.append(((camera_id, room_id), room))
            for key, room in updates:
                histories.setdefault(key, []).append((timestamp, room["occupancy_count"], observation.get("source_type")))
                if key not in latest or timestamp > latest[key][0]:
                    latest[key] = (timestamp, room, observation.get("source_type", "unknown"))
        except (KeyError, ValueError, TypeError, AttributeError):
            rejected += 1
    states = []
    for configured in floor["rooms"]:
        state = {**configured, "status": "unknown", "occupancy_count": None,
                 "last_known_count": None, "observed_at": None, "age_seconds": None,
                 "brightness": None, "brightness_scope": None, "source_type": None, "vacant_seconds": None}
        entry = latest.get((configured["camera_id"], configured["vision_room_id"]))
        if entry:
            timestamp, room, source_type = entry
            age = max(0, (now - timestamp).total_seconds())
            fresh = age <= stale_seconds
            state.update({"status": ("occupied" if room["occupancy_count"] else "empty") if fresh else "stale",
                          "occupancy_count": room["occupancy_count"] if fresh else None,
                          "last_known_count": room["occupancy_count"], "observed_at": timestamp.isoformat(),
                          "age_seconds": round(age, 1), "source_type": source_type})
            if fresh and room["occupancy_count"] == 0 and source_type in ("camera", "demo"):
                earliest = timestamp
                history = histories[(configured["camera_id"], configured["vision_room_id"])]
                for previous, count, previous_source in sorted(history, key=lambda entry: entry[0], reverse=True):
                    if count != 0 or previous_source != source_type or (earliest - previous).total_seconds() > stale_seconds:
                        break
                    earliest = previous
                state["vacant_seconds"] = max(0, round((timestamp - earliest).total_seconds(), 1))
            lighting = room.get("lighting")
            brightness = lighting.get("mean_brightness") if isinstance(lighting, dict) else None
            if fresh and isinstance(brightness, (int, float)) and not isinstance(brightness, bool):
                if math.isfinite(brightness) and 0 <= brightness <= 1:
                    state["brightness"] = brightness
                    state["brightness_scope"] = lighting.get("scope", "unspecified")
        states.append(state)
    return {"generated_at": now.isoformat(), "floor": {key: value for key, value in floor.items() if key != "rooms"},
            "rooms": states, "stale_seconds": stale_seconds, "rejected_records": rejected,
            "summary": {"visible_people": sum(room["occupancy_count"] or 0 for room in states),
                        "reporting_rooms": sum(room["status"] in ("occupied", "empty") for room in states),
                        "total_rooms": len(states),
                        "occupied_rooms": sum(room["status"] == "occupied" for room in states)}}