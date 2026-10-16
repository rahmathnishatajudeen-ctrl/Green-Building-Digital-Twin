"""Local HTTP service for the floor-plan twin."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from threading import RLock
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel, Field

from gbdt_vision.lighting import attach_lighting, classify_brightness
from gbdt_vision.recommendations import RecommendationInput, RecommendationService
from gbdt_vision.twin import build_snapshot, read_observations


class LightState(BaseModel):
    state: Literal["on", "off"]


class LightReading(LightState):
    room_id: str = Field(min_length=1)
    observed_at: datetime | None = None


class RecommendationRequest(BaseModel):
    mode: Literal["live", "demo", "manual"] = "manual"
    room_id: str | None = Field(default=None, max_length=128)
    people_count: int | None = Field(default=None, strict=True, ge=0, le=10000)
    lighting_condition: Literal["low", "normal", "excessive"] | None = None


def load_floor(path):
    floor = json.loads(Path(path).read_text(encoding="utf-8"))
    for dimension in ("width", "height"):
        if not isinstance(floor[dimension], (int, float)) or not math.isfinite(floor[dimension]) or floor[dimension] <= 0:
            raise ValueError("Floor dimensions must be finite and positive")
    if not floor["rooms"]:
        raise ValueError("Floor must contain rooms")
    identifiers = set()
    bindings = set()
    for room in floor["rooms"]:
        for field in ("id", "name", "label", "use", "camera_id", "vision_room_id"):
            if not isinstance(room[field], str) or not room[field].strip():
                raise ValueError(f"Room {field} must be a nonempty string")
        binding = (room["camera_id"], room["vision_room_id"])
        if room["id"] in identifiers or binding in bindings:
            raise ValueError("Room IDs and camera-room bindings must be unique")
        identifiers.add(room["id"])
        bindings.add(binding)
        if type(room["capacity"]) is not int or room["capacity"] <= 0:
            raise ValueError("Room capacity must be a positive integer")
        lighting = room.get("lighting", {})
        if not isinstance(lighting, dict):
            raise ValueError("Lighting configuration must be an object")
        classify_brightness(None, lighting.get("low_brightness_threshold", 0.30),
                            lighting.get("high_brightness_threshold", 0.75))
        if not all(isinstance(room[field], (int, float)) and math.isfinite(room[field]) for field in ("x", "y", "width", "height")):
            raise ValueError("Room geometry must be finite")
        if not (room["x"] >= 0 and room["y"] >= 0 and room["width"] > 0 and room["height"] > 0
                and room["x"] + room["width"] <= floor["width"]
                and room["y"] + room["height"] <= floor["height"]):
            raise ValueError("Room geometry must fit inside the floor")
    return floor


def demo_observations(floor, now):
    records = []
    for index, room in enumerate(floor["rooms"]):
        count = min(4, room["capacity"]) if index % 3 == 0 else 0
        records.append({"schema_version": "1.0", "status": "observed", "source_type": "demo",
                        "observed_at": now.isoformat(), "camera_id": room["camera_id"],
                        "rooms": [{"room_id": room["vision_room_id"], "occupancy_count": count,
                                   "lighting": {"mean_brightness": (0.15, 0.90, 0.50)[index % 3], "scope": "full_frame"}}]})
    return records


def create_app(floor_path=Path("config/floor-plan.json"), observations_path=Path("data/observations.jsonl"), stale_seconds=15,
               lighting_path=Path("data/lighting.jsonl"), recommendation_service=None):
    if not math.isfinite(stale_seconds) or stale_seconds <= 0:
        raise ValueError("stale_seconds must be positive and finite")
    floor = load_floor(floor_path)
    advisor = recommendation_service if recommendation_service is not None else RecommendationService()
    lock = RLock()
    app = FastAPI(title="Green Building Floor-plan Twin")
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static, check_dir=False), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static / "index.html")

    def current_snapshot(mode):
        now = datetime.now(timezone.utc)
        invalid = 0
        ingestion_error = None
        if mode == "demo":
            observations = demo_observations(floor, now)
        else:
            try:
                observations, invalid = read_observations(Path(observations_path))
            except OSError:
                observations = []
                ingestion_error = "Observation file is unavailable"
        result = build_snapshot(floor, observations, now, stale_seconds)
        attach_lighting(result, [], now, stale_seconds)
        result.update({"mode": mode, "ingestion_error": ingestion_error,
                       "invalid_lines": invalid, "sample_layout": True})
        return result

    @app.get("/api/twin")
    def snapshot(response: Response, mode: Literal["live", "demo"] = "live"):
        response.headers["Cache-Control"] = "no-store"
        return current_snapshot(mode)

    @app.get("/api/recommendations/status")
    def recommendation_status(response: Response):
        response.headers["Cache-Control"] = "no-store"
        return advisor.status()

    def recommendation_room(mode, room_id):
        room = next((room for room in current_snapshot(mode)["rooms"] if room["id"] == room_id), None)
        if room is None:
            raise HTTPException(404, "Unknown room")
        if room["occupancy_count"] is None or room["lighting_analysis"]["status"] == "unknown":
            raise HTTPException(409, "Fresh occupancy and frame brightness are required")
        return room

    @app.post("/api/recommendations")
    def recommend(request: RecommendationRequest, response: Response):
        response.headers["Cache-Control"] = "no-store"
        basis = {"mode": request.mode, "room_id": None, "room_name": None, "observed_at": None, "source_type": "manual"}
        if request.mode == "manual":
            if request.people_count is None or request.lighting_condition is None:
                raise HTTPException(422, "Manual scenarios require people_count and lighting_condition")
            inputs = RecommendationInput(people_count=request.people_count, lighting_condition=request.lighting_condition)
        else:
            if request.room_id is None:
                raise HTTPException(422, "Select a room for observed recommendations")
            room = recommendation_room(request.mode, request.room_id)
            inputs = RecommendationInput(people_count=room["occupancy_count"], lighting_condition=room["lighting_analysis"]["status"])
            basis.update({"room_id": room["id"], "room_name": room["name"], "observed_at": room["observed_at"],
                          "source_type": room["source_type"]})
        result = advisor.recommend(inputs)
        if request.mode != "manual":
            latest = recommendation_room(request.mode, request.room_id)
            if (latest["occupancy_count"], latest["lighting_analysis"]["status"]) != (inputs.people_count, inputs.lighting_condition):
                raise HTTPException(409, "Room inputs changed during generation; generate again")
        return {**result, "basis": basis}

    @app.post("/api/lighting/readings", status_code=201)
    def ingest_light(reading: LightReading):
        if reading.room_id not in {room["id"] for room in floor["rooms"]}:
            raise HTTPException(404, "Unknown floor-plan room ID")
        now = datetime.now(timezone.utc)
        timestamp = reading.observed_at or now
        if timestamp.tzinfo is None or (timestamp - now).total_seconds() > 5:
            raise HTTPException(422, "observed_at must be timezone-aware and not in the future")
        record = {"room_id": reading.room_id, "state": reading.state, "source": "sensor", "observed_at": timestamp.isoformat()}
        try:
            with lock:
                path = Path(lighting_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record) + "\n")
        except OSError:
            raise HTTPException(503, "Lighting observation storage is unavailable") from None
        return record

    return app


def main():
    parser = argparse.ArgumentParser(description="Green Building floor-plan twin dashboard")
    parser.add_argument("--floor-plan", type=Path, default=Path("config/floor-plan.json"))
    parser.add_argument("--observations", type=Path, default=Path("data/observations.jsonl"))
    parser.add_argument("--lighting-data", type=Path, default=Path("data/lighting.jsonl"))
    parser.add_argument("--stale-seconds", type=float, default=15)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        app = create_app(args.floor_plan, args.observations, args.stale_seconds, args.lighting_data)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Twin configuration error: {error}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()