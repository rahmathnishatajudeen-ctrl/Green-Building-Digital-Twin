"""Camera-independent detection and room-state contracts."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Zone:
    room_id: str
    bounds: tuple[float, float, float, float]
    lighting_bounds: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.room_id.strip():
            raise ValueError("room_id must not be empty")
        for bounds in (self.bounds, self.lighting_bounds):
            if bounds is None:
                continue
            if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
                raise ValueError("Bounds must contain four finite numbers")
            left, top, right, bottom = bounds
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise ValueError("Bounds must be normalized [left, top, right, bottom]")


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    confidence: float


def assign_occupancy(
    detections: list[Detection], zones: list[Zone], width: int, height: int
) -> tuple[dict[str, int], int]:
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    counts = {zone.room_id: 0 for zone in zones}
    if len(counts) != len(zones):
        raise ValueError("Room IDs must be unique")
    unassigned = 0
    for detection in detections:
        left, _top, right, bottom = detection.box
        anchor_x = (left + right) / (2 * width)
        anchor_y = bottom / height
        for zone in zones:
            zone_left, zone_top, zone_right, zone_bottom = zone.bounds
            inside_x = zone_left <= anchor_x < zone_right or anchor_x == zone_right == 1
            inside_y = zone_top <= anchor_y < zone_bottom or anchor_y == zone_bottom == 1
            if inside_x and inside_y:
                counts[zone.room_id] += 1
                break
        else:
            unassigned += 1
    return counts, unassigned