"""Classify camera brightness without inferring lux or light-switch state."""

import math


def classify_brightness(brightness, low_threshold=0.30, high_threshold=0.75):
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
               for value in (low_threshold, high_threshold)) or not 0 <= low_threshold < high_threshold <= 1:
        raise ValueError("Brightness thresholds must satisfy 0 <= low < high <= 1")
    result = {"status": "unknown", "label": "Unknown lighting", "brightness": None,
              "low_threshold": low_threshold, "high_threshold": high_threshold,
              "message": "No fresh frame brightness is available."}
    if not isinstance(brightness, (int, float)) or isinstance(brightness, bool) or not math.isfinite(brightness) or not 0 <= brightness <= 1:
        return result
    if brightness < low_threshold:
        status, label, message = "low", "Low lighting", "The scene is too dark."
    elif brightness > high_threshold:
        status, label, message = "excessive", "Excessive lighting", "The scene is too bright."
    else:
        status, label, message = "normal", "Normal lighting", "The scene has normal lighting."
    result.update(status=status, label=label, message=message, brightness=brightness)
    return result


def evaluate_lighting(room, reading, now, stale_seconds=15):
    config = room.get("lighting", {})
    age = room.get("age_seconds")
    fresh = room.get("status") in ("empty", "occupied") and age is not None and age <= stale_seconds
    result = classify_brightness(room.get("brightness") if fresh else None,
                                 config.get("low_brightness_threshold", 0.30),
                                 config.get("high_brightness_threshold", 0.75))
    result.update({"evidence": "none" if result["status"] == "unknown" else
                   "simulated" if room.get("source_type") == "demo" else "camera_brightness",
                   "reading_age_seconds": age, "scope": room.get("brightness_scope", "unspecified")})
    return result


def attach_lighting(snapshot, records, now, stale_seconds=15):
    for room in snapshot["rooms"]:
        room["lighting_analysis"] = evaluate_lighting(room, None, now, stale_seconds)
    snapshot["lighting_summary"] = {f"{status}_rooms": sum(room["lighting_analysis"]["status"] == status
                                                          for room in snapshot["rooms"])
                                     for status in ("low", "normal", "excessive", "unknown")}
    return snapshot