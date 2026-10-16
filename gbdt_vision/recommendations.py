"""Grounded lighting advice from occupancy and camera-brightness scenarios."""

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from threading import RLock
from time import monotonic
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class RecommendationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    people_count: int = Field(strict=True, ge=0, le=10000)
    lighting_condition: Literal["low", "normal", "excessive"]


class Advice(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    summary: str = Field(min_length=1, max_length=500)
    actions: list[Annotated[str, Field(min_length=1, max_length=400)]] = Field(min_length=1, max_length=3)


CAUTION = (
    "Camera brightness is not lux or proof of light-switch state. Verify actual conditions before acting. "
    "Never disable emergency lighting, safety systems or required ventilation."
)


def rule_advice(inputs: RecommendationInput) -> Advice:
    people = inputs.people_count
    condition = inputs.lighting_condition
    if people == 0:
        actions = [
            "Verify the room is unoccupied, then switch off or dim only nonessential lights that are actually on.",
            "Keep emergency and egress lighting operational; do not brighten an unused room just to change the camera reading.",
        ]
        if condition == "excessive":
            actions.append("Check whether daylight or camera exposure explains the bright scene before assuming energy waste.")
        return Advice(summary=f"No people are visible and lighting is {condition}; prioritize avoiding unnecessary lighting.", actions=actions)
    if condition == "low":
        return Advice(summary=f"{people} people are visible in a dark scene; check visual comfort and task visibility.", actions=[
            "Verify the scene is genuinely too dark rather than underexposed by the camera.",
            "Use available daylight where appropriate, then increase local task lighting only where people need it.",
            "Confirm adequate visibility without introducing glare or lighting unused areas.",
        ])
    if condition == "excessive":
        return Advice(summary=f"{people} people are visible in a bright scene; check glare and unnecessary illumination.", actions=[
            "Check daylight, glare and camera exposure before concluding the electric lights are excessive.",
            "If nonessential lights are on, dim them gradually or adjust blinds while preserving comfortable task visibility.",
            "Keep occupied work areas and circulation routes adequately illuminated.",
        ])
    return Advice(summary=f"{people} people are visible and scene brightness is normal; maintain comfortable lighting.", actions=[
        "Keep the current lighting if occupants have adequate task visibility and no glare.",
        "Use daylight and occupied-zone lighting where practical; avoid illuminating unused zones.",
        "Reassess when occupancy or brightness changes rather than making an unnecessary adjustment.",
    ])


SYSTEM_PROMPT = (
    "You advise a building operator using ONLY people_count and lighting_condition. "
    "Use the supplied grounded guidance as the basis for concise, practical recommendations. "
    "Treat camera brightness as a heuristic, not lux, switch state or measured energy. "
    "A visible count of zero is not proof the room is empty: require human verification. "
    "For an empty dark room, never recommend increasing lighting merely to brighten the camera. "
    "For occupied rooms, preserve task visibility and comfort. Never advise disabling emergency/egress "
    "lighting, safety systems or required ventilation. Do not invent sensors, temperatures, equipment, "
    "tariffs, kWh, cost or carbon savings. No equipment is controlled by this advice. "
    "Return JSON with exactly summary (a string, maximum 500 characters) and actions "
    "(1 to 3 strings, each maximum 400 characters). Do not include Markdown."
)


class RecommendationService:
    def __init__(self, provider=None, model=None, client=None):
        self.provider = (provider or os.getenv("GBDT_AI_PROVIDER", "rules")).lower()
        self.model = model or (os.getenv("AZURE_OPENAI_DEPLOYMENT") if self.provider == "azure"
                               else os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self.client = client
        self.cache = OrderedDict()
        self.lock = RLock()
        self.retry_after = 0

    def status(self):
        configured = self.client is not None or (
            self.provider == "openai" and bool(os.getenv("OPENAI_API_KEY"))
        ) or (
            self.provider == "azure" and all(os.getenv(name) for name in (
                "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"))
        )
        configured = bool(configured and self.provider in ("openai", "azure"))
        return {"provider": self.provider if self.provider in ("openai", "azure") else "rules",
                "configured": configured, "engine": "genai" if configured else "rules"}

    def _generate(self, inputs, grounded):
        try:
            from openai import AzureOpenAI, OpenAI, OpenAIError
        except ImportError:
            raise RuntimeError("sdk_unavailable") from None
        try:
            if self.client is None:
                if self.provider == "azure":
                    self.client = AzureOpenAI(
                        api_key=os.environ["AZURE_OPENAI_API_KEY"],
                        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                        timeout=12, max_retries=0,
                    )
                else:
                    self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=12, max_retries=0)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": json.dumps({"inputs": inputs.model_dump(),
                                                                  "grounded_guidance": grounded.model_dump()})}],
                response_format={"type": "json_object"}, max_completion_tokens=600,
            )
            choice = completion.choices[0]
            if choice.finish_reason != "stop" or getattr(choice.message, "refusal", None):
                raise ValueError("Incomplete or refused recommendation")
            return Advice.model_validate_json(choice.message.content)
        except OpenAIError:
            raise RuntimeError("provider_unavailable") from None

    def recommend(self, inputs: RecommendationInput):
        grounded = rule_advice(inputs)
        key = (inputs.people_count, inputs.lighting_condition)
        with self.lock:
            now = monotonic()
            cached = self.cache.get(key)
            if cached and now - cached[0] < 300:
                self.cache.move_to_end(key)
                return {**deepcopy(cached[1]), "cached": True}
            advice = grounded
            source = "rules"
            reason = "not_configured"
            if self.status()["configured"]:
                reason = "provider_cooldown"
                if now >= self.retry_after:
                    try:
                        advice = self._generate(inputs, grounded)
                        source, reason = "genai", None
                    except (RuntimeError, ValueError, TypeError, IndexError, AttributeError, KeyError, OSError):
                        self.retry_after = monotonic() + 60
                        reason = "provider_unavailable_or_invalid_response"
            result = {"inputs": inputs.model_dump(), "source": source,
                      "provider": self.provider if source == "genai" else "rules", "fallback_reason": reason,
                      "summary": advice.summary, "actions": advice.actions, "caution": CAUTION,
                      "generated_at": datetime.now(timezone.utc).isoformat(), "cached": False}
            if source == "genai":
                self.cache[key] = (monotonic(), deepcopy(result))
                self.cache.move_to_end(key)
                while len(self.cache) > 128:
                    self.cache.popitem(last=False)
            return result