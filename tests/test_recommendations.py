import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError
from fastapi.testclient import TestClient

from gbdt_vision.recommendations import CAUTION, RecommendationInput, RecommendationService, rule_advice
from gbdt_vision.twin_server import create_app


class RecommendationApiTests(unittest.TestCase):
    def setUp(self):
        self.service = RecommendationService(provider="rules")
        self.client = TestClient(create_app(recommendation_service=self.service))

    def test_manual_inputs_and_source_label(self):
        response = self.client.post("/api/recommendations", json={"mode": "manual", "people_count": 5, "lighting_condition": "low"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.json()["inputs"], {"people_count": 5, "lighting_condition": "low"})
        self.assertEqual(response.json()["source"], "rules")
        self.assertEqual(response.json()["basis"]["mode"], "manual")

    def test_observed_demo_inputs_come_from_server_not_request(self):
        result = self.client.post("/api/recommendations", json={"mode": "demo", "room_id": "workspace",
                                                               "people_count": 999, "lighting_condition": "normal"}).json()
        self.assertEqual(result["inputs"], {"people_count": 4, "lighting_condition": "low"})
        self.assertEqual(result["basis"]["source_type"], "demo")

    def test_invalid_manual_input_is_rejected(self):
        for body in ({"people_count": -1, "lighting_condition": "low"},
                     {"people_count": True, "lighting_condition": "low"},
                     {"people_count": 1, "lighting_condition": "unknown"}, {}):
            self.assertEqual(self.client.post("/api/recommendations", json=body).status_code, 422)

    def test_stale_room_does_not_generate_advice(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "observations.jsonl"
            record = {"schema_version": "1.0", "status": "observed", "camera_id": "camera-1", "source_type": "camera",
                      "observed_at": (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat(),
                      "rooms": [{"room_id": "room-1", "occupancy_count": 0, "lighting": {"mean_brightness": 0.1}}]}
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            service = Mock(wraps=self.service)
            client = TestClient(create_app(observations_path=path, recommendation_service=service))
            self.assertEqual(client.post("/api/recommendations", json={"mode": "live", "room_id": "workspace"}).status_code, 409)
            service.recommend.assert_not_called()

    def test_polling_and_status_do_not_trigger_generation(self):
        service = Mock(wraps=self.service)
        client = TestClient(create_app(recommendation_service=service))
        for _ in range(2):
            client.get("/api/twin?mode=demo")
        self.assertEqual(client.get("/api/recommendations/status").json()["engine"], "rules")
        service.recommend.assert_not_called()

    def test_changed_room_inputs_during_generation_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vision.jsonl"
            record = {"schema_version": "1.0", "status": "observed", "camera_id": "camera-1", "source_type": "camera",
                      "observed_at": datetime.now(timezone.utc).isoformat(),
                      "rooms": [{"room_id": "room-1", "occupancy_count": 2, "lighting": {"mean_brightness": 0.1}}]}
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            service = Mock()
            def generate(inputs):
                record["rooms"][0]["occupancy_count"] = 5
                path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                return self.service.recommend(inputs)
            service.recommend.side_effect = generate
            client = TestClient(create_app(observations_path=path, recommendation_service=service))
            response = client.post("/api/recommendations", json={"mode": "live", "room_id": "workspace"})
            self.assertEqual(response.status_code, 409)


class RecommendationTests(unittest.TestCase):
    def test_all_six_occupancy_lighting_combinations(self):
        for people in (0, 4):
            for condition in ("low", "normal", "excessive"):
                advice = rule_advice(RecommendationInput(people_count=people, lighting_condition=condition))
                self.assertTrue(advice.summary)
                self.assertTrue(1 <= len(advice.actions) <= 3)

    def test_empty_dark_room_is_not_told_to_increase_lighting(self):
        advice = rule_advice(RecommendationInput(people_count=0, lighting_condition="low"))
        self.assertNotIn("increase", " ".join(advice.actions).lower())
        self.assertIn("unoccupied", advice.actions[0])

    def test_occupied_dark_room_addresses_task_visibility(self):
        advice = rule_advice(RecommendationInput(people_count=5, lighting_condition="low"))
        self.assertIn("5", advice.summary)
        self.assertIn("task lighting", " ".join(advice.actions))

    def test_invalid_people_and_conditions_are_rejected(self):
        for people in (-1, True, 1.5, "2", 10001):
            with self.assertRaises(ValidationError):
                RecommendationInput(people_count=people, lighting_condition="normal")
        with self.assertRaises(ValidationError):
            RecommendationInput(people_count=2, lighting_condition="unknown")

    def test_caution_distinguishes_brightness_and_protects_safety(self):
        self.assertIn("not lux", CAUTION)
        self.assertIn("emergency lighting", CAUTION)

    def test_unconfigured_service_is_explicitly_rule_based(self):
        service = RecommendationService(provider="rules")
        result = service.recommend(RecommendationInput(people_count=2, lighting_condition="normal"))
        self.assertEqual(result["source"], "rules")
        self.assertEqual(result["fallback_reason"], "not_configured")

    def test_provider_output_is_validated_and_cached(self):
        client = Mock()
        content = json.dumps({"summary": "Check task visibility.", "actions": ["Verify camera exposure before increasing task lighting."]})
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content, refusal=None))])
        service = RecommendationService(provider="openai", model="test-model", client=client)
        inputs = RecommendationInput(people_count=3, lighting_condition="low")
        first = service.recommend(inputs)
        second = service.recommend(inputs)
        self.assertEqual(first["source"], "genai")
        self.assertTrue(second["cached"])
        self.assertEqual(first["caution"], CAUTION)
        client.chat.completions.create.assert_called_once()
        sent = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(sent["inputs"], {"people_count": 3, "lighting_condition": "low"})

    def test_invalid_ai_response_and_timeout_fall_back(self):
        for failure in ("invalid_json", "timeout", "refusal"):
            client = Mock()
            if failure == "timeout":
                client.chat.completions.create.side_effect = TimeoutError("Private provider details")
            else:
                client.chat.completions.create.return_value = SimpleNamespace(choices=[
                    SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="not JSON", refusal="Refused" if failure == "refusal" else None))])
            service = RecommendationService(provider="openai", model="test-model", client=client)
            inputs = RecommendationInput(people_count=0, lighting_condition="low")
            result = service.recommend(inputs)
            self.assertEqual(result["source"], "rules")
            self.assertNotIn("Private", json.dumps(result))
            service.recommend(inputs)
            client.chat.completions.create.assert_called_once()

    def test_openai_and_azure_environment_configuration(self):
        for provider, constructor_name, environment in (
            ("openai", "openai.OpenAI", {"OPENAI_API_KEY": "test-placeholder", "OPENAI_MODEL": "test-model"}),
            ("azure", "openai.AzureOpenAI", {"AZURE_OPENAI_API_KEY": "test-placeholder",
                                            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
                                            "AZURE_OPENAI_DEPLOYMENT": "test-model"}),
        ):
            client = Mock()
            client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content='{"summary":"Check task visibility.","actions":["Verify actual conditions."]}', refusal=None))])
            with patch.dict(os.environ, {"GBDT_AI_PROVIDER": provider, **environment}, clear=True), patch(constructor_name, return_value=client) as constructor:
                service = RecommendationService()
                self.assertTrue(service.status()["configured"])
                result = service.recommend(RecommendationInput(people_count=3, lighting_condition="low"))
                self.assertEqual(result["provider"], provider)
                self.assertEqual(result["source"], "genai")
                self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], "test-model")
                self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)


if __name__ == "__main__":
    unittest.main()