import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import SecretStr

from casepilot.config import ProviderSettings, RunConfig
from casepilot.integration import provider_health


@pytest.fixture
def provider_server():
    class Handler(BaseHTTPRequestHandler):
        status = 200
        payload = {"choices": [{"message": {"content": "OK"}}]}
        requests = []

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.requests.append((self.path, body, self.headers.get("Authorization")))
            self.send_response(self.status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.payload).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    settings = ProviderSettings(
        api_key=SecretStr("test-only-provider-key"),
        base_url=SecretStr(f"http://127.0.0.1:{server.server_port}/v1"),
    )
    try:
        yield settings, Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_real_chat_probe_uses_current_settings_but_keeps_them_out_of_results(provider_server):
    settings, handler = provider_server
    result = provider_health.probe_provider(settings, model="openai/gpt-6-luna")
    assert result["operational"] and result["http_status"] == 200
    path, body, authorization = handler.requests[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "gpt-6-luna" and body["reasoning_effort"] == "none"
    assert authorization == "Bearer test-only-provider-key"
    serialized = json.dumps(result)
    assert settings.api_key.get_secret_value() not in serialized
    assert settings.base_url.get_secret_value() not in serialized


def test_http_403_does_not_leak_provider_error_body(provider_server):
    settings, handler = provider_server
    handler.status = 403
    handler.payload = {"error": "test-only-provider-key private backend information"}
    result = provider_health.probe_provider(settings, model="openai/gpt-6-luna")
    assert not result["operational"] and result["http_status"] == 403
    assert "private" not in json.dumps(result)
    assert "test-only-provider-key" not in json.dumps(result)


def test_responses_probe_requires_generated_output(provider_server):
    settings, handler = provider_server
    handler.payload = {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}]
    }
    result = provider_health.probe_provider(
        settings, model="openai/responses/gpt-6-luna", api="responses"
    )
    assert result["operational"]
    assert handler.requests[0][1]["model"] == "gpt-6-luna"
    handler.payload = {"data": [{"id": "gpt-6-luna"}]}
    assert not provider_health.probe_provider(settings, model="gpt-6-luna")["operational"]


def test_model_visibility_cannot_satisfy_inference_preflight(monkeypatch):
    calls = []

    def denied(_settings, *, model, api):
        calls.append((model, api))
        return {"operational": False, "http_status": 403}

    monkeypatch.setattr(provider_health, "probe_provider", denied)
    config = RunConfig(model="openai/gpt-6-luna", user_model="openai/gpt-6-luna")
    with pytest.raises(ValueError, match="chat: 403"):
        provider_health.require_provider(None, [config])
    assert calls == [(config.model, "chat")]


def test_distinct_models_and_judge_are_checked_once(monkeypatch):
    calls = []

    def available(_settings, *, model, api):
        calls.append((model, api))
        return {"operational": True}

    monkeypatch.setattr(provider_health, "probe_provider", available)
    config = RunConfig(
        track="custom",
        model="openai/gpt-6-luna",
        user_model="openai/gpt-6-luna",
        judge_model="openai/responses/gpt-6-luna",
    )
    provider_health.require_provider(None, [config, config])
    assert len(calls) == 2
    assert (config.judge_model, "responses") in calls
