"""Bounded provider probes. Never persist credentials, URLs, prompts or error bodies."""

import json
import urllib.error
import urllib.request
from datetime import UTC, datetime


class ProviderUnavailable(ValueError):
    """Safe public preflight failure with no upstream payload or private settings."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _model_id(model):
    for prefix in ("openai/responses/", "openai/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def probe_provider(settings, *, model, api="chat", timeout=30):
    """A tiny real inference request; model-list visibility alone is not inference health."""
    if api not in {"chat", "responses", "models"}:
        raise ValueError("Unsupported provider probe API")
    if not model or not 0 < timeout <= 60:
        raise ValueError("Model and a timeout in (0, 60] are required")
    model_id = _model_id(model)
    route = {"chat": "/chat/completions", "responses": "/responses", "models": "/models"}[api]
    base = (
        settings.base_url.get_secret_value() if settings.base_url else "https://api.openai.com/v1"
    )
    payload = None
    if api == "chat":
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply OK."}],
            "max_completion_tokens": 32,
        }
        if model_id.startswith("gpt-6"):
            payload["reasoning_effort"] = "none"
    elif api == "responses":
        payload = {"model": model_id, "input": "Reply OK.", "max_output_tokens": 32}
    request = urllib.request.Request(
        base.rstrip("/") + route,
        data=json.dumps(payload).encode() if payload else None,
        headers={
            "Authorization": "Bearer " + settings.api_key.get_secret_value(),
            "Content-Type": "application/json",
            "User-Agent": "CasePilot/0.1",
        },
    )
    result = {
        "checked_at": datetime.now(UTC).isoformat(),
        "api": api,
        "model": model,
        "http_status": None,
        "operational": False,
        "error_type": None,
    }
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            result["http_status"] = response.status
            content = response.read(1024 * 1024 + 1)
            if len(content) > 1024 * 1024:
                raise ValueError("Probe response exceeds limit")
            value = json.loads(content)
            if api == "models":
                result["operational"] = isinstance(value, dict) and isinstance(
                    value.get("data"), list
                )
            elif api == "chat":
                choices = value.get("choices") if isinstance(value, dict) else None
                result["operational"] = bool(
                    choices
                    and isinstance(choices, list)
                    and isinstance(choices[0], dict)
                    and isinstance(choices[0].get("message"), dict)
                    and choices[0]["message"].get("content")
                )
            else:
                output = value.get("output") if isinstance(value, dict) else None
                result["operational"] = bool(
                    output
                    and isinstance(output, list)
                    and any(
                        isinstance(item, dict)
                        and item.get("type") == "message"
                        and any(
                            isinstance(part, dict)
                            and part.get("type") == "output_text"
                            and part.get("text")
                            for part in item.get("content", [])
                        )
                        for item in output
                    )
                )
            if not result["operational"]:
                result["error_type"] = "InvalidResponse"
    except urllib.error.HTTPError as exc:
        result.update(http_status=exc.code, error_type="HTTPError")
        exc.close()
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    return result


def require_provider(settings, configs):
    """Check every distinct requested model/API before scheduling any benchmark work."""
    required = set()
    for config in configs:
        for model in (config.model, config.user_model):
            if model:
                required.add((model, "chat"))
        if config.judge_model:
            api = "responses" if "/responses/" in config.judge_model else "chat"
            required.add((config.judge_model, api))
    if not required:
        raise ValueError("No provider models configured")
    results = []
    for model, api in sorted(required):
        result = probe_provider(settings, model=model, api=api)
        results.append(result)
        if not result["operational"]:
            status = result["http_status"] or result["error_type"]
            raise ProviderUnavailable(f"Provider inference unavailable ({api}: {status})")
    return results
