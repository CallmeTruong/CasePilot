"""Giskard generator over the existing OpenAI-compatible endpoint, with explicit budgets.

Native OpenAI SDK avoids upgrading the LiteLLM revision required by the pinned Retail benchmark.
The translator is covered by contract tests and pinned by the optional security dependency group.
"""

from giskard.agents import Generator
from giskard.llm.translators.openai_chat import OpenAIChatTranslator
from openai import AsyncOpenAI
from pydantic import PrivateAttr


def strict_schema(value):
    """Make optional Pydantic fields explicit for OpenAI strict structured outputs."""
    if isinstance(value, list):
        return [strict_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: strict_schema(item) for key, item in value.items() if key != "default"}
    if result.get("type") == "object" or "properties" in result:
        result["additionalProperties"] = False
        result["required"] = list(result.get("properties", {}))
    return result


class BudgetedGenerator(Generator):
    _client: AsyncOpenAI = PrivateAttr()
    _budget: object = PrivateAttr()

    def __init__(self, model, settings, budget):
        if not model.startswith("openai/") or model.count("/") != 1:
            raise ValueError(
                "Red-team model must use the configured OpenAI-compatible chat provider"
            )
        super().__init__(
            model=model,
            retry_policy=None,
            params={"temperature": 0, "max_tokens": 2048, "timeout": 45},
        )
        self._client = AsyncOpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url.get_secret_value() if settings.base_url else None,
            timeout=45,
            max_retries=0,
        )
        self._budget = budget

    async def _call_model(self, messages, params, metadata=None):
        self._budget.consume("scanner")
        kwargs = OpenAIChatTranslator.to_openai(
            self.model.split("/", 1)[1],
            messages,
            tools=self._serialize_tools(params.tools) if params.tools else None,
            **params.model_dump(exclude={"tools"}, exclude_unset=True),
        )
        maximum = min(int(kwargs.pop("max_tokens", 2048) or 2048), 2048)
        kwargs.update(
            max_completion_tokens=maximum, timeout=min(45, self._budget.remaining_time()), seed=42
        )
        response_format = kwargs.get("response_format", {})
        if response_format.get("type") == "json_schema":
            schema = response_format["json_schema"]
            schema["schema"] = strict_schema(schema["schema"])
            schema["strict"] = True
        if self.model.split("/")[-1].startswith("gpt-6"):
            kwargs["reasoning_effort"] = "none"
        reply = await self._client.chat.completions.create(**kwargs)
        return OpenAIChatTranslator.from_openai(reply)

    async def close(self):
        await self._client.close()
