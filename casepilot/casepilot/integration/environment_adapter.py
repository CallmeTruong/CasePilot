"""Only public context crosses the Agent boundary."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PublicRetailContext:
    policy: str
    tools: list

    @classmethod
    def from_environment(cls, environment):
        return cls(policy=environment.get_policy(), tools=environment.get_tools())
