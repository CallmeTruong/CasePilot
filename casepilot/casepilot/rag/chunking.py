import hashlib
import re
from datetime import date

from pydantic import Field, model_validator

from casepilot.config import NonEmpty, StrictModel


class PolicyChunk(StrictModel):
    chunk_id: NonEmpty
    section: NonEmpty
    text: NonEmpty
    source: NonEmpty
    version: NonEmpty
    text_hash: NonEmpty
    effective_from: date | None = None
    effective_to: date | None = None
    synthetic: bool = False

    @model_validator(mode="after")
    def dates_in_source_order(self):
        if self.effective_from and self.effective_to and self.effective_from > self.effective_to:
            raise ValueError("Policy effective_from must not exceed effective_to")
        return self


class RetrievalHit(StrictModel):
    chunk: PolicyChunk
    score: float = Field(allow_inf_nan=False)
    rank: int = Field(ge=1)
    component_scores: dict[str, float] = Field(default_factory=dict)


def chunk_policy(text: str, *, source: str, version: str) -> list[PolicyChunk]:
    """Split on source headings, retaining ancestor headings for short clauses."""
    chunks, buffer, headings = [], [], []

    def flush():
        content = "\n".join(buffer).strip()
        if not content:
            return
        section = " / ".join(headings) or "Policy"
        digest = hashlib.sha256(content.encode()).hexdigest()
        identifier = hashlib.sha256(f"{source}:{version}:{section}:{digest}".encode()).hexdigest()
        chunks.append(
            PolicyChunk(
                chunk_id="POLICY_" + identifier[:16],
                section=section,
                text=content,
                source=source,
                version=version,
                text_hash=digest,
            )
        )

    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush()
            buffer.clear()
            depth, title = len(heading[1]), heading[2].strip()
            headings[:] = headings[: depth - 1] + [title]
        buffer.append(line)
    flush()
    return chunks
