from __future__ import annotations

import json

from app.schemas import ReviewContext


class PromptBuilder:
    """Build grounded prompts for the LLM review stage."""

    def __init__(self, template: str) -> None:
        self._template = template

    def build_system_prompt(self) -> str:
        """Return the system prompt used for review generation."""
        return self._template

    def build_user_prompt(self, context: ReviewContext) -> str:
        """Serialize the review context into a model-friendly JSON string."""
        payload = {
            "pull_request": context.pull_request.model_dump(),
            "parsed_diffs": [item.model_dump() for item in context.parsed_diffs],
            "ast_analyses": [item.model_dump() for item in context.ast_analyses],
            "heuristic_findings": [item.model_dump() for item in context.heuristic_findings],
        }
        return json.dumps(payload, indent=2)
