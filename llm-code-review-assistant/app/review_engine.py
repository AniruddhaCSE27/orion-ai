from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import APIError, OpenAI

from app.prompt_builder import PromptBuilder
from app.config import Settings
from app.schemas import LLMReviewPayload, ReviewContext, ReviewFinding, ReviewResult


class ReviewEngine:
    """Coordinate LLM review generation with strict validation and fallback behavior."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key)
        prompt_template = Path(settings.prompt_path).read_text(encoding="utf-8")
        self._prompt_builder = PromptBuilder(template=prompt_template)

    def generate_review(self, context: ReviewContext) -> ReviewResult:
        """Generate a review from the LLM and safely fall back to deterministic findings."""
        fallback_review = self._build_fallback_review(context.heuristic_findings, context)

        try:
            response = self._client.chat.completions.create(
                model=self._settings.openai_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._prompt_builder.build_system_prompt()},
                    {"role": "user", "content": self._prompt_builder.build_user_prompt(context)},
                ],
            )
        except APIError as exc:
            return fallback_review
        content = response.choices[0].message.content
        if content is None:
            return fallback_review

        try:
            payload = LLMReviewPayload.model_validate(self._parse_json(content))
        except Exception:
            return fallback_review

        grounded_findings = self._validate_findings(payload.findings, context)
        if not grounded_findings and context.heuristic_findings:
            return fallback_review

        if not grounded_findings:
            return self._no_significant_issues_review()

        return ReviewResult(
            summary=self._normalize_summary(payload.summary, grounded_findings),
            overall_risk=self._derive_overall_risk(grounded_findings),
            findings=grounded_findings,
        )

    @staticmethod
    def _parse_json(raw_content: str) -> Any:
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OpenAI response was not valid JSON: {exc}") from exc

    def _validate_findings(self, findings: list[ReviewFinding], context: ReviewContext) -> list[ReviewFinding]:
        valid_files = {file.filename for file in context.pull_request.files}
        diff_map = {diff.file_path: diff for diff in context.parsed_diffs}
        validated: list[ReviewFinding] = []

        for finding in findings:
            if finding.file not in valid_files:
                continue
            diff = diff_map.get(finding.file)
            if diff is None:
                continue
            if not finding.line_hint or not diff.contains_added_line(finding.line_hint):
                continue
            if not self._passes_final_output_gate(finding):
                continue
            validated.append(finding)

        return self._sort_findings(self._deduplicate(validated))

    def _passes_final_output_gate(self, finding: ReviewFinding) -> bool:
        """Final output override: only exact, high-signal PR findings may survive."""
        if finding.issue_type in {"readability", "best_practice"}:
            return False
        if finding.issue_type == "maintainability" and finding.severity != "high":
            return False
        if self._is_low_signal_finding(finding):
            return False
        if finding.evidence_source == "ast":
            return finding.issue_type == "bug_risk" and finding.severity in {"medium", "high"}
        return True

    def _build_fallback_review(self, heuristic_findings: list[ReviewFinding], context: ReviewContext) -> ReviewResult:
        deduplicated_findings = self._validate_findings(heuristic_findings, context)
        if not deduplicated_findings:
            return self._no_significant_issues_review()
        return ReviewResult(
            summary=(
                f"Deterministic review surfaced {len(deduplicated_findings)} high-signal issue(s) "
                "after the LLM response was unavailable or failed validation."
            ),
            overall_risk=self._derive_overall_risk(deduplicated_findings),
            findings=deduplicated_findings,
        )

    @staticmethod
    def _derive_overall_risk(findings: list[ReviewFinding]) -> str:
        if any(item.severity == "high" for item in findings):
            return "high"
        if any(item.severity == "medium" for item in findings):
            return "medium"
        return "low"

    @staticmethod
    def _deduplicate(findings: list[ReviewFinding]) -> list[ReviewFinding]:
        deduplicated: dict[tuple[str, int, str], ReviewFinding] = {}
        for finding in findings:
            key = (finding.file, finding.line_hint, finding.title)
            current = deduplicated.get(key)
            if current is None or finding.confidence > current.confidence:
                deduplicated[key] = finding
        return list(deduplicated.values())

    @staticmethod
    def _sort_findings(findings: list[ReviewFinding]) -> list[ReviewFinding]:
        severity_order = {"high": 0, "medium": 1, "low": 2}
        issue_order = {
            "bug_risk": 0,
            "security": 1,
            "maintainability": 2,
            "best_practice": 3,
            "performance": 4,
            "readability": 5,
        }
        return sorted(
            findings,
            key=lambda item: (
                severity_order[item.severity],
                issue_order[item.issue_type],
                -item.confidence,
                item.file,
                item.line_hint,
            ),
        )

    @staticmethod
    def _is_low_signal_finding(finding: ReviewFinding) -> bool:
        text = " ".join(
            [
                finding.title.lower(),
                finding.impact.lower(),
                finding.explanation.lower(),
                finding.suggestion.lower(),
            ]
        )
        low_signal_patterns = [
            "missing docstring",
            "newline at end of file",
            "end with a newline",
            "formatting preference",
            "minor formatting",
            "could be improved",
            "might be better",
            "docstring",
            "long function",
            "function is long",
            "large function",
            "deep nesting",
            "too many parameters",
            "parameter count",
            "maintainability",
            "readability",
            "clearer",
            "test added",
            "no changes recommended",
            "looks good",
            "works correctly",
            "test added to verify",
            "strictly positive",
            "improves confidence",
        ]
        if any(pattern in text for pattern in low_signal_patterns):
            return True
        generic_explanation_patterns = [
            "may cause issues",
            "could lead to problems",
            "harder to maintain",
        ]
        return any(pattern in text for pattern in generic_explanation_patterns)

    @staticmethod
    def _no_significant_issues_review() -> ReviewResult:
        return ReviewResult(
            summary="The pull request is focused and no significant issues were found in the changed lines.",
            overall_risk="low",
            findings=[],
        )

    @staticmethod
    def _normalize_summary(summary: str, findings: list[ReviewFinding]) -> str:
        if not findings:
            return "The pull request is focused and no significant issues were found in the changed lines."

        cleaned = " ".join(summary.split())
        if not cleaned:
            return f"Identified {len(findings)} high-signal issue(s) in the changed lines."

        lowered = cleaned.lower()
        banned_summary_patterns = [
            "docstring",
            "long function",
            "too many parameters",
            "deep nesting",
            "maintainability",
            "readability",
            "no changes recommended",
            "looks good",
            "correctly",
            "test added",
        ]
        if any(pattern in lowered for pattern in banned_summary_patterns):
            return f"Identified {len(findings)} high-signal issue(s) in the changed lines."
        return cleaned
