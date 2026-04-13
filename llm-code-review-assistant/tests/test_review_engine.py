from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.review_engine import ReviewEngine
from app.schemas import (
    DiffHunk,
    FileASTAnalysis,
    ParsedDiffFile,
    PullRequestData,
    PullRequestFile,
    PullRequestMetadata,
    ReviewContext,
    ReviewFinding,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, **_: object) -> _FakeResponse:
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = _FakeChat(content)


def _build_settings() -> Settings:
    return Settings(
        github_token=None,
        openai_api_key="test-key",
        openai_model="gpt-4.1-mini",
        github_api_base_url="https://api.github.com",
        request_timeout_seconds=30.0,
        max_files_for_review=20,
    )


def _build_context() -> ReviewContext:
    heuristic_finding = ReviewFinding(
        title="Broad exception handler hides failure modes",
        issue_type="bug_risk",
        severity="high",
        confidence=0.94,
        file="app/service.py",
        line_hint=2,
        impact="Swallowed failures are hard to debug.",
        explanation="The code uses a bare except block.",
        suggestion="Catch explicit exception types.",
        evidence_source="heuristic",
    )
    return ReviewContext(
        pull_request=PullRequestData(
            metadata=PullRequestMetadata(
                owner="octo",
                repo="demo",
                pr_number=1,
                title="Update service",
                body="Adds retry handling.",
                base_ref="main",
                head_ref="feature",
                head_sha="abc123",
            ),
            files=[
                PullRequestFile(
                    filename="app/service.py",
                    status="modified",
                    additions=2,
                    deletions=1,
                    changes=3,
                    patch="@@ -1,2 +1,3 @@\n line_1\n+line_2\n line_3\n",
                    contents="def run():\n    pass\n",
                )
            ],
        ),
        parsed_diffs=[
            ParsedDiffFile(
                file_path="app/service.py",
                added_lines=[2],
                changed_line_ranges=[(1, 3)],
                hunks=[
                    DiffHunk(
                        old_start=1,
                        old_count=2,
                        new_start=1,
                        new_count=3,
                        added_lines=[2],
                    )
                ],
                patch="@@ -1,2 +1,3 @@\n line_1\n+line_2\n line_3\n",
            )
        ],
        ast_analyses=[FileASTAnalysis(file="app/service.py", issues=[])],
        heuristic_findings=[heuristic_finding],
    )


def test_review_engine_falls_back_when_llm_json_is_invalid() -> None:
    engine = ReviewEngine(_build_settings())
    engine._client = _FakeClient("not-json")

    result = engine.generate_review(_build_context())

    assert result.overall_risk == "high"
    assert result.findings[0].evidence_source == "heuristic"


def test_review_engine_filters_hallucinated_files() -> None:
    engine = ReviewEngine(_build_settings())
    engine._client = _FakeClient(
        """
        {
          "summary": "Found one issue.",
          "overall_risk": "medium",
          "findings": [
            {
              "title": "Fake file",
              "issue_type": "bug_risk",
              "severity": "high",
              "confidence": 0.9,
              "file": "does/not/exist.py",
              "line_hint": 9,
              "impact": "Bad impact",
              "explanation": "Bad explanation",
              "suggestion": "Bad suggestion",
              "evidence_source": "llm"
            }
          ]
        }
        """
    )

    result = engine.generate_review(_build_context())

    assert result.findings[0].file == "app/service.py"
    assert result.findings[0].evidence_source == "heuristic"


def test_review_engine_filters_low_signal_nits() -> None:
    engine = ReviewEngine(_build_settings())
    engine._client = _FakeClient(
        """
        {
          "summary": "Only a minor nit was found.",
          "overall_risk": "low",
          "findings": [
            {
              "title": "No newline at end of file",
              "issue_type": "readability",
              "severity": "low",
              "confidence": 0.95,
              "file": "app/service.py",
              "line_hint": 2,
              "impact": "Formatting preference only.",
              "explanation": "The file should end with a newline.",
              "suggestion": "Add a newline at end of file.",
              "evidence_source": "llm"
            }
          ]
        }
        """
    )

    result = engine.generate_review(_build_context())

    assert result.findings[0].evidence_source == "heuristic"


def test_review_engine_rejects_findings_outside_changed_lines() -> None:
    engine = ReviewEngine(_build_settings())
    engine._client = _FakeClient(
        """
        {
          "summary": "Found one issue.",
          "overall_risk": "medium",
          "findings": [
            {
              "title": "Outside diff",
              "issue_type": "bug_risk",
              "severity": "high",
              "confidence": 0.9,
              "file": "app/service.py",
              "line_hint": 99,
              "impact": "Bad impact",
              "explanation": "This points to unchanged code.",
              "suggestion": "Fix it.",
              "evidence_source": "llm"
            }
          ]
        }
        """
    )

    result = engine.generate_review(_build_context())

    assert result.findings[0].evidence_source == "heuristic"


def test_review_engine_returns_no_significant_issues_when_only_generic_feedback_exists() -> None:
    engine = ReviewEngine(_build_settings())
    engine._client = _FakeClient(
        """
        {
          "summary": "Only generic advice found.",
          "overall_risk": "low",
          "findings": [
            {
              "title": "Improve readability",
              "issue_type": "readability",
              "severity": "low",
              "confidence": 0.8,
              "file": "app/service.py",
              "line_hint": 2,
              "impact": "Could be clearer.",
              "explanation": "This may cause issues because it could be improved.",
              "suggestion": "Consider improving readability.",
              "evidence_source": "llm"
            }
          ]
        }
        """
    )

    context = _build_context()
    context.heuristic_findings = []
    result = engine.generate_review(context)

    assert result.summary == "No significant issues found in the changed lines."
    assert result.findings == []
