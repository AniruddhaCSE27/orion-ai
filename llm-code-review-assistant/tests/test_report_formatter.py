from __future__ import annotations

from pathlib import Path

from app.report_formatter import ReportFormatter
from app.schemas import ReviewFinding, ReviewResult


def test_report_formatter_writes_markdown_and_json(tmp_path: Path) -> None:
    formatter = ReportFormatter()
    review = ReviewResult(
        summary="Risk is concentrated in one exception path.",
        overall_risk="high",
        findings=[
            ReviewFinding(
                title="Broad exception handler hides failure modes",
                issue_type="bug_risk",
                severity="high",
                confidence=0.94,
                file="app/service.py",
                line_hint=24,
                impact="Silent fallback behavior can mask production issues.",
                explanation="The code catches every exception type.",
                suggestion="Catch expected exceptions only.",
                evidence_source="heuristic",
            )
        ],
    )

    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"
    formatter.save_outputs(review, str(markdown_path), str(json_path))

    assert "Overall Risk" in markdown_path.read_text(encoding="utf-8")
    assert '"overall_risk": "high"' in json_path.read_text(encoding="utf-8")
