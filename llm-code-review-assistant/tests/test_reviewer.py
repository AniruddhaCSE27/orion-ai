from __future__ import annotations

from app.ast_analyzer import ASTAnalyzer
from app.config import AnalyzerThresholds
from app.diff_parser import DiffParser
from app.report_formatter import ReportFormatter
from app.reviewer import Reviewer
from app.rule_engine import RuleBasedReviewer
from app.schemas import PullRequestData, PullRequestFile, PullRequestMetadata, ReviewContext, ReviewFinding, ReviewResult


class FakeGitHubClient:
    def fetch_pull_request(self, owner: str, repo: str, pr_number: int) -> PullRequestData:
        return PullRequestData(
            metadata=PullRequestMetadata(
                owner=owner,
                repo=repo,
                pr_number=pr_number,
                title="Refactor payment flow",
                body="Adds retry logic and conditional charge handling.",
                base_ref="main",
                head_ref="feature/refactor-payment",
                head_sha="abc123",
            ),
            files=[
                PullRequestFile(
                    filename="app/service.py",
                    status="modified",
                    additions=5,
                    deletions=1,
                    changes=6,
                    patch="@@ -1,2 +1,6 @@\n line_1\n+line_2\n+line_3\n+line_4\n+line_5\n line_6\n",
                    contents=(
                        "def sample(a, b, c, d, e, f):\n"
                        "    try:\n"
                        "        return a\n"
                        "    except:\n"
                        "        return 0\n"
                    ),
                )
            ],
        )


class FakeReviewEngine:
    def generate_review(self, context: ReviewContext) -> ReviewResult:
        assert context.pull_request.metadata.pr_number == 42
        assert context.parsed_diffs[0].file_path == "app/service.py"
        assert context.ast_analyses[0].file == "app/service.py"
        assert context.heuristic_findings
        return ReviewResult(
            summary="One maintainability issue identified from the provided evidence.",
            overall_risk="medium",
            findings=[
                ReviewFinding(
                    title="Helper signature is getting hard to use safely",
                    issue_type="maintainability",
                    severity="medium",
                    confidence=0.88,
                    file="app/service.py",
                    line_hint=1,
                    impact="Wide signatures are harder to evolve safely.",
                    explanation="The function accepts too many parameters, which makes the API harder to evolve safely.",
                    suggestion="Group related inputs into a data structure or split responsibilities into smaller helpers.",
                    evidence_source="llm",
                )
            ],
        )


def test_reviewer_pipeline_returns_markdown_and_json_model() -> None:
    reviewer = Reviewer(
        github_client=FakeGitHubClient(),
        diff_parser=DiffParser(),
        ast_analyzer=ASTAnalyzer(AnalyzerThresholds()),
        rule_based_reviewer=RuleBasedReviewer(),
        review_engine=FakeReviewEngine(),
        report_formatter=ReportFormatter(),
    )

    markdown_report, review_result = reviewer.review_pull_request("octo", "demo", 42)

    assert "LLM Code Review Report" in markdown_report
    assert "Overall Risk" in markdown_report
    assert review_result.findings[0].file == "app/service.py"
    assert review_result.overall_risk == "medium"
