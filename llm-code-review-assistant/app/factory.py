from __future__ import annotations

from dataclasses import dataclass

from app.ast_analyzer import ASTAnalyzer
from app.config import Settings, get_settings
from app.diff_parser import DiffParser
from app.github_client import GitHubClient
from app.report_formatter import ReportFormatter
from app.review_engine import ReviewEngine
from app.reviewer import Reviewer
from app.rule_engine import RuleBasedReviewer


@dataclass
class ReviewerServices:
    """Shared dependency container for CLI and Streamlit execution."""

    settings: Settings
    github_client: GitHubClient
    report_formatter: ReportFormatter
    reviewer: Reviewer

    def close(self) -> None:
        self.github_client.close()


def create_reviewer(settings: Settings | None = None) -> ReviewerServices:
    """Build the full reviewer pipeline once for any runtime entrypoint."""
    resolved_settings = settings or get_settings()
    report_formatter = ReportFormatter()
    github_client = GitHubClient(resolved_settings)
    reviewer = Reviewer(
        github_client=github_client,
        diff_parser=DiffParser(),
        ast_analyzer=ASTAnalyzer(resolved_settings.thresholds),
        rule_based_reviewer=RuleBasedReviewer(),
        review_engine=ReviewEngine(resolved_settings),
        report_formatter=report_formatter,
    )
    return ReviewerServices(
        settings=resolved_settings,
        github_client=github_client,
        report_formatter=report_formatter,
        reviewer=reviewer,
    )
