from __future__ import annotations

from app.ast_analyzer import ASTAnalyzer
from app.diff_parser import DiffParser
from app.github_client import GitHubClient
from app.report_formatter import ReportFormatter
from app.review_engine import ReviewEngine
from app.rule_engine import RuleBasedReviewer
from app.schemas import FileASTAnalysis, ParsedDiffFile, PullRequestData, ReviewContext, ReviewResult


class Reviewer:
    """Orchestrate the hybrid review workflow."""

    def __init__(
        self,
        github_client: GitHubClient,
        diff_parser: DiffParser,
        ast_analyzer: ASTAnalyzer,
        rule_based_reviewer: RuleBasedReviewer,
        review_engine: ReviewEngine,
        report_formatter: ReportFormatter,
    ) -> None:
        self._github_client = github_client
        self._diff_parser = diff_parser
        self._ast_analyzer = ast_analyzer
        self._rule_based_reviewer = rule_based_reviewer
        self._review_engine = review_engine
        self._report_formatter = report_formatter

    def review_pull_request(self, owner: str, repo: str, pr_number: int) -> tuple[str, ReviewResult]:
        """Fetch and review a pull request end to end."""
        pull_request = self._github_client.fetch_pull_request(owner=owner, repo=repo, pr_number=pr_number)
        parsed_diffs = self._parse_diffs(pull_request)
        ast_analyses = self._analyze_python_files(pull_request)
        heuristic_findings = self._rule_based_reviewer.build_findings(parsed_diffs=parsed_diffs, ast_analyses=ast_analyses)
        context = ReviewContext(
            pull_request=pull_request,
            parsed_diffs=parsed_diffs,
            ast_analyses=ast_analyses,
            heuristic_findings=heuristic_findings,
        )
        review = self._review_engine.generate_review(context)
        markdown_report = self._report_formatter.to_markdown(review)
        return markdown_report, review

    def _parse_diffs(self, pull_request: PullRequestData) -> list[ParsedDiffFile]:
        """Parse patches for all files that include diff hunks."""
        parsed_files: list[ParsedDiffFile] = []
        for file in pull_request.files:
            if file.patch:
                parsed_files.append(self._diff_parser.parse_patch(file.filename, file.patch))
        return parsed_files

    def _analyze_python_files(self, pull_request: PullRequestData) -> list[FileASTAnalysis]:
        """Run AST analysis for Python files with retrievable contents."""
        analyses: list[FileASTAnalysis] = []
        for file in pull_request.files:
            if file.filename.endswith(".py") and file.contents:
                analyses.append(self._ast_analyzer.analyze_file(file.filename, file.contents))
        return analyses
