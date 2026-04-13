from __future__ import annotations

import argparse
from typing import Optional

try:
    from rich.console import Console
except ImportError:  # pragma: no cover - exercised via runtime fallback
    Console = None  # type: ignore[assignment]

from app.ast_analyzer import ASTAnalyzer
from app.config import get_settings
from app.diff_parser import DiffParser
from app.github_client import GitHubClient, GitHubClientError
from app.report_formatter import ReportFormatter
from app.review_engine import ReviewEngine
from app.reviewer import Reviewer
from app.rule_engine import RuleBasedReviewer


def build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for pull request review execution."""
    parser = argparse.ArgumentParser(description="LLM-assisted GitHub pull request reviewer.")
    parser.add_argument("--owner", required=True, help="GitHub repository owner")
    parser.add_argument("--repo", required=True, help="GitHub repository name")
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    parser.add_argument("--markdown-output", help="Optional path to save the markdown report")
    parser.add_argument("--json-output", help="Optional path to save the JSON report")
    parser.add_argument("--no-rich", action="store_true", help="Disable Rich terminal formatting")
    return parser


def main() -> None:
    """Run the CLI review workflow."""
    args = build_argument_parser().parse_args()
    error_console: Optional[Console] = Console(stderr=True) if Console is not None else None
    try:
        settings = get_settings()
    except ValueError as exc:
        if error_console is None:
            print(f"Configuration error: {exc}")
        else:
            error_console.print(f"[red]Configuration error:[/red] {exc}")
        raise SystemExit(1) from exc

    report_formatter = ReportFormatter()
    github_client = GitHubClient(settings)
    reviewer = Reviewer(
        github_client=github_client,
        diff_parser=DiffParser(),
        ast_analyzer=ASTAnalyzer(settings.thresholds),
        rule_based_reviewer=RuleBasedReviewer(),
        review_engine=ReviewEngine(settings),
        report_formatter=report_formatter,
    )

    try:
        markdown_report, json_result = reviewer.review_pull_request(args.owner, args.repo, args.pr)
    except (GitHubClientError, RuntimeError) as exc:
        if error_console is None:
            print(f"Runtime error: {exc}")
        else:
            error_console.print(f"[red]Runtime error:[/red] {exc}")
        raise SystemExit(1) from exc
    finally:
        github_client.close()

    report_formatter.save_outputs(
        json_result,
        markdown_path=args.markdown_output,
        json_path=args.json_output,
    )

    if args.no_rich:
        print(markdown_report)
        print()
        print("JSON Output")
        print(report_formatter.to_json(json_result))
    else:
        output_console = Console() if Console is not None else None
        report_formatter.render_terminal(json_result, console=output_console)
        if output_console is None:
            print()
            print(markdown_report)
            print()
            print("JSON Output")
            print(report_formatter.to_json(json_result))
        else:
            output_console.rule("Markdown Report")
            output_console.print(markdown_report)
            output_console.rule("JSON Output")
            report_formatter.render_json(json_result, console=output_console)
