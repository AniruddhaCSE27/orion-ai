from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.schemas import ReviewResult

try:
    from rich.console import Console
    from rich.json import JSON
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover - exercised via runtime fallback
    Console = None  # type: ignore[assignment]
    JSON = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]


class ReportFormatter:
    """Format review results for markdown, JSON, and terminal output."""

    def to_markdown(self, review: ReviewResult) -> str:
        """Render a polished markdown report."""
        lines = [
            "# LLM Code Review Report",
            "",
            "## Summary",
            review.summary,
            "",
            f"**Overall Risk:** `{review.overall_risk.upper()}`",
            "",
            "## Findings",
        ]

        if not review.findings:
            lines.append("- No actionable issues identified from the supplied evidence.")
            return "\n".join(lines)

        for finding in review.findings:
            lines.extend(
                [
                    f"### {finding.title}",
                    "",
                    f"- Severity: `{finding.severity}`",
                    f"- Confidence: `{finding.confidence:.2f}`",
                    f"- Type: `{finding.issue_type}`",
                    f"  - File: `{finding.file}`",
                    f"  - Line hint: `{finding.line_hint}`",
                    f"  - Evidence source: `{finding.evidence_source}`",
                    f"  - Impact: {finding.impact}",
                    f"  - Why this matters: {finding.impact}",
                    f"  - Explanation: {finding.explanation}",
                    f"  - Suggestion: {finding.suggestion}",
                    "",
                ]
            )

        return "\n".join(lines)

    def to_json(self, review: ReviewResult) -> str:
        """Serialize a review result to stable JSON."""
        return json.dumps(review.model_dump(), indent=2)

    def render_terminal(self, review: ReviewResult, console: Optional[Console]) -> None:
        """Render a professional terminal summary with Rich."""
        if console is None or Table is None or Panel is None:
            print(self.to_markdown(review))
            return
        console.print(
            Panel.fit(
                f"[bold]LLM Code Review Assistant[/bold]\n\n{review.summary}\n\nOverall risk: [bold]{review.overall_risk.upper()}[/bold]",
                title="Review Summary",
                border_style="cyan",
            )
        )
        if not review.findings:
            console.print("[green]No actionable issues identified from the supplied evidence.[/green]")
            return

        table = Table(title="Review Findings", show_lines=True)
        table.add_column("Severity", style="bold")
        table.add_column("Type")
        table.add_column("File:Line")
        table.add_column("Confidence")
        table.add_column("Title")
        for finding in review.findings:
            table.add_row(
                finding.severity.upper(),
                finding.issue_type,
                f"{finding.file}:{finding.line_hint}",
                f"{finding.confidence:.2f}",
                finding.title,
            )
        console.print(table)

    def save_outputs(
        self,
        review: ReviewResult,
        markdown_path: Optional[str] = None,
        json_path: Optional[str] = None,
    ) -> None:
        """Optionally persist markdown and JSON reports to disk."""
        if markdown_path:
            Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
            Path(markdown_path).write_text(self.to_markdown(review), encoding="utf-8")
        if json_path:
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            Path(json_path).write_text(self.to_json(review), encoding="utf-8")

    def render_json(self, review: ReviewResult, console: Optional[Console]) -> None:
        """Render JSON professionally when Rich is available."""
        if console is None or JSON is None:
            print(self.to_json(review))
            return
        console.print(Panel.fit(JSON(self.to_json(review)), title="Structured JSON", border_style="blue"))
