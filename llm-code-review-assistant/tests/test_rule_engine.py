from __future__ import annotations

from app.rule_engine import RuleBasedReviewer
from app.schemas import ASTIssue, DiffHunk, FileASTAnalysis, ParsedDiffFile


def test_rule_engine_excludes_unchanged_whole_file_ast_issues() -> None:
    reviewer = RuleBasedReviewer()
    parsed_diffs = [
        ParsedDiffFile(
            file_path="app/service.py",
            added_lines=[20],
            changed_line_ranges=[(18, 22)],
            hunks=[DiffHunk(old_start=18, old_count=2, new_start=18, new_count=5, added_lines=[20])],
            patch="@@ -18,2 +18,5 @@\n line\n+new_line\n line\n",
        )
    ]
    ast_analyses = [
        FileASTAnalysis(
            file="app/service.py",
            issues=[
                ASTIssue.model_construct(
                    issue_type="broad_exception",
                    file="app/service.py",
                    line=19,
                    message="legacy issue",
                    evidence="unchanged context line",
                )
            ],
        )
    ]

    findings = reviewer.build_findings(parsed_diffs=parsed_diffs, ast_analyses=ast_analyses)

    assert findings == []


def test_rule_engine_ignores_low_signal_ast_issue_types_even_if_present() -> None:
    reviewer = RuleBasedReviewer()
    parsed_diffs = [
        ParsedDiffFile(
            file_path="app/service.py",
            added_lines=[10],
            changed_line_ranges=[(10, 10)],
            hunks=[DiffHunk(old_start=10, old_count=0, new_start=10, new_count=1, added_lines=[10])],
            patch="@@ -10,0 +10,1 @@\n+line\n",
        )
    ]
    ast_analyses = [
        FileASTAnalysis(
            file="app/service.py",
            issues=[
                ASTIssue.model_construct(
                    issue_type="missing_docstring",
                    file="app/service.py",
                    line=10,
                    message="missing docstring",
                    evidence="legacy docstring issue",
                ),
                ASTIssue.model_construct(
                    issue_type="long_function",
                    file="app/service.py",
                    line=10,
                    message="long function",
                    evidence="legacy length issue",
                ),
                ASTIssue.model_construct(
                    issue_type="deep_nesting",
                    file="app/service.py",
                    line=10,
                    message="deep nesting",
                    evidence="legacy nesting issue",
                ),
            ],
        )
    ]

    findings = reviewer.build_findings(parsed_diffs=parsed_diffs, ast_analyses=ast_analyses)

    assert findings == []
