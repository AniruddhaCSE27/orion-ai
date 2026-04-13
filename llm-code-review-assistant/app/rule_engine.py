from __future__ import annotations

from typing import Optional

from app.schemas import FileASTAnalysis, ParsedDiffFile, ReviewFinding


class RuleBasedReviewer:
    """Generate deterministic findings before the LLM step."""

    def build_findings(
        self,
        parsed_diffs: list[ParsedDiffFile],
        ast_analyses: list[FileASTAnalysis],
    ) -> list[ReviewFinding]:
        """Map AST and diff evidence to high-signal deterministic findings."""
        diff_map = {diff.file_path: diff for diff in parsed_diffs}
        findings: list[ReviewFinding] = []

        for analysis in ast_analyses:
            diff = diff_map.get(analysis.file)
            for issue in analysis.issues:
                if diff is not None and diff.added_lines and not diff.contains_line(issue.line):
                    continue
                finding = self._issue_to_finding(
                    issue_type=issue.issue_type,
                    file_path=analysis.file,
                    line=issue.line,
                    evidence=issue.evidence,
                )
                if finding is not None:
                    findings.append(finding)

        findings.extend(self._detect_suspicious_diff_patterns(parsed_diffs))
        return self._deduplicate(findings)

    def _issue_to_finding(self, issue_type: str, file_path: str, line: int, evidence: str) -> Optional[ReviewFinding]:
        mapping = {
            "broad_exception": ReviewFinding(
                title="Broad exception handler hides failure modes",
                issue_type="bug_risk",
                severity="high",
                confidence=0.94,
                file=file_path,
                line_hint=line,
                impact="Catching every exception can mask real defects and turn unexpected failures into silent fallback behavior.",
                explanation=f"The updated code uses a bare `except:` block. {evidence}",
                suggestion="Catch expected exception types explicitly and log or re-raise unexpected failures.",
                evidence_source="heuristic",
            ),
            "deep_nesting": ReviewFinding(
                title="New branching depth increases edge-case risk",
                issue_type="bug_risk",
                severity="medium",
                confidence=0.79,
                file=file_path,
                line_hint=line,
                impact="Additional decision depth makes it easier for new branches to bypass validation or produce inconsistent outcomes on edge cases.",
                explanation=f"The diff introduces control flow that exceeds the configured nesting threshold. {evidence}",
                suggestion="Reduce the number of nested branches in the changed path so each outcome is validated explicitly.",
                evidence_source="heuristic",
            ),
            "parse_error": ReviewFinding(
                title="Updated Python could not be parsed",
                issue_type="bug_risk",
                severity="high",
                confidence=0.99,
                file=file_path,
                line_hint=line,
                impact="If the source cannot be parsed, the change may fail before runtime or hide more serious analysis gaps.",
                explanation=f"Static analysis could not parse the file. {evidence}",
                suggestion="Fix the syntax issue first, then rerun the review pipeline.",
                evidence_source="ast",
            ),
            "mutable_default": ReviewFinding(
                title="Mutable default parameter can leak state between calls",
                issue_type="bug_risk",
                severity="high",
                confidence=0.97,
                file=file_path,
                line_hint=line,
                impact="Mutable defaults are shared across invocations and can cause surprising cross-request state leaks.",
                explanation=f"The function defines a mutable default value. {evidence}",
                suggestion="Use `None` as the default and initialize the mutable value inside the function body.",
                evidence_source="heuristic",
            ),
        }
        return mapping.get(issue_type)

    def _detect_suspicious_diff_patterns(self, parsed_diffs: list[ParsedDiffFile]) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        for diff in parsed_diffs:
            patch_lower = diff.patch.lower()
            if "return none" in patch_lower and "raise" not in patch_lower and diff.added_lines:
                findings.append(
                    ReviewFinding(
                        title="New silent failure path returns None",
                        issue_type="bug_risk",
                        severity="medium",
                        confidence=0.72,
                        file=diff.file_path,
                        line_hint=diff.added_lines[0],
                        impact="Returning `None` in a new path can shift error handling to distant callers and make failures harder to diagnose.",
                        explanation="The diff introduces a `return None` path without matching evidence of an explicit nullable contract in the supplied change.",
                        suggestion="Prefer a typed result, explicit exception, or targeted tests that prove callers handle the new nullable path safely.",
                        evidence_source="diff",
                    )
                )
        return findings

    def _deduplicate(self, findings: list[ReviewFinding]) -> list[ReviewFinding]:
        deduplicated: dict[tuple[str, int, str], ReviewFinding] = {}
        for finding in findings:
            key = (finding.file, finding.line_hint, finding.title)
            existing = deduplicated.get(key)
            if existing is None or finding.confidence > existing.confidence:
                deduplicated[key] = finding

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
            deduplicated.values(),
            key=lambda item: (
                severity_order[item.severity],
                issue_order[item.issue_type],
                item.file,
                item.line_hint,
            ),
        )
