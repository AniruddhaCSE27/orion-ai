from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


IssueType = Literal[
    "bug_risk",
    "security",
    "maintainability",
    "best_practice",
    "performance",
    "readability",
]
Severity = Literal["low", "medium", "high"]
OverallRisk = Literal["low", "medium", "high"]
EvidenceSource = Literal["diff", "ast", "heuristic", "llm"]


class ReviewFinding(BaseModel):
    """A single review finding suitable for humans and automation."""

    model_config = ConfigDict(extra="forbid")

    title: str
    issue_type: IssueType
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    file: str
    line_hint: int = Field(ge=0)
    impact: str
    explanation: str
    suggestion: str
    evidence_source: EvidenceSource


class ReviewResult(BaseModel):
    """Final review payload returned by the pipeline."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    overall_risk: OverallRisk
    findings: list[ReviewFinding]


class DiffHunk(BaseModel):
    """A unified diff hunk."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    added_lines: list[int]


class ParsedDiffFile(BaseModel):
    """Parsed diff metadata for a single file."""

    file_path: str
    added_lines: list[int]
    changed_line_ranges: list[tuple[int, int]]
    hunks: list[DiffHunk]
    patch: str

    def contains_line(self, line_number: int) -> bool:
        """Return whether a line number falls inside the changed ranges."""
        return any(start <= line_number <= end for start, end in self.changed_line_ranges)


class PullRequestFile(BaseModel):
    """File metadata fetched from GitHub."""

    filename: str
    status: str
    additions: int
    deletions: int
    changes: int
    patch: Optional[str] = None
    contents: Optional[str] = None


class PullRequestMetadata(BaseModel):
    """Top-level pull request metadata."""

    owner: str
    repo: str
    pr_number: int
    title: str
    body: str
    base_ref: str
    head_ref: str
    head_sha: str


class PullRequestData(BaseModel):
    """Full pull request data used during review."""

    metadata: PullRequestMetadata
    files: list[PullRequestFile]


class ASTIssue(BaseModel):
    """A deterministic AST analysis issue."""

    issue_type: Literal[
        "long_function",
        "deep_nesting",
        "broad_exception",
        "missing_docstring",
        "too_many_parameters",
        "parse_error",
        "mutable_default",
    ]
    file: str
    line: int
    message: str
    evidence: str


class FileASTAnalysis(BaseModel):
    """AST issues collected for a specific file."""

    file: str
    issues: list[ASTIssue]


class ReviewContext(BaseModel):
    """Normalized review evidence shared across pipeline stages."""

    pull_request: PullRequestData
    parsed_diffs: list[ParsedDiffFile]
    ast_analyses: list[FileASTAnalysis]
    heuristic_findings: list[ReviewFinding]


class LLMReviewPayload(BaseModel):
    """Strict validation model for the LLM response."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    overall_risk: OverallRisk
    findings: list[ReviewFinding]

    @field_validator("findings")
    @classmethod
    def findings_must_be_reasonable(cls, findings: list[ReviewFinding]) -> list[ReviewFinding]:
        if len(findings) > 20:
            raise ValueError("LLM returned too many findings for a single review.")
        return findings
