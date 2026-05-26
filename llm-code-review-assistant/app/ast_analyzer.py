from __future__ import annotations

import ast
from typing import Union

from app.config import AnalyzerThresholds
from app.schemas import ASTIssue, FileASTAnalysis


FunctionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]


class ASTAnalyzer:
    """Run deterministic AST-based checks on Python files."""

    def __init__(self, thresholds: AnalyzerThresholds) -> None:
        self._thresholds = thresholds

    def analyze_file(self, file_path: str, source_code: str) -> FileASTAnalysis:
        """Analyze a Python source file and return structured issues."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as exc:
            return FileASTAnalysis(
                file=file_path,
                issues=[
                    ASTIssue(
                        issue_type="parse_error",
                        file=file_path,
                        line=exc.lineno or 1,
                        message="Python source could not be parsed for AST analysis.",
                        evidence=str(exc),
                    )
                ],
            )

        visitor = _AnalyzerVisitor(file_path=file_path, thresholds=self._thresholds)
        visitor.visit(tree)
        return FileASTAnalysis(file=file_path, issues=visitor.issues)


class _AnalyzerVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, thresholds: AnalyzerThresholds) -> None:
        self.file_path = file_path
        self.thresholds = thresholds
        self.issues: list[ASTIssue] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        for handler in node.handlers:
            if handler.type is None:
                self.issues.append(
                    ASTIssue(
                        issue_type="broad_exception",
                        file=self.file_path,
                        line=handler.lineno,
                        message="Broad exception handling makes failures harder to reason about.",
                        evidence="`except:` catches all exception types and can hide programming errors.",
                    )
                )
        self.generic_visit(node)

    def _handle_function(self, node: FunctionNode) -> None:
        for default_node in [*node.args.defaults, *node.args.kw_defaults]:
            if default_node is None:
                continue
            if isinstance(default_node, (ast.List, ast.Dict, ast.Set)):
                self.issues.append(
                    ASTIssue(
                        issue_type="mutable_default",
                        file=self.file_path,
                        line=getattr(default_node, "lineno", node.lineno),
                        message="Function defines a mutable default argument.",
                        evidence=f"{node.name} uses a mutable default value, which is shared between calls.",
                    )
                )
