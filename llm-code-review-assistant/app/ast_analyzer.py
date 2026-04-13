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
        self._nesting_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self._visit_nested_node(node)

    def visit_For(self, node: ast.For) -> None:
        self._visit_nested_node(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_nested_node(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_nested_node(node)

    def visit_With(self, node: ast.With) -> None:
        self._visit_nested_node(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_nested_node(node)

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
        self._visit_nested_node(node)

    def _handle_function(self, node: FunctionNode) -> None:
        function_length = self._function_length(node)
        if function_length > self.thresholds.long_function_lines:
            self.issues.append(
                ASTIssue(
                    issue_type="long_function",
                    file=self.file_path,
                    line=node.lineno,
                    message="Function exceeds the configured length threshold.",
                    evidence=f"{node.name} spans {function_length} lines.",
                )
            )

        parameter_count = len(node.args.args) + len(node.args.kwonlyargs)
        if node.args.vararg is not None:
            parameter_count += 1
        if node.args.kwarg is not None:
            parameter_count += 1
        if parameter_count > self.thresholds.too_many_parameters:
            self.issues.append(
                ASTIssue(
                    issue_type="too_many_parameters",
                    file=self.file_path,
                    line=node.lineno,
                    message="Function signature has many parameters.",
                    evidence=f"{node.name} defines {parameter_count} parameters.",
                )
            )

        if ast.get_docstring(node) is None:
            self.issues.append(
                ASTIssue(
                    issue_type="missing_docstring",
                    file=self.file_path,
                    line=node.lineno,
                    message="Function is missing a docstring.",
                    evidence=f"{node.name} does not declare a docstring.",
                )
            )

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

    def _visit_nested_node(self, node: ast.AST) -> None:
        self._nesting_depth += 1
        try:
            if self._nesting_depth > self.thresholds.deep_nesting_level and hasattr(node, "lineno"):
                self.issues.append(
                    ASTIssue(
                        issue_type="deep_nesting",
                        file=self.file_path,
                        line=getattr(node, "lineno", 1),
                        message="Control flow nesting is deeper than the configured threshold.",
                        evidence=f"Nesting depth reached {self._nesting_depth}.",
                    )
                )
            self.generic_visit(node)
        finally:
            self._nesting_depth -= 1

    @staticmethod
    def _function_length(node: FunctionNode) -> int:
        end_lineno = getattr(node, "end_lineno", node.lineno)
        return end_lineno - node.lineno + 1
