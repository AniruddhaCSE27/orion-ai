from __future__ import annotations

from app.ast_analyzer import ASTAnalyzer
from app.config import AnalyzerThresholds


def test_ast_analyzer_detects_expected_issues() -> None:
    source_code = """
def complicated(a, b, c, d, e, f):
    if a:
        for item in b:
            while c:
                if d:
                    print(item)
    try:
        return a + 1
    except:
        return 0
"""

    analyzer = ASTAnalyzer(AnalyzerThresholds())
    analysis = analyzer.analyze_file("app/example.py", source_code)
    issue_types = {issue.issue_type for issue in analysis.issues}

    assert "broad_exception" in issue_types
    assert "mutable_default" not in issue_types


def test_ast_analyzer_reports_parse_error() -> None:
    analyzer = ASTAnalyzer(AnalyzerThresholds())
    analysis = analyzer.analyze_file("app/broken.py", "def broken(:\n    pass\n")

    assert analysis.issues[0].issue_type == "parse_error"


def test_ast_analyzer_detects_mutable_default_arguments() -> None:
    analyzer = ASTAnalyzer(AnalyzerThresholds())
    analysis = analyzer.analyze_file(
        "app/defaults.py",
        "def build_payload(items=[]):\n    return items\n",
    )

    assert any(issue.issue_type == "mutable_default" for issue in analysis.issues)
