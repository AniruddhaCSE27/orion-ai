from __future__ import annotations

from types import SimpleNamespace

from app import main as cli_main
from app.schemas import ReviewResult


class _FakeReviewer:
    def review_pull_request(self, owner: str, repo: str, pr_number: int):
        assert owner == "octo"
        assert repo == "demo"
        assert pr_number == 7
        review = ReviewResult(
            summary="No significant issues found in the changed lines.",
            overall_risk="low",
            findings=[],
        )
        return "# report", review


class _FakeFormatter:
    def __init__(self) -> None:
        self.saved_review: ReviewResult | None = None

    def save_outputs(self, review: ReviewResult, markdown_path=None, json_path=None) -> None:
        self.saved_review = review

    def to_json(self, review: ReviewResult) -> str:
        return review.model_dump_json()

    def render_terminal(self, review: ReviewResult, console=None) -> None:
        self.saved_review = review

    def render_json(self, review: ReviewResult, console=None) -> None:
        self.saved_review = review


class _FakeServices:
    def __init__(self) -> None:
        self.report_formatter = _FakeFormatter()
        self.reviewer = _FakeReviewer()
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeParser:
    def parse_args(self):
        return SimpleNamespace(
            owner="octo",
            repo="demo",
            pr=7,
            markdown_output=None,
            json_output=None,
            no_rich=True,
        )


def test_cli_main_uses_shared_reviewer_factory(monkeypatch) -> None:
    fake_services = _FakeServices()

    monkeypatch.setattr(cli_main, "get_settings", lambda: object())
    monkeypatch.setattr(cli_main, "create_reviewer", lambda settings: fake_services)
    monkeypatch.setattr(cli_main, "build_argument_parser", lambda: _FakeParser())

    cli_main.main()

    assert fake_services.report_formatter.saved_review is not None
    assert fake_services.closed is True
