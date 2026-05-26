from __future__ import annotations

import streamlit_app


def test_parse_pr_url_extracts_owner_repo_and_number() -> None:
    parsed = streamlit_app._parse_pr_url("https://github.com/psf/requests/pull/7315")

    assert parsed == ("psf", "requests", 7315)


def test_parse_pr_url_rejects_invalid_input() -> None:
    assert streamlit_app._parse_pr_url("not-a-pr-url") is None
