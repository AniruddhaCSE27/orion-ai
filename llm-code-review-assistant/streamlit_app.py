from __future__ import annotations

import os
import re
from typing import Optional

import streamlit as st

from app.config import get_settings
from app.factory import create_reviewer
from app.github_client import GitHubClientError


PR_URL_RE = re.compile(r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)", re.IGNORECASE)


def _load_streamlit_secrets() -> None:
    for key in ("OPENAI_API_KEY", "GITHUB_TOKEN", "OPENAI_MODEL"):
        if key not in os.environ:
            value = st.secrets.get(key)
            if value:
                os.environ[key] = str(value)


def _parse_pr_url(pr_url: str) -> tuple[str, str, int] | None:
    match = PR_URL_RE.search((pr_url or "").strip())
    if not match:
        return None
    owner, repo, pr_number = match.groups()
    return owner, repo, int(pr_number)


def _render_findings(findings) -> None:
    if not findings:
        st.success("No significant issues found in the changed lines.")
        return

    for finding in findings:
        with st.container(border=True):
            st.markdown(f"### {finding.title}")
            st.caption(
                f"{finding.severity.upper()} | {finding.issue_type} | "
                f"{finding.file}:{finding.line_hint} | confidence {finding.confidence:.2f}"
            )
            st.write(f"Impact: {finding.impact}")
            st.write(f"Explanation: {finding.explanation}")
            st.write(f"Suggestion: {finding.suggestion}")
            st.caption(f"Evidence source: {finding.evidence_source}")


def main() -> None:
    st.set_page_config(
        page_title="ORION AI Reviewer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _load_streamlit_secrets()

    st.title("ORION AI Pull Request Reviewer")
    st.caption("Strict diff-aware code review for changed lines only.")

    with st.sidebar:
        st.header("How It Works")
        st.write("Review is limited to changed diff lines and grounded by diff, AST, heuristic, and LLM evidence.")
        st.write("Set `OPENAI_API_KEY`, `GITHUB_TOKEN`, and optionally `OPENAI_MODEL` in environment variables or Streamlit secrets.")

    try:
        settings = get_settings()
    except ValueError as exc:
        st.error(f"Configuration error: {exc}")
        st.info("Add the required keys in your environment or Streamlit app secrets before running a review.")
        return

    st.caption(f"Model: `{settings.openai_model}`")

    left, right = st.columns([2, 1])
    with left:
        pr_url = st.text_input("GitHub PR URL", placeholder="https://github.com/owner/repo/pull/123")
    with right:
        st.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
        parse_url = st.checkbox("Parse PR URL", value=True)

    parsed = _parse_pr_url(pr_url) if parse_url and pr_url.strip() else None

    col1, col2, col3 = st.columns(3)
    with col1:
        owner = st.text_input("Owner", value=parsed[0] if parsed else "", placeholder="openai")
    with col2:
        repo = st.text_input("Repo", value=parsed[1] if parsed else "", placeholder="openai-python")
    with col3:
        pr_value = str(parsed[2]) if parsed else ""
        pr_number_raw = st.text_input("PR Number", value=pr_value, placeholder="123")

    if not owner and not repo and not pr_number_raw:
        st.info("Enter a GitHub PR URL or provide owner, repo, and PR number to start a review.")

    if st.button("Run Review", type="primary", use_container_width=True):
        if not owner.strip() or not repo.strip() or not pr_number_raw.strip():
            st.error("Owner, repo, and PR number are required.")
            return
        try:
            pr_number = int(pr_number_raw)
        except ValueError:
            st.error("PR number must be a valid integer.")
            return

        with st.spinner("Reviewing pull request and grounding findings against changed lines..."):
            services = None
            try:
                services = create_reviewer(settings)
                markdown_report, review = services.reviewer.review_pull_request(owner.strip(), repo.strip(), pr_number)
                json_output = services.report_formatter.to_json(review)
            except GitHubClientError as exc:
                st.error(f"GitHub request failed: {exc}")
                return
            except RuntimeError as exc:
                st.error(f"Review generation failed: {exc}")
                return
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                return
            finally:
                if services is not None:
                    services.close()

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Overall Risk", review.overall_risk.upper())
        metric2.metric("Findings", len(review.findings))
        metric3.metric("Scope", "Changed Lines Only")

        st.subheader("Summary")
        st.write(review.summary)

        st.subheader("Findings")
        _render_findings(review.findings)

        with st.expander("Markdown Report", expanded=False):
            st.markdown(markdown_report)

        with st.expander("JSON Output", expanded=False):
            st.json(review.model_dump())

        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "Download JSON",
                data=json_output,
                file_name=f"{owner}-{repo}-pr-{pr_number}-review.json",
                mime="application/json",
                use_container_width=True,
            )
        with download_col2:
            st.download_button(
                "Download Markdown",
                data=markdown_report,
                file_name=f"{owner}-{repo}-pr-{pr_number}-review.md",
                mime="text/markdown",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
