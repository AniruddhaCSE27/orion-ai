# LLM Code Review Assistant

LLM Code Review Assistant is a Codex-style AI system for GitHub pull request review. It does not treat a PR as a single prompt. Instead, it runs a structured review pipeline that parses diffs, performs deterministic code analysis, applies review rules, and then asks the LLM for grounded synthesis.

The result is a production-oriented reviewer that behaves more like an engineering assistant than a generic chat wrapper.

## Overview

- Diff-aware review restricted to changed lines
- Codex-style modular pipeline with shared builder wiring
- Deterministic AST and rule-based checks before the LLM step
- Strict LLM output validation and fallback behavior
- CLI and Streamlit interfaces backed by the exact same reviewer factory
- Markdown and JSON outputs for automation, demos, and reporting

## Codex-Style Architecture

High-level flow:

`CLI / Streamlit -> Shared Reviewer Factory -> GitHub Client -> Diff Parser -> AST Analyzer -> Rule Engine -> LLM Review Engine -> Report Formatter`

Text architecture diagram:

1. Input layer
   CLI or Streamlit collects `owner`, `repo`, and `pr_number`.
2. Ingestion layer
   GitHub metadata, changed files, patches, and Python file contents are fetched.
3. Diff grounding layer
   Unified diffs are parsed into hunks, changed ranges, and added line references.
4. Deterministic analysis layer
   Python AST checks and rule-based heuristics extract concrete signals from the changed code path.
5. LLM review layer
   The model receives only structured review context and must return strict JSON.
6. Validation layer
   Findings are filtered against real files, changed-line ranges, and low-signal review policies.
7. Output layer
   Results are rendered consistently in terminal, Markdown, JSON, and Streamlit.

## Review Philosophy

The reviewer is intentionally strict about signal quality.

- It reviews only changed diff lines.
- It ignores unchanged code and pre-existing issues outside the patch.
- It filters out docstring suggestions, generic readability comments, and vague “function too long” style feedback.
- It prioritizes correctness, regression risk, security, API behavior, and performance impact.
- If nothing strong is found, it returns: `No significant issues found in the changed lines.`

## Core Modules

- `app/config.py`
  Loads environment variables, prompt location, runtime settings, and non-fatal integration warnings.
- `app/factory.py`
  Central shared builder for the full reviewer pipeline. Both CLI and Streamlit use this same function.
- `app/github_client.py`
  Pull request ingestion via the GitHub REST API.
- `app/diff_parser.py`
  Converts unified diffs into structured hunk metadata and changed line ranges.
- `app/ast_analyzer.py`
  Runs deterministic Python checks for parse failures, broad exception handling, deep nesting, and mutable defaults.
- `app/rule_engine.py`
  Converts deterministic evidence into senior-engineer-style findings.
- `app/prompt_builder.py`
  Serializes the grounded review context for the LLM.
- `app/review_engine.py`
  Executes the LLM review, validates structured JSON, rejects hallucinated findings, and falls back safely when needed.
- `app/reviewer.py`
  Orchestrates the full analysis workflow.
- `app/report_formatter.py`
  Produces terminal, Markdown, and JSON output.
- `streamlit_app.py`
  Deployment-ready Streamlit entrypoint.
- `run.py`
  CLI entrypoint.

## Features

- Pull request review by owner, repo, and PR number
- Optional GitHub PR URL parsing in Streamlit
- Structured findings with severity, confidence, file, and line hint
- Empty-state handling for clean PRs
- Downloadable Markdown and JSON reports
- Clear runtime errors for missing configuration
- Safe deterministic fallback when the OpenAI response is malformed or unavailable

## Diff-Aware Review Behavior

This project is intentionally patch-scoped.

- The diff parser records changed line ranges per file.
- Heuristic and LLM findings are validated against those ranges.
- Hallucinated file paths are discarded.
- Findings that point outside changed lines are discarded.
- Low-signal comments are discarded even if the LLM returns them.

That keeps the system focused on what changed in the PR instead of drifting into full-file review.

## Example Output

```markdown
# LLM Code Review Report

## Summary
The new retry path introduces a broad exception handler that can turn unexpected failures into silent fallback behavior.

**Overall Risk:** `HIGH`

## Findings
### Broad exception handler hides failure modes

- Severity: `high`
- Confidence: `0.94`
- Type: `bug_risk`
- File: `app/service.py`
- Line hint: `24`
- Evidence source: `heuristic`
- Impact: Unexpected integration failures can be swallowed and converted into silent fallback behavior.
- Explanation: The changed branch catches every exception type, so operational failures and programmer errors are treated identically.
- Suggestion: Catch the expected exception types explicitly and surface unexpected failures.
```

## Environment Setup

1. Create a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env`.
4. Set the required environment variables.

Required:

```env
OPENAI_API_KEY=your_openai_api_key
```

Recommended:

```env
GITHUB_TOKEN=your_github_token
OPENAI_MODEL=gpt-4.1-mini
GITHUB_API_BASE_URL=https://api.github.com
REQUEST_TIMEOUT_SECONDS=30
MAX_FILES_FOR_REVIEW=20
```

## CLI Usage

Basic review:

```bash
python run.py --owner openai --repo openai-python --pr 123
```

Save both report formats:

```bash
python run.py --owner openai --repo openai-python --pr 123 --markdown-output review.md --json-output review.json
```

Disable Rich output:

```bash
python run.py --owner openai --repo openai-python --pr 123 --no-rich
```

## Streamlit Usage

Run locally:

```bash
streamlit run streamlit_app.py
```

The Streamlit app supports:

- owner, repo, and PR number input
- optional GitHub PR URL parsing
- loading spinner and runtime validation
- summary and findings display
- empty state for clean reviews
- JSON and Markdown downloads

## Deployment

### Streamlit Community Cloud

1. Push the project to GitHub.
2. Create a new Streamlit app from the repository.
3. Set the entrypoint to `streamlit_app.py`.
4. Add secrets for:
   `OPENAI_API_KEY`
   `GITHUB_TOKEN`
   `OPENAI_MODEL`
5. Optionally add:
   `GITHUB_API_BASE_URL`
   `REQUEST_TIMEOUT_SECONDS`
   `MAX_FILES_FOR_REVIEW`
6. Deploy.

### What Makes It Deployment-Ready

- No hardcoded secrets
- No local-only file paths
- Standard environment variable configuration
- Shared service construction for CLI and UI
- Streamlit entrypoint at `streamlit_app.py`

## Testing

Run the full suite:

```bash
pytest -q
```

Useful validation commands:

```bash
python run.py --help
streamlit run streamlit_app.py
```

Current coverage includes:

- diff parsing
- AST analysis
- review engine validation and fallback behavior
- reviewer orchestration
- formatter output
- CLI execution path
- config validation

## Final Runtime Notes

- Final findings are validated against actual changed new-file lines, not just diff hunk context.
- Whole-file AST legacy issues are excluded from the final review output.
- If no valid PR-specific findings remain, the exact summary is:
  `The pull request is focused and no significant issues were found in the changed lines.`
- Only changed lines are reviewed.
