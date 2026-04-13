# LLM Code Review Assistant

LLM Code Review Assistant is a CLI-first AI developer tool that reviews GitHub pull requests using a hybrid pipeline. It also includes a deployment-ready Streamlit interface powered by the same reviewer factory used by the CLI.

- GitHub API ingestion for PR metadata and changed files
- unified diff parsing for line-level grounding
- AST-based static analysis for deterministic Python findings
- rule-based heuristics for trusted pre-LLM review comments
- OpenAI reasoning for prioritization, synthesis, and clearer explanations
- Rich terminal output plus Markdown and JSON reporting
- Streamlit UI for hosted PR review workflows

The project is intentionally designed to look and feel like a credible internal engineering tool rather than a prompt-only demo.

## Why This Is Stronger Than A Prompt-Only Reviewer

Most LLM code review demos send a diff to a model and trust whatever comes back. This project adds a trust layer before the model:

- deterministic AST analysis surfaces concrete structural issues
- rule-based heuristics convert static signals into actionable findings
- the LLM is grounded by diff data, AST evidence, and heuristic findings
- model output is validated strictly and filtered against known files and changed line ranges
- the system falls back safely to deterministic review output if the LLM response is malformed or unavailable

That design makes the tool easier to explain in interviews and more believable as production-minded engineering work.

## Architecture

### High-Level Flow

`CLI / Streamlit -> Shared Reviewer Factory -> GitHubClient -> DiffParser -> ASTAnalyzer -> RuleBasedReviewer -> ReviewEngine -> ReportFormatter`

### Module Responsibilities

- `app/config.py`
  Loads environment variables, model settings, and AST thresholds.
- `app/github_client.py`
  Fetches pull request metadata, changed files, patch hunks, and Python file contents from the GitHub REST API.
- `app/diff_parser.py`
  Parses unified diff hunks into added lines and changed ranges for grounding.
- `app/ast_analyzer.py`
  Runs deterministic Python AST checks including long functions, deep nesting, broad exceptions, missing docstrings, too many parameters, parse errors, and mutable defaults.
- `app/rule_engine.py`
  Maps AST and diff evidence into deterministic review findings with severity, confidence, and impact.
- `app/prompt_builder.py`
  Builds a grounded system prompt and review payload for the LLM.
- `app/review_engine.py`
  Calls the OpenAI API, validates structured output, filters hallucinated files and line references, and falls back to deterministic findings when needed.
- `app/reviewer.py`
  Orchestrates the full review pipeline.
- `app/factory.py`
  Centralizes reviewer dependency wiring so CLI and Streamlit use the exact same pipeline.
- `app/report_formatter.py`
  Produces Markdown, JSON, optional file exports, and Rich terminal rendering.
- `app/schemas.py`
  Defines Pydantic models for diffs, AST evidence, review context, and final review output.
- `streamlit_app.py`
  Streamlit entrypoint for reviewing PRs in a browser with JSON and Markdown downloads.

## Hybrid Review Pipeline

1. The CLI or Streamlit UI receives `owner`, `repo`, and `pr`.
2. GitHub ingestion fetches PR metadata and changed files.
3. Diff parsing extracts changed line ranges for grounding.
4. AST analysis runs on changed Python files with retrievable contents.
5. Rule-based heuristics convert deterministic issues into trusted findings.
6. The LLM receives PR metadata, diff context, AST issues, and heuristic findings.
7. The model returns strict JSON only.
8. The response is validated and filtered to reject unknown files or suspicious line references.
9. If the model fails or returns invalid JSON, the tool falls back to deterministic review output.
10. Results are rendered to the terminal and can be saved as Markdown or JSON.

## Structured Output

The review engine validates this schema:

```json
{
  "summary": "string",
  "overall_risk": "low | medium | high",
  "findings": [
    {
      "title": "string",
      "issue_type": "bug_risk | security | maintainability | best_practice | performance | readability",
      "severity": "low | medium | high",
      "confidence": 0.0,
      "file": "string",
      "line_hint": 0,
      "impact": "string",
      "explanation": "string",
      "suggestion": "string",
      "evidence_source": "diff | ast | heuristic | llm"
    }
  ]
}
```

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env`.
4. Add your credentials.

## Environment Variables

- `OPENAI_API_KEY`
  Required. OpenAI API key.
- `OPENAI_MODEL`
  Optional. Defaults to `gpt-4.1-mini`.
- `GITHUB_TOKEN`
  Optional but recommended to avoid low GitHub rate limits.
- `GITHUB_API_BASE_URL`
  Optional. Defaults to `https://api.github.com`.
- `REQUEST_TIMEOUT_SECONDS`
  Optional HTTP timeout.
- `MAX_FILES_FOR_REVIEW`
  Optional cap on the number of PR files analyzed.

## CLI Usage

Basic usage:

```bash
python run.py --owner <owner> --repo <repo> --pr <number>
```

Save reports to disk:

```bash
python run.py --owner psf --repo requests --pr 6710 --markdown-output review.md --json-output review.json
```

Disable Rich rendering:

```bash
python run.py --owner psf --repo requests --pr 6710 --no-rich
```

## Streamlit Usage

Run locally:

```bash
streamlit run streamlit_app.py
```

The Streamlit app supports:

- owner, repo, and PR number input
- GitHub PR URL parsing
- review summary and findings
- empty-state message when no significant issues are found in changed lines
- JSON and Markdown downloads

The UI is still grounded by the same strict diff-aware reviewer pipeline used by the CLI.

## Deployment

### Streamlit Community Cloud

1. Push this project to GitHub.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Set the entrypoint to:

```text
streamlit_app.py
```

4. Add these secrets or environment variables:

- `OPENAI_API_KEY`
- `GITHUB_TOKEN`
- `OPENAI_MODEL`

Optional:

- `GITHUB_API_BASE_URL`
- `REQUEST_TIMEOUT_SECONDS`
- `MAX_FILES_FOR_REVIEW`

The app reads standard environment variables and also maps Streamlit secrets into environment variables at startup, so no code changes are needed for hosted deployment.

### Local Environment Example

Copy `.env.example` to `.env` and populate:

```bash
OPENAI_API_KEY=your_openai_api_key
GITHUB_TOKEN=your_github_token
OPENAI_MODEL=gpt-4.1-mini
```

## Sample Output

Terminal and Markdown summary:

```markdown
# LLM Code Review Report

## Summary
The pull request introduces retry logic in the payment path, but the new error handling can hide real failures and makes the control flow harder to reason about.

**Overall Risk:** `HIGH`

## Findings
### Broad exception handler hides failure modes

- Severity: `high`
- Confidence: `0.94`
- Type: `bug_risk`
  - File: `app/service.py`
  - Line hint: `24`
  - Evidence source: `heuristic`
  - Impact: Catching every exception can mask real defects and turn unexpected failures into silent fallback behavior.
  - Explanation: The updated code uses a bare `except:` block, which means programmer errors and operational failures are handled identically.
  - Suggestion: Catch expected exception types explicitly and log or re-raise unexpected failures.
```

JSON:

```json
{
  "summary": "The pull request introduces retry logic in the payment path, but the new error handling can hide real failures and makes the control flow harder to reason about.",
  "overall_risk": "high",
  "findings": [
    {
      "title": "Broad exception handler hides failure modes",
      "issue_type": "bug_risk",
      "severity": "high",
      "confidence": 0.94,
      "file": "app/service.py",
      "line_hint": 24,
      "impact": "Catching every exception can mask real defects and turn unexpected failures into silent fallback behavior.",
      "explanation": "The updated code uses a bare `except:` block, which means programmer errors and operational failures are handled identically.",
      "suggestion": "Catch expected exception types explicitly and log or re-raise unexpected failures.",
      "evidence_source": "heuristic"
    }
  ]
}
```

## Testing

Run the full test suite:

```bash
pytest tests
```

Quick validation checks:

```bash
python run.py --help
streamlit run streamlit_app.py
```

Coverage includes:

- diff parsing edge cases
- AST analyzer edge cases
- reviewer orchestration
- report formatting
- LLM JSON validation and fallback behavior

## Future Improvements

- language-specific analyzers beyond Python
- patch chunk ranking for very large pull requests
- inline GitHub review comment publishing
- caching of file contents and repeated reviews
- richer semantic heuristics for risky control flow and error handling
