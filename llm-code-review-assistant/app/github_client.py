from __future__ import annotations

import base64
from typing import Any, Optional

import httpx

from app.config import Settings
from app.schemas import PullRequestData, PullRequestFile, PullRequestMetadata


class GitHubClientError(RuntimeError):
    """Raised when GitHub API access fails."""

    pass


class GitHubClient:
    """Thin GitHub REST client for pull request review data."""

    def __init__(self, settings: Settings) -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "llm-code-review-assistant",
        }
        if settings.github_token:
            headers["Authorization"] = f"Bearer {settings.github_token}"

        self._settings = settings
        self._client = httpx.Client(
            base_url=settings.github_api_base_url.rstrip("/"),
            headers=headers,
            timeout=settings.request_timeout_seconds,
        )

    def fetch_pull_request(self, owner: str, repo: str, pr_number: int) -> PullRequestData:
        """Fetch pull request metadata, changed files, patches, and Python contents."""
        pull_payload = self._get_json(f"/repos/{owner}/{repo}/pulls/{pr_number}")
        files_payload = self._get_paginated_json(f"/repos/{owner}/{repo}/pulls/{pr_number}/files")

        metadata = PullRequestMetadata(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            title=pull_payload["title"],
            body=pull_payload.get("body") or "",
            base_ref=pull_payload["base"]["ref"],
            head_ref=pull_payload["head"]["ref"],
            head_sha=pull_payload["head"]["sha"],
        )

        files: list[PullRequestFile] = []
        for item in files_payload[: self._settings.max_files_for_review]:
            filename = item["filename"]
            status = item["status"]
            contents = None
            if filename.endswith(".py") and status != "removed":
                contents = self._fetch_file_contents(owner, repo, filename, metadata.head_sha)

            files.append(
                PullRequestFile(
                    filename=filename,
                    status=status,
                    additions=item.get("additions", 0),
                    deletions=item.get("deletions", 0),
                    changes=item.get("changes", 0),
                    patch=item.get("patch"),
                    contents=contents,
                )
            )

        return PullRequestData(metadata=metadata, files=files)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def _fetch_file_contents(self, owner: str, repo: str, path: str, ref: str) -> Optional[str]:
        try:
            payload = self._get_json(f"/repos/{owner}/{repo}/contents/{path}", params={"ref": ref})
        except GitHubClientError:
            return None

        encoded = payload.get("content")
        if not encoded:
            return None

        try:
            return base64.b64decode(encoded).decode("utf-8")
        except UnicodeDecodeError:
            return None

    def _get_paginated_json(self, path: str) -> list[dict[str, Any]]:
        page = 1
        results: list[dict[str, Any]] = []

        while True:
            payload = self._get_json(path, params={"page": page, "per_page": 100})
            if not isinstance(payload, list):
                raise GitHubClientError(f"Expected list response from GitHub for {path}")
            if not payload:
                break
            results.extend(payload)
            if len(payload) < 100:
                break
            page += 1

        return results

    def _get_json(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        try:
            response = self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise GitHubClientError(f"GitHub API request failed: {exc}") from exc
        if response.status_code >= 400:
            raise GitHubClientError(
                f"GitHub API request failed with status {response.status_code}: {response.text}"
            )
        return response.json()
