from __future__ import annotations

import re
from typing import Optional

from app.schemas import DiffHunk, ParsedDiffFile


HUNK_HEADER_PATTERN = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


class DiffParser:
    """Parse unified diffs into structured hunk metadata."""

    def parse_patch(self, file_path: str, patch: str) -> ParsedDiffFile:
        """Parse a single-file patch into changed ranges and added line numbers."""
        hunks: list[DiffHunk] = []
        current_hunk: Optional[DiffHunk] = None
        new_line_number = 0

        for raw_line in patch.splitlines():
            header_match = HUNK_HEADER_PATTERN.match(raw_line)
            if header_match:
                if current_hunk is not None:
                    hunks.append(current_hunk)
                current_hunk = DiffHunk(
                    old_start=int(header_match.group("old_start")),
                    old_count=int(header_match.group("old_count") or 1),
                    new_start=int(header_match.group("new_start")),
                    new_count=int(header_match.group("new_count") or 1),
                    added_lines=[],
                )
                new_line_number = current_hunk.new_start
                continue

            if current_hunk is None:
                continue

            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                current_hunk.added_lines.append(new_line_number)
                new_line_number += 1
                continue

            if raw_line.startswith("-") and not raw_line.startswith("---"):
                continue

            if raw_line.startswith("\\"):
                continue

            new_line_number += 1

        if current_hunk is not None:
            hunks.append(current_hunk)

        added_lines = [line for hunk in hunks for line in hunk.added_lines]
        changed_line_ranges = [
            (hunk.new_start, hunk.new_start + max(hunk.new_count, 1) - 1)
            for hunk in hunks
        ]
        return ParsedDiffFile(
            file_path=file_path,
            added_lines=added_lines,
            changed_line_ranges=changed_line_ranges,
            hunks=hunks,
            patch=patch,
        )
