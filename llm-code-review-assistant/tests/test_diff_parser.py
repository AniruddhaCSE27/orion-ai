from __future__ import annotations

from app.diff_parser import DiffParser


def test_parse_patch_extracts_added_lines_and_ranges() -> None:
    patch = """@@ -1,3 +1,5 @@
 line_1
+line_2
+line_3
 line_4
"""

    parsed = DiffParser().parse_patch("app/example.py", patch)

    assert parsed.file_path == "app/example.py"
    assert parsed.added_lines == [2, 3]
    assert parsed.changed_line_ranges == [(1, 5)]


def test_parse_patch_ignores_removed_line_numbers() -> None:
    patch = """@@ -10,3 +10,4 @@
 line_10
-old_line
+new_line
 line_12
+line_13
"""

    parsed = DiffParser().parse_patch("app/example.py", patch)

    assert parsed.added_lines == [11, 13]


def test_contains_line_matches_changed_ranges() -> None:
    patch = """@@ -3,2 +3,3 @@
 line_3
+line_4
 line_5
"""

    parsed = DiffParser().parse_patch("app/example.py", patch)

    assert parsed.contains_line(3) is True
    assert parsed.contains_line(4) is True
    assert parsed.contains_line(5) is True
    assert parsed.contains_line(9) is False
