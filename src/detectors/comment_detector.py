"""
comment_detector.py - Hardening Fix #1: Single-line comment deprecation detection.

Detects bare '# deprecated' comments attached to functions or classes using
Python's tokenize module paired with AST node boundaries.

Fixes APIScanner failure mode where deprecated APIs documented solely through
comments (without @deprecated decorator, warnings.warn, or docstring) were missed.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from src.detectors.legacy_heuristics import DeprecationCandidate


class CommentDeprecationDetector:
    """
    Scans Python source code using tokenize and AST to detect single-line comments
    attached to functions or classes that signal deprecation.
    """

    DEPRECATION_PATTERN = re.compile(r"(?i)\bdeprecat")
    NEGATION_PATTERN = re.compile(r"(?i)\b(not|never|has not been|is not|no|won't be)\s+deprecat")

    def __init__(self, filename: str = "<string>", package_prefix: str = "") -> None:
        self.filename = filename
        self.package_prefix = package_prefix

    def detect(self, source_code: str) -> List[DeprecationCandidate]:
        """Detects comment-style deprecations in source_code."""
        try:
            tree = ast.parse(source_code, filename=self.filename)
        except SyntaxError:
            return []

        # Extract all comments via tokenize
        comments_by_line: Dict[int, str] = {}
        try:
            token_gen = tokenize.tokenize(io.BytesIO(source_code.encode("utf-8")).readline)
            for tok in token_gen:
                if tok.type == tokenize.COMMENT:
                    comments_by_line[tok.start[0]] = tok.string
        except (tokenize.TokenError, IndentationError):
            return []

        if not comments_by_line:
            return []

        # Collect all statement boundary lines
        stmt_lines: Set[int] = set()
        for node in ast.walk(tree):
            if hasattr(node, "lineno"):
                stmt_lines.add(node.lineno)

        candidates: List[DeprecationCandidate] = []

        # Scope tracker: (node, qualified_name)
        def walk_scopes(
            parent_node: ast.AST, current_scope: List[str]
        ) -> List[Tuple[Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef], str]]:
            items = []
            for child in getattr(parent_node, "body", []):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    qname_parts = []
                    if self.package_prefix:
                        qname_parts.append(self.package_prefix)
                    qname_parts.extend(current_scope)
                    qname_parts.append(child.name)
                    qname = ".".join(qname_parts)
                    items.append((child, qname))
                    items.extend(walk_scopes(child, current_scope + [child.name]))
            return items

        targets = walk_scopes(tree, [])

        for node, qname in targets:
            start_line = node.lineno
            if getattr(node, "decorator_list", None):
                start_line = min([node.lineno] + [d.lineno for d in node.decorator_list])

            attached_comments: List[Tuple[int, str]] = []

            # 1. Preceding comments: look up to 5 lines backwards, stopping if reaching another statement
            for line_no in range(start_line - 1, max(0, start_line - 6), -1):
                if line_no in stmt_lines and line_no not in comments_by_line:
                    break
                if line_no in comments_by_line:
                    attached_comments.append((line_no, comments_by_line[line_no]))

            # 2. Inline comments on the def / class line
            for line_no in range(start_line, node.lineno + 1):
                if line_no in comments_by_line:
                    attached_comments.append((line_no, comments_by_line[line_no]))

            # 3. Leading comments inside body (before first statement or docstring)
            if getattr(node, "body", None):
                first_stmt = node.body[0]
                cutoff = first_stmt.lineno
                for line_no in range(node.lineno + 1, cutoff):
                    if line_no in comments_by_line:
                        attached_comments.append((line_no, comments_by_line[line_no]))

            # Check if any attached comment matches deprecation (and not negated)
            matched_lines: List[int] = []
            matched_texts: List[str] = []

            for line_no, text in attached_comments:
                if self.DEPRECATION_PATTERN.search(text) and not self.NEGATION_PATTERN.search(text):
                    matched_lines.append(line_no)
                    matched_texts.append(text.strip())

            if matched_lines:
                evidence = " ; ".join(matched_texts)
                primary_line = matched_lines[0]
                candidates.append(
                    DeprecationCandidate(
                        qualified_name=qname,
                        origin="comment",
                        location=f"{self.filename}:{primary_line}",
                        raw_evidence=evidence,
                        line=primary_line,
                    )
                )

        return candidates


def detect_comment_deprecations(
    source_code: str,
    filename: str = "<string>",
    package_prefix: str = "",
) -> List[DeprecationCandidate]:
    """Detects single-line comment deprecations from source code string."""
    detector = CommentDeprecationDetector(filename=filename, package_prefix=package_prefix)
    return detector.detect(source_code)


def detect_comment_deprecations_from_file(
    filepath: Union[str, Path],
    package_prefix: str = "",
) -> List[DeprecationCandidate]:
    """Detects single-line comment deprecations from a file on disk."""
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    source_code = path.read_text(encoding="utf-8", errors="replace")
    return detect_comment_deprecations(
        source_code=source_code,
        filename=str(path),
        package_prefix=package_prefix,
    )
