"""
legacy_heuristics.py - Port of APIScanner's three legacy deprecation heuristics:
1. Decorator matcher (@deprecated, @deprecate, etc.)
2. Warning call matcher (warnings.warn(...) with DeprecationWarning/FutureWarning)
3. Docstring keyword matcher (paragraphs containing 'deprecat*')

Provides a unified interface returning DeprecationCandidate objects.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Union


@dataclass
class DeprecationCandidate:
    """Represents a deprecation candidate detected by static heuristics or checkers."""

    qualified_name: str
    origin: str  # e.g., "decorator", "warning", "docstring", "pep702", "comment"
    location: str  # e.g., "module.py:42"
    raw_evidence: str
    line: int = 0
    origins: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.origins and self.origin:
            self.origins = {self.origin}


def _flatten_ast_attr(node: ast.AST) -> str:
    """Recursively reconstructs dotted attribute strings like 'warnings.warn'."""
    if isinstance(node, ast.Attribute):
        parent = _flatten_ast_attr(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Call):
        return _flatten_ast_attr(node.func)
    return ""


def _extract_string_value(node: ast.AST) -> str:
    """Extracts string content from Constant, Str, or formatted values."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    elif hasattr(ast, "Str") and isinstance(node, ast.Str):  # Python < 3.8 back-compat
        return getattr(node, "s", "")
    elif isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                parts.append(part.value)
            elif isinstance(part, ast.FormattedValue):
                parts.append("{...}")
        return "".join(parts)
    return ""


class LegacyHeuristicsVisitor(ast.NodeVisitor):
    """
    AST Visitor implementing APIScanner's three legacy detection heuristics:
    - Decorator matching
    - warnings.warn matching
    - Docstring keyword matching
    """

    DEPRECATION_PATTERN = re.compile(r"(?i)deprecat")
    WARNING_CATEGORIES = {"DeprecationWarning", "PendingDeprecationWarning", "FutureWarning"}

    def __init__(self, filename: str = "<string>", package_prefix: str = "") -> None:
        self.filename = filename
        self.package_prefix = package_prefix
        self.scope_stack: List[str] = []
        self.candidates: List[DeprecationCandidate] = []

    def _current_qualified_name(self, name: str) -> str:
        parts = []
        if self.package_prefix:
            parts.append(self.package_prefix)
        parts.extend(self.scope_stack)
        parts.append(name)
        return ".".join(parts)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qname = self._current_qualified_name(node.name)

        # 1. Docstring heuristic on class
        self._check_docstring(node, qname)

        # 2. Decorator heuristic on class
        self._check_decorators(node, qname)

        # Push class scope and visit children (methods, nested classes)
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._inspect_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._inspect_function(node)

    def _inspect_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        qname = self._current_qualified_name(node.name)

        # 1. Decorator heuristic
        self._check_decorators(node, qname)

        # 2. Docstring heuristic
        self._check_docstring(node, qname)

        # 3. Warning call heuristic inside function body
        self._check_warnings(node, qname)

        # Visit nested functions/classes if any
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def _check_decorators(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef], qname: str) -> None:
        for dec in node.decorator_list:
            dec_id = ""
            args_repr = ""

            if isinstance(dec, ast.Name):
                dec_id = dec.id
            elif isinstance(dec, ast.Attribute):
                dec_id = _flatten_ast_attr(dec)
            elif isinstance(dec, ast.Call):
                dec_id = _flatten_ast_attr(dec.func)
                # Extract first string argument if present (e.g. deprecation message)
                for arg in dec.args:
                    val = _extract_string_value(arg)
                    if val:
                        args_repr = f"({val!r})"
                        break

            if dec_id and self.DEPRECATION_PATTERN.search(dec_id):
                evidence = f"@{dec_id}{args_repr}"
                self.candidates.append(
                    DeprecationCandidate(
                        qualified_name=qname,
                        origin="decorator",
                        location=f"{self.filename}:{node.lineno}",
                        raw_evidence=evidence,
                        line=node.lineno,
                    )
                )

    def _check_docstring(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef], qname: str) -> None:
        doc = ast.get_docstring(node)
        if not doc:
            return

        if self.DEPRECATION_PATTERN.search(doc):
            # Extract paragraphs or lines matching 'deprecat'
            paragraphs = doc.split("\n\n")
            matching_paras = [
                " ".join(p.split()) for p in paragraphs if self.DEPRECATION_PATTERN.search(p)
            ]
            evidence = " ~ ".join(matching_paras) if matching_paras else " ".join(doc.split())
            self.candidates.append(
                DeprecationCandidate(
                    qualified_name=qname,
                    origin="docstring",
                    location=f"{self.filename}:{node.lineno}",
                    raw_evidence=evidence,
                    line=node.lineno,
                )
            )

    def _check_warnings(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], qname: str) -> None:
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue

            func_name = _flatten_ast_attr(child.func)
            is_warn_call = func_name.endswith("warn") or "warn" in func_name.lower()

            warning_cat = ""
            warn_msg = ""

            # Walk all nodes inside the call to find warning categories or message strings
            for arg_child in ast.walk(child):
                if isinstance(arg_child, ast.Name) and arg_child.id in self.WARNING_CATEGORIES:
                    warning_cat = arg_child.id
                elif isinstance(arg_child, ast.Attribute) and arg_child.attr in self.WARNING_CATEGORIES:
                    warning_cat = arg_child.attr

            # Extract message string from first positional arg or 'message' keyword arg
            if child.args:
                warn_msg = _extract_string_value(child.args[0])
            for kw in child.keywords:
                if kw.arg == "message":
                    warn_msg = _extract_string_value(kw.value)
                elif kw.arg == "category":
                    cat_val = _flatten_ast_attr(kw.value)
                    if any(c in cat_val for c in self.WARNING_CATEGORIES):
                        warning_cat = cat_val

            # Trigger if it's a deprecation category, or warn call with 'deprecat' in msg/call
            is_dep_warning = bool(warning_cat and any(c in warning_cat for c in self.WARNING_CATEGORIES))
            is_dep_message = bool(warn_msg and self.DEPRECATION_PATTERN.search(warn_msg))

            if is_warn_call and (is_dep_warning or is_dep_message):
                category_info = f", category={warning_cat}" if warning_cat else ""
                evidence = f"warnings.warn({warn_msg!r}{category_info})"
                self.candidates.append(
                    DeprecationCandidate(
                        qualified_name=qname,
                        origin="warning",
                        location=f"{self.filename}:{child.lineno}",
                        raw_evidence=evidence,
                        line=child.lineno,
                    )
                )


def detect_legacy_deprecations(
    source_code: str,
    filename: str = "<string>",
    package_prefix: str = "",
) -> List[DeprecationCandidate]:
    """
    Analyzes Python source code and detects deprecated APIs using the three legacy heuristics:
    1. Decorator matching
    2. warnings.warn call matching
    3. Docstring keyword matching

    Returns a list of DeprecationCandidate objects.
    """
    try:
        tree = ast.parse(source_code, filename=filename)
    except SyntaxError:
        return []

    visitor = LegacyHeuristicsVisitor(filename=filename, package_prefix=package_prefix)
    visitor.visit(tree)
    return visitor.candidates


def detect_legacy_deprecations_from_file(
    filepath: Union[str, Path],
    package_prefix: str = "",
) -> List[DeprecationCandidate]:
    """Reads a Python file from disk and detects deprecated APIs using legacy heuristics."""
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    source_code = path.read_text(encoding="utf-8", errors="replace")
    return detect_legacy_deprecations(
        source_code=source_code,
        filename=str(path),
        package_prefix=package_prefix,
    )
