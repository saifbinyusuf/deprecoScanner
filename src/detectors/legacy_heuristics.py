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
from typing import List, Optional, Set, Tuple, Union


@dataclass
class DeprecationCandidate:
    """Represents a deprecation candidate detected by static heuristics or checkers."""

    qualified_name: str
    origin: str  # e.g., "decorator", "warning", "docstring", "pep702", "comment"
    location: str  # e.g., "module.py:42"
    raw_evidence: str
    line: int = 0
    origins: Set[str] = field(default_factory=set)
    scope: str = "function"  # "function" or "parameter"
    param_name: Optional[str] = None
    function_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.origins and self.origin:
            self.origins = {self.origin}
        if not self.function_name:
            if self.scope == "parameter" and "::" in self.qualified_name:
                self.function_name = self.qualified_name.split("::")[0]
            else:
                self.function_name = self.qualified_name

    @property
    def scoped_key(self) -> Tuple[str, Optional[str]]:
        """Returns (function_name, param_name) tuple for call-site resolution."""
        return (self.function_name or self.qualified_name, self.param_name)


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
    PARAM_DECORATOR_PATTERN = re.compile(r"(?i)(deprecat.*(kwarg|param|arg)|(kwarg|param|arg).*deprecat)")
    STD_DEPRECATION_WARNINGS = {"DeprecationWarning", "PendingDeprecationWarning", "FutureWarning"}
    WARNING_CATEGORY_PATTERN = re.compile(r"(?i)(deprecat|future|pendingdeprecat)")

    COMMON_WARNING_MODULES = (
        "pandas.errors",
        "pandas",
        "scipy",
        "scipy._lib.deprecation",
        "numpy.exceptions",
        "numpy",
    )

    @classmethod
    def is_deprecation_warning_category(cls, cat_name: str, local_warning_classes: Optional[Set[str]] = None) -> bool:
        """
        Universally checks if a warning category is or subclasses a deprecation warning:
        1. Exact match against standard library deprecation warnings.
        2. Local AST-defined warning class inheriting directly/transitively from DeprecationWarning/FutureWarning.
        3. Dynamic runtime introspection: issubclass(cls, (DeprecationWarning, FutureWarning)) for imported symbols.
        4. Name-based lexical fallback for unimported external classes: matches 'deprecat' or 'future'.
        """
        if not cat_name:
            return False

        # 1. Standard library deprecation warnings
        if cat_name in cls.STD_DEPRECATION_WARNINGS:
            return True

        # 2. Local AST-defined warning subclasses
        if local_warning_classes and cat_name in local_warning_classes:
            return True

        # 3. Dynamic runtime subclass check if symbol is in sys.modules, common libraries, or builtins
        try:
            import sys
            import importlib

            # Check already loaded modules
            for mod in list(sys.modules.values()):
                if mod and hasattr(mod, cat_name):
                    obj = getattr(mod, cat_name)
                    if isinstance(obj, type) and issubclass(obj, (DeprecationWarning, FutureWarning)):
                        return True

            # Check known library exception modules if available
            for mod_name in cls.COMMON_WARNING_MODULES:
                try:
                    mod = importlib.import_module(mod_name)
                    if hasattr(mod, cat_name):
                        obj = getattr(mod, cat_name)
                        if isinstance(obj, type) and issubclass(obj, (DeprecationWarning, FutureWarning)):
                            return True
                except Exception:
                    pass
        except Exception:
            pass

        # 4. Lexical pattern fallback (e.g. ScipyDeprecationWarning, PytestDeprecationWarning)
        return bool(cls.WARNING_CATEGORY_PATTERN.search(cat_name))

    def __init__(self, filename: str = "<string>", package_prefix: str = "") -> None:
        self.filename = filename
        self.package_prefix = package_prefix
        self.scope_stack: List[str] = []
        self.candidates: List[DeprecationCandidate] = []
        self.local_warning_classes: Set[str] = set()

    def _current_qualified_name(self, name: str) -> str:
        parts = []
        if self.package_prefix:
            parts.append(self.package_prefix)
        parts.extend(self.scope_stack)
        parts.append(name)
        return ".".join(parts)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qname = self._current_qualified_name(node.name)

        # Track if this class defines a custom deprecation warning subclass
        for base in node.bases:
            base_name = _flatten_ast_attr(base)
            if (
                base_name in self.STD_DEPRECATION_WARNINGS
                or base_name in self.local_warning_classes
                or self.is_deprecation_warning_category(base_name, self.local_warning_classes)
            ):
                self.local_warning_classes.add(node.name)
                break

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
        param_names: Set[str] = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for a in getattr(node.args, "posonlyargs", []) + getattr(node.args, "args", []) + getattr(node.args, "kwonlyargs", []):
                param_names.add(a.arg)
            if node.args.vararg:
                param_names.add(node.args.vararg.arg)
            if node.args.kwarg:
                param_names.add(node.args.kwarg.arg)

        for dec in node.decorator_list:
            dec_id = ""
            args_repr = ""
            dec_param: Optional[str] = None

            if isinstance(dec, ast.Name):
                dec_id = dec.id
            elif isinstance(dec, ast.Attribute):
                dec_id = _flatten_ast_attr(dec)
            elif isinstance(dec, ast.Call):
                dec_id = _flatten_ast_attr(dec.func)
                # First string argument for representation display
                for arg in dec.args:
                    val = _extract_string_value(arg)
                    if val:
                        args_repr = f"({val!r})"
                        break

            if not dec_id:
                continue

            # Scope determination is strictly bound to the decorator's symbol/name (dec_id).
            # Reason messages passed as arguments (e.g. @deprecated("argument x is obsolete"))
            # must NEVER convert a whole-function decorator into a parameter decorator.
            is_param_dec = bool(self.PARAM_DECORATOR_PATTERN.search(dec_id))
            is_func_dec = bool(self.DEPRECATION_PATTERN.search(dec_id))

            if is_param_dec:
                # Inspect arguments/keywords to identify the targeted parameter
                if isinstance(dec, ast.Call):
                    for kw in dec.keywords:
                        if kw.arg in ("old_arg_name", "old_arg", "old_name", "param", "param_name", "arg_name", "name", "kwarg"):
                            val = _extract_string_value(kw.value)
                            if val:
                                dec_param = val
                                args_repr = f"({kw.arg}={val!r})"
                                break
                    if not dec_param:
                        for arg in dec.args:
                            val = _extract_string_value(arg)
                            if val:
                                dec_param = val
                                args_repr = f"({val!r})"
                                break

                evidence = f"@{dec_id}{args_repr}"
                target_param = dec_param or "kwarg"
                evidence += f" (parameter: '{target_param}')"
                self.candidates.append(
                    DeprecationCandidate(
                        qualified_name=f"{qname}::{target_param}",
                        origin="decorator",
                        location=f"{self.filename}:{node.lineno}",
                        raw_evidence=evidence,
                        line=node.lineno,
                        scope="parameter",
                        param_name=target_param,
                        function_name=qname,
                    )
                )
            elif is_func_dec:
                # Whole-function deprecation decorator (e.g. @deprecated, @deprecate).
                # Even if the reason message contains words like 'argument', 'param', or 'kwarg',
                # it stays strictly function-scoped.
                evidence = f"@{dec_id}{args_repr}"
                self.candidates.append(
                    DeprecationCandidate(
                        qualified_name=qname,
                        origin="decorator",
                        location=f"{self.filename}:{node.lineno}",
                        raw_evidence=evidence,
                        line=node.lineno,
                        scope="function",
                        param_name=None,
                        function_name=qname,
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
                    scope="function",
                    param_name=None,
                    function_name=qname,
                )
            )

    def _check_warnings(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], qname: str) -> None:
        param_names: Set[str] = set()
        kwarg_names: Set[str] = set()
        for a in getattr(node.args, "posonlyargs", []) + getattr(node.args, "args", []) + getattr(node.args, "kwonlyargs", []):
            param_names.add(a.arg)
        if node.args.vararg:
            param_names.add(node.args.vararg.arg)
        if node.args.kwarg:
            kwarg_names.add(node.args.kwarg.arg)
            param_names.add(node.args.kwarg.arg)

        warning_visitor = FunctionWarningVisitor(
            param_names=param_names,
            kwarg_names=kwarg_names,
            local_warning_classes=self.local_warning_classes,
        )
        for stmt in node.body:
            warning_visitor.visit(stmt)

        for call_node, warn_msg, warning_cat, conditioned_param, condition_repr in warning_visitor.warning_calls:
            category_info = f", category={warning_cat}" if warning_cat else ""
            if conditioned_param:
                evidence = (
                    f"warnings.warn({warn_msg!r}{category_info}) "
                    f"conditioned on param '{conditioned_param}' (if {condition_repr})"
                )
                self.candidates.append(
                    DeprecationCandidate(
                        qualified_name=f"{qname}::{conditioned_param}",
                        origin="warning",
                        location=f"{self.filename}:{call_node.lineno}",
                        raw_evidence=evidence,
                        line=call_node.lineno,
                        scope="parameter",
                        param_name=conditioned_param,
                        function_name=qname,
                    )
                )
            else:
                evidence = f"warnings.warn({warn_msg!r}{category_info})"
                self.candidates.append(
                    DeprecationCandidate(
                        qualified_name=qname,
                        origin="warning",
                        location=f"{self.filename}:{call_node.lineno}",
                        raw_evidence=evidence,
                        line=call_node.lineno,
                        scope="function",
                        param_name=None,
                        function_name=qname,
                    )
                )


class FunctionWarningVisitor(ast.NodeVisitor):
    """
    Walks a function's body statements while tracking enclosing `ast.If` conditions.
    Determines if a warnings.warn call is conditioned on a parameter (parameter-level)
    or unconditional (function-level). Handles both if-branches and else-branches,
    and differentiates AND vs OR conditions.
    """

    def __init__(
        self,
        param_names: Set[str],
        kwarg_names: Set[str],
        local_warning_classes: Optional[Set[str]] = None,
    ) -> None:
        self.param_names = param_names
        self.kwarg_names = kwarg_names
        self.local_warning_classes = local_warning_classes or set()
        self.if_stack: List[Tuple[ast.expr, bool]] = []  # (cond_expr, is_if_branch)
        self.warning_calls: List[Tuple[ast.Call, str, str, Optional[str], Optional[str]]] = []
        # list of (call_node, warn_msg, warning_cat, conditioned_param, condition_repr)

    def visit_If(self, node: ast.If) -> None:
        self.if_stack.append((node.test, True))
        for stmt in node.body:
            self.visit(stmt)
        self.if_stack.pop()

        if node.orelse:
            self.if_stack.append((node.test, False))
            for stmt in node.orelse:
                self.visit(stmt)
            self.if_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Stop at nested function boundary
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass

    def visit_Call(self, node: ast.Call) -> None:
        func_name = _flatten_ast_attr(node.func)
        is_warn_call = func_name.endswith("warn") or "warn" in func_name.lower()
        if not is_warn_call:
            self.generic_visit(node)
            return

        warning_cat = ""
        warn_msg = ""
        for arg_child in ast.walk(node):
            if isinstance(arg_child, ast.Name) and LegacyHeuristicsVisitor.is_deprecation_warning_category(arg_child.id, self.local_warning_classes):
                warning_cat = arg_child.id
            elif isinstance(arg_child, ast.Attribute) and LegacyHeuristicsVisitor.is_deprecation_warning_category(arg_child.attr, self.local_warning_classes):
                warning_cat = arg_child.attr

        if node.args:
            warn_msg = _extract_string_value(node.args[0])
        for kw in node.keywords:
            if kw.arg == "message":
                warn_msg = _extract_string_value(kw.value)
            elif kw.arg == "category":
                cat_val = _flatten_ast_attr(kw.value)
                if LegacyHeuristicsVisitor.is_deprecation_warning_category(cat_val, self.local_warning_classes):
                    warning_cat = cat_val

        is_dep_warning = bool(warning_cat and LegacyHeuristicsVisitor.is_deprecation_warning_category(warning_cat, self.local_warning_classes))
        is_dep_message = bool(warn_msg and LegacyHeuristicsVisitor.DEPRECATION_PATTERN.search(warn_msg))

        if is_dep_warning or is_dep_message:
            conditioned_params: List[str] = []
            condition_repr: Optional[str] = None

            for cond_expr, is_if_branch in reversed(self.if_stack):
                cond_code = ""
                try:
                    cond_code = ast.unparse(cond_expr)
                except Exception:
                    pass

                branch_repr = cond_code if is_if_branch else f"else: not ({cond_code})"

                # Harvest all parameters referenced in this condition
                params_in_cond: List[str] = []
                for child in ast.walk(cond_expr):
                    if isinstance(child, ast.Name) and child.id in self.param_names:
                        if child.id in self.kwarg_names:
                            # e.g., if "param" in kwargs:
                            for sub in ast.walk(cond_expr):
                                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                                    if sub.value not in params_in_cond:
                                        params_in_cond.append(sub.value)
                        else:
                            if child.id not in params_in_cond:
                                params_in_cond.append(child.id)
                    elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                        if child.value in self.param_names and child.value not in params_in_cond:
                            params_in_cond.append(child.value)

                if params_in_cond:
                    # Differentiate AND vs OR conditions:
                    # - If condition requires multiple parameters simultaneously (AND), emit a joint parameter candidate
                    #   to avoid falsely flagging client calls that only supply one of the parameters.
                    # - If condition triggers on any parameter independently (OR) or has a single parameter,
                    #   emit independent parameter candidates.
                    is_conjunction = False
                    if len(params_in_cond) > 1:
                        if isinstance(cond_expr, ast.BoolOp) and isinstance(cond_expr.op, ast.And):
                            is_conjunction = True
                        elif any(isinstance(n, ast.And) for n in ast.walk(cond_expr)) and not any(isinstance(n, ast.Or) for n in ast.walk(cond_expr)):
                            is_conjunction = True

                    if is_conjunction:
                        joint_param = "+".join(sorted(params_in_cond))
                        conditioned_params = [joint_param]
                    else:
                        conditioned_params = params_in_cond

                    condition_repr = branch_repr
                    break

            if conditioned_params:
                for p in conditioned_params:
                    self.warning_calls.append((node, warn_msg, warning_cat, p, condition_repr))
            else:
                self.warning_calls.append((node, warn_msg, warning_cat, None, None))

        self.generic_visit(node)



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
