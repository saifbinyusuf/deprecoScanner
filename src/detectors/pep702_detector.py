"""
pep702_detector.py - PEP 702 Deprecation Detection.

Extracts PEP 702 deprecation metadata via two complementary paths:
1. Path A (Type Checker Diagnostics): Runs mypy (--show-error-codes --enable-error-code deprecated)
   or pyright (--outputjson) to extract deprecation-specific diagnostics ([deprecated] / reportDeprecated).
2. Path B (Programmatic Declaration Extraction): Uses `griffe` with the
   `griffe-warnings-deprecated` extension to statically extract `@warnings.deprecated` /
   `@typing_extensions.deprecated` metadata (message, version, line).

Both paths normalize into `DeprecationCandidate` objects with `origin="pep702"`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Set, Union

import griffe

from src.detectors.legacy_heuristics import DeprecationCandidate


MYPY_DEPRECATION_PATTERN = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):\s*(?:error|warning|note):\s*"
    r"(?:(?:function|method|class|attribute|variable)\s+)?"
    r"(?P<symbol>[^\s:]+)\s+is deprecated(?::\s*(?P<msg>.*?))?\s*\[deprecated\]\s*$",
    re.IGNORECASE,
)

PYRIGHT_DEPRECATION_PATTERN = re.compile(
    r'^(?:The\s+(?:function|method|class|variable|symbol|property|attribute)\s+)?'
    r'\"(?P<symbol>[^\"]+)\"'
    r'(?:\s+in\s+class\s+\"(?P<class_name>[^\"]+)\")?'
    r'\s+is deprecated(?::|\s*\n)?\s*(?P<msg>.*)?$',
    re.DOTALL | re.IGNORECASE,
)


def _clean_symbol_name(symbol: str, mod_name: str, package_prefix: str) -> str:
    """Normalizes symbol names to match legacy heuristics qualified naming."""
    clean = symbol
    if mod_name and clean.startswith(f"{mod_name}."):
        clean = clean[len(mod_name) + 1 :]
    if package_prefix:
        if not clean.startswith(f"{package_prefix}."):
            clean = f"{package_prefix}.{clean}"
    return clean


def _collect_griffe_members(
    obj: griffe.Object | griffe.Alias,
    filepath: str,
    mod_name: str,
    package_prefix: str,
    results: List[DeprecationCandidate],
    seen: Set[str],
) -> None:
    """Recursively traverses Griffe objects to collect PEP 702 deprecated symbols."""
    if getattr(obj, "is_alias", False):
        return

    msg = getattr(obj, "deprecated", None)
    if msg is not None:
        lineno = getattr(obj, "lineno", 0) or 0
        raw_path = obj.path
        qname = _clean_symbol_name(raw_path, mod_name, package_prefix)
        scope = "class" if getattr(obj, "is_class", False) else "function"

        key = f"{qname}:{lineno}"
        if key not in seen:
            seen.add(key)
            results.append(
                DeprecationCandidate(
                    qualified_name=qname,
                    origin="pep702:griffe",
                    location=f"{filepath}:{lineno}",
                    raw_evidence=f"@warnings.deprecated({msg!r})",
                    line=lineno,
                    origins={"pep702", "pep702:griffe"},
                    scope=scope,
                    param_name=None,
                    function_name=qname,
                    message=msg,
                )
            )

    try:
        members = getattr(obj, "members", {})
    except Exception:
        members = {}

    for member_name, member in members.items():
        if getattr(member, "is_alias", False):
            continue
        _collect_griffe_members(member, filepath, mod_name, package_prefix, results, seen)


def detect_pep702_via_griffe(
    source_or_path: Union[str, Path],
    filename: str = "<string>",
    package_prefix: str = "",
) -> List[DeprecationCandidate]:
    """
    Path B: Statically extracts PEP 702 @warnings.deprecated metadata using griffe
    and griffe-warnings-deprecated.

    Can accept either a file Path or Python source code string.
    """
    if isinstance(source_or_path, Path) or (
        isinstance(source_or_path, str) and "\n" not in source_or_path and Path(source_or_path).is_file()
    ):
        p = Path(source_or_path).resolve()
        code = p.read_text(encoding="utf-8")
        filepath_str = str(p)
        mod_name = package_prefix or p.stem
    else:
        code = str(source_or_path)
        filepath_str = filename
        mod_name = package_prefix or (Path(filename).stem if filename != "<string>" else "module")

    try:
        exts = griffe.load_extensions("griffe_warnings_deprecated")
        mod = griffe.visit(
            module_name=mod_name,
            filepath=Path(filepath_str) if filepath_str != "<string>" else None,
            code=code,
            extensions=exts,
        )
    except Exception:
        return []

    results: List[DeprecationCandidate] = []
    seen: Set[str] = set()
    _collect_griffe_members(mod, filepath_str, mod_name if not package_prefix else "", package_prefix, results, seen)
    return results


def detect_pep702_via_mypy(
    source_or_path: Union[str, Path],
    package_prefix: str = "",
) -> List[DeprecationCandidate]:
    """
    Path A (Mypy): Runs `mypy --show-error-codes --enable-error-code deprecated <filepath>`
    and parses deprecation diagnostics into DeprecationCandidate objects.
    """
    cleanup_temp = False
    if isinstance(source_or_path, Path) or (
        isinstance(source_or_path, str) and "\n" not in source_or_path and Path(source_or_path).is_file()
    ):
        p = Path(source_or_path).resolve()
        mod_name = p.stem
    else:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tf:
            tf.write(str(source_or_path))
            p = Path(tf.name)
        cleanup_temp = True
        mod_name = p.stem

    try:
        cmd = [
            sys.executable,
            "-m",
            "mypy",
            "--show-error-codes",
            "--enable-error-code",
            "deprecated",
            "--check-untyped-defs",
            str(p),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        candidates: List[DeprecationCandidate] = []
        seen: Set[str] = set()

        for line in proc.stdout.splitlines():
            line = line.strip()
            m = MYPY_DEPRECATION_PATTERN.match(line)
            if m:
                d = m.groupdict()
                file_loc = d["file"]
                lineno = int(d["line"])
                raw_symbol = d["symbol"]
                symbol = _clean_symbol_name(raw_symbol, mod_name, package_prefix)
                msg = (d.get("msg") or "").strip()
                evidence = f"mypy [deprecated]: {symbol} is deprecated"
                if msg:
                    evidence += f" ({msg})"

                key = f"{symbol}:{lineno}"
                if key not in seen:
                    seen.add(key)
                    candidates.append(
                        DeprecationCandidate(
                            qualified_name=symbol,
                            origin="pep702:mypy",
                            location=f"{file_loc}:{lineno}",
                            raw_evidence=evidence,
                            line=lineno,
                            origins={"pep702", "pep702:mypy"},
                            scope="function",
                            param_name=None,
                            function_name=symbol,
                            message=msg if msg else None,
                        )
                    )

        return candidates
    finally:
        if cleanup_temp:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def detect_pep702_via_pyright(
    source_or_path: Union[str, Path],
    package_prefix: str = "",
) -> List[DeprecationCandidate]:
    """
    Path A (Pyright): Runs `pyright --outputjson` with reportDeprecated enabled
    and parses deprecation diagnostics into DeprecationCandidate objects.
    """
    cleanup_temp = False
    if isinstance(source_or_path, Path) or (
        isinstance(source_or_path, str) and "\n" not in source_or_path and Path(source_or_path).is_file()
    ):
        p = Path(source_or_path).resolve()
        mod_name = p.stem
    else:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tf:
            tf.write(str(source_or_path))
            p = Path(tf.name)
        cleanup_temp = True
        mod_name = p.stem

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as cf:
        json.dump({"reportDeprecated": "warning"}, cf)
        cfg_path = Path(cf.name)

    try:
        cmd = [
            sys.executable,
            "-m",
            "pyright",
            "--project",
            str(cfg_path),
            "--outputjson",
            str(p),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        try:
            data = json.loads(proc.stdout)
        except Exception:
            return []

        candidates: List[DeprecationCandidate] = []
        seen: Set[str] = set()

        for diag in data.get("generalDiagnostics", []):
            if diag.get("rule") == "reportDeprecated":
                msg_full = diag.get("message", "")
                m = PYRIGHT_DEPRECATION_PATTERN.match(msg_full)
                if m:
                    sym = m.group("symbol")
                    cls_name = m.group("class_name")
                    raw_symbol = f"{cls_name}.{sym}" if cls_name else sym
                    msg_body = m.group("msg") or ""
                else:
                    raw_symbol = "deprecated_symbol"
                    msg_body = msg_full

                symbol = _clean_symbol_name(raw_symbol, mod_name, package_prefix)
                clean_msg = msg_body.replace("\xa0", " ").strip()
                lineno = int(diag.get("range", {}).get("start", {}).get("line", 0)) + 1
                evidence = f"pyright [reportDeprecated]: {clean_msg}" if clean_msg else f"pyright [reportDeprecated]: {raw_symbol} is deprecated"

                key = f"{symbol}:{lineno}"
                if key not in seen:
                    seen.add(key)
                    candidates.append(
                        DeprecationCandidate(
                            qualified_name=symbol,
                            origin="pep702:pyright",
                            location=f"{p}:{lineno}",
                            raw_evidence=evidence,
                            line=lineno,
                            origins={"pep702", "pep702:pyright"},
                            scope="function",
                            param_name=None,
                            function_name=symbol,
                            message=clean_msg if clean_msg else None,
                        )
                    )

        return candidates
    finally:
        try:
            cfg_path.unlink(missing_ok=True)
        except Exception:
            pass
        if cleanup_temp:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def detect_pep702_via_typechecker(
    source_or_path: Union[str, Path],
    package_prefix: str = "",
    checker: str = "mypy",
) -> List[DeprecationCandidate]:
    """Runs the specified type checker ('mypy' or 'pyright') to extract PEP 702 diagnostics."""
    if checker == "pyright":
        return detect_pep702_via_pyright(source_or_path, package_prefix=package_prefix)
    return detect_pep702_via_mypy(source_or_path, package_prefix=package_prefix)


def detect_pep702_deprecations(
    source_or_path: Union[str, Path],
    package_prefix: str = "",
    include_typechecker: bool = True,
    checker: str = "mypy",
) -> List[DeprecationCandidate]:
    """
    Unified PEP 702 detector running both Griffe (declarations) and type checker (diagnostics).
    """
    candidates = detect_pep702_via_griffe(source_or_path, package_prefix=package_prefix)
    if include_typechecker:
        checker_cands = detect_pep702_via_typechecker(
            source_or_path, package_prefix=package_prefix, checker=checker
        )
        candidates.extend(checker_cands)
    return candidates
