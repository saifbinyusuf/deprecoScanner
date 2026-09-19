"""
src/resolution/jedi_resolver.py - Stage 2: Jedi Type Resolution & Call-Site Matching.

Resolves client call sites (file, line, column) to their fully-qualified target symbols,
and matches them against the Stage 1 historical deprecation catalog.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
import textwrap
from typing import Any, Collection, Dict, List, Optional, Set

import jedi

from src.detectors.union_dedup import _is_symbol_compatible

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BENCHMARK_LIBS_DIR = REPO_ROOT / "data" / "benchmark_libs"

COVERING_SNAPSHOTS = [
    "numpy-1.26.4",
    "pandas-1.5.3",
    "pandas-2.2.3",
    "pandas-0.22.0",
    "scipy-1.12.0",
    "scipy-1.7.3",
    "scipy-1.2.3",
    "scipy-0.19.1",
]


@dataclass(frozen=True)
class ResolvedCallSite:
    """Represents a resolved call-site symbol from client code."""

    qualified_name: str
    name: str
    module_name: Optional[str]
    type: str
    line: Optional[int]
    column: Optional[int]
    file_path: Optional[str]
    description: str
    matched_catalog_symbol: Optional[str] = None
    is_deprecated: bool = False
    call_site_snippet: Optional[str] = None
    client_line: Optional[int] = None


@dataclass(frozen=True)
class LowConfidenceCandidate:
    """
    Represents a candidate call site that could not be confidently resolved by Jedi.
    Preserved for recall accounting rather than silently dropped.
    """

    call_site_snippet: str
    callee_name: str
    line: int
    column: int
    failure_reason: str  # e.g., "empty_goto", "unresolved_receiver", "dynamic_dispatch", "ambiguous_definitions", "syntax_error", "jedi_exception"
    file_path: Optional[str] = None
    matched_catalog_symbol: Optional[str] = None
    details: Optional[str] = None
    sample_id: Optional[str] = None


@dataclass(frozen=True)
class Stage2Result:
    """Container for Stage 2 call-site resolution across a client unit of code."""

    resolved_deprecated: List[ResolvedCallSite] = field(default_factory=list)
    resolved_benign: List[ResolvedCallSite] = field(default_factory=list)
    low_confidence: List[LowConfidenceCandidate] = field(default_factory=list)

    @property
    def total_candidates(self) -> int:
        return len(self.resolved_deprecated) + len(self.low_confidence)



def get_clean_sys_path(extra_paths: Optional[List[str]] = None) -> List[str]:
    """
    Constructs a sys_path prioritizing historical libraries/stubs while filtering out
    the current virtual environment's site-packages to prevent modern versions from shadowing.
    """
    default_sys_path = jedi.get_default_environment().get_sys_path()
    clean_stdlib = [p for p in default_sys_path if "site-packages" not in p and ".venv" not in p]
    paths: List[str] = []
    if extra_paths:
        paths.extend(extra_paths)
    paths.extend(clean_stdlib)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def build_snapshot_project(snapshot_name: str, base_dir: Optional[Path] = None) -> jedi.Project:
    """Builds a jedi.Project for a specific historical snapshot with stubs."""
    base = (base_dir or DEFAULT_BENCHMARK_LIBS_DIR).resolve()
    stubs_dir = base / "stubs"
    snap_dir = base / snapshot_name

    extra = []
    if stubs_dir.exists():
        extra.append(str(stubs_dir))
    if snap_dir.exists():
        extra.append(str(snap_dir))

    sys_path = get_clean_sys_path(extra)
    return jedi.Project(path=str(REPO_ROOT), sys_path=sys_path)


def build_historical_project_pool(base_dir: Optional[Path] = None) -> List[jedi.Project]:
    """
    Builds a pool of jedi.Projects covering each historical library era.
    Using separate projects prevents cross-version package shadowing on sys_path.
    """
    base = (base_dir or DEFAULT_BENCHMARK_LIBS_DIR).resolve()
    pool = []
    for snap in COVERING_SNAPSHOTS:
        if (base / snap).exists():
            pool.append(build_snapshot_project(snap, base))
    return pool


def build_historical_project(base_dir: Optional[Path] = None) -> jedi.Project:
    """Builds a single primary project from the historical pool (convenience wrapper)."""
    pool = build_historical_project_pool(base_dir)
    return pool[0] if pool else jedi.Project(path=str(REPO_ROOT))


def match_symbol_against_catalog(
    resolved_symbol: str, catalog_symbols: Collection[str]
) -> Optional[str]:
    """
    Matches a resolved symbol against candidate catalog symbols using symbol compatibility rules.
    Returns the matching catalog symbol, or None if no match.
    """
    if resolved_symbol in catalog_symbols:
        return resolved_symbol

    for cat_sym in catalog_symbols:
        if _is_symbol_compatible(cat_sym, resolved_symbol) or _is_symbol_compatible(resolved_symbol, cat_sym):
            return cat_sym

    return None


def normalize_snippet_indentation(code: str) -> str:
    """
    Normalizes indentation for isolated code snippets extracted from classes or functions.
    Handles mixed tabs and spaces, and leading space on line 1 that prevents standard
    textwrap.dedent() from finding a common indentation prefix.
    """
    # 1. Standard dedent
    candidate = textwrap.dedent(code)
    try:
        ast.parse(candidate)
        return candidate
    except SyntaxError:
        pass

    # 2. Expand tabs (4 spaces) then dedent
    candidate = textwrap.dedent(code.expandtabs(4))
    try:
        ast.parse(candidate)
        return candidate
    except SyntaxError:
        pass

    # 3. Expand tabs (8 spaces) then dedent
    candidate = textwrap.dedent(code.expandtabs(8))
    try:
        ast.parse(candidate)
        return candidate
    except SyntaxError:
        pass

    # 4. If line 1 has stray leading whitespace (common when extracting ' def func():'),
    # strip line 1 leading space and expandtabs+dedent
    lines = code.splitlines()
    if lines and lines[0].startswith(" "):
        candidate = textwrap.dedent(("\n".join([lines[0].lstrip()] + lines[1:])).expandtabs(4))
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            pass

    return textwrap.dedent(code)


class JediResolver:
    """
    Stage 2 Call-Site Resolver using Jedi.
    Resolves client code call sites to definitions and cross-references Stage 1 catalogs.
    Supports a project pool to seamlessly resolve across non-overlapping historical library eras.
    """

    def __init__(
        self,
        project: Optional[jedi.Project] = None,
        projects: Optional[List[jedi.Project]] = None,
        catalog_symbols: Optional[Collection[str]] = None,
        base_dir: Optional[Path] = None,
    ):
        if projects:
            self.projects = projects
        elif project:
            self.projects = [project]
        else:
            self.projects = build_historical_project_pool(base_dir)

        self.catalog_symbols: Set[str] = set(catalog_symbols or [])

        # Index projects by library for fast targeted lookup
        self.projects_by_lib: Dict[str, List[jedi.Project]] = {"numpy": [], "pandas": [], "scipy": []}
        for p in self.projects:
            p_str = " ".join(str(x) for x in (getattr(p, "_sys_path", []) or [])) + " " + str(getattr(p, "path", ""))
            for lib in ["numpy", "pandas", "scipy"]:
                if lib in p_str.lower():
                    self.projects_by_lib[lib].append(p)

    def set_catalog_symbols(self, symbols: Collection[str]) -> None:
        self.catalog_symbols = set(symbols)

    def resolve(
        self,
        code: str,
        line: int,
        column: int,
        path: Optional[str] = None,
        library_hint: Optional[str] = None,
    ) -> Optional[ResolvedCallSite]:
        """
        Resolves a call site at (line, column) in the provided code snippet.
        Line is 1-indexed, column is 0-indexed (matching Jedi convention).
        """
        target_projects = self.projects
        if library_hint and library_hint.lower() in self.projects_by_lib and self.projects_by_lib[library_hint.lower()]:
            target_projects = self.projects_by_lib[library_hint.lower()]

        best_unmatched: Optional[ResolvedCallSite] = None

        for proj in target_projects:
            defs = []
            try:
                script = jedi.Script(code, project=proj, path=path)
                defs = script.goto(line, column)
            except Exception as e:
                logger.debug(f"Jedi goto failed on project {proj}: {e}")

            if not defs:
                try:
                    names = script.infer(line, column)
                    if names:
                        defs = names
                except Exception:
                    pass

            if not defs:
                continue

            d = defs[0]
            full_name = getattr(d, "full_name", "") or ""
            name = getattr(d, "name", "") or ""
            module_name = getattr(d, "module_name", None)
            dtype = getattr(d, "type", "unknown") or "unknown"
            def_line = getattr(d, "line", None)
            def_col = getattr(d, "column", None)
            file_path = str(d.module_path) if getattr(d, "module_path", None) else None
            description = getattr(d, "description", "") or ""

            matched_catalog = None
            if self.catalog_symbols and full_name:
                matched_catalog = match_symbol_against_catalog(full_name, self.catalog_symbols)

            site = ResolvedCallSite(
                qualified_name=full_name,
                name=name,
                module_name=module_name,
                type=dtype,
                line=def_line,
                column=def_col,
                file_path=file_path,
                description=description,
                matched_catalog_symbol=matched_catalog,
                is_deprecated=matched_catalog is not None,
            )

            # If this project matched a deprecated symbol in the catalog, return immediately
            if site.is_deprecated:
                return site

            if best_unmatched is None:
                best_unmatched = site

        return best_unmatched

    def resolve_with_diagnostics(
        self,
        code: str,
        line: int,
        column: int,
        receiver_name: Optional[str] = None,
        path: Optional[str] = None,
        library_hint: Optional[str] = None,
    ) -> tuple[Optional[ResolvedCallSite], Optional[str], Optional[str]]:
        """
        Attempts to resolve a call site. If resolution succeeds, returns (site, None, None).
        If resolution fails or is ambiguous, returns (None, failure_reason, details).
        """
        site = self.resolve(code, line, column, path=path, library_hint=library_hint)
        if site is not None:
            return site, None, None

        target_projects = self.projects
        if library_hint and library_hint.lower() in self.projects_by_lib and self.projects_by_lib[library_hint.lower()]:
            target_projects = self.projects_by_lib[library_hint.lower()]

        # Diagnose failure reason
        for proj in target_projects:
            try:
                script = jedi.Script(code, project=proj, path=path)
                defs = script.goto(line, column)
                if not defs:
                    defs = script.infer(line, column)
                if len(defs) > 1:
                    mods = {getattr(d, "module_name", None) for d in defs if getattr(d, "module_name", None)}
                    if len(mods) > 1:
                        return None, "ambiguous_definitions", f"Multiple definitions across modules: {mods}"
            except Exception as e:
                return None, "jedi_exception", str(e)

        if receiver_name:
            return None, "unresolved_receiver", f"Receiver '{receiver_name}' could not be statically resolved"

        return None, "empty_goto", "Jedi script.goto returned no definitions"

    def analyze_client_snippet(
        self,
        code: str,
        file_path: Optional[str] = None,
        sample_id: Optional[str] = None,
        preamble: Optional[str] = None,
        library_hint: Optional[str] = None,
    ) -> Stage2Result:
        """
        Analyzes a client code snippet or file for deprecated API call sites.
        Extracts call sites, attempts Jedi resolution, and bins unresolved candidates
        into the low_confidence bucket with explicit failure reasons.
        """
        dedented_code = normalize_snippet_indentation(code)
        lines = dedented_code.splitlines()

        try:
            tree = ast.parse(dedented_code)
        except SyntaxError as e:
            return Stage2Result(
                resolved_deprecated=[],
                resolved_benign=[],
                low_confidence=[
                    LowConfidenceCandidate(
                        call_site_snippet=lines[0].strip() if lines else code[:80],
                        callee_name="<syntax_error>",
                        line=getattr(e, "lineno", 1) or 1,
                        column=getattr(e, "offset", 0) or 0,
                        failure_reason="syntax_error",
                        file_path=file_path,
                        details=str(e),
                        sample_id=sample_id,
                    )
                ],
            )

        catalog_callee_names = {sym.split(".")[-1] for sym in self.catalog_symbols} if self.catalog_symbols else set()

        resolved_deprecated: List[ResolvedCallSite] = []
        resolved_benign: List[ResolvedCallSite] = []
        low_confidence: List[LowConfidenceCandidate] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            callee_name = None
            line = getattr(node.func, "lineno", getattr(node, "lineno", 1))
            col = getattr(node.func, "col_offset", getattr(node, "col_offset", 0))
            receiver_name = None

            if isinstance(node.func, ast.Name):
                callee_name = node.func.id
                col = node.func.col_offset
            elif isinstance(node.func, ast.Attribute):
                callee_name = node.func.attr
                receiver_name = getattr(node.func.value, "id", getattr(node.func.value, "attr", None))
                line_text = lines[line - 1] if 0 <= line - 1 < len(lines) else ""
                end_col = getattr(node.func, "end_col_offset", len(line_text))
                found_col = line_text.rfind(callee_name, 0, end_col)
                col = found_col if found_col != -1 else node.func.col_offset
            elif isinstance(node.func, ast.Call) and getattr(node.func.func, "id", "") == "getattr":
                line_text = lines[line - 1] if 0 <= line - 1 < len(lines) else ""
                dynamic_name = "<dynamic>"
                if len(node.func.args) >= 2 and isinstance(node.func.args[1], ast.Constant):
                    dynamic_name = str(node.func.args[1].value)

                matched_sym = None
                for sym in self.catalog_symbols:
                    if sym == dynamic_name or sym.endswith("." + dynamic_name):
                        matched_sym = sym
                        break

                if not self.catalog_symbols or dynamic_name in catalog_callee_names or matched_sym:
                    low_confidence.append(
                        LowConfidenceCandidate(
                            call_site_snippet=line_text.strip(),
                            callee_name=dynamic_name,
                            line=line,
                            column=col,
                            failure_reason="dynamic_dispatch",
                            file_path=file_path,
                            matched_catalog_symbol=matched_sym,
                            details="Dynamic invocation via getattr",
                            sample_id=sample_id,
                        )
                    )
                continue
            else:
                continue

            if self.catalog_symbols and callee_name not in catalog_callee_names:
                continue

            # Candidate match in catalog (deterministic sort)
            candidate_matches = [
                sym for sym in sorted(self.catalog_symbols)
                if sym == callee_name or sym.endswith("." + callee_name)
            ]
            matched_catalog_cand = None
            if candidate_matches:
                if receiver_name and "df" in receiver_name.lower():
                    df_cands = [s for s in candidate_matches if "dataframe" in s.lower()]
                    matched_catalog_cand = df_cands[0] if df_cands else candidate_matches[0]
                elif receiver_name and "series" in receiver_name.lower():
                    s_cands = [s for s in candidate_matches if "series" in s.lower()]
                    matched_catalog_cand = s_cands[0] if s_cands else candidate_matches[0]
                else:
                    matched_catalog_cand = candidate_matches[0]


            line_text = lines[line - 1] if 0 <= line - 1 < len(lines) else ""
            snippet = line_text.strip()

            # Determine library hint
            call_lib_hint = library_hint
            if not call_lib_hint and matched_catalog_cand:
                prefix = matched_catalog_cand.split(".")[0].lower()
                if prefix in ("numpy", "pandas", "scipy"):
                    call_lib_hint = prefix

            site, reason, details = self.resolve_with_diagnostics(
                dedented_code, line, col, receiver_name=receiver_name, path=file_path, library_hint=call_lib_hint
            )

            if site is None and preamble:
                preamble_lines = preamble.strip().splitlines()
                preamble_code = preamble.strip() + "\n" + dedented_code
                p_line = line + len(preamble_lines)
                p_site, p_reason, p_details = self.resolve_with_diagnostics(
                    preamble_code, p_line, col, receiver_name=receiver_name, path=file_path, library_hint=call_lib_hint
                )
                if p_site is not None:
                    site = p_site
                    reason = None
                    details = None
                else:
                    reason = p_reason
                    details = p_details

            if site is not None:
                site = ResolvedCallSite(
                    qualified_name=site.qualified_name,
                    name=site.name,
                    module_name=site.module_name,
                    type=site.type,
                    line=site.line,
                    column=site.column,
                    file_path=site.file_path,
                    description=site.description,
                    matched_catalog_symbol=site.matched_catalog_symbol,
                    is_deprecated=site.is_deprecated,
                    call_site_snippet=snippet,
                    client_line=line,
                )
                if site.is_deprecated:
                    resolved_deprecated.append(site)
                else:
                    resolved_benign.append(site)
            else:
                low_confidence.append(
                    LowConfidenceCandidate(
                        call_site_snippet=snippet,
                        callee_name=callee_name,
                        line=line,
                        column=col,
                        failure_reason=reason or "empty_goto",
                        file_path=file_path,
                        matched_catalog_symbol=matched_catalog_cand,
                        details=details,
                        sample_id=sample_id,
                    )
                )

        return Stage2Result(
            resolved_deprecated=resolved_deprecated,
            resolved_benign=resolved_benign,
            low_confidence=low_confidence,
        )


def resolve_call_site(
    code: str,
    line: int,
    column: int,
    project: Optional[jedi.Project] = None,
    catalog_symbols: Optional[Collection[str]] = None,
) -> Optional[ResolvedCallSite]:
    """Functional convenience wrapper for resolving a call site."""
    resolver = JediResolver(project=project, catalog_symbols=catalog_symbols)
    return resolver.resolve(code, line, column)

