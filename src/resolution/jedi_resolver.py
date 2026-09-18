"""
src/resolution/jedi_resolver.py - Stage 2: Jedi Type Resolution & Call-Site Matching.

Resolves client call sites (file, line, column) to their fully-qualified target symbols,
and matches them against the Stage 1 historical deprecation catalog.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
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

    def set_catalog_symbols(self, symbols: Collection[str]) -> None:
        self.catalog_symbols = set(symbols)

    def resolve(
        self,
        code: str,
        line: int,
        column: int,
        path: Optional[str] = None,
    ) -> Optional[ResolvedCallSite]:
        """
        Resolves a call site at (line, column) in the provided code snippet.
        Line is 1-indexed, column is 0-indexed (matching Jedi convention).
        """
        best_unmatched: Optional[ResolvedCallSite] = None

        for proj in self.projects:
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
