"""
scripts/run_stage1_pipeline.py - Stage 1 End-to-End Execution on NumPy, SciPy, Pandas.

Executes the full Stage 1 detection and deduplication stack across the installed
source trees of NumPy, SciPy, and Pandas in the local Python environment.

Produces:
1. Candidate counts per library, broken down by origin and multi-origin intersections.
2. Cross-referencing against the 32 benchmark target APIs.
3. Coverage gap audit (tracking any unparseable files).
4. Serialized candidate catalogs exported to data/stage1_candidates_<library>.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Set

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detectors.comment_detector import detect_comment_deprecations
from src.detectors.legacy_heuristics import (
    DeprecationCandidate,
    detect_legacy_deprecations,
)
from src.detectors.parameter_detector import detect_parameter_deprecations
from src.detectors.pep702_detector import detect_pep702_via_griffe
from src.detectors.union_dedup import union_stage1_candidates


BENCHMARK_TARGET_APIS = {
    "numpy": [
        "numpy.alltrue",
        "numpy.product",
        "numpy.cumproduct",
    ],
    "scipy": [
        "scipy.misc.logsumexp",
        "scipy.misc.comb",
        "scipy.integrate.cumtrapz",
        "scipy.integrate.simps",
        "scipy.integrate.trapz",
        "scipy.interpolate.interp2d",
        "scipy.linalg.pinv2",
        "scipy.misc.factorial",
        "scipy.stats.itemfreq",
        "scipy.signal.hanning",
        "scipy.special.sph_jn",
        "scipy.stats.betai",
        "scipy.stats.chisqprob",
        "scipy.misc.face",
        "scipy.misc.factorial2",
        "scipy.special.sph_yn",
        "scipy.special.errprint",
        "scipy.stats.rvs_ratio_uniforms",
    ],
    "pandas": [
        "pandas.io.formats.style.Styler.render",
        "pandas.DataFrame.swapaxes",
        "pandas.DataFrame.applymap",
        "pandas.DataFrame.pad",
        "pandas.Series.iteritems",
        "pandas.DataFrame.iteritems",
        "pandas.DataFrame.select",
        "pandas.DataFrame.first",
        "pandas.DataFrame.last",
        "pandas.Series.pad",
    ],
}


def get_library_root(lib_name: str) -> Path:
    """Finds the package root directory for an installed library."""
    mod = __import__(lib_name)
    file_path = getattr(mod, "__file__", None)
    if not file_path:
        raise RuntimeError(f"Cannot find __file__ for {lib_name}")
    p = Path(file_path).resolve()
    return p.parent if p.name == "__init__.py" else p


def compute_package_prefix(file_path: Path, lib_root: Path, lib_name: str) -> str:
    """Computes a dotted Python module name for a file inside a package tree."""
    rel = file_path.relative_to(lib_root)
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts.pop()
    if not parts:
        return lib_name
    return f"{lib_name}.{'.'.join(parts)}"


def scan_library(lib_name: str, lib_root: Path) -> Dict[str, Any]:
    """Scans all Python files in a library root using all Stage 1 detectors."""
    py_files = sorted(list(lib_root.rglob("*.py")))
    print(f"\nScanning {lib_name} ({len(py_files)} files at {lib_root})...")

    raw_candidates: List[DeprecationCandidate] = []
    coverage_gaps: List[Dict[str, str]] = []

    counts_by_detector = {
        "decorator": 0,
        "warning": 0,
        "docstring": 0,
        "comment": 0,
        "parameter": 0,
        "pep702:griffe": 0,
    }

    start_time = time.time()

    for idx, f in enumerate(py_files):
        pkg_prefix = compute_package_prefix(f, lib_root, lib_name)
        try:
            source_code = f.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            coverage_gaps.append({"file": str(f), "error": f"Read error: {e}"})
            continue

        # 1. Legacy heuristics (decorator, warning, docstring)
        try:
            legacy_cands = detect_legacy_deprecations(
                source_code=source_code,
                filename=str(f),
                package_prefix=pkg_prefix,
            )
            raw_candidates.extend(legacy_cands)
            for c in legacy_cands:
                counts_by_detector[c.origin] = counts_by_detector.get(c.origin, 0) + 1
        except Exception as e:
            coverage_gaps.append({"file": str(f), "error": f"legacy_heuristics: {e}"})

        # 2. Comment detector
        try:
            comment_cands = detect_comment_deprecations(
                source_code=source_code,
                filename=str(f),
                package_prefix=pkg_prefix,
            )
            raw_candidates.extend(comment_cands)
            counts_by_detector["comment"] += len(comment_cands)
        except Exception as e:
            coverage_gaps.append({"file": str(f), "error": f"comment_detector: {e}"})

        # 3. Parameter detector
        try:
            param_cands = detect_parameter_deprecations(
                source_code=source_code,
                filename=str(f),
                package_prefix=pkg_prefix,
            )
            raw_candidates.extend(param_cands)
            counts_by_detector["parameter"] += len(param_cands)
        except Exception as e:
            coverage_gaps.append({"file": str(f), "error": f"parameter_detector: {e}"})

        # 4. PEP 702 Griffe detector
        try:
            pep702_cands = detect_pep702_via_griffe(
                source_or_path=source_code,
                filename=str(f),
                package_prefix=pkg_prefix,
            )
            raw_candidates.extend(pep702_cands)
            counts_by_detector["pep702:griffe"] += len(pep702_cands)
        except Exception as e:
            coverage_gaps.append({"file": str(f), "error": f"pep702_detector: {e}"})

    elapsed = time.time() - start_time
    print(f"Scanned {len(py_files)} files in {elapsed:.2f}s.")
    print(f"Total raw candidate hits across all detectors: {len(raw_candidates)}")

    # Deduplicate into unique Stage 1 catalog
    dedup_start = time.time()
    surviving_candidates = union_stage1_candidates(raw_candidates)
    dedup_elapsed = time.time() - dedup_start
    print(f"Deduplicated to {len(surviving_candidates)} unique API candidates in {dedup_elapsed:.2f}s.")

    # Accounting & Multi-origin analysis
    origin_frequencies: Dict[str, int] = {}
    multi_origin_counts: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for c in surviving_candidates:
        active_origins = {o for o in c.origins if o != "pep702"}  # normalize generic tag
        k = len(active_origins) if len(active_origins) <= 5 else 5
        multi_origin_counts[k] = multi_origin_counts.get(k, 0) + 1

        for o in active_origins:
            origin_frequencies[o] = origin_frequencies.get(o, 0) + 1

    # Check target benchmark APIs
    target_apis = BENCHMARK_TARGET_APIS.get(lib_name, [])
    catalog_by_name: Dict[str, DeprecationCandidate] = {}
    for c in surviving_candidates:
        catalog_by_name[c.qualified_name] = c

    target_results: List[Dict[str, Any]] = []
    for target in target_apis:
        # Match either exact or suffix
        matched = None
        for name, c in catalog_by_name.items():
            if name == target or name.endswith(f".{target}") or target.endswith(f".{name}"):
                matched = c
                break

        if matched:
            target_results.append(
                {
                    "target_api": target,
                    "matched_symbol": matched.qualified_name,
                    "status": "DETECTED",
                    "origins": sorted(list(matched.origins)),
                    "multi_origin": len(matched.origins) > 1,
                    "location": matched.location,
                    "evidence_snippet": matched.raw_evidence[:120],
                }
            )
        else:
            target_results.append(
                {
                    "target_api": target,
                    "matched_symbol": None,
                    "status": "NOT_IN_CURRENT_TREE",
                    "origins": [],
                    "multi_origin": False,
                    "location": None,
                    "evidence_snippet": "Removed or refactored in modern library version",
                }
            )

    return {
        "library": lib_name,
        "files_scanned": len(py_files),
        "raw_detector_counts": counts_by_detector,
        "total_raw_hits": len(raw_candidates),
        "surviving_unique_candidates": len(surviving_candidates),
        "origin_frequencies_in_catalog": origin_frequencies,
        "multi_origin_overlap_distribution": multi_origin_counts,
        "coverage_gaps_count": len(coverage_gaps),
        "coverage_gaps": coverage_gaps,
        "benchmark_targets": target_results,
        "candidates": [
            {
                "qualified_name": c.qualified_name,
                "origin": c.origin,
                "origins": sorted(list(c.origins)),
                "location": c.location,
                "scope": c.scope,
                "param_name": c.param_name,
                "message": c.message,
                "raw_evidence": c.raw_evidence,
            }
            for c in surviving_candidates
        ],
    }


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    summary_report: Dict[str, Any] = {}

    for lib in ["numpy", "scipy", "pandas"]:
        try:
            lib_root = get_library_root(lib)
            res = scan_library(lib, lib_root)
            candidates_list = res.pop("candidates")

            # Serialize candidate catalog
            cat_file = data_dir / f"stage1_candidates_{lib}.json"
            cat_file.write_text(json.dumps(candidates_list, indent=2), encoding="utf-8")
            print(f"Saved {len(candidates_list)} candidates to {cat_file}")

            summary_report[lib] = res
        except Exception as e:
            print(f"Error scanning {lib}: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    summary_file = data_dir / "stage1_summary.json"
    summary_file.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")
    print(f"\nSaved Stage 1 summary report to {summary_file}")


if __name__ == "__main__":
    main()
