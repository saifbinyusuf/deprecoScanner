#!/usr/bin/env python3
"""
scripts/run_stage1_historical.py - Stage 1 End-to-End Execution on Historical Benchmarks.

Scans the minimal covering set of historical source trees across NumPy, SciPy, and Pandas:
- NumPy: numpy-1.26.4
- Pandas: pandas-0.22.0, pandas-1.5.3, pandas-2.2.3
- SciPy: scipy-0.19.1, scipy-1.2.3, scipy-1.7.3, scipy-1.12.0

Deduplicates candidates per library across snapshots using union_stage1_candidates,
cross-references the 31 benchmark target APIs, and exports candidate catalogs.
"""

from __future__ import annotations

import json
import sys
import time
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
from src.detectors.union_dedup import _is_symbol_compatible, union_stage1_candidates


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

SNAPSHOT_CONFIG = {
    "numpy": ["numpy-1.26.4"],
    "pandas": ["pandas-0.22.0", "pandas-1.5.3", "pandas-2.2.3"],
    "scipy": ["scipy-0.19.1", "scipy-1.2.3", "scipy-1.7.3", "scipy-1.12.0"],
}


def compute_package_prefix(file_path: Path, lib_root: Path, lib_name: str) -> str:
    rel = file_path.relative_to(lib_root)
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts.pop()
    if not parts:
        return lib_name
    return f"{lib_name}.{'.'.join(parts)}"


def scan_tree(pkg_dir: Path, lib_name: str) -> List[DeprecationCandidate]:
    py_files = sorted(list(pkg_dir.rglob("*.py")))
    raw_candidates: List[DeprecationCandidate] = []

    for f in py_files:
        pkg_prefix = compute_package_prefix(f, pkg_dir, lib_name)
        try:
            source_code = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # 1. Legacy heuristics
        try:
            cands = detect_legacy_deprecations(source_code, str(f), pkg_prefix)
            raw_candidates.extend(cands)
        except (SyntaxError, UnicodeDecodeError):
            pass
        except Exception as e:
            print(f"Warning: error in legacy_heuristics on {f}: {e}")

        # 2. Comment detector
        try:
            cands = detect_comment_deprecations(source_code, str(f), pkg_prefix)
            raw_candidates.extend(cands)
        except (SyntaxError, UnicodeDecodeError):
            pass
        except Exception as e:
            print(f"Warning: error in comment_detector on {f}: {e}")

        # 3. Parameter detector
        try:
            cands = detect_parameter_deprecations(source_code, str(f), pkg_prefix)
            raw_candidates.extend(cands)
        except (SyntaxError, UnicodeDecodeError):
            pass
        except Exception as e:
            print(f"Warning: error in parameter_detector on {f}: {e}")

        # 4. PEP 702 Griffe detector
        try:
            cands = detect_pep702_via_griffe(source_code, str(f), pkg_prefix)
            raw_candidates.extend(cands)
        except (SyntaxError, UnicodeDecodeError):
            pass
        except Exception as e:
            print(f"Warning: error in pep702_detector on {f}: {e}")

    return raw_candidates


def scan_library_historical(lib_name: str, base_dir: Path) -> Dict[str, Any]:
    snapshots = SNAPSHOT_CONFIG.get(lib_name, [])
    all_raw_candidates: List[DeprecationCandidate] = []
    total_files_scanned = 0
    snapshot_stats = {}

    start_time = time.time()

    for snap in snapshots:
        pkg_dir = base_dir / snap / lib_name
        if not pkg_dir.exists():
            print(f"Warning: snapshot dir {pkg_dir} does not exist!")
            continue

        py_files = list(pkg_dir.rglob("*.py"))
        total_files_scanned += len(py_files)

        cands = scan_tree(pkg_dir, lib_name)
        snapshot_stats[snap] = {
            "files_scanned": len(py_files),
            "raw_candidates": len(cands),
        }
        all_raw_candidates.extend(cands)

    scan_elapsed = time.time() - start_time
    print(f"\n[{lib_name}] Scanned {total_files_scanned} files across {len(snapshots)} snapshots in {scan_elapsed:.2f}s.")
    print(f"[{lib_name}] Total raw candidates across snapshots: {len(all_raw_candidates)}")

    # Deduplicate across snapshots
    dedup_start = time.time()
    surviving_candidates = union_stage1_candidates(all_raw_candidates)
    dedup_elapsed = time.time() - dedup_start
    print(f"[{lib_name}] Deduplicated to {len(surviving_candidates)} unique candidates in {dedup_elapsed:.2f}s.")

    # Accounting
    origin_frequencies: Dict[str, int] = {}
    multi_origin_counts: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    param_candidates = [c for c in surviving_candidates if c.scope == "parameter"]
    func_candidates = [c for c in surviving_candidates if c.scope == "function"]
    param_origins: Dict[str, int] = {}

    for c in surviving_candidates:
        active_origins = {o for o in c.origins if o != "pep702"}
        k = len(active_origins) if len(active_origins) <= 5 else 5
        multi_origin_counts[k] = multi_origin_counts.get(k, 0) + 1
        for o in active_origins:
            origin_frequencies[o] = origin_frequencies.get(o, 0) + 1

    for c in param_candidates:
        for o in c.origins:
            if o != "pep702":
                param_origins[o] = param_origins.get(o, 0) + 1

    origin_frequencies["parameter"] = len(param_candidates)

    # Cross-reference benchmark target APIs
    targets = BENCHMARK_TARGET_APIS.get(lib_name, [])
    catalog_by_name = {c.qualified_name: c for c in surviving_candidates}

    target_results: List[Dict[str, Any]] = []
    detected_count = 0

    for target in targets:
        matched = None
        # Try exact or symbol compatibility
        for name, c in catalog_by_name.items():
            if _is_symbol_compatible(target, name) or _is_symbol_compatible(name, target):
                matched = c
                break

        if matched:
            detected_count += 1
            target_results.append(
                {
                    "target_api": target,
                    "matched_symbol": matched.qualified_name,
                    "status": "DETECTED",
                    "origins": sorted(list(matched.origins)),
                    "multi_origin": len(matched.origins) > 1,
                    "location": matched.location,
                    "evidence_snippet": matched.raw_evidence[:150] if matched.raw_evidence else "",
                }
            )
        else:
            target_results.append(
                {
                    "target_api": target,
                    "matched_symbol": None,
                    "status": "NOT_FOUND",
                    "origins": [],
                    "multi_origin": False,
                    "location": None,
                    "evidence_snippet": "Not detected in scanned snapshots",
                }
            )

    print(f"[{lib_name}] Benchmark Target Detection: {detected_count}/{len(targets)} DETECTED")

    return {
        "library": lib_name,
        "snapshots_scanned": snapshots,
        "total_files_scanned": total_files_scanned,
        "snapshot_stats": snapshot_stats,
        "total_raw_hits": len(all_raw_candidates),
        "surviving_unique_candidates": len(surviving_candidates),
        "parameter_scoped_candidates": len(param_candidates),
        "function_scoped_candidates": len(func_candidates),
        "parameter_origins": param_origins,
        "origin_frequencies_in_catalog": origin_frequencies,
        "multi_origin_overlap_distribution": multi_origin_counts,
        "benchmark_targets_detected": detected_count,
        "benchmark_targets_total": len(targets),
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
    base_dir = Path("data/benchmark_libs").resolve()
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    summary_report: Dict[str, Any] = {}
    total_detected = 0
    total_targets = 0

    print("======================================================================")
    print("STAGE 1 HISTORICAL BENCHMARK SCAN & DEDUPLICATION")
    print("======================================================================")

    for lib in ["numpy", "pandas", "scipy"]:
        res = scan_library_historical(lib, base_dir)
        candidates_list = res.pop("candidates")

        cat_file = data_dir / f"stage1_candidates_historical_{lib}.json"
        cat_file.write_text(json.dumps(candidates_list, indent=2), encoding="utf-8")
        print(f"Saved {len(candidates_list)} candidates to {cat_file}")

        total_detected += res["benchmark_targets_detected"]
        total_targets += res["benchmark_targets_total"]
        summary_report[lib] = res

    summary_file = data_dir / "stage1_historical_summary.json"
    summary_report["overall"] = {
        "total_targets": total_targets,
        "total_detected": total_detected,
        "coverage_pct": round((total_detected / total_targets) * 100, 1) if total_targets else 0,
        "total_unique_candidates": sum(summary_report[l]["surviving_unique_candidates"] for l in ["numpy", "pandas", "scipy"]),
        "total_function_scoped_candidates": sum(summary_report[l]["function_scoped_candidates"] for l in ["numpy", "pandas", "scipy"]),
        "total_parameter_scoped_candidates": sum(summary_report[l]["parameter_scoped_candidates"] for l in ["numpy", "pandas", "scipy"]),
    }
    summary_file.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")
    print(f"\n======================================================================")
    print(f"FINAL HISTORICAL VERIFICATION: {total_detected}/{total_targets} ({summary_report['overall']['coverage_pct']}%) TARGET APIS DETECTED")
    print(f"======================================================================")


if __name__ == "__main__":
    main()
