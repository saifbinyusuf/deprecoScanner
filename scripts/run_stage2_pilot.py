#!/usr/bin/env python3
"""
scripts/run_stage2_pilot.py - Stage 2: Jedi Type Resolution on Pilot Benchmark Data.

Runs Stage 2 (Jedi type resolution and candidate matching) across the full
probing-inputs benchmark dataset (NumPy, SciPy, Pandas).

Preserves all unresolved or ambiguous call sites in the low_confidence bucket
with explicit failure reasons (empty_goto, unresolved_receiver, dynamic_dispatch,
syntax_error, etc.) for recall accounting.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.resolution.jedi_resolver import (
    COVERING_SNAPSHOTS,
    DEFAULT_BENCHMARK_LIBS_DIR,
    JediResolver,
    LowConfidenceCandidate,
    ResolvedCallSite,
    Stage2Result,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_stage2_pilot")

BENCHMARK_TARGETS = {
    "numpy.alltrue",
    "numpy.product",
    "numpy.cumproduct",
    "pandas.DataFrame.iteritems",
    "pandas.Series.iteritems",
    "pandas.DataFrame.swapaxes",
    "pandas.io.formats.style.Styler.render",
    "scipy.misc.comb",
    "scipy.misc.logsumexp",
}

DEFAULT_PREAMBLE = """import numpy as np
import scipy
import scipy.misc
import pandas as pd
"""


def load_stage1_catalog() -> Set[str]:
    """Loads all unique qualified names from Stage 1 historical candidate catalogs."""
    catalog: Set[str] = set(BENCHMARK_TARGETS)
    data_dir = REPO_ROOT / "data"
    for lib in ["numpy", "pandas", "scipy"]:
        path = data_dir / f"stage1_candidates_historical_{lib}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                cands = json.load(f)
                for c in cands:
                    qname = c.get("qualified_name")
                    if qname:
                        catalog.add(qname)
    logger.info(f"Loaded Stage 1 catalog: {len(catalog)} unique symbols across all libraries.")
    return catalog


# Worker function for process pool
_worker_resolver: Optional[JediResolver] = None


def _init_worker(catalog_symbols: List[str]):
    global _worker_resolver
    _worker_resolver = JediResolver(catalog_symbols=catalog_symbols)


def _process_sample(args: tuple[str, str, Dict[str, Any], str]) -> Dict[str, Any]:
    global _worker_resolver
    lib, sample_id, sample, preamble = args
    code = sample.get("function", "")
    gt_category = sample.get("category", "unknown")
    gt_dep_api = sample.get("deprecated api", [])

    if _worker_resolver is None:
        raise RuntimeError("Worker resolver not initialized")

    result = _worker_resolver.analyze_client_snippet(
        code=code,
        sample_id=sample_id,
        preamble=preamble,
        library_hint=lib,
    )


    return {
        "library": lib,
        "sample_id": sample_id,
        "gt_category": gt_category,
        "gt_deprecated_api": gt_dep_api,
        "resolved_deprecated": [
            {
                "qualified_name": s.qualified_name,
                "matched_catalog_symbol": s.matched_catalog_symbol,
                "line": s.line,
                "col": s.column,
            }
            for s in result.resolved_deprecated
        ],
        "resolved_benign": [
            {
                "qualified_name": s.qualified_name,
                "line": s.line,
                "col": s.column,
            }
            for s in result.resolved_benign
        ],
        "low_confidence": [asdict(lc) for lc in result.low_confidence],
    }


def run_stage2(
    libraries: List[str],
    max_samples: Optional[int] = None,
    workers: int = 4,
    preamble: str = DEFAULT_PREAMBLE,
) -> Dict[str, Any]:
    catalog = load_stage1_catalog()
    catalog_list = sorted(list(catalog))

    raw_dir = REPO_ROOT / "data" / "raw" / "llm-dep-api" / "probing-inputs"

    tasks = []
    total_samples = 0
    for lib in libraries:
        lib_path = raw_dir / lib / "samples.json"
        if not lib_path.exists():
            logger.warning(f"File not found: {lib_path}")
            continue
        with open(lib_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        if max_samples:
            samples = samples[:max_samples]
        total_samples += len(samples)
        for idx, s in enumerate(samples):
            sample_id = f"{lib}_{idx}"
            tasks.append((lib, sample_id, s, preamble))

    logger.info(f"Starting Stage 2 resolution on {total_samples} samples across {len(libraries)} libraries with {workers} workers...")
    t0 = time.time()

    results_by_lib: Dict[str, List[Dict[str, Any]]] = {lib: [] for lib in libraries}
    processed_count = 0

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(catalog_list,)) as executor:
            futures = [executor.submit(_process_sample, t) for t in tasks]
            for future in as_completed(futures):
                res = future.result()
                results_by_lib[res["library"]].append(res)
                processed_count += 1
                if processed_count % 500 == 0 or processed_count == total_samples:
                    logger.info(f"Processed {processed_count} / {total_samples} samples ({time.time() - t0:.1f}s)...")
    else:
        _init_worker(catalog_list)
        for t in tasks:
            res = _process_sample(t)
            results_by_lib[res["library"]].append(res)
            processed_count += 1
            if processed_count % 500 == 0 or processed_count == total_samples:
                logger.info(f"Processed {processed_count} / {total_samples} samples ({time.time() - t0:.1f}s)...")

    elapsed = time.time() - t0
    logger.info(f"Completed Stage 2 analysis in {elapsed:.2f}s ({elapsed / max(1, total_samples):.4f}s/sample).")

    # Aggregate metrics
    summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": total_samples,
        "elapsed_seconds": round(elapsed, 2),
        "per_library": {},
        "overall_reasons": Counter(),
        "bucket_samples": {},
    }

    sample_pool_for_reasons: Dict[str, List[Dict[str, Any]]] = {}

    for lib in libraries:
        lib_results = results_by_lib[lib]
        dep_count = sum(len(r["resolved_deprecated"]) for r in lib_results)
        benign_count = sum(len(r["resolved_benign"]) for r in lib_results)
        lc_count = sum(len(r["low_confidence"]) for r in lib_results)
        reasons = Counter()

        for r in lib_results:
            for lc in r["low_confidence"]:
                reason = lc.get("failure_reason", "unknown")
                reasons[reason] += 1
                summary["overall_reasons"][reason] += 1
                if reason not in sample_pool_for_reasons:
                    sample_pool_for_reasons[reason] = []
                if len(sample_pool_for_reasons[reason]) < 5:
                    sample_pool_for_reasons[reason].append(lc)

        summary["per_library"][lib] = {
            "samples_analyzed": len(lib_results),
            "resolved_deprecated_calls": dep_count,
            "resolved_benign_calls": benign_count,
            "low_confidence_calls": lc_count,
            "reasons": dict(reasons),
        }

    summary["overall_reasons"] = dict(summary["overall_reasons"])
    summary["bucket_samples"] = sample_pool_for_reasons

    # Save outputs
    out_dir = REPO_ROOT / "data"
    summary_path = out_dir / "stage2_pilot_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved Stage 2 summary to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Stage 2 on pilot benchmark data.")
    parser.add_argument("--libraries", nargs="+", default=["numpy", "scipy", "pandas"], help="Libraries to analyze")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per library (for testing)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    summary = run_stage2(
        libraries=args.libraries,
        max_samples=args.max_samples,
        workers=args.workers,
    )

    print("\n" + "=" * 70)
    print("STAGE 2 PILOT BENCHMARK ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total Samples Analyzed: {summary['total_samples']}")
    print(f"Elapsed Time:           {summary['elapsed_seconds']}s")
    print("-" * 70)
    print(f"{'Library':<12} | {'Samples':<8} | {'Resolved Dep':<14} | {'Resolved Benign':<16} | {'Low Confidence':<14}")
    print("-" * 70)
    for lib, stats in summary["per_library"].items():
        print(f"{lib:<12} | {stats['samples_analyzed']:<8} | {stats['resolved_deprecated_calls']:<14} | {stats['resolved_benign_calls']:<16} | {stats['low_confidence_calls']:<14}")
    print("-" * 70)
    print("LOW CONFIDENCE REASON BREAKDOWN:")
    for reason, count in summary["overall_reasons"].items():
        print(f"  - {reason:<24}: {count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
