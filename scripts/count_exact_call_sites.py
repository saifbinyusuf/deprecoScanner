#!/usr/bin/env python3
"""
scripts/count_exact_call_sites.py - Fast Exact Benchmark Candidate Call-Site Census.

Computes the exact integer count of:
1. Resolved deprecated call sites across the benchmark dataset.
2. Low-confidence call sites matching canonical benchmark target APIs.
3. Total candidate call sites to be verified in Stage 3 (Task 4.4).
Uses ProcessPoolExecutor (8 workers) for fast parallel processing.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.detectors.union_dedup import _is_symbol_compatible
from src.resolution.jedi_resolver import JediResolver
from scripts.run_stage2_pilot import (
    BENCHMARK_TARGETS,
    build_reconstructed_preamble,
    load_stage1_catalog,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("count_exact_call_sites")

TARGET_SHORT_NAMES: Set[str] = {t.split(".")[-1] for t in BENCHMARK_TARGETS}

_worker_resolver: Optional[JediResolver] = None


def _init_worker(catalog_symbols: List[str]):
    global _worker_resolver
    _worker_resolver = JediResolver(catalog_symbols=catalog_symbols)


def is_target_candidate(callee_name: str, matched_sym: str | None) -> bool:
    """Checks if a low-confidence call site is a candidate for a benchmark target API."""
    if callee_name in TARGET_SHORT_NAMES:
        return True
    if matched_sym:
        for t in BENCHMARK_TARGETS:
            if _is_symbol_compatible(t, matched_sym):
                return True
    return False


def _process_sample(args: tuple[str, str, Dict[str, Any]]) -> Dict[str, Any]:
    global _worker_resolver
    if _worker_resolver is None:
        raise RuntimeError("Worker resolver not initialized")

    lib, sample_id, sample = args
    code = sample.get("function", "")
    preamble = build_reconstructed_preamble(sample)

    res = _worker_resolver.analyze_client_snippet(
        code=code,
        sample_id=sample_id,
        preamble=preamble,
        library_hint=lib,
    )

    resolved_dep_count = len(res.resolved_deprecated)
    target_lc_count = 0
    all_lc_count = len(res.low_confidence)

    for lc in res.low_confidence:
        if is_target_candidate(lc.callee_name, lc.matched_catalog_symbol):
            target_lc_count += 1

    return {
        "library": lib,
        "sample_id": sample_id,
        "resolved_deprecated_calls": resolved_dep_count,
        "target_low_confidence_calls": target_lc_count,
        "all_low_confidence_calls": all_lc_count,
    }


def main():
    catalog = load_stage1_catalog()
    catalog_list = sorted(list(catalog))

    raw_dir = REPO_ROOT / "data" / "raw" / "llm-dep-api" / "probing-inputs"
    tasks = []
    total_samples = 0

    libraries = ["numpy", "scipy", "pandas"]
    for lib in libraries:
        lib_path = raw_dir / lib / "samples.json"
        with open(lib_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        total_samples += len(samples)
        for idx, s in enumerate(samples):
            sample_id = f"{lib}_{idx}"
            tasks.append((lib, sample_id, s))

    logger.info(f"Starting parallel exact call-site census on {total_samples} samples (8 workers)...")
    t0 = time.time()

    per_lib_stats = {
        lib: {
            "samples": 0,
            "resolved_deprecated_calls": 0,
            "target_low_confidence_calls": 0,
            "total_candidate_calls": 0,
            "all_low_confidence_calls": 0,
        }
        for lib in libraries
    }

    processed = 0
    with ProcessPoolExecutor(max_workers=8, initializer=_init_worker, initargs=(catalog_list,)) as executor:
        futures = [executor.submit(_process_sample, t) for t in tasks]
        for future in as_completed(futures):
            res = future.result()
            lib = res["library"]
            per_lib_stats[lib]["samples"] += 1
            per_lib_stats[lib]["resolved_deprecated_calls"] += res["resolved_deprecated_calls"]
            per_lib_stats[lib]["target_low_confidence_calls"] += res["target_low_confidence_calls"]
            per_lib_stats[lib]["total_candidate_calls"] += (
                res["resolved_deprecated_calls"] + res["target_low_confidence_calls"]
            )
            per_lib_stats[lib]["all_low_confidence_calls"] += res["all_low_confidence_calls"]

            processed += 1
            if processed % 1000 == 0 or processed == total_samples:
                logger.info(f"Census progress: {processed}/{total_samples} samples ({time.time() - t0:.1f}s)...")

    elapsed = time.time() - t0
    logger.info(f"Completed exact census in {elapsed:.1f}s.")

    total_resolved = sum(st["resolved_deprecated_calls"] for st in per_lib_stats.values())
    total_target_lc = sum(st["target_low_confidence_calls"] for st in per_lib_stats.values())
    total_all_lc = sum(st["all_low_confidence_calls"] for st in per_lib_stats.values())
    grand_total = total_resolved + total_target_lc

    output = {
        "benchmark_samples_total": total_samples,
        "elapsed_seconds": round(elapsed, 2),
        "resolved_deprecated_calls": total_resolved,
        "target_low_confidence_calls": total_target_lc,
        "exact_candidate_call_sites_total": grand_total,
        "all_low_confidence_calls": total_all_lc,
        "per_library": per_lib_stats,
    }

    print("\n" + "=" * 80)
    print("EXACT BENCHMARK CANDIDATE CALL-SITE CENSUS RESULT:")
    print("=" * 80)
    print(json.dumps(output, indent=2))

    out_file = REPO_ROOT / "results" / "stage3_exact_call_site_census.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Census written to {out_file}")


if __name__ == "__main__":
    main()
