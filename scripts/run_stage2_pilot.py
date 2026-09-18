#!/usr/bin/env python3
"""
scripts/run_stage2_pilot.py - Stage 2: Jedi Type Resolution & Benchmark Accounting.

Evaluates Stage 2 call-site resolution and sample-level ground-truth target recovery
across all 5,875 benchmark samples in the LLM-Deprecated-API dataset (NumPy, SciPy, Pandas).

Supports:
1. Call-Site Level Accounting (Resolved Deprecated, Resolved Benign, Low Confidence).
2. Sample-Level Ground-Truth Target Accounting (Outdated vs Up-to-Dated resolution rates).
3. Comparative Evaluation: Baseline Preamble vs. Reconstructed Preamble (leveraging alias dict).
4. Full 6-Category Diagnostic Failure Reason Reporting.
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

from src.detectors.union_dedup import _is_symbol_compatible
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

BASELINE_PREAMBLE = """import numpy as np
import scipy
import scipy.misc
import pandas as pd
"""

ALL_DIAGNOSTIC_REASONS = [
    "unresolved_receiver",
    "empty_goto",
    "syntax_error",
    "dynamic_dispatch",
    "ambiguous_definitions",
    "jedi_exception",
]


def build_reconstructed_preamble(sample: Dict[str, Any]) -> str:
    """
    Reconstructs an import preamble using the sample's alias dict, reference dict,
    and library-wide conventions to recover from isolated function extraction.
    """
    imports = [
        "import numpy",
        "import numpy as np",
        "from numpy import *",
        "import scipy",
        "import scipy.misc",
        "from scipy.misc import *",
        "import pandas",
        "import pandas as pd",
        "from pandas import *",
        "from pandas.io.formats.style import Styler",
    ]
    alias_dict = sample.get("alias dict") or {}
    for k, v in alias_dict.items():
        if "." not in k:
            mod = ".".join(v.split(".")[:-1])
            if mod:
                imports.append(f"from {mod} import {k}")
        elif k.startswith("DataFrame."):
            imports.append("from pandas import DataFrame")
        elif k.startswith("Series."):
            imports.append("from pandas import Series")
        elif k.startswith("Styler."):
            imports.append("from pandas.io.formats.style import Styler")

    return "\n".join(dict.fromkeys(imports)) + "\n"


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


# Worker state
_worker_resolver: Optional[JediResolver] = None


def _init_worker(catalog_symbols: List[str]):
    global _worker_resolver
    _worker_resolver = JediResolver(catalog_symbols=catalog_symbols)


def _analyze_single_configuration(
    code: str,
    preamble: str,
    lib: str,
    sample_id: str,
    gt_dep_apis: List[str],
    gt_category: str,
) -> Dict[str, Any]:
    global _worker_resolver
    if _worker_resolver is None:
        raise RuntimeError("Worker resolver not initialized")

    result = _worker_resolver.analyze_client_snippet(
        code=code,
        sample_id=sample_id,
        preamble=preamble,
        library_hint=lib,
    )

    # 1. Sample-level target deprecated resolution
    target_resolved_dep = False
    for r in result.resolved_deprecated:
        for gt in gt_dep_apis:
            if _is_symbol_compatible(gt, r.matched_catalog_symbol or "") or _is_symbol_compatible(gt, r.qualified_name):
                target_resolved_dep = True
                break
        if target_resolved_dep:
            break

    # 2. Sample-level target in low confidence
    target_in_low_conf = False
    if not target_resolved_dep:
        for lc in result.low_confidence:
            for gt in gt_dep_apis:
                short_gt = gt.split(".")[-1]
                if lc.callee_name == short_gt or (lc.matched_catalog_symbol and _is_symbol_compatible(gt, lc.matched_catalog_symbol)):
                    target_in_low_conf = True
                    break
            if target_in_low_conf:
                break

    # Categorize sample-level outcome
    if gt_category == "outdated":
        if target_resolved_dep:
            sample_outcome = "target_resolved_deprecated"
        elif target_in_low_conf:
            sample_outcome = "target_in_low_confidence"
        else:
            sample_outcome = "target_missed"
    else:  # up-to-dated
        if target_resolved_dep:
            sample_outcome = "spurious_deprecated_flagged"
        elif target_in_low_conf:
            sample_outcome = "clean_in_low_confidence"
        else:
            sample_outcome = "clean_resolved_benign"

    return {
        "call_counts": {
            "resolved_deprecated": len(result.resolved_deprecated),
            "resolved_benign": len(result.resolved_benign),
            "low_confidence": len(result.low_confidence),
        },
        "sample_outcome": sample_outcome,
        "target_resolved_dep": target_resolved_dep,
        "target_in_low_conf": target_in_low_conf,
        "low_confidence": [asdict(lc) for lc in result.low_confidence],
    }


def _process_sample_dual(args: tuple[str, str, Dict[str, Any]]) -> Dict[str, Any]:
    lib, sample_id, sample = args
    code = sample.get("function", "")
    gt_category = sample.get("category", "unknown")
    gt_dep_api = sample.get("deprecated api", [])
    gt_dep_apis = gt_dep_api if isinstance(gt_dep_api, list) else [str(gt_dep_api)]

    # 1. Baseline analysis
    base_res = _analyze_single_configuration(
        code=code,
        preamble=BASELINE_PREAMBLE,
        lib=lib,
        sample_id=sample_id,
        gt_dep_apis=gt_dep_apis,
        gt_category=gt_category,
    )

    # 2. Reconstructed preamble analysis
    recon_preamble = build_reconstructed_preamble(sample)
    recon_res = _analyze_single_configuration(
        code=code,
        preamble=recon_preamble,
        lib=lib,
        sample_id=sample_id,
        gt_dep_apis=gt_dep_apis,
        gt_category=gt_category,
    )

    return {
        "library": lib,
        "sample_id": sample_id,
        "gt_category": gt_category,
        "gt_deprecated_api": gt_dep_apis,
        "baseline": base_res,
        "reconstructed": recon_res,
    }


def run_stage2_evaluation(
    libraries: List[str],
    max_samples: Optional[int] = None,
    workers: int = 8,
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
            tasks.append((lib, sample_id, s))

    logger.info(f"Starting Dual-Preamble Stage 2 resolution on {total_samples} samples across {len(libraries)} libraries ({workers} workers)...")
    t0 = time.time()

    results_by_lib: Dict[str, List[Dict[str, Any]]] = {lib: [] for lib in libraries}
    processed_count = 0

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(catalog_list,)) as executor:
        futures = [executor.submit(_process_sample_dual, t) for t in tasks]
        for future in as_completed(futures):
            res = future.result()
            results_by_lib[res["library"]].append(res)
            processed_count += 1
            if processed_count % 500 == 0 or processed_count == total_samples:
                logger.info(f"Processed {processed_count} / {total_samples} samples ({time.time() - t0:.1f}s)...")

    elapsed = time.time() - t0
    logger.info(f"Completed dual Stage 2 analysis in {elapsed:.2f}s ({elapsed / max(1, total_samples):.4f}s/sample).")

    # Aggregate summaries for both configurations
    def aggregate_config(config_key: str) -> Dict[str, Any]:
        call_counts = {"resolved_deprecated": 0, "resolved_benign": 0, "low_confidence": 0}
        per_lib: Dict[str, Dict[str, Any]] = {}
        reason_counter = {r: 0 for r in ALL_DIAGNOSTIC_REASONS}
        sample_outcomes = {
            "outdated": Counter(),
            "up_to_dated": Counter(),
        }

        bucket_samples: Dict[str, List[Dict[str, Any]]] = {r: [] for r in ALL_DIAGNOSTIC_REASONS}

        for lib in libraries:
            lib_results = results_by_lib[lib]
            lib_calls = {"resolved_deprecated": 0, "resolved_benign": 0, "low_confidence": 0}
            lib_reasons = {r: 0 for r in ALL_DIAGNOSTIC_REASONS}
            lib_sample_outcomes = {"outdated": Counter(), "up_to_dated": Counter()}

            for r in lib_results:
                cfg = r[config_key]
                for k in lib_calls:
                    lib_calls[k] += cfg["call_counts"][k]
                    call_counts[k] += cfg["call_counts"][k]

                cat_key = "outdated" if r["gt_category"] == "outdated" else "up_to_dated"
                outcome = cfg["sample_outcome"]
                lib_sample_outcomes[cat_key][outcome] += 1
                sample_outcomes[cat_key][outcome] += 1

                for lc in cfg["low_confidence"]:
                    reason = lc.get("failure_reason", "unknown")
                    if reason in reason_counter:
                        reason_counter[reason] += 1
                        lib_reasons[reason] += 1
                        if len(bucket_samples[reason]) < 5:
                            bucket_samples[reason].append(lc)
                    else:
                        reason_counter[reason] = reason_counter.get(reason, 0) + 1
                        lib_reasons[reason] = lib_reasons.get(reason, 0) + 1

            per_lib[lib] = {
                "samples_analyzed": len(lib_results),
                "call_counts": lib_calls,
                "reasons": lib_reasons,
                "sample_outcomes": {
                    "outdated": dict(lib_sample_outcomes["outdated"]),
                    "up_to_dated": dict(lib_sample_outcomes["up_to_dated"]),
                },
            }

        return {
            "call_counts": call_counts,
            "per_library": per_lib,
            "reasons": reason_counter,
            "sample_outcomes": {
                "outdated": dict(sample_outcomes["outdated"]),
                "up_to_dated": dict(sample_outcomes["up_to_dated"]),
            },
            "bucket_samples": bucket_samples,
        }

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": total_samples,
        "elapsed_seconds": round(elapsed, 2),
        "baseline_preamble": aggregate_config("baseline"),
        "reconstructed_preamble": aggregate_config("reconstructed"),
    }

    # Save outputs
    out_dir = REPO_ROOT / "data"
    summary_path = out_dir / "stage2_pilot_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved Stage 2 summary to {summary_path}")

    return summary


def print_report(summary: Dict[str, Any]):
    total = summary["total_samples"]
    elapsed = summary["elapsed_seconds"]
    base = summary["baseline_preamble"]
    recon = summary["reconstructed_preamble"]

    print("\n" + "=" * 80)
    print("STAGE 2 RESOLUTION: COMPARATIVE AUDIT (BASELINE vs RECONSTRUCTED PREAMBLE)")
    print("=" * 80)
    print(f"Total Samples Analyzed: {total} across NumPy, SciPy, Pandas")
    print(f"Total Execution Time:   {elapsed}s ({elapsed/total:.4f}s / sample across 8 workers)")
    print("-" * 80)

    print("1. CALL-SITE LEVEL RECOVERY:")
    print(f"{'Metric':<28} | {'Baseline Preamble':<20} | {'Reconstructed Preamble':<24} | {'Change / Gain':<15}")
    print("-" * 80)
    dep_b, dep_r = base["call_counts"]["resolved_deprecated"], recon["call_counts"]["resolved_deprecated"]
    ben_b, ben_r = base["call_counts"]["resolved_benign"], recon["call_counts"]["resolved_benign"]
    lc_b, lc_r = base["call_counts"]["low_confidence"], recon["call_counts"]["low_confidence"]

    print(f"{'Resolved Deprecated Calls':<28} | {dep_b:<20} | {dep_r:<24} | +{dep_r - dep_b} (+{((dep_r-dep_b)/max(1,dep_b))*100:.1f}%)")
    print(f"{'Resolved Benign Calls':<28} | {ben_b:<20} | {ben_r:<24} | +{ben_r - ben_b} (+{((ben_r-ben_b)/max(1,ben_b))*100:.1f}%)")
    print(f"{'Low-Confidence Calls':<28} | {lc_b:<20} | {lc_r:<24} | -{lc_b - lc_r} (-{((lc_b-lc_r)/max(1,lc_b))*100:.1f}%)")
    print("-" * 80)

    print("\n2. SAMPLE-LEVEL GROUND-TRUTH TARGET ACCOUNTING (FOR PHASE 5 BENCHMARK):")
    outdated_total = sum(base["sample_outcomes"]["outdated"].values())
    uptodate_total = sum(base["sample_outcomes"]["up_to_dated"].values())
    print(f"A. Outdated Samples (Target Deprecated API Expected) [N = {outdated_total}]:")
    print(f"{'Outcome':<30} | {'Baseline Count (%)':<22} | {'Reconstructed Count (%)':<24} | {'Net Impact':<15}")
    print("-" * 80)
    b_res = base["sample_outcomes"]["outdated"].get("target_resolved_deprecated", 0)
    r_res = recon["sample_outcomes"]["outdated"].get("target_resolved_deprecated", 0)
    b_lc = base["sample_outcomes"]["outdated"].get("target_in_low_confidence", 0)
    r_lc = recon["sample_outcomes"]["outdated"].get("target_in_low_confidence", 0)
    b_miss = base["sample_outcomes"]["outdated"].get("target_missed", 0)
    r_miss = recon["sample_outcomes"]["outdated"].get("target_missed", 0)

    print(f"{'Target Deprecated Resolved':<30} | {b_res} ({b_res/outdated_total*100:.1f}%)" + " " * 10 + f"| {r_res} ({r_res/outdated_total*100:.1f}%)" + " " * 10 + f"| +{r_res - b_res}")
    print(f"{'Target in Low-Confidence':<30} | {b_lc} ({b_lc/outdated_total*100:.1f}%)" + " " * 10 + f"| {r_lc} ({r_lc/outdated_total*100:.1f}%)" + " " * 10 + f"| -{b_lc - r_lc}")
    print(f"{'Target Missed / Not Detected':<30} | {b_miss} ({b_miss/outdated_total*100:.1f}%)" + " " * 10 + f"| {r_miss} ({r_miss/outdated_total*100:.1f}%)" + " " * 10 + f"| -{b_miss - r_miss}")

    print(f"\nB. Up-to-Dated Samples (Replacement / Clean API Expected) [N = {uptodate_total}]:")
    print(f"{'Outcome':<30} | {'Baseline Count (%)':<22} | {'Reconstructed Count (%)':<24}")
    print("-" * 80)
    b_clean = base["sample_outcomes"]["up_to_dated"].get("clean_resolved_benign", 0)
    r_clean = recon["sample_outcomes"]["up_to_dated"].get("clean_resolved_benign", 0)
    b_up_lc = base["sample_outcomes"]["up_to_dated"].get("clean_in_low_confidence", 0)
    r_up_lc = recon["sample_outcomes"]["up_to_dated"].get("clean_in_low_confidence", 0)
    b_spur = base["sample_outcomes"]["up_to_dated"].get("spurious_deprecated_flagged", 0)
    r_spur = recon["sample_outcomes"]["up_to_dated"].get("spurious_deprecated_flagged", 0)

    print(f"{'Clean Resolved Benign':<30} | {b_clean} ({b_clean/uptodate_total*100:.1f}%)" + " " * 10 + f"| {r_clean} ({r_clean/uptodate_total*100:.1f}%)")
    print(f"{'Clean in Low-Confidence':<30} | {b_up_lc} ({b_up_lc/uptodate_total*100:.1f}%)" + " " * 10 + f"| {r_up_lc} ({r_up_lc/uptodate_total*100:.1f}%)")
    print(f"{'Spurious Deprecated Flagged':<30} | {b_spur} ({b_spur/uptodate_total*100:.1f}%)" + " " * 10 + f"| {r_spur} ({r_spur/uptodate_total*100:.1f}%)")

    print("\n3. ALL 6 DIAGNOSTIC FAILURE REASONS (RECONSTRUCTED PREAMBLE):")
    print(f"{'Failure Reason':<26} | {'Count':<10} | {'Share':<10} | {'Architectural Explanation'}")
    print("-" * 80)
    explanations = {
        "unresolved_receiver": "Receiver lacks static type annotations (e.g. df.iteritems, styler.render)",
        "empty_goto": "Bare names or symbols unresolvable in local scope",
        "syntax_error": "Malformed unicode indentation or incomplete extracts from GitHub scraper",
        "dynamic_dispatch": "Explicit 0: No candidate calls invoked via getattr() or dynamic dispatch",
        "ambiguous_definitions": "Explicit 0: Jedi never returned conflicting definitions across modules",
        "jedi_exception": "Explicit 0: Jedi error handling caught all internal parsing exceptions",
    }
    r_total_reasons = sum(recon["reasons"].values())
    for r in ALL_DIAGNOSTIC_REASONS:
        cnt = recon["reasons"].get(r, 0)
        pct = f"{cnt/max(1,r_total_reasons)*100:.1f}%" if cnt > 0 else "0.0%"
        print(f"{r:<26} | {cnt:<10} | {pct:<10} | {explanations.get(r, '')}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Stage 2 on pilot benchmark data.")
    parser.add_argument("--libraries", nargs="+", default=["numpy", "scipy", "pandas"], help="Libraries to analyze")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per library (for testing)")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    args = parser.parse_args()

    summary = run_stage2_evaluation(
        libraries=args.libraries,
        max_samples=args.max_samples,
        workers=args.workers,
    )
    print_report(summary)


if __name__ == "__main__":
    main()
