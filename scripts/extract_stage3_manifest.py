#!/usr/bin/env python3
"""
scripts/extract_stage3_manifest.py - Extracts the exact 2,932 candidate call sites for Stage 3.

Runs parallel extraction across all 5,875 benchmark samples and writes:
data/stage3_candidates_manifest.json

Asserts:
- Total candidate call sites == 2,932
- Resolved deprecated calls == 2,533
- Target low-confidence calls == 399
- Exact per-library match (NumPy: 1,113; SciPy: 1,605; Pandas: 214)
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
from src.verification.evidence_retriever import EvidenceRetriever
from scripts.run_stage2_pilot import (
    BENCHMARK_TARGETS,
    build_reconstructed_preamble,
    load_stage1_catalog,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_stage3_manifest")

TARGET_SHORT_NAMES: Set[str] = {t.split(".")[-1] for t in BENCHMARK_TARGETS}

_worker_resolver: Optional[JediResolver] = None
_worker_evidence: Optional[EvidenceRetriever] = None


def _init_worker(catalog_symbols: List[str]):
    global _worker_resolver, _worker_evidence
    _worker_resolver = JediResolver(catalog_symbols=catalog_symbols)
    _worker_evidence = EvidenceRetriever()


def match_canonical_target(sym: Optional[str], callee: str, lib: str) -> str:
    """Associates a candidate call with the appropriate canonical benchmark target API."""
    if sym:
        for t in BENCHMARK_TARGETS:
            if _is_symbol_compatible(t, sym):
                return t
    short_matches = [t for t in BENCHMARK_TARGETS if t.split(".")[-1] == callee and t.startswith(lib)]
    if short_matches:
        return short_matches[0]
    # Fallback to any matching target
    all_matches = [t for t in BENCHMARK_TARGETS if t.split(".")[-1] == callee]
    if all_matches:
        return all_matches[0]
    return sym or callee


def is_target_candidate(callee_name: str, matched_sym: str | None) -> bool:
    if callee_name in TARGET_SHORT_NAMES:
        return True
    if matched_sym:
        for t in BENCHMARK_TARGETS:
            if _is_symbol_compatible(t, matched_sym):
                return True
    return False


def _extract_sample_candidates(args: tuple[str, str, int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    global _worker_resolver, _worker_evidence
    if _worker_resolver is None or _worker_evidence is None:
        raise RuntimeError("Worker not initialized")

    lib, sample_id, sample_idx, sample = args
    code = sample.get("function", "")
    preamble = build_reconstructed_preamble(sample)
    label = sample.get("label", "unknown")  # outdated vs up-to-date cohort

    res = _worker_resolver.analyze_client_snippet(
        code=code,
        sample_id=sample_id,
        preamble=preamble,
        library_hint=lib,
    )

    candidates = []

    # 1. Resolved Deprecated Candidates (Confirmation Mode)
    for idx, r in enumerate(res.resolved_deprecated):
        sym = r.matched_catalog_symbol or r.qualified_name
        target_api = match_canonical_target(sym, sym.split(".")[-1], lib)
        ev = _worker_evidence.get_evidence(target_api)
        candidates.append(
            {
                "candidate_id": f"{sample_id}_res_{idx}_{r.client_line}_{r.column}",
                "sample_id": sample_id,
                "sample_idx": sample_idx,
                "library": lib,
                "cohort": label,
                "stage2_status": "resolved_deprecated",
                "mode": "confirmation",
                "target_api": target_api,
                "resolved_symbol": sym,
                "client_line": r.client_line,
                "column": r.column,
                "call_site_snippet": r.call_site_snippet or "",
                "failure_reason": None,
                "failure_details": None,
                "recommended_replacement": ev.get("recommended_replacement"),
                "evidence_warning": ev.get("warning"),
                "evidence_docstring": ev.get("docstring"),
                "enclosing_code": code,
            }
        )

    # 2. Target Low-Confidence Candidates (Inference Mode)
    for idx, lc in enumerate(res.low_confidence):
        if is_target_candidate(lc.callee_name, lc.matched_catalog_symbol):
            target_api = match_canonical_target(lc.matched_catalog_symbol, lc.callee_name, lib)
            ev = _worker_evidence.get_evidence(target_api)
            candidates.append(
                {
                    "candidate_id": f"{sample_id}_lc_{idx}_{lc.line}_{lc.column}",
                    "sample_id": sample_id,
                    "sample_idx": sample_idx,
                    "library": lib,
                    "cohort": label,
                    "stage2_status": "low_confidence",
                    "mode": "inference",
                    "target_api": target_api,
                    "resolved_symbol": lc.matched_catalog_symbol,
                    "client_line": lc.line,
                    "column": lc.column,
                    "call_site_snippet": lc.call_site_snippet or "",
                    "failure_reason": lc.failure_reason,
                    "failure_details": lc.details,
                    "recommended_replacement": ev.get("recommended_replacement"),
                    "evidence_warning": ev.get("warning"),
                    "evidence_docstring": ev.get("docstring"),
                    "enclosing_code": code,
                }
            )

    return candidates


def main():
    catalog = load_stage1_catalog()
    catalog_list = sorted(list(catalog))

    raw_dir = REPO_ROOT / "data" / "raw" / "llm-dep-api" / "probing-inputs"
    tasks = []
    libraries = ["numpy", "scipy", "pandas"]
    for lib in libraries:
        lib_path = raw_dir / lib / "samples.json"
        with open(lib_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        for idx, s in enumerate(samples):
            sample_id = f"{lib}_{idx}"
            tasks.append((lib, sample_id, idx, s))

    logger.info(f"Extracting Stage 3 candidates from {len(tasks)} samples using 8 workers...")
    t0 = time.time()

    all_candidates: List[Dict[str, Any]] = []
    processed = 0

    with ProcessPoolExecutor(max_workers=8, initializer=_init_worker, initargs=(catalog_list,)) as executor:
        futures = [executor.submit(_extract_sample_candidates, t) for t in tasks]
        for future in as_completed(futures):
            cand_list = future.result()
            all_candidates.extend(cand_list)
            processed += 1
            if processed % 1000 == 0 or processed == len(tasks):
                logger.info(f"Progress: {processed}/{len(tasks)} samples processed ({time.time() - t0:.1f}s)...")

    # Sort deterministically by (library, sample_idx, client_line, column, candidate_id)
    all_candidates.sort(key=lambda c: (c["library"], c["sample_idx"], c["client_line"], c["column"], c["candidate_id"]))

    logger.info(f"Extracted {len(all_candidates)} candidates in {time.time() - t0:.1f}s.")

    # Reconcile counts against census
    resolved_count = sum(1 for c in all_candidates if c["stage2_status"] == "resolved_deprecated")
    lc_count = sum(1 for c in all_candidates if c["stage2_status"] == "low_confidence")

    by_lib = {lib: sum(1 for c in all_candidates if c["library"] == lib) for lib in libraries}
    logger.info(f"Resolved deprecated calls: {resolved_count}")
    logger.info(f"Target low-confidence calls: {lc_count}")
    logger.info(f"Per-library totals: {by_lib}")
    logger.info(f"Grand total candidate call sites: {len(all_candidates)}")

    assert len(all_candidates) == 2932, f"Expected 2932 candidates, got {len(all_candidates)}"
    assert resolved_count == 2533, f"Expected 2533 resolved candidates, got {resolved_count}"
    assert lc_count == 399, f"Expected 399 low-confidence candidates, got {lc_count}"
    assert by_lib == {"numpy": 1113, "scipy": 1605, "pandas": 214}, f"Per-lib mismatch: {by_lib}"

    out_file = REPO_ROOT / "data" / "stage3_candidates_manifest.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, indent=2)

    logger.info(f"Successfully saved manifest with exactly 2,932 candidates to {out_file}")


if __name__ == "__main__":
    main()
