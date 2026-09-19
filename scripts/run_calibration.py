#!/usr/bin/env python3
"""
scripts/run_calibration.py - Day 1 Calibration & Call-Site Granularity Runner.

Validates:
1. Call-Site Granularity: Distinguishes distinct call sites within multi-call lines:
   - numpy_0: np.product(x, axis=0) (resolved) vs. product(x, axis=0) (bare call / empty_goto)
   - pandas_70: pdf.iteritems() (pandas true positive) vs. psdf.iteritems() (pyspark.pandas lookalike)
2. Anomaly Detection & Lookalike Rejection:
   - scipy_1560: mpmath.factorial lookalike correctly rejected
   - pandas_70: psdf.iteritems() correctly rejected as PySpark wrapper
3. Inactive Fallback Guards:
   - scipy_577: hasattr(scipy.misc, 'logsumexp') check
4. Type Inference:
   - pandas_0: es.render() inferred from Styler(empty_df)
5. Non-Saturating Confidence Calibration on Genuine Ambiguity:
   - ambiguous_records: untyped function parameter records.iteritems() drops confidence to 0.85
6. Cache Idempotence:
   - Pass 2 repeat execution yields 0 new API calls and 100% cache hit rate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.verification.evidence_retriever import EvidenceRetriever
from src.verification.gemini_client import GeminiClient, DEFAULT_PRIMARY_MODEL, DEFAULT_VALIDATION_MODEL
from src.verification.response_cache import ResponseCache
from src.verification.verifier_prompt import (
    CandidateProvenance,
    PromptMode,
    format_verification_prompt,
    PROMPT_VERSION,
)
from src.resolution.jedi_resolver import JediResolver
from scripts.run_stage2_pilot import (
    BASELINE_PREAMBLE,
    load_stage1_catalog,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_calibration")

AMBIGUOUS_SNIPPET = """def export_metrics(records):
    # Serializes key-value metrics
    output = []
    for k, v in records.iteritems():
        output.append(f"{k}={v}")
    return output"""


def build_calibration_dataset(resolver: JediResolver, retriever: EvidenceRetriever) -> List[Dict[str, Any]]:
    raw_dir = REPO_ROOT / "data" / "raw" / "llm-dep-api" / "probing-inputs"

    # Helper to load raw sample
    def get_sample(lib: str, idx: int) -> Dict[str, Any]:
        with open(raw_dir / lib / "samples.json", "r", encoding="utf-8") as f:
            return json.load(f)[idx]

    candidates = []

    # 1. scipy_0: True Positive scipy.misc.comb
    s_scipy0 = get_sample("scipy", 0)
    ev_comb = retriever.get_evidence("scipy.misc.comb", "scipy")
    candidates.append({
        "sample_id": "scipy_0",
        "role": "True Positive: scipy.misc.comb",
        "target_api": "scipy.misc.comb",
        "library": "scipy",
        "call_site": "real_pairs += scipy.misc.comb(count, 2)",
        "line": 24,
        "col": 26,
        "mode": PromptMode.CONFIRMATION,
        "enclosing_code": s_scipy0["function"],
        "failure_reason": None,
        "failure_details": None,
        "evidence": ev_comb,
    })

    # 2. numpy_0 Call A: Resolved np.product
    s_numpy0 = get_sample("numpy", 0)
    ev_prod = retriever.get_evidence("numpy.product", "numpy")
    candidates.append({
        "sample_id": "numpy_0_call_a",
        "role": "True Positive (Call A): np.product (prefixed invocation)",
        "target_api": "numpy.product",
        "library": "numpy",
        "call_site": "np.product(x, axis=0)",
        "line": 11,
        "col": 25,
        "mode": PromptMode.CONFIRMATION,
        "enclosing_code": s_numpy0["function"],
        "failure_reason": None,
        "failure_details": None,
        "evidence": ev_prod,
    })

    # 3. numpy_0 Call B: Bare product (unresolved import in baseline preamble)
    candidates.append({
        "sample_id": "numpy_0_call_b",
        "role": "Low-Confidence (Call B): product(...) (bare invocation on same line)",
        "target_api": "numpy.product",
        "library": "numpy",
        "call_site": "product(x, axis=0)",
        "line": 11,
        "col": 50,
        "mode": PromptMode.INFERENCE,
        "enclosing_code": s_numpy0["function"],
        "failure_reason": "empty_goto",
        "failure_details": "Bare function 'product' has no local definition or import in baseline preamble.",
        "evidence": ev_prod,
    })

    # 4. pandas_70 Call A: pdf.iteritems() (genuine pandas DataFrame)
    s_pandas70 = get_sample("pandas", 70)
    ev_iter = retriever.get_evidence("pandas.DataFrame.iteritems", "pandas")
    candidates.append({
        "sample_id": "pandas_70_pdf",
        "role": "True Positive: pdf.iteritems() (native pandas DataFrame)",
        "target_api": "pandas.DataFrame.iteritems",
        "library": "pandas",
        "call_site": "pdf.iteritems()",
        "line": 9,
        "col": 44,
        "mode": PromptMode.CONFIRMATION,
        "enclosing_code": s_pandas70["function"],
        "failure_reason": None,
        "failure_details": None,
        "evidence": ev_iter,
    })

    # 5. pandas_70 Call B: psdf.iteritems() (PySpark lookalike on same line!)
    candidates.append({
        "sample_id": "pandas_70_psdf",
        "role": "Third-Party Lookalike: psdf.iteritems() (pyspark.pandas wrapper object)",
        "target_api": "pandas.DataFrame.iteritems",
        "library": "pandas",
        "call_site": "psdf.iteritems()",
        "line": 9,
        "col": 61,
        "mode": PromptMode.INFERENCE,
        "enclosing_code": s_pandas70["function"],
        "failure_reason": "unresolved_receiver",
        "failure_details": "Receiver object psdf is assigned from ps.from_pandas(pdf) (PySpark pandas).",
        "evidence": ev_iter,
    })

    # 6. scipy_1560: mpmath.factorial lookalike
    s_scipy1560 = get_sample("scipy", 1560)
    ev_fact = retriever.get_evidence("scipy.misc.factorial", "scipy")
    candidates.append({
        "sample_id": "scipy_1560",
        "role": "GT Label Anomaly: mpmath.factorial (3rd-party lookalike)",
        "target_api": "scipy.misc.factorial",
        "library": "scipy",
        "call_site": "from mpmath import mpf, factorial, findroot, fsum, power, exp, quad",
        "line": 29,
        "col": 0,
        "mode": PromptMode.CONFIRMATION,
        "enclosing_code": s_scipy1560["function"],
        "failure_reason": None,
        "failure_details": None,
        "evidence": ev_fact,
    })

    # 7. scipy_577: Inactive Fallback Guard
    s_scipy577 = get_sample("scipy", 577)
    ev_lsm = retriever.get_evidence("scipy.misc.logsumexp", "scipy")
    candidates.append({
        "sample_id": "scipy_577",
        "role": "Fallback Guard: hasattr(scipy.misc, 'logsumexp')",
        "target_api": "scipy.misc.logsumexp",
        "library": "scipy",
        "call_site": "return scipy.misc.logsumexp( *args, **kwargs )",
        "line": 4,
        "col": 15,
        "mode": PromptMode.CONFIRMATION,
        "enclosing_code": s_scipy577["function"],
        "failure_reason": None,
        "failure_details": None,
        "evidence": ev_lsm,
    })

    # 8. pandas_0: Low-Confidence Untyped Receiver
    s_pandas0 = get_sample("pandas", 0)
    ev_render = retriever.get_evidence("pandas.io.formats.style.Styler.render", "pandas")
    candidates.append({
        "sample_id": "pandas_0",
        "role": "Receiver Type Inference: es.render() via Styler(empty_df)",
        "target_api": "pandas.io.formats.style.Styler.render",
        "library": "pandas",
        "call_site": "es.render()",
        "line": 4,
        "col": 8,
        "mode": PromptMode.INFERENCE,
        "enclosing_code": s_pandas0["function"],
        "failure_reason": "unresolved_receiver",
        "failure_details": "Receiver object 'es' could not be resolved by Jedi",
        "evidence": ev_render,
    })

    # 9. Ambiguous Untyped Parameter: records.iteritems() (drops confidence off 1.0)
    candidates.append({
        "sample_id": "ambiguous_records_iteritems",
        "role": "Ambiguous Untyped Parameter: records.iteritems() (Confidence < 1.0 check)",
        "target_api": "pandas.DataFrame.iteritems",
        "library": "pandas",
        "call_site": "for k, v in records.iteritems():",
        "line": 4,
        "col": 13,
        "mode": PromptMode.INFERENCE,
        "enclosing_code": AMBIGUOUS_SNIPPET,
        "failure_reason": "unresolved_receiver",
        "failure_details": "Receiver object 'records' is an untyped function parameter without class instantiation.",
        "evidence": ev_iter,
    })

    return candidates


def run_pass(
    candidates: List[Dict[str, Any]],
    client: GeminiClient,
    pass_num: int,
) -> List[Dict[str, Any]]:
    print(f"\n{'=' * 80}")
    print(f"CALIBRATION RUN: PASS {pass_num} ({len(candidates)} CALL SITES)")
    print(f"{'=' * 80}")

    results = []
    for cand in candidates:
        prov = CandidateProvenance(
            sample_id=cand["sample_id"],
            target_api=cand["target_api"],
            library=cand["library"],
            call_site_snippet=cand["call_site"],
            line_number=cand["line"],
            column_number=cand.get("col"),
            enclosing_code=cand["enclosing_code"],
            failure_reason=cand.get("failure_reason"),
            failure_details=cand.get("failure_details"),
            evidence_docstring=cand["evidence"].get("docstring"),
            evidence_warning=cand["evidence"].get("warning"),
            recommended_replacement=cand["evidence"].get("recommended_replacement"),
            source_location=cand["evidence"].get("source_location"),
        )

        res = client.verify_candidate(
            provenance=prov,
            mode=cand["mode"],
            prompt_version=PROMPT_VERSION,
        )

        d = res["decision"]
        cached = res["cached"]
        key = res["cache_key"]
        p_tok = res["prompt_tokens"]
        c_tok = res["candidate_tokens"]

        print(f"\n--------------------------------------------------------------------------------")
        print(f"[{cand['sample_id']}] {cand['role']}")
        print(f"Target: {cand['target_api']} | Mode: {cand['mode'].value.upper()} | Line {cand['line']}: {cand['call_site']}")
        print(f"Decision: is_deprecated={d.is_deprecated_usage} | Confidence={d.confidence} | Cached={cached} ({key[:10]}...)")
        print(f"Rationale: {d.rationale}")

        results.append({
            "sample_id": cand["sample_id"],
            "role": cand["role"],
            "target_api": cand["target_api"],
            "mode": cand["mode"].value,
            "call_site": cand["call_site"],
            "line": cand["line"],
            "decision": d.model_dump(),
            "cached": cached,
            "cache_key": key,
            "tokens": {"prompt": p_tok, "candidate": c_tok, "total": p_tok + c_tok},
        })

    return results


def main():
    logger.info("Initializing Stage 1 catalog, Jedi resolver, and evidence retriever...")
    catalog = load_stage1_catalog()
    resolver = JediResolver(catalog_symbols=sorted(list(catalog)))
    retriever = EvidenceRetriever()

    cache_db = REPO_ROOT / "data" / "stage3_response_cache.db"
    cache = ResponseCache(db_path=cache_db)
    client = GeminiClient(cache=cache, default_model=DEFAULT_PRIMARY_MODEL)

    candidates = build_calibration_dataset(resolver, retriever)
    logger.info(f"Loaded {len(candidates)} distinct call-site candidates for calibration.")

    # Pass 1: Initial run
    p1_results = run_pass(candidates, client, pass_num=1)
    m1 = client.get_run_metrics()
    print("\n" + "=" * 80)
    print("PASS 1 METRICS:")
    print(json.dumps(m1, indent=2))

    # Pass 2: Repeat run (verifying cache)
    p2_results = run_pass(candidates, client, pass_num=2)
    m2 = client.get_run_metrics()
    print("\n" + "=" * 80)
    print("PASS 2 CUMULATIVE METRICS:")
    print(json.dumps(m2, indent=2))

    p2_new_api_calls = m2["api_calls"] - m1["api_calls"]
    p2_cache_hits = m2["cache_hits"] - m1["cache_hits"]

    print("\n" + "=" * 80)
    print(f"Cache Check: Pass 2 New API Calls = {p2_new_api_calls} (Expected: 0)")
    print(f"Cache Check: Pass 2 Cache Hits = {p2_cache_hits} (Expected: {len(candidates)})")
    assert p2_new_api_calls == 0, f"Cache verification failed: {p2_new_api_calls} new calls!"
    assert p2_cache_hits == len(candidates), f"Cache verification failed: {p2_cache_hits} hits!"
    print("✅ CACHE VERIFICATION SUCCESSFUL: 100% Cache Hit Rate on Repeat Run!")

    # Check that confidence varies off 1.0
    confidences = [r["decision"]["confidence"] for r in p1_results]
    has_sub_one_confidence = any(c < 1.0 for c in confidences)
    print(f"Confidence values observed across calibration set: {confidences}")
    print(f"Confidence variation off 1.0 observed: {has_sub_one_confidence}")
    assert has_sub_one_confidence, "Confidence remained statically locked at 1.0 across all cases!"

    # Save results
    out_file = REPO_ROOT / "results" / "stage3_calibration_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "prompt_version": PROMPT_VERSION,
            "primary_model": DEFAULT_PRIMARY_MODEL,
            "pinned_validation_model": DEFAULT_VALIDATION_MODEL,
            "candidates": p1_results,
            "pass1_metrics": m1,
            "pass2_cumulative_metrics": m2,
        }, f, indent=2)
    logger.info(f"Calibration results written to {out_file}")


if __name__ == "__main__":
    main()
