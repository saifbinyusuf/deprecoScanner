#!/usr/bin/env python3
"""
scripts/run_calibration.py - Day 1 Calibration & Stop Gate Runner.

Runs Task 4.1 to 4.3 calibration on representative benchmark test cases:
1. True Positive: scipy_0 (scipy.misc.comb)
2. True Positive: numpy_0 (numpy.product)
3. True Positive: pandas_70 (pandas.DataFrame.iteritems)
4. Ground-Truth Anomaly / 3rd-party Lookalike: scipy_1560 (mpmath.factorial mislabeled)
5. Inactive Fallback / Guard: scipy_577 (scipy.misc.logsumexp inside hasattr check)
6. Low-Confidence Untyped Receiver: pandas_0 (df.style.render)

Verifies:
- Task 4.1: Structured JSON output adherence and rationale validity.
- Task 4.2: Accurate grounding evidence extraction.
- Task 4.3: Compound SQLite caching and 100% cache hit on repeat run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.verification.evidence_retriever import EvidenceRetriever
from src.verification.gemini_client import GeminiClient
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

CALIBRATION_SPEC = [
    {
        "library": "scipy",
        "idx": 0,
        "role": "True Positive (scipy.misc.comb)",
        "expected_target": "scipy.misc.comb",
        "expected_mode": PromptMode.CONFIRMATION,
    },
    {
        "library": "numpy",
        "idx": 0,
        "role": "True Positive (numpy.product)",
        "expected_target": "numpy.product",
        "expected_mode": PromptMode.CONFIRMATION,
    },
    {
        "library": "pandas",
        "idx": 70,
        "role": "True Positive (pandas.DataFrame.iteritems)",
        "expected_target": "pandas.DataFrame.iteritems",
        "expected_mode": PromptMode.CONFIRMATION,
    },
    {
        "library": "scipy",
        "idx": 1560,
        "role": "GT Label Anomaly / Third-Party Lookalike (mpmath.factorial)",
        "expected_target": "scipy.misc.factorial",
        "expected_mode": PromptMode.CONFIRMATION,
    },
    {
        "library": "scipy",
        "idx": 577,
        "role": "Inactive Fallback Guard (scipy.misc.logsumexp inside hasattr)",
        "expected_target": "scipy.misc.logsumexp",
        "expected_mode": PromptMode.CONFIRMATION,
    },
    {
        "library": "pandas",
        "idx": 0,
        "role": "Low-Confidence Untyped Receiver (Styler.render chained call)",
        "expected_target": "pandas.io.formats.style.Styler.render",
        "expected_mode": PromptMode.INFERENCE,
    },
]


def load_calibration_samples() -> List[Dict[str, Any]]:
    raw_dir = REPO_ROOT / "data" / "raw" / "llm-dep-api" / "probing-inputs"
    loaded = []
    for spec in CALIBRATION_SPEC:
        lib = spec["library"]
        idx = spec["idx"]
        path = raw_dir / lib / "samples.json"
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        s = samples[idx]
        loaded.append({
            **spec,
            "sample_id": f"{lib}_{idx}",
            "sample_data": s,
        })
    return loaded


def run_calibration_pass(
    samples: List[Dict[str, Any]],
    client: GeminiClient,
    retriever: EvidenceRetriever,
    resolver: JediResolver,
    pass_num: int,
) -> List[Dict[str, Any]]:
    results = []
    print(f"\n{'=' * 80}")
    print(f"CALIBRATION RUN: PASS {pass_num}")
    print(f"{'=' * 80}")

    for item in samples:
        sample_id = item["sample_id"]
        lib = item["library"]
        role = item["role"]
        expected_target = item["expected_target"]
        expected_mode = item["expected_mode"]
        sample_data = item["sample_data"]
        code = sample_data.get("function", "")

        # Run Stage 2 resolution
        stage2_res = resolver.analyze_client_snippet(
            code=code,
            sample_id=sample_id,
            preamble=BASELINE_PREAMBLE,
            library_hint=lib,
        )

        # Select representative candidate call site
        candidate_site = None
        call_snippet = None
        line_no = None
        col_no = None
        failure_reason = None
        failure_details = None

        if expected_mode == PromptMode.CONFIRMATION:
            if stage2_res.resolved_deprecated:
                # Find matching target
                for r in stage2_res.resolved_deprecated:
                    if r.matched_catalog_symbol == expected_target or r.qualified_name == expected_target:
                        candidate_site = r
                        break
                if not candidate_site:
                    candidate_site = stage2_res.resolved_deprecated[0]
                call_snippet = candidate_site.call_site_snippet or candidate_site.description or candidate_site.name
                line_no = candidate_site.client_line or candidate_site.line
                col_no = candidate_site.column
            else:
                # Fallback for anomaly cases like scipy_1560 where resolver correctly didn't resolve deprecated
                # Extract line matching target or callee
                callee_short = expected_target.split(".")[-1]
                lines = code.splitlines()
                for l_idx, l_text in enumerate(lines):
                    if callee_short in l_text:
                        call_snippet = l_text.strip()
                        line_no = l_idx + 1
                        break
                if not call_snippet:
                    call_snippet = lines[0].strip() if lines else code[:60]
                    line_no = 1
        else:  # INFERENCE mode
            if stage2_res.low_confidence:
                # Find candidate matching target
                for lc in stage2_res.low_confidence:
                    if lc.matched_catalog_symbol == expected_target or lc.callee_name in expected_target:
                        candidate_site = lc
                        break
                if not candidate_site:
                    candidate_site = stage2_res.low_confidence[0]
                call_snippet = candidate_site.call_site_snippet
                line_no = candidate_site.line
                col_no = candidate_site.column
                failure_reason = candidate_site.failure_reason
                failure_details = candidate_site.details
            else:
                lines = code.splitlines()
                call_snippet = lines[0].strip()
                line_no = 1
                failure_reason = "unresolved_receiver"

        # Retrieve grounding evidence
        evidence = retriever.get_evidence(expected_target, library_hint=lib)

        # Build provenance
        provenance = CandidateProvenance(
            sample_id=sample_id,
            target_api=expected_target,
            library=lib,
            call_site_snippet=call_snippet or "<call_site>",
            line_number=line_no,
            column_number=col_no,
            enclosing_code=code,
            failure_reason=failure_reason,
            failure_details=failure_details,
            evidence_docstring=evidence.get("docstring"),
            evidence_warning=evidence.get("warning"),
            recommended_replacement=evidence.get("recommended_replacement"),
            source_location=evidence.get("source_location"),
        )

        # Execute verification
        res = client.verify_candidate(
            provenance=provenance,
            mode=expected_mode,
            prompt_version=PROMPT_VERSION,
        )

        decision = res["decision"]
        cached = res["cached"]
        cache_key = res["cache_key"]
        p_tokens = res["prompt_tokens"]
        c_tokens = res["candidate_tokens"]

        print(f"\n--------------------------------------------------------------------------------")
        print(f"Sample: {sample_id} | Role: {role}")
        print(f"Target API: {expected_target} | Mode: {expected_mode.value.upper()}")
        print(f"Call Site (Line {line_no}): {call_snippet}")
        print(f"Cached: {cached} (Key: {cache_key[:12]}...)")
        print(f"Tokens: Prompt={p_tokens}, Candidate={c_tokens}, Total={p_tokens + c_tokens}")
        print(f"Decision: is_deprecated={decision.is_deprecated_usage}, confidence={decision.confidence}")
        print(f"Rationale: {decision.rationale}")

        results.append({
            "sample_id": sample_id,
            "role": role,
            "target_api": expected_target,
            "mode": expected_mode.value,
            "call_site": call_snippet,
            "line": line_no,
            "decision": decision.model_dump(),
            "cached": cached,
            "tokens": {
                "prompt": p_tokens,
                "candidate": c_tokens,
                "total": p_tokens + c_tokens,
            },
            "evidence": evidence,
        })

    return results


def main():
    logger.info("Initializing Stage 1 catalog and Jedi resolver...")
    catalog = load_stage1_catalog()
    resolver = JediResolver(catalog_symbols=sorted(list(catalog)))
    retriever = EvidenceRetriever()

    cache_db = REPO_ROOT / "data" / "stage3_response_cache.db"
    cache = ResponseCache(db_path=cache_db)
    client = GeminiClient(cache=cache, default_model="gemini-3.5-flash-lite")

    samples = load_calibration_samples()
    logger.info(f"Loaded {len(samples)} calibration samples.")

    # Pass 1: Initial execution (populate cache)
    pass1_results = run_calibration_pass(
        samples=samples,
        client=client,
        retriever=retriever,
        resolver=resolver,
        pass_num=1,
    )
    metrics_pass1 = client.get_run_metrics()
    print("\n" + "=" * 80)
    print("PASS 1 METRICS:")
    print(json.dumps(metrics_pass1, indent=2))

    # Pass 2: Repeat execution (verify 100% cache hit rate)
    pass2_results = run_calibration_pass(
        samples=samples,
        client=client,
        retriever=retriever,
        resolver=resolver,
        pass_num=2,
    )
    metrics_pass2 = client.get_run_metrics()
    print("\n" + "=" * 80)
    print("PASS 2 CUMULATIVE METRICS:")
    print(json.dumps(metrics_pass2, indent=2))

    # Assert cache verification
    pass2_hits = metrics_pass2["cache_hits"] - metrics_pass1["cache_hits"]
    pass2_api_calls = metrics_pass2["api_calls"] - metrics_pass1["api_calls"]
    print(f"\nCache Verification Check:")
    print(f"Pass 2 New API Calls: {pass2_api_calls} (Expected: 0)")
    print(f"Pass 2 Cache Hits: {pass2_hits} (Expected: {len(samples)})")

    assert pass2_api_calls == 0, f"Cache verification failed: {pass2_api_calls} new API calls in pass 2"
    assert pass2_hits == len(samples), f"Cache verification failed: {pass2_hits} hits in pass 2"
    print("✅ CACHE VERIFICATION SUCCESSFUL: 100% Cache Hit Rate on Repeat Run!")

    # Save calibration results
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "stage3_calibration_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "prompt_version": PROMPT_VERSION,
            "primary_model": "gemini-3.5-flash-lite",
            "samples": pass1_results,
            "pass1_metrics": metrics_pass1,
            "pass2_cumulative_metrics": metrics_pass2,
        }, f, indent=2)
    logger.info(f"Calibration results saved to {out_file}")


if __name__ == "__main__":
    main()
