#!/usr/bin/env python3
"""
scripts/run_stage3_lc_validation.py - Dedicated Validation of Low-Confidence Inference-Mode Candidates.

Selects an explicitly stratified sample of 25 candidates from the 117-item low-confidence pool:
- NumPy (10): 8 Deprecated, 2 Benign
- SciPy (10): 7 Deprecated, 3 Benign
- Pandas (5): 2 Deprecated, 3 Benign
Total: 17 Deprecated, 8 Benign (68.0% prevalence, matching global 68.38%).

Evaluates using gemini-3.1-pro-preview (with response caching and 2.0s pacing).
Computes:
1. Standalone low-confidence concordance, Cohen's Kappa, PABAK, and balanced accuracy (N=25).
2. Composite validation metrics merging the 104 resolved candidates + 25 low-confidence candidates (N=129).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.verification.gemini_client import GeminiClient
from src.verification.response_cache import ResponseCache
from src.verification.verifier_prompt import CandidateProvenance, PromptMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(REPO_ROOT / "results" / "stage3_lc_validation.log", mode="w"),
    ],
)
logger = logging.getLogger("run_stage3_lc_validation")


def compute_metrics(table: List[List[int]]) -> Dict[str, Any]:
    """
    table: [[both_true, prim_true_val_false], [prim_false_val_true, both_false]]
    """
    both_true = table[0][0]
    p_t_v_f = table[0][1]
    p_f_v_t = table[1][0]
    both_false = table[1][1]

    n = both_true + p_t_v_f + p_f_v_t + both_false
    if n == 0:
        return {}

    agreed = both_true + both_false
    po = agreed / n

    p_dep_prim = (both_true + p_t_v_f) / n
    p_ben_prim = (p_f_v_t + both_false) / n
    p_dep_val = (both_true + p_f_v_t) / n
    p_ben_val = (p_t_v_f + both_false) / n

    pe = (p_dep_prim * p_dep_val) + (p_ben_prim * p_ben_val)
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) > 1e-9 else 0.0
    pabak = 2.0 * po - 1.0

    sens = both_true / (both_true + p_f_v_t) if (both_true + p_f_v_t) > 0 else 0.0
    spec = both_false / (both_false + p_t_v_f) if (both_false + p_t_v_f) > 0 else 0.0
    bal_acc = (sens + spec) / 2.0

    return {
        "n": n,
        "agreed": agreed,
        "concordance_pct": round(po * 100.0, 2),
        "chance_agreement_pe": round(pe, 4),
        "cohens_kappa": round(kappa, 4),
        "pabak": round(pabak, 4),
        "sensitivity_pct": round(sens * 100.0, 2),
        "specificity_pct": round(spec * 100.0, 2),
        "balanced_accuracy_pct": round(bal_acc * 100.0, 2),
        "contingency_table": table,
    }


def main():
    logger.info("=" * 80)
    logger.info("STAGE 3: LOW-CONFIDENCE VALIDATION SUBSAMPLE (TASK 4.4 EXTENSION)")
    logger.info("=" * 80)

    manifest_path = REPO_ROOT / "data" / "stage3_candidates_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_map = {c["candidate_id"]: c for c in manifest}

    preds_path = REPO_ROOT / "results" / "stage3_predictions.jsonl"
    with open(preds_path, "r", encoding="utf-8") as f:
        preds = {p["candidate_id"]: p for p in (json.loads(line) for line in f)}

    # Identify existing validation records
    val_path = REPO_ROOT / "results" / "stage3_validation_predictions.jsonl"
    existing_val_records = []
    if val_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                existing_val_records.append(json.loads(line))

    # Clean existing validation: resolved candidates only
    val_resolved = [
        r for r in existing_val_records
        if manifest_map.get(r["candidate_id"], {}).get("stage2_status") == "resolved_deprecated"
    ]
    logger.info(f"Loaded {len(val_resolved)} surviving clean resolved validation records.")

    # 1. Select Stratified Low-Confidence Subsample (N = 25)
    lc_cands = [c for c in manifest if c["stage2_status"] == "low_confidence"]
    logger.info(f"Total Low-Confidence pool size: {len(lc_cands)}")

    # Strata buckets
    strata = {}
    for c in lc_cands:
        p = preds[c["candidate_id"]]
        key = (c["library"], p["is_deprecated_usage"])
        strata.setdefault(key, []).append(c)

    # Reused IDs from existing LC validation
    val_lc_existing_ids = {
        r["candidate_id"] for r in existing_val_records
        if manifest_map.get(r["candidate_id"], {}).get("stage2_status") == "low_confidence"
    }

    allocations = {
        ("numpy", True): 8,
        ("numpy", False): 2,
        ("scipy", True): 7,
        ("scipy", False): 3,
        ("pandas", True): 2,
        ("pandas", False): 3,
    }

    random.seed(42)
    selected_lc = []
    for key, target_n in allocations.items():
        pool = strata[key]
        already_in_val = [c for c in pool if c["candidate_id"] in val_lc_existing_ids]
        others = [c for c in pool if c["candidate_id"] not in val_lc_existing_ids]
        others.sort(key=lambda x: x["candidate_id"])
        random.shuffle(others)
        chosen = (already_in_val + others)[:target_n]
        selected_lc.extend(chosen)

    logger.info(f"Selected {len(selected_lc)} stratified low-confidence candidates.")

    # 2. Run Validation with gemini-3.1-pro-preview
    val_model = "gemini-3.1-pro-preview"
    cache = ResponseCache()
    client = GeminiClient(cache=cache, min_interval_seconds=2.0)

    lc_val_results = []
    t0 = time.time()
    for idx, cand in enumerate(selected_lc, 1):
        prov = CandidateProvenance(
            sample_id=cand["sample_id"],
            target_api=cand["target_api"],
            library=cand["library"],
            call_site_snippet=cand["call_site_snippet"],
            line_number=cand["client_line"],
            enclosing_code=cand["enclosing_code"],
            recommended_replacement=cand.get("recommended_replacement"),
            evidence_warning=cand.get("evidence_warning"),
            evidence_docstring=cand.get("evidence_docstring"),
            failure_reason=cand.get("failure_reason"),
            failure_details=cand.get("failure_details"),
        )
        # Low confidence candidates use INFERENCE mode
        resp = client.verify_candidate(prov, mode=PromptMode.INFERENCE, model=val_model)
        dec = resp["decision"]
        prim = preds[cand["candidate_id"]]

        rec = {
            "candidate_id": cand["candidate_id"],
            "sample_id": cand["sample_id"],
            "library": cand["library"],
            "target_api": cand["target_api"],
            "stage2_status": "low_confidence",
            "cohort": cand.get("cohort", "outdated"),
            "primary_model": "gemini-3.5-flash-lite",
            "primary_decision": prim["is_deprecated_usage"],
            "primary_confidence": prim["confidence"],
            "primary_rationale": prim["rationale"],
            "validation_model": val_model,
            "validation_decision": dec.is_deprecated_usage,
            "validation_confidence": dec.confidence,
            "validation_rationale": dec.rationale,
            "agreement": (prim["is_deprecated_usage"] == dec.is_deprecated_usage),
            "cached": resp["cached"],
        }
        lc_val_results.append(rec)
        logger.info(
            f"[{idx:2d}/{len(selected_lc)}] {cand['candidate_id']} ({cand['library']}) | "
            f"Prim: {rec['primary_decision']} vs Val: {rec['validation_decision']} | "
            f"Agree: {rec['agreement']} (cached={resp['cached']})"
        )

    logger.info(f"Low-confidence validation complete in {time.time() - t0:.1f}s.")

    # Save standalone LC validation predictions
    lc_val_file = REPO_ROOT / "results" / "stage3_lc_validation_predictions.jsonl"
    with open(lc_val_file, "w", encoding="utf-8") as f:
        for r in lc_val_results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved {len(lc_val_results)} LC validation predictions to {lc_val_file}")

    # Compute Standalone LC Metrics
    lc_table = [[0, 0], [0, 0]]
    for r in lc_val_results:
        p = r["primary_decision"]
        v = r["validation_decision"]
        if p and v: lc_table[0][0] += 1
        elif p and not v: lc_table[0][1] += 1
        elif not p and v: lc_table[1][0] += 1
        else: lc_table[1][1] += 1

    lc_metrics = compute_metrics(lc_table)
    print("\n" + "=" * 80)
    print("STANDALONE LOW-CONFIDENCE VALIDATION RESULTS (N = 25)")
    print("=" * 80)
    print(f"Observed Agreement (Po): {lc_metrics['agreed']}/{lc_metrics['n']} = {lc_metrics['concordance_pct']}%")
    print(f"Expected Chance Agreement (Pe): {lc_metrics['chance_agreement_pe']}")
    print(f"Raw Cohen's Kappa: {lc_metrics['cohens_kappa']}")
    print(f"Prevalence-Adjusted Kappa (PABAK): {lc_metrics['pabak']}")
    print(f"Sensitivity: {lc_metrics['sensitivity_pct']}%")
    print(f"Specificity: {lc_metrics['specificity_pct']}%")
    print(f"Balanced Accuracy: {lc_metrics['balanced_accuracy_pct']}%")
    print(f"Contingency Table: {lc_metrics['contingency_table']}")

    # 3. Create Composite Re-Weighted Validation Set (104 Resolved + 25 Low-Confidence = 129 Total)
    all_val_records = val_resolved + lc_val_results
    composite_val_file = REPO_ROOT / "results" / "stage3_validation_predictions.jsonl"
    with open(composite_val_file, "w", encoding="utf-8") as f:
        for r in all_val_records:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved {len(all_val_records)} composite validation predictions to {composite_val_file}")

    comp_table = [[0, 0], [0, 0]]
    for r in all_val_records:
        p = r["primary_decision"]
        v = r["validation_decision"]
        if p and v: comp_table[0][0] += 1
        elif p and not v: comp_table[0][1] += 1
        elif not p and v: comp_table[1][0] += 1
        else: comp_table[1][1] += 1

    comp_metrics = compute_metrics(comp_table)
    print("\n" + "=" * 80)
    print("COMPOSITE RE-WEIGHTED VALIDATION RESULTS (N = 129: 104 Resolved + 25 Low-Confidence)")
    print("=" * 80)
    print(f"Observed Agreement (Po): {comp_metrics['agreed']}/{comp_metrics['n']} = {comp_metrics['concordance_pct']}%")
    print(f"Expected Chance Agreement (Pe): {comp_metrics['chance_agreement_pe']}")
    print(f"Raw Cohen's Kappa: {comp_metrics['cohens_kappa']}")
    print(f"Prevalence-Adjusted Kappa (PABAK): {comp_metrics['pabak']}")
    print(f"Sensitivity: {comp_metrics['sensitivity_pct']}%")
    print(f"Specificity: {comp_metrics['specificity_pct']}%")
    print(f"Balanced Accuracy: {comp_metrics['balanced_accuracy_pct']}%")
    print(f"Contingency Table: {comp_metrics['contingency_table']}")

    # Update final report JSON
    report_file = REPO_ROOT / "results" / "stage3_final_report.json"
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            report = json.load(f)
        report["low_confidence_validation_metrics"] = lc_metrics
        report["composite_validation_metrics"] = comp_metrics
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Updated {report_file} with low-confidence and composite validation metrics.")


if __name__ == "__main__":
    main()
