"""
scripts/reconcile_stage3_predictions.py - Reconciles Stage 3 Predictions & Final Report.

Filters Stage 3 primary predictions (from 2,932 to 2,263) and validation predictions
(from 150 to 125) to strictly match the canonical 31 benchmark target APIs.

Recomputes:
- Semantic verdicts across resolved deprecated (N=1,864) and target low-confidence (N=399).
- Dual-model validation concordance, Cohen's Kappa, PABAK, and Balanced Accuracy on in-scope slice (N=125).
- Updates results/stage3_predictions.jsonl, results/stage3_validation_predictions.jsonl,
  and results/stage3_final_report.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.verification.batch_verifier import compute_cohens_kappa
from src.verification.verifier_prompt import PROMPT_VERSION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reconcile_stage3")


def main():
    manifest_path = REPO_ROOT / "data" / "stage3_candidates_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    logger.info(f"Loaded manifest with {len(manifest)} candidates.")
    manifest_ids = {c["candidate_id"] for c in manifest}
    manifest_dict = {c["candidate_id"]: c for c in manifest}

    # 1. Filter primary predictions
    raw_preds_path = REPO_ROOT / "results" / "stage3_predictions.jsonl"
    all_preds = []
    with open(raw_preds_path, "r", encoding="utf-8") as f:
        for line in f:
            all_preds.append(json.loads(line))

    in_scope_preds = []
    for p in all_preds:
        cid = p["candidate_id"]
        if cid in manifest_ids:
            # Update target_api from clean manifest
            p["target_api"] = manifest_dict[cid]["target_api"]
            in_scope_preds.append(p)

    logger.info(f"Filtered primary predictions: {len(all_preds)} -> {len(in_scope_preds)}.")
    assert len(in_scope_preds) == len(manifest), f"Mismatch: {len(in_scope_preds)} vs {len(manifest)}"

    # Sort deterministically
    in_scope_preds.sort(key=lambda p: (p["library"], p["sample_idx"], p["client_line"], p["column"], p["candidate_id"]))

    with open(raw_preds_path, "w", encoding="utf-8") as f:
        for p in in_scope_preds:
            f.write(json.dumps(p) + "\n")
    logger.info(f"Overwrote {raw_preds_path} with {len(in_scope_preds)} in-scope predictions.")

    # 2. Filter validation predictions
    raw_val_path = REPO_ROOT / "results" / "stage3_validation_predictions.jsonl"
    all_val = []
    with open(raw_val_path, "r", encoding="utf-8") as f:
        for line in f:
            all_val.append(json.loads(line))

    in_scope_val = []
    for v in all_val:
        cid = v["candidate_id"]
        if cid in manifest_ids:
            v["target_api"] = manifest_dict[cid]["target_api"]
            in_scope_val.append(v)

    logger.info(f"Filtered validation predictions: {len(all_val)} -> {len(in_scope_val)}.")
    assert len(in_scope_val) == 110, f"Expected 110 in-scope validation calls, got {len(in_scope_val)}"

    with open(raw_val_path, "w", encoding="utf-8") as f:
        for v in in_scope_val:
            f.write(json.dumps(v) + "\n")
    logger.info(f"Overwrote {raw_val_path} with {len(in_scope_val)} in-scope predictions.")

    # 3. Recompute Validation Metrics
    n_val = len(in_scope_val)
    table = [[0, 0], [0, 0]]
    agreed = 0
    disagreements = []

    for v in in_scope_val:
        p_dec = v["primary_decision"]
        val_dec = v["validation_decision"]
        if p_dec == val_dec:
            agreed += 1
        else:
            disagreements.append(v)

        if p_dec and val_dec:
            table[0][0] += 1
        elif p_dec and not val_dec:
            table[0][1] += 1
        elif not p_dec and val_dec:
            table[1][0] += 1
        else:
            table[1][1] += 1

    raw_concordance = agreed / n_val * 100.0
    kappa = compute_cohens_kappa(table)
    pabak = round(2.0 * (raw_concordance / 100.0) - 1.0, 4)
    sens = table[0][0] / (table[0][0] + table[1][0]) if (table[0][0] + table[1][0]) > 0 else 0.0
    spec = table[1][1] / (table[1][1] + table[0][1]) if (table[1][1] + table[0][1]) > 0 else 0.0
    bal_acc = round((sens + spec) / 2.0 * 100.0, 2)

    val_summary = {
        "total_evaluated": n_val,
        "agreed_count": agreed,
        "disagreed_count": n_val - agreed,
        "raw_concordance_pct": round(raw_concordance, 2),
        "cohens_kappa": round(kappa, 4),
        "pabak": pabak,
        "balanced_accuracy_pct": bal_acc,
        "contingency_table": {
            "both_deprecated": table[0][0],
            "primary_deprecated_val_benign": table[0][1],
            "primary_benign_val_deprecated": table[1][0],
            "both_benign": table[1][1],
        },
        "disagreements": disagreements,
    }

    # 4. Assemble Final Report
    dep_count = sum(1 for p in in_scope_preds if p["is_deprecated_usage"])
    ben_count = len(in_scope_preds) - dep_count

    by_status = {}
    for p in in_scope_preds:
        st = p["stage2_status"]
        by_status.setdefault(st, {"total": 0, "deprecated": 0, "benign": 0})
        by_status[st]["total"] += 1
        if p["is_deprecated_usage"]:
            by_status[st]["deprecated"] += 1
        else:
            by_status[st]["benign"] += 1

    by_lib = {}
    for p in in_scope_preds:
        lib = p["library"]
        by_lib.setdefault(lib, {"total": 0, "deprecated": 0, "benign": 0})
        by_lib[lib]["total"] += 1
        if p["is_deprecated_usage"]:
            by_lib[lib]["deprecated"] += 1
        else:
            by_lib[lib]["benign"] += 1

    benign_spot_path = REPO_ROOT / "results" / "stage3_benign_spot_check.json"
    benign_summary = {}
    if benign_spot_path.exists():
        with open(benign_spot_path, "r", encoding="utf-8") as f:
            benign_summary = json.load(f)

    report = {
        "evaluation_stage": "Stage 3: LLM Verification (Phase 4)",
        "scope_policy": "Strictly anchored to 31 canonical benchmark target APIs",
        "primary_model": "gemini-3.5-flash-lite",
        "validation_model": "gemini-3.1-pro-preview",
        "prompt_version": PROMPT_VERSION,
        "total_candidates_verified": len(in_scope_preds),
        "semantic_verdicts": {
            "verified_deprecated_count": dep_count,
            "verified_deprecated_pct": round(dep_count / len(in_scope_preds) * 100.0, 2),
            "rejected_benign_or_anomaly_count": ben_count,
            "rejected_benign_or_anomaly_pct": round(ben_count / len(in_scope_preds) * 100.0, 2),
        },
        "by_stage2_status": by_status,
        "by_library": by_lib,
        "validation_subsample": val_summary,
        "benign_spot_check": benign_summary,
    }

    report_path = REPO_ROOT / "results" / "stage3_final_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Reconciled final report saved to {report_path}")
    print("\n" + "=" * 80)
    print("RECONCILED STAGE 3 EVALUATION REPORT:")
    print("=" * 80)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
