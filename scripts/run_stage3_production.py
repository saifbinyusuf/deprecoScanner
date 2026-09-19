#!/usr/bin/env python3
"""
scripts/run_stage3_production.py - Day 2 Full Production Execution (Task 4.4).

Executes:
1. Primary Bulk Verification across all 2,932 candidate call sites (gemini-3.5-flash-lite).
2. Validation Subsample across 150 stratified call sites (gemini-3.1-pro-preview).
3. Concordance & Cohen's Kappa evaluation.
4. Latent False-Negative Benign Spot-Check across 40 clean resolved-benign samples.
5. Final summary reporting (results/stage3_final_report.json).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.verification.batch_verifier import (
    BatchVerifier,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_VALIDATION_PATH,
    DEFAULT_BENIGN_SPOT_CHECK_PATH,
    DEFAULT_REPORT_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(REPO_ROOT / "results" / "stage3_production.log", mode="w"),
    ],
)
logger = logging.getLogger("run_stage3_production")


def main():
    logger.info("=" * 80)
    logger.info("STAGE 3: LLM VERIFICATION — FULL PRODUCTION RUN (TASK 4.4)")
    logger.info("=" * 80)

    t_start = time.time()
    verifier = BatchVerifier(
        primary_model="gemini-3.5-flash-lite",
        validation_model="gemini-3.1-pro-preview",
        max_workers=5,
        min_interval_seconds=0.2,  # ~300 RPM rate pacing
    )

    # 1. Load Candidates Manifest
    logger.info(f"Loading candidates manifest from {DEFAULT_MANIFEST_PATH}...")
    candidates = verifier.load_manifest(DEFAULT_MANIFEST_PATH)
    logger.info(f"Loaded {len(candidates)} candidate call sites.")

    resolved_count = sum(1 for c in candidates if c["stage2_status"] == "resolved_deprecated")
    lc_count = sum(1 for c in candidates if c["stage2_status"] == "low_confidence")
    logger.info(f"Breakdown: {resolved_count} resolved deprecated + {lc_count} target low-confidence.")

    # 2. Run Primary Bulk Verification Batch (gemini-3.5-flash-lite)
    logger.info("\n" + "-" * 80)
    logger.info(f"STEP 1: Primary Bulk Verification ({len(candidates)} call sites)...")
    logger.info("-" * 80)
    t0 = time.time()
    primary_results = verifier.run_primary_verification(
        candidates=candidates,
        output_path=DEFAULT_PREDICTIONS_PATH,
    )
    t_primary = time.time() - t0
    logger.info(f"Step 1 Complete in {t_primary:.1f}s.")

    # 3. Run Validation Subsample (gemini-3.1-pro-preview)
    logger.info("\n" + "-" * 80)
    logger.info("STEP 2: Stratified Validation Subsample (150 call sites)...")
    logger.info("-" * 80)
    t0 = time.time()
    val_summary = verifier.run_validation_subsample(
        candidates=candidates,
        primary_predictions=primary_results,
        sample_size=150,
        output_path=DEFAULT_VALIDATION_PATH,
    )
    t_val = time.time() - t0
    logger.info(f"Step 2 Complete in {t_val:.1f}s.")

    # 4. Run Benign Spot-Check
    logger.info("\n" + "-" * 80)
    logger.info("STEP 3: Latent False-Negative Benign Spot-Check (40 samples)...")
    logger.info("-" * 80)
    t0 = time.time()
    benign_summary = verifier.run_benign_spot_check(
        sample_size=40,
        output_path=DEFAULT_BENIGN_SPOT_CHECK_PATH,
    )
    t_spot = time.time() - t0
    logger.info(f"Step 3 Complete in {t_spot:.1f}s.")

    # 5. Generate Comprehensive Final Report
    logger.info("\n" + "-" * 80)
    logger.info("STEP 4: Generating Final Evaluation Report...")
    logger.info("-" * 80)
    report = verifier.generate_final_report(
        primary_results=primary_results,
        validation_summary=val_summary,
        benign_summary=benign_summary,
        output_path=DEFAULT_REPORT_PATH,
    )

    total_wall_time = time.time() - t_start
    logger.info("\n" + "=" * 80)
    logger.info(f"STAGE 3 FULL PRODUCTION RUN COMPLETE (Total Wall Time: {total_wall_time:.1f}s)")
    logger.info("=" * 80)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
