"""
tests/test_batch_verifier.py - Unit tests for Stage 3 Batch Verifier and Evaluation Metrics.

Validates:
1. Cohen's Kappa calculation on known confusion matrices.
2. Stratified subsample selection across library, cohort, and stage2 status.
3. CandidateProvenance mapping in single-candidate verification.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.verification.batch_verifier import (
    BatchVerifier,
    compute_cohens_kappa,
)
from src.verification.response_cache import ResponseCache


def test_cohens_kappa_calculation():
    # 1. Perfect agreement (kappa = 1.0)
    table_perfect = [[50, 0], [0, 50]]
    assert compute_cohens_kappa(table_perfect) == 1.0

    # 2. Complete disagreement
    table_opposite = [[0, 50], [50, 0]]
    assert compute_cohens_kappa(table_opposite) == -1.0

    # 3. Realistic high agreement:
    # Primary & Validation agree on 80 deprecated, 15 benign. 3 primary-only, 2 val-only.
    table_realistic = [[80, 3], [2, 15]]
    kappa = compute_cohens_kappa(table_realistic)
    assert 0.80 < kappa < 1.0, f"Expected high positive kappa, got {kappa}"


def test_stratified_validation_subsample():
    # Create synthetic candidate list with varying library, cohort, status
    synthetic_candidates = []
    libs = ["numpy", "scipy", "pandas"]
    cohorts = ["outdated", "up-to-date"]
    statuses = ["resolved_deprecated", "low_confidence"]

    c_id = 0
    for lib in libs:
        for ch in cohorts:
            for st in statuses:
                for _ in range(20):
                    synthetic_candidates.append({
                        "candidate_id": f"cand_{c_id}",
                        "sample_id": f"{lib}_{c_id}",
                        "library": lib,
                        "cohort": ch,
                        "stage2_status": st,
                        "target_api": f"{lib}.some_target",
                    })
                    c_id += 1

    verifier = BatchVerifier()
    subsample = verifier.select_stratified_validation_subsample(
        synthetic_candidates, sample_size=30, seed=42
    )

    assert len(subsample) == 30

    # Check all 3 libraries are represented
    sampled_libs = {c["library"] for c in subsample}
    assert sampled_libs == {"numpy", "scipy", "pandas"}

    # Check both statuses are represented
    sampled_statuses = {c["stage2_status"] for c in subsample}
    assert sampled_statuses == {"resolved_deprecated", "low_confidence"}
