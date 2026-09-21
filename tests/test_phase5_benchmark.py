"""
tests/test_phase5_benchmark.py - Validates the frozen Phase 5 ground-truth benchmark dataset (N = 150).
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = REPO_ROOT / "data" / "benchmark" / "phase5_benchmark_unannotated.jsonl"

EXCLUDED_CALIBRATION_IDS = {
    "numpy_0", "numpy_3", "scipy_0", "scipy_58", "scipy_577",
    "pandas_0", "pandas_70", "scipy_1560", "pandas_32", "pandas_33"
}


@pytest.fixture(scope="module")
def benchmark_data():
    assert BENCHMARK_PATH.exists(), f"Benchmark file not found at {BENCHMARK_PATH}"
    items = []
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def test_total_benchmark_size(benchmark_data):
    """Asserts exact total size of N = 150 call sites."""
    assert len(benchmark_data) == 150


def test_ground_truth_class_balance(benchmark_data):
    """Asserts exact class balance: 108 True Deprecations (72.0%), 42 True Benign (28.0%)."""
    true_deprecations = [b for b in benchmark_data if b["expected_ground_truth"] is True]
    true_benign = [b for b in benchmark_data if b["expected_ground_truth"] is False]
    assert len(true_deprecations) == 108
    assert len(true_benign) == 42
    assert len(true_deprecations) / len(benchmark_data) == pytest.approx(0.72, abs=1e-4)
    assert len(true_benign) / len(benchmark_data) == pytest.approx(0.28, abs=1e-4)


def test_strata_breakdown(benchmark_data):
    """Asserts exact item counts per stratum."""
    counts = {}
    for b in benchmark_data:
        s = b["stratum"]
        counts[s] = counts.get(s, 0) + 1

    assert counts.get("candidate_positive") == 100
    assert counts.get("pipeline_miss") == 8
    assert counts.get("label_anomaly") == 2
    assert counts.get("hard_negative") == 40


def test_calibration_contamination_guard(benchmark_data):
    """Asserts zero contamination between hard negatives and Phase 4 calibration/audit instances."""
    hard_negs = [b for b in benchmark_data if b["stratum"] == "hard_negative"]
    hard_neg_sample_ids = {b["sample_id"] for b in hard_negs}
    contamination = hard_neg_sample_ids & EXCLUDED_CALIBRATION_IDS
    assert not contamination, f"Found calibration sample IDs in hard negatives: {contamination}"


def test_per_library_deprecation_quotas(benchmark_data):
    """Asserts exact per-library quotas for deprecation call sites: NumPy: 12, Pandas: 27, SciPy: 69."""
    dep_items = [b for b in benchmark_data if b["expected_ground_truth"] is True]
    by_lib = {}
    for b in dep_items:
        lib = b["library"]
        by_lib[lib] = by_lib.get(lib, 0) + 1

    assert by_lib.get("numpy") == 12
    assert by_lib.get("pandas") == 27
    assert by_lib.get("scipy") == 69


def test_scipy_parity_quotas(benchmark_data):
    """Asserts all 18 SciPy target APIs receive quota k=4 (pinv2 capped at 1)."""
    dep_items = [b for b in benchmark_data if b["expected_ground_truth"] is True and b["library"] == "scipy"]
    by_target = {}
    for b in dep_items:
        t = b["target_api"]
        by_target[t] = by_target.get(t, 0) + 1

    # 17 targets with quota 4, 1 target (pinv2) with quota 1
    assert len(by_target) == 18
    assert by_target["scipy.special.errprint"] == 4
    assert by_target["scipy.stats.rvs_ratio_uniforms"] == 4
    assert by_target["scipy.linalg.pinv2"] == 1
    for target, count in by_target.items():
        if target == "scipy.linalg.pinv2":
            assert count == 1
        else:
            assert count == 4, f"Target {target} count {count} != 4"


def test_field_completeness(benchmark_data):
    """Asserts all items contain required metadata and non-empty code contexts."""
    for b in benchmark_data:
        assert b["benchmark_id"].startswith("bench_")
        assert b["candidate_id"]
        assert b["sample_id"]
        assert isinstance(b["client_line"], int) and b["client_line"] >= 1
        assert isinstance(b["column"], int) and b["column"] >= 0
        assert b["call_site_snippet"]
        assert b["enclosing_code"]
        assert b["target_api"]
        assert b["stratum"] in ("candidate_positive", "pipeline_miss", "label_anomaly", "hard_negative")
        assert b["expected_ground_truth"] in (True, False)
        assert b["annotation_status"] in ("unannotated", "pre_labeled")


def test_hard_negative_sub_strata_breakdown(benchmark_data):
    """Asserts exact 16/8/8/8 composition among all 40 fresh hard negatives."""
    hard_negs = [b for b in benchmark_data if b["stratum"] == "hard_negative"]
    assert len(hard_negs) == 40

    replacements = [b for b in hard_negs if "replacement_overload" in b.get("sub_stratum", "")]
    submodules = [b for b in hard_negs if b.get("sub_stratum") in ("submodule_non_deprecated_pinv", "misc_non_benchmark_symbol")]
    stdlib = [b for b in hard_negs if b.get("sub_stratum") == "stdlib_name_collision_product"]
    wrappers = [b for b in hard_negs if b.get("sub_stratum") == "wrapper_lookalike_pyspark"]

    assert len(replacements) == 16, f"Expected 16 replacements, found {len(replacements)}"
    assert len(submodules) == 8, f"Expected 8 submodules/misc, found {len(submodules)}"
    assert len(stdlib) == 8, f"Expected 8 stdlib collisions, found {len(stdlib)}"
    assert len(wrappers) == 8, f"Expected 8 wrappers, found {len(wrappers)}"
    assert len(replacements) + len(submodules) + len(stdlib) + len(wrappers) == 40


FROZEN_BENCHMARK_PATH = REPO_ROOT / "data" / "benchmark" / "ground_truth_benchmark.jsonl"
SECOND_PASS_PATH = REPO_ROOT / "data" / "benchmark" / "phase5_second_pass_blinded.jsonl"


def test_frozen_ground_truth_dataset():
    """Asserts that ground_truth_benchmark.jsonl is frozen with 150 non-null items and exact class counts."""
    assert FROZEN_BENCHMARK_PATH.exists(), f"Frozen benchmark not found at {FROZEN_BENCHMARK_PATH}"
    items = []
    with open(FROZEN_BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    assert len(items) == 150
    for b in items:
        assert b["is_deprecated_call"] in (True, False), f"Item {b['benchmark_id']} has invalid verdict"
        assert b["annotation_status"] in ("annotated", "pre_labeled")
        assert b["annotator"] in ("HUMAN_USER", "HUMAN_EXPERT_PRELABELED", "HUMAN_AUDITED_PRE_ESTABLISHED")

    dep_count = sum(1 for b in items if b["is_deprecated_call"] is True)
    ben_count = sum(1 for b in items if b["is_deprecated_call"] is False)
    assert dep_count == 107, f"Expected 107 True Deprecations, found {dep_count}"
    assert ben_count == 43, f"Expected 43 True Benign calls, found {ben_count}"


def test_second_pass_inter_annotator_agreement():
    """Asserts that 30-item blinded second-pass review achieved perfect or high inter-annotator concordance."""
    assert SECOND_PASS_PATH.exists(), f"Second pass file not found at {SECOND_PASS_PATH}"
    items = []
    with open(SECOND_PASS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    assert len(items) == 30
    agreements = [r for r in items if r.get("agreement") is True]
    concordance = len(agreements) / len(items)
    assert concordance >= 0.95, f"Observed concordance {concordance:.2%} below 95%"


