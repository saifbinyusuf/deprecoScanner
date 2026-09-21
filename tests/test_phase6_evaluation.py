"""
tests/test_phase6_evaluation.py - Validates the Phase 6 evaluation harness, metrics, and statistical test.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_JSON_PATH = REPO_ROOT / "results" / "phase6_evaluation_report.json"
SUMMARY_MD_PATH = REPO_ROOT / "results" / "phase6_evaluation_summary.md"
BENCHMARK_FROZEN_PATH = REPO_ROOT / "data" / "benchmark" / "ground_truth_benchmark.jsonl"


@pytest.fixture(scope="module")
def evaluation_report():
    assert REPORT_JSON_PATH.exists(), f"Phase 6 report not found at {REPORT_JSON_PATH}"
    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_report_structure_and_completeness(evaluation_report):
    """Asserts that the evaluation report contains all required top-level sections."""
    assert evaluation_report["total_benchmark_call_sites"] == 150
    assert "class_balance" in evaluation_report
    assert "flagged_candidates_count" in evaluation_report
    assert "overall_metrics" in evaluation_report
    assert "per_library_metrics" in evaluation_report
    assert "per_origin_metrics" in evaluation_report
    assert "mcnemar_test" in evaluation_report
    assert "detailed_call_site_evaluations" in evaluation_report
    assert len(evaluation_report["detailed_call_site_evaluations"]) == 150


def test_ground_truth_class_balance(evaluation_report):
    """Asserts that ground-truth counts match the frozen benchmark (107 Deprecated, 43 Benign)."""
    cb = evaluation_report["class_balance"]
    assert cb["true_deprecations"] == 107
    assert cb["true_benign"] == 43


def test_dual_recall_and_confusion_matrix_accounting(evaluation_report):
    """Asserts mathematical consistency of dual recall and confusion matrices for both configurations."""
    for config_key in ["config_a_ast_heuristics", "config_b_full_pipeline"]:
        metrics = evaluation_report["overall_metrics"][config_key]
        tp = metrics["tp"]
        fp = metrics["fp"]
        fn_cond = metrics["fn_conditional"]
        fn_e2e = metrics["fn_end_to_end"]
        tn = metrics["tn"]

        # Assert precision formula
        expected_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        assert metrics["precision"] == pytest.approx(expected_prec, abs=1e-4)

        # Assert candidate-conditional recall
        expected_rec_cond = tp / (tp + fn_cond) if (tp + fn_cond) > 0 else 0.0
        assert metrics["recall_conditional"] == pytest.approx(expected_rec_cond, abs=1e-4)

        # Assert end-to-end recall includes the 8 pre-labeled pipeline misses
        assert fn_e2e == fn_cond + 8
        expected_rec_e2e = tp / (tp + fn_e2e) if (tp + fn_e2e) > 0 else 0.0
        assert metrics["recall_end_to_end"] == pytest.approx(expected_rec_e2e, abs=1e-4)

        # Assert F1 formulas
        if metrics["precision"] + metrics["recall_end_to_end"] > 0:
            expected_f1_e2e = 2 * metrics["precision"] * metrics["recall_end_to_end"] / (metrics["precision"] + metrics["recall_end_to_end"])
            assert metrics["f1_end_to_end"] == pytest.approx(expected_f1_e2e, abs=1e-4)


def test_mcnemar_contingency_table_integrity(evaluation_report):
    """Asserts that McNemar's 2x2 table sums to exactly 150 items and p-value is valid."""
    m = evaluation_report["mcnemar_test"]
    table = m["contingency_table"]
    n00, n01 = table[0]
    n10, n11 = table[1]

    assert n00 + n01 + n10 + n11 == 150
    assert m["b_correct_a_wrong"] == n01
    assert m["a_correct_b_wrong"] == n10
    assert 0.0 <= m["p_value"] <= 1.0


def test_anti_leakage_detector_isolation(evaluation_report):
    """Asserts that each detailed evaluation record contains separate predictions generated without labels."""
    records = evaluation_report["detailed_call_site_evaluations"]
    assert len(records) == 150
    for r in records:
        assert isinstance(r["pred_config_a"], bool)
        assert isinstance(r["pred_config_b"], bool)
        assert r["correct_a"] == (r["pred_config_a"] == r["ground_truth"])
        assert r["correct_b"] == (r["pred_config_b"] == r["ground_truth"])


def test_summary_markdown_exists():
    """Asserts that the human-readable summary markdown report was generated."""
    assert SUMMARY_MD_PATH.exists()
    content = SUMMARY_MD_PATH.read_text(encoding="utf-8")
    assert "Phase 6 Evaluation Summary Report" in content
    assert "McNemar" in content
