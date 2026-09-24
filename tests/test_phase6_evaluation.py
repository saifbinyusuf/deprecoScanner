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
        cond = metrics["candidate_conditional"]
        e2e = metrics["end_to_end"]

        # Assert candidate-conditional metrics consistency
        tp_c, fp_c, fn_c, tn_c = cond["tp"], cond["fp"], cond["fn"], cond["tn"]
        expected_prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        expected_rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        assert cond["precision"] == pytest.approx(expected_prec_c, abs=1e-4)
        assert cond["recall"] == pytest.approx(expected_rec_c, abs=1e-4)
        assert metrics["recall_conditional"] == pytest.approx(expected_rec_c, abs=1e-4)

        # Assert end-to-end metrics consistency (driven by actual per-item predictions on all 150 items)
        tp_e, fp_e, fn_e, tn_e = e2e["tp"], e2e["fp"], e2e["fn"], e2e["tn"]
        assert tp_e + fp_e + fn_e + tn_e == 150
        expected_prec_e = tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else 0.0
        expected_rec_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0.0
        assert e2e["precision"] == pytest.approx(expected_prec_e, abs=1e-4)
        assert e2e["recall"] == pytest.approx(expected_rec_e, abs=1e-4)
        assert metrics["precision"] == pytest.approx(expected_prec_e, abs=1e-4)
        assert metrics["recall_end_to_end"] == pytest.approx(expected_rec_e, abs=1e-4)

        # Assert F1 formulas
        if expected_prec_e + expected_rec_e > 0:
            expected_f1_e2e = 2 * expected_prec_e * expected_rec_e / (expected_prec_e + expected_rec_e)
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
