"""
tests/test_prompt_v2_wrappers.py - Regression test for Prompt v2 Receiver Rule.

Verifies:
1. Prompt v2 contains the explicit receiver rule.
2. Cache keys for v2 differ from v1 (no stale cache contamination).
3. Zero benchmark items are included in fresh wrapper evaluations.
"""

import json
from pathlib import Path
import pytest

from src.verification.verifier_prompt import (
    CandidateProvenance,
    PromptMode,
    PROMPT_VERSION,
    PROMPT_VERSION_V2,
    format_verification_prompt,
)
from src.verification.response_cache import ResponseCache

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_prompt_v2_cache_key_isolation():
    """Verify that changing prompt version alters the SHA-256 cache key."""
    call_site = "modin_df.swapaxes(axis1, axis2)"
    api = "pandas.DataFrame.swapaxes"
    evidence = "Deprecated in 2.1.0."

    key_v1 = ResponseCache.compute_cache_key("gemini-3.5-flash-lite", PROMPT_VERSION, call_site, api, evidence)
    key_v2 = ResponseCache.compute_cache_key("gemini-3.5-flash-lite", PROMPT_VERSION_V2, call_site, api, evidence)

    assert key_v1 != key_v2
    assert len(key_v1) == 64
    assert len(key_v2) == 64


def test_prompt_v2_text_contains_receiver_rule():
    """Verify that prompt v2 explicitly instructs on native vs wrapper receivers in comparison tests."""
    prov = CandidateProvenance(
        sample_id="test_sample",
        target_api="pandas.DataFrame.last",
        library="pandas",
        call_site_snippet="pdf.last('3D')",
        enclosing_code="assert_eq(pdf.last('3D'), modin_df.last('3D'))",
    )

    prompt = format_verification_prompt(prov, PromptMode.CONFIRMATION, prompt_version=PROMPT_VERSION_V2)
    assert "PROMPT VERSION: v2.0" in prompt
    assert "native-library receiver (pandas) is a deprecated usage" in prompt
    assert "third-party wrapper receiver (such as PySpark/Koalas, Modin, Dask, cuDF) is a benign lookalike" in prompt
    assert "evaluate the flagged call site strictly on its own receiver" in prompt


def test_fresh_wrapper_eval_excludes_benchmark():
    """Verify that results/prompt_v2_fresh_wrapper_eval.json has 0 benchmark items."""
    eval_file = REPO_ROOT / "results" / "prompt_v2_fresh_wrapper_eval.json"
    bench_file = REPO_ROOT / "data" / "benchmark" / "ground_truth_benchmark_v2.jsonl"

    assert eval_file.exists()
    assert bench_file.exists()

    with open(bench_file, encoding="utf-8") as f:
        bench_sample_ids = {json.loads(line)["sample_id"] for line in f}

    with open(eval_file, encoding="utf-8") as f:
        eval_data = json.load(f)

    for rec in eval_data["records"]:
        assert rec["sample_id"] not in bench_sample_ids
