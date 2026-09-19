"""
tests/test_stage3_verifier.py - Unit tests for Stage 3 LLM Verification Stack.

Validates:
1. Provenance-branching prompt generation (Confirmation vs Inference mode).
2. Pydantic VerificationDecision JSON parsing and validation.
3. Grounding evidence retrieval across benchmark targets.
4. SQLite compound caching, compound SHA-256 key stability, and version busting.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.verification.verifier_prompt import (
    CandidateProvenance,
    PromptMode,
    VerificationDecision,
    format_verification_prompt,
    PROMPT_VERSION,
)
from src.verification.evidence_retriever import (
    EvidenceRetriever,
    CANONICAL_REPLACEMENTS,
    TARGET_BENCHMARK_GROUNDING,
)
from src.verification.response_cache import ResponseCache


def test_verifier_prompt_formatting_resolved():
    prov = CandidateProvenance(
        sample_id="scipy_0",
        target_api="scipy.misc.comb",
        library="scipy",
        call_site_snippet="scipy.misc.comb(count, 2)",
        line_number=42,
        enclosing_code="for count in series: scipy.misc.comb(count, 2)",
        recommended_replacement="scipy.special.comb",
        evidence_warning="Importing `comb` from scipy.misc is deprecated in scipy 1.0.0.",
        evidence_docstring=".. deprecated:: 1.0.0 Use scipy.special.comb instead.",
    )
    prompt = format_verification_prompt(prov, mode=PromptMode.CONFIRMATION)

    assert "Branch A: Confirmation Mode" in prompt
    assert "TARGET API: `scipy.misc.comb`" in prompt
    assert "Recommended Replacement: `scipy.special.comb`" in prompt
    assert "Fallback & Compatibility Guards" in prompt
    assert "Third-Party Lookalikes" in prompt
    assert f"PROMPT VERSION: {PROMPT_VERSION}" in prompt


def test_verifier_prompt_formatting_low_confidence():
    prov = CandidateProvenance(
        sample_id="pandas_0",
        target_api="pandas.io.formats.style.Styler.render",
        library="pandas",
        call_site_snippet="es.render()",
        line_number=15,
        enclosing_code="empty_df = DataFrame(); es = Styler(empty_df); es.render()",
        failure_reason="unresolved_receiver",
        failure_details="Receiver object 'es' could not be resolved by Jedi",
        recommended_replacement="pandas.io.formats.style.Styler.to_html",
    )
    prompt = format_verification_prompt(prov, mode=PromptMode.INFERENCE)

    assert "Branch B: Type Inference Mode" in prompt
    assert "STATIC RESOLUTION FAILURE REASON: unresolved_receiver" in prompt
    assert "Receiver object 'es' could not be resolved by Jedi" in prompt
    assert "Type & Receiver Inference" in prompt


def test_json_schema_validation():
    # Valid output
    valid_data = {
        "is_deprecated_usage": True,
        "confidence": 0.95,
        "rationale": "Direct invocation of deprecated function in active loop.",
    }
    decision = VerificationDecision(**valid_data)
    assert decision.is_deprecated_usage is True
    assert decision.confidence == 0.95

    # Invalid confidence range
    with pytest.raises(ValidationError):
        VerificationDecision(is_deprecated_usage=True, confidence=1.5, rationale="Too confident")

    # Missing required field
    with pytest.raises(ValidationError):
        VerificationDecision(is_deprecated_usage=True, rationale="Missing confidence")


def test_evidence_retriever_grounding():
    retriever = EvidenceRetriever()

    # High-frequency targets should have replacements and warnings/docstrings
    targets = [
        "scipy.misc.comb",
        "scipy.misc.logsumexp",
        "numpy.alltrue",
        "numpy.product",
        "pandas.DataFrame.iteritems",
        "pandas.io.formats.style.Styler.render",
    ]
    for t in targets:
        ev = retriever.get_evidence(t)
        assert ev["recommended_replacement"] is not None, f"Missing replacement for {t}"
        assert ev["warning"] is not None or ev["docstring"] is not None, f"Missing warning/docstring for {t}"


def test_response_cache_compound_key_and_invalidation():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_cache.db"
        cache = ResponseCache(db_path=db_path)

        key1 = cache.compute_cache_key(
            model_name="gemini-3.5-flash-lite",
            prompt_version="v1.0",
            call_site_text="scipy.misc.comb(count, 2)",
            api_name="scipy.misc.comb",
            evidence_snippet="evidence_1",
        )

        key2 = cache.compute_cache_key(
            model_name="gemini-pro-latest",  # Different model
            prompt_version="v1.0",
            call_site_text="scipy.misc.comb(count, 2)",
            api_name="scipy.misc.comb",
            evidence_snippet="evidence_1",
        )

        key3 = cache.compute_cache_key(
            model_name="gemini-3.5-flash-lite",
            prompt_version="v2.0",  # Different prompt version (busts cache)
            call_site_text="scipy.misc.comb(count, 2)",
            api_name="scipy.misc.comb",
            evidence_snippet="evidence_1",
        )

        # All compound keys must be distinct
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

        # Put entry under key1
        cache.put(
            model_name="gemini-3.5-flash-lite",
            prompt_version="v1.0",
            call_site_text="scipy.misc.comb(count, 2)",
            api_name="scipy.misc.comb",
            evidence_snippet="evidence_1",
            response_json='{"is_deprecated_usage": true, "confidence": 1.0, "rationale": "deprecated"}',
            prompt_tokens=100,
            candidate_tokens=20,
            total_tokens=120,
        )

        # Test cache hit on identical query
        res = cache.get(
            model_name="gemini-3.5-flash-lite",
            prompt_version="v1.0",
            call_site_text="scipy.misc.comb(count, 2)",
            api_name="scipy.misc.comb",
            evidence_snippet="evidence_1",
        )
        assert res is not None
        assert res["cached"] is True
        assert res["response"]["is_deprecated_usage"] is True

        # Test cache miss on different model
        res_pro = cache.get(
            model_name="gemini-pro-latest",
            prompt_version="v1.0",
            call_site_text="scipy.misc.comb(count, 2)",
            api_name="scipy.misc.comb",
            evidence_snippet="evidence_1",
        )
        assert res_pro is None

        # Test cache miss on bumped prompt version
        res_v2 = cache.get(
            model_name="gemini-3.5-flash-lite",
            prompt_version="v2.0",
            call_site_text="scipy.misc.comb(count, 2)",
            api_name="scipy.misc.comb",
            evidence_snippet="evidence_1",
        )
        assert res_v2 is None

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["total_records"] == 1

        cache.close()
