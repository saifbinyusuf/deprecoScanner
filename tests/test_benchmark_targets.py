"""
tests/test_benchmark_targets.py - Regression & Scope Tests for Benchmark Target Matching.

Guarantees:
1. Rejects modern replacement APIs (e.g. scipy.special.comb, scipy.integrate.simpson).
2. Rejects out-of-scope historical symbols (e.g. numpy.expand_dims, scipy.linalg.pinv).
3. Correctly resolves legitimate variations of the 31 canonical benchmark targets.
4. Prevents evidence retriever from fuzzy-matching replacement APIs to deprecated targets.
"""

from __future__ import annotations

import pytest

from src.resolution.benchmark_targets import (
    ALL_BENCHMARK_TARGETS,
    BENCHMARK_TARGET_APIS,
    match_benchmark_target,
)
from src.verification.evidence_retriever import EvidenceRetriever


def test_canonical_benchmark_targets_count_and_composition():
    assert len(ALL_BENCHMARK_TARGETS) == 31
    assert len(BENCHMARK_TARGET_APIS["numpy"]) == 3
    assert len(BENCHMARK_TARGET_APIS["scipy"]) == 18
    assert len(BENCHMARK_TARGET_APIS["pandas"]) == 10


def test_match_benchmark_target_rejects_special_comb_and_replacements():
    # Priority 1 bug regression: scipy.special.comb must NEVER map to scipy.misc.comb
    assert match_benchmark_target("scipy.special.comb") is None
    assert match_benchmark_target("scipy.special._basic.comb") is None
    assert match_benchmark_target("scipy.special._basic.comb::legacy") is None

    # Modern replacement APIs must never map to deprecated targets
    assert match_benchmark_target("scipy.integrate._quadrature.simpson") is None
    assert match_benchmark_target("scipy.integrate.simpson") is None
    assert match_benchmark_target("scipy.integrate._quadrature.cumulative_trapezoid") is None
    assert match_benchmark_target("scipy.integrate.cumulative_trapezoid") is None
    assert match_benchmark_target("pandas.core.frame.DataFrame.map") is None
    assert match_benchmark_target("pandas.DataFrame.map") is None


def test_match_benchmark_target_rejects_out_of_scope_historical_symbols():
    # Priority 2 leak symbols must be rejected
    assert match_benchmark_target("numpy.lib.shape_base.expand_dims") is None
    assert match_benchmark_target("numpy.expand_dims") is None
    assert match_benchmark_target("scipy.linalg._basic.pinv") is None
    assert match_benchmark_target("scipy.linalg.pinv") is None
    assert match_benchmark_target("numpy.core.records.fromstring.shape") is None
    assert match_benchmark_target("numpy.core.records.fromarrays.shape") is None
    assert match_benchmark_target("scipy.stats._continuous_distns.frechet_r_gen.sf") is None
    assert match_benchmark_target("numpy.lib.function_base.percentile") is None


def test_match_benchmark_target_accepts_valid_benchmark_targets():
    # Legitimate canonical targets and submodule variations
    assert match_benchmark_target("numpy.alltrue") == "numpy.alltrue"
    assert match_benchmark_target("numpy.product") == "numpy.product"
    assert match_benchmark_target("numpy.core.fromnumeric.product") == "numpy.product"
    assert match_benchmark_target("numpy.core.fromnumeric.cumproduct") == "numpy.cumproduct"
    assert match_benchmark_target("scipy.misc.comb") == "scipy.misc.comb"
    assert match_benchmark_target("scipy.misc.logsumexp") == "scipy.misc.logsumexp"
    assert match_benchmark_target("scipy.linalg.pinv2") == "scipy.linalg.pinv2"
    assert match_benchmark_target("pandas.DataFrame.iteritems") == "pandas.DataFrame.iteritems"
    assert match_benchmark_target("pandas.core.frame.DataFrame.iteritems") == "pandas.DataFrame.iteritems"
    assert match_benchmark_target("pandas.DataFrame.swapaxes") == "pandas.DataFrame.swapaxes"
    assert match_benchmark_target("pandas.io.formats.style.Styler.render") == "pandas.io.formats.style.Styler.render"


def test_evidence_retriever_does_not_confuse_special_comb_with_misc_comb():
    retriever = EvidenceRetriever()

    # Querying for deprecated target gets replacement
    ev_misc = retriever.get_evidence("scipy.misc.comb")
    assert ev_misc["recommended_replacement"] == "scipy.special.comb"

    # Querying for modern replacement does NOT get replacement or false warning
    ev_special = retriever.get_evidence("scipy.special.comb")
    assert ev_special["recommended_replacement"] is None
