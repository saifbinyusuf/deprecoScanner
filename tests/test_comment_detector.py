"""
test_comment_detector.py - Unit tests for Task 2.2: Hardening Fix #1 (single-line comments).

Verifies detection of bare '# deprecated' comments attached to functions or classes
using the tokenize module paired with AST node boundaries.
"""

from pathlib import Path
import pytest

from src.detectors.comment_detector import (
    detect_comment_deprecations,
    detect_comment_deprecations_from_file,
)


SYNTHETIC_BARE_COMMENT_SNIPPET = """
# Deprecated: use modern_func instead. Will be removed in v2.0.
def legacy_func(x):
    return x * 10

def active_func(y):
    # Just an implementation note: fast path
    return y + 1

# Not deprecated: this function is active and maintained
def maintained_func(z):
    return z ** 2

def inline_comment_func(a):  # deprecated in 1.5
    return a - 1
"""


def test_bare_comment_only_synthetic_case():
    """Verify that a bare-comment-only synthetic function is detected with origin 'comment'."""
    candidates = detect_comment_deprecations(SYNTHETIC_BARE_COMMENT_SNIPPET, filename="synthetic_comments.py")

    assert len(candidates) == 2, f"Expected 2 candidates, got {len(candidates)}: {candidates}"

    names = {c.qualified_name for c in candidates}
    assert names == {"legacy_func", "inline_comment_func"}

    for c in candidates:
        assert c.origin == "comment"
        assert "deprecated" in c.raw_evidence.lower()
        assert c.location.startswith("synthetic_comments.py:")


def test_negation_comments_not_flagged():
    """Verify that comments containing negations ('not deprecated') are not flagged."""
    code = """
# This API is not deprecated and remains supported
def supported_api(x):
    return x

# has not been deprecated in this version
def legacy_named_api(y):
    return y
"""
    candidates = detect_comment_deprecations(code, filename="negation.py")
    assert len(candidates) == 0, f"Expected 0 candidates, got {candidates}"


def test_class_level_comment_deprecation():
    """Verify comments attached to class definitions."""
    code = """
# DEPRECATED: Use ModernContainer instead
class LegacyContainer:
    def __init__(self, val):
        self.val = val
"""
    candidates = detect_comment_deprecations(code, filename="class_comment.py", package_prefix="mylib")
    assert len(candidates) == 1
    assert candidates[0].qualified_name == "mylib.LegacyContainer"
    assert candidates[0].origin == "comment"


def test_real_scipy_file_spot_check():
    """
    Spot-check against real SciPy files:
    1. scipy/sparse/_sputils.py uses comment-style deprecation on matrix wrappers.
    2. scipy/stats/tests/test_stats.py has a negation comment ('has not been deprecated')
       and should produce 0 false positives.
    """
    scipy_sparse_sputils = Path(".venv/lib/python3.13/site-packages/scipy/sparse/_sputils.py")
    if not scipy_sparse_sputils.exists():
        pytest.skip("SciPy not installed in .venv, skipping real-file check")

    # 1. Check sputils.py detects comment deprecation on matrix
    candidates = detect_comment_deprecations_from_file(scipy_sparse_sputils, package_prefix="scipy.sparse._sputils")
    assert len(candidates) >= 1
    matrix_cands = [c for c in candidates if "matrix" in c.qualified_name]
    assert len(matrix_cands) >= 1
    assert matrix_cands[0].origin == "comment"
    assert "deprecated" in matrix_cands[0].raw_evidence.lower()

    # 2. Check test_stats.py negation comment does not trigger false-positive explosion
    scipy_test_stats = Path(".venv/lib/python3.13/site-packages/scipy/stats/tests/test_stats.py")
    if scipy_test_stats.exists():
        test_cands = detect_comment_deprecations_from_file(scipy_test_stats, package_prefix="scipy.stats.tests")
        # Ensure test_rename_moment_order is NOT falsely flagged
        falsely_flagged = [c for c in test_cands if "test_rename_moment_order" in c.qualified_name]
        assert len(falsely_flagged) == 0, f"False positive detected: {falsely_flagged}"
