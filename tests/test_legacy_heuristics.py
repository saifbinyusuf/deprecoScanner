"""
test_legacy_heuristics.py - Unit tests for Task 2.1 legacy heuristics.

Verifies detection of APIScanner's three legacy deprecation heuristics:
1. Decorator matcher (@deprecated, @deprecate, etc.)
2. Warning call matcher (warnings.warn(...) with DeprecationWarning/FutureWarning)
3. Docstring keyword matcher (paragraphs containing 'deprecat*')
"""

import pytest
from src.detectors.legacy_heuristics import (
    DeprecationCandidate,
    detect_legacy_deprecations,
)


SYNTHETIC_SNIPPET = """
import warnings

@deprecated("Use modern_func() instead.")
def func_with_decorator(x):
    return x * 2

def func_with_warning(y):
    warnings.warn("func_with_warning is deprecated, use modern_y()", DeprecationWarning)
    return y + 1

def func_with_docstring(z):
    \"\"\"
    Calculate something.

    .. deprecated:: 1.4.0
       func_with_docstring is deprecated. Please use modern_z() instead.
    \"\"\"
    return z ** 2

def normal_function(w):
    \"\"\"Standard clean function.\"\"\"
    return w
"""


def test_all_three_legacy_patterns_detected():
    """Verify that a synthetic snippet containing all three patterns detects all three."""
    candidates = detect_legacy_deprecations(SYNTHETIC_SNIPPET, filename="synthetic.py")

    assert len(candidates) == 3, f"Expected exactly 3 candidates, got {len(candidates)}: {candidates}"

    origins = {c.origin for c in candidates}
    assert origins == {"decorator", "warning", "docstring"}, f"Expected all three origins, got {origins}"

    by_origin = {c.origin: c for c in candidates}

    # 1. Decorator candidate
    dec_c = by_origin["decorator"]
    assert dec_c.qualified_name == "func_with_decorator"
    assert "@deprecated" in dec_c.raw_evidence
    assert dec_c.location.startswith("synthetic.py:")

    # 2. Warning candidate
    warn_c = by_origin["warning"]
    assert warn_c.qualified_name == "func_with_warning"
    assert "DeprecationWarning" in warn_c.raw_evidence
    assert "func_with_warning is deprecated" in warn_c.raw_evidence
    assert warn_c.location.startswith("synthetic.py:")

    # 3. Docstring candidate
    doc_c = by_origin["docstring"]
    assert doc_c.qualified_name == "func_with_docstring"
    assert "deprecated" in doc_c.raw_evidence.lower()
    assert doc_c.location.startswith("synthetic.py:")


def test_single_function_with_all_three_patterns():
    """Verify that a function with all three patterns yields candidates for each origin."""
    code = """
import warnings

@deprecated("Triple pattern")
def triple_deprecated():
    \"\"\"
    This function is deprecated in docstring.
    \"\"\"
    warnings.warn("Triple warning", DeprecationWarning)
    return 42
"""
    candidates = detect_legacy_deprecations(code, filename="triple.py", package_prefix="mypkg")
    assert len(candidates) == 3

    origins = {c.origin for c in candidates}
    assert origins == {"decorator", "warning", "docstring"}
    for c in candidates:
        assert c.qualified_name == "mypkg.triple_deprecated"


def test_class_and_method_scoping():
    """Verify qualified names for classes and methods."""
    code = """
class OldClass:
    \"\"\"Deprecated class.\"\"\"
    def old_method(self):
        import warnings
        warnings.warn("old_method is deprecated", FutureWarning)
"""
    candidates = detect_legacy_deprecations(code, filename="classes.py", package_prefix="lib")
    assert len(candidates) == 2

    by_origin = {c.origin: c for c in candidates}
    assert "docstring" in by_origin
    assert by_origin["docstring"].qualified_name == "lib.OldClass"

    assert "warning" in by_origin
    assert by_origin["warning"].qualified_name == "lib.OldClass.old_method"


def test_benign_code_produces_no_candidates():
    """Verify clean code does not trigger false positives."""
    code = """
import warnings

def regular_function(x):
    \"\"\"Computes regular value.\"\"\"
    if x < 0:
        warnings.warn("Negative value encountered", UserWarning)
    return x * 10
"""
    candidates = detect_legacy_deprecations(code, filename="clean.py")
    assert len(candidates) == 0


def test_custom_warning_subclasses_detected():
    """
    Verify that ANY subclass of DeprecationWarning or FutureWarning is recognized,
    even with an arbitrary name not containing 'deprecat' or 'future', both direct
    and transitive.
    """
    code = """
import warnings

class ObsoleteAlert(DeprecationWarning):
    pass

class SubAlert(ObsoleteAlert):
    pass

def custom_warn_func(x):
    warnings.warn("something changed", SubAlert)
    return x
"""
    candidates = detect_legacy_deprecations(code, filename="custom_warn.py")
    assert len(candidates) == 1, f"Expected 1 candidate, got {len(candidates)}: {candidates}"
    assert candidates[0].qualified_name == "custom_warn_func"
    assert candidates[0].origin == "warning"
    assert "SubAlert" in candidates[0].raw_evidence
