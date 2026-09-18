"""
tests/test_pep702_detector.py - Test suite for Task 2.4 PEP 702 deprecation detection.

Verifies:
1. Path B (Griffe + griffe-warnings-deprecated): declaration-site metadata extraction.
2. Path A (Mypy and Pyright): diagnostic-based call-site detection.
3. Unified detection and side-by-side consistency between both paths.
4. Support for both stdlib warnings.deprecated (3.13+) and typing_extensions.deprecated.
"""

import tempfile
from pathlib import Path
import pytest

from src.detectors.pep702_detector import (
    detect_pep702_via_griffe,
    detect_pep702_via_mypy,
    detect_pep702_via_pyright,
    detect_pep702_via_typechecker,
    detect_pep702_deprecations,
)


SYNTHETIC_CODE = '''
import warnings
from typing_extensions import deprecated

@deprecated("use new_foo() instead")
def old_foo():
    pass

@warnings.deprecated("use new_calc() instead")
def old_calc():
    pass

class DataProcessor:
    @deprecated("use compute() instead")
    def process_data(self):
        pass

def new_foo():
    pass

def new_calc():
    pass

def run():
    old_foo()
    old_calc()
    dp = DataProcessor()
    dp.process_data()
'''


def test_griffe_extraction_stdlib_and_typing_extensions():
    """Path B: Verify Griffe detects both @warnings.deprecated and @typing_extensions.deprecated."""
    candidates = detect_pep702_via_griffe(SYNTHETIC_CODE, filename="sample.py")
    
    by_name = {c.qualified_name: c for c in candidates}
    assert "old_foo" in by_name
    assert "old_calc" in by_name
    assert "DataProcessor.process_data" in by_name

    # Check candidate properties
    cand_foo = by_name["old_foo"]
    assert cand_foo.origin == "pep702:griffe"
    assert "pep702" in cand_foo.origins
    assert cand_foo.line == 5
    assert "use new_foo() instead" in cand_foo.raw_evidence
    assert cand_foo.scope == "function"

    cand_calc = by_name["old_calc"]
    assert cand_calc.origin == "pep702:griffe"
    assert "pep702" in cand_calc.origins
    assert cand_calc.line == 9
    assert "use new_calc() instead" in cand_calc.raw_evidence

    cand_proc = by_name["DataProcessor.process_data"]
    assert cand_proc.origin == "pep702:griffe"
    assert "pep702" in cand_proc.origins
    assert cand_proc.line == 14
    assert "use compute() instead" in cand_proc.raw_evidence


def test_griffe_class_level_deprecation():
    """Path B: Verify class-level @warnings.deprecated is assigned class scope."""
    code = '''
import warnings

@warnings.deprecated("LegacyModel is deprecated, use ModernModel")
class LegacyModel:
    pass
'''
    candidates = detect_pep702_via_griffe(code, filename="models.py")
    assert len(candidates) == 1
    c = candidates[0]
    assert c.qualified_name == "LegacyModel"
    assert c.origin == "pep702:griffe"
    assert "pep702" in c.origins
    assert c.scope == "class"
    assert "LegacyModel is deprecated" in c.raw_evidence


def test_mypy_diagnostics():
    """Path A: Verify Mypy detects call-site PEP 702 deprecations."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tf:
        tf.write(SYNTHETIC_CODE)
        p = Path(tf.name)

    try:
        candidates = detect_pep702_via_mypy(p)
        names = {c.qualified_name for c in candidates}
        assert "old_foo" in names
        assert "old_calc" in names
        assert "DataProcessor.process_data" in names

        for c in candidates:
            assert c.origin == "pep702:mypy"
            assert "pep702" in c.origins
            assert "mypy [deprecated]" in c.raw_evidence
    finally:
        p.unlink(missing_ok=True)


def test_pyright_diagnostics():
    """Path A: Verify Pyright detects call-site PEP 702 deprecations."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tf:
        tf.write(SYNTHETIC_CODE)
        p = Path(tf.name)

    try:
        candidates = detect_pep702_via_pyright(p)
        names = {c.qualified_name for c in candidates}
        assert "old_foo" in names
        assert "old_calc" in names
        assert "DataProcessor.process_data" in names

        for c in candidates:
            assert c.origin == "pep702:pyright"
            assert "pep702" in c.origins
            assert "pyright [reportDeprecated]" in c.raw_evidence
    finally:
        p.unlink(missing_ok=True)


def test_side_by_side_consistency():
    """
    Task 2.4 Gate Requirement:
    Both Path A (typechecker) and Path B (Griffe) identify the same deprecated symbols
    and extract matching deprecation messages.
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tf:
        tf.write(SYNTHETIC_CODE)
        p = Path(tf.name)

    try:
        griffe_cands = detect_pep702_via_griffe(p)
        mypy_cands = detect_pep702_via_mypy(p)
        pyright_cands = detect_pep702_via_pyright(p)

        griffe_symbols = {c.qualified_name for c in griffe_cands}
        mypy_symbols = {c.qualified_name for c in mypy_cands}
        pyright_symbols = {c.qualified_name for c in pyright_cands}

        # All paths agree on the exact set of deprecated symbols
        expected_symbols = {"old_foo", "old_calc", "DataProcessor.process_data"}
        assert griffe_symbols == expected_symbols
        assert mypy_symbols == expected_symbols
        assert pyright_symbols == expected_symbols

        # All paths extract the message
        griffe_msgs = {c.qualified_name: c.raw_evidence for c in griffe_cands}
        mypy_msgs = {c.qualified_name: c.raw_evidence for c in mypy_cands}
        pyright_msgs = {c.qualified_name: c.raw_evidence for c in pyright_cands}

        assert "use new_foo() instead" in griffe_msgs["old_foo"]
        assert "use new_foo() instead" in mypy_msgs["old_foo"]
        assert "use new_foo() instead" in pyright_msgs["old_foo"]

        assert "use new_calc() instead" in griffe_msgs["old_calc"]
        assert "use new_calc() instead" in mypy_msgs["old_calc"]
        assert "use new_calc() instead" in pyright_msgs["old_calc"]

        assert "use compute() instead" in griffe_msgs["DataProcessor.process_data"]
        assert "use compute() instead" in mypy_msgs["DataProcessor.process_data"]
        assert "use compute() instead" in pyright_msgs["DataProcessor.process_data"]

        # Exact normalized reason message match via candidate.message field
        g_clean = {c.qualified_name: c.message for c in griffe_cands}
        m_clean = {c.qualified_name: c.message for c in mypy_cands}
        p_clean = {c.qualified_name: c.message for c in pyright_cands}
        assert g_clean["old_foo"] == m_clean["old_foo"] == p_clean["old_foo"] == "use new_foo() instead"
        assert g_clean["old_calc"] == m_clean["old_calc"] == p_clean["old_calc"] == "use new_calc() instead"
        assert g_clean["DataProcessor.process_data"] == m_clean["DataProcessor.process_data"] == p_clean["DataProcessor.process_data"] == "use compute() instead"
    finally:
        p.unlink(missing_ok=True)


def test_package_prefix_propagation():
    """Verify package_prefix propagates cleanly into qualified names."""
    code = '''
import warnings

@warnings.deprecated("use new_func()")
def legacy_func():
    pass
'''
    candidates = detect_pep702_via_griffe(code, filename="legacy.py", package_prefix="my_pkg.sub")
    assert len(candidates) == 1
    assert candidates[0].qualified_name == "my_pkg.sub.legacy_func"


def test_syntax_error_resilience():
    """Verify malformed Python does not crash the detectors."""
    bad_code = "def broken(:"
    assert detect_pep702_via_griffe(bad_code) == []
    assert detect_pep702_via_mypy(bad_code) == []
    assert detect_pep702_via_pyright(bad_code) == []


def test_griffe_category_none_extraction():
    """
    Verify Griffe correctly extracts deprecation message when category=None,
    matching the real-world Pydantic BaseModel.dict pattern.
    """
    code = '''
from typing_extensions import deprecated

@deprecated("The `dict` method is deprecated; use `model_dump` instead.", category=None)
def dict_method():
    pass
'''
    candidates = detect_pep702_via_griffe(code, filename="models.py")
    assert len(candidates) == 1
    c = candidates[0]
    assert c.qualified_name == "dict_method"
    assert c.origin == "pep702:griffe"
    assert "pep702" in c.origins
    assert c.message == "The `dict` method is deprecated; use `model_dump` instead."

