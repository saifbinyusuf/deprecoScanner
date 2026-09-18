"""
tests/test_jedi_resolver.py - Verification for Task 3.1: Jedi Call-Site Resolution.

Tests the three mandatory guideline edge cases:
1. Aliased import (`import numpy as np; np.alltrue(...)`)
2. Subclass override and inheritance (`class MyDF(pd.DataFrame): ...; df.iteritems()`)
3. Wildcard import (`from numpy import *; alltrue(...)`)
Plus negative controls (benign, non-deprecated APIs) and catalog matching.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.resolution.jedi_resolver import JediResolver, build_historical_project, resolve_call_site

BENCHMARK_TARGETS = [
    "numpy.alltrue",
    "numpy.product",
    "numpy.cumproduct",
    "pandas.DataFrame.iteritems",
    "pandas.Series.iteritems",
    "pandas.DataFrame.swapaxes",
    "scipy.misc.comb",
    "scipy.misc.logsumexp",
]


def find_pos(code: str, target: str) -> tuple[int, int]:
    """Finds 1-indexed line and 0-indexed column of target substring."""
    for line_no, line in enumerate(code.splitlines(), 1):
        if target in line:
            col = line.index(target)
            return line_no, col
    raise ValueError(f"Target '{target}' not found in code snippet")


@pytest.fixture(scope="module")
def resolver():
    return JediResolver(catalog_symbols=BENCHMARK_TARGETS)


def test_edge_case_1_aliased_import(resolver: JediResolver):
    """Guideline Edge Case 1: Aliased import (`import numpy as np`)."""
    code = """import numpy as np

def compute(arr):
    return np.alltrue(arr)
"""
    line, col = find_pos(code, "alltrue")
    res = resolver.resolve(code, line=line, column=col)
    assert res is not None, "Failed to resolve np.alltrue call site"
    assert "alltrue" in res.name
    assert "numpy" in res.qualified_name
    assert res.is_deprecated is True
    assert res.matched_catalog_symbol == "numpy.alltrue"


def test_edge_case_2_subclass_inheritance(resolver: JediResolver):
    """Guideline Edge Case 2a: Subclass inheritance without override (`class CustomDF(pd.DataFrame)`)."""
    code = """import pandas as pd

class CustomDataFrame(pd.DataFrame):
    pass

def iterate(df: CustomDataFrame):
    for k, v in df.iteritems():
        pass
"""
    line, col = find_pos(code, "iteritems")
    res = resolver.resolve(code, line=line, column=col)
    assert res is not None, "Failed to resolve df.iteritems on custom subclass"
    assert res.name == "iteritems"
    assert "pandas" in res.qualified_name
    assert res.is_deprecated is True
    assert res.matched_catalog_symbol == "pandas.DataFrame.iteritems"


def test_edge_case_2_subclass_override_with_super(resolver: JediResolver):
    """Guideline Edge Case 2b: Subclass method override delegating to super()."""
    code = """import pandas as pd

class CustomDataFrame(pd.DataFrame):
    def iteritems(self):
        return super().iteritems()
"""
    line, col = find_pos(code, "super().iteritems")
    # Position on 'iteritems' inside super().iteritems
    col += len("super().")
    res = resolver.resolve(code, line=line, column=col)
    assert res is not None, "Failed to resolve super().iteritems in subclass override"
    assert res.name == "iteritems"
    assert "pandas" in res.qualified_name
    assert res.is_deprecated is True
    assert res.matched_catalog_symbol == "pandas.DataFrame.iteritems"


def test_edge_case_3_wildcard_import(resolver: JediResolver):
    """Guideline Edge Case 3: Wildcard import (`from numpy import *`)."""
    code = """from numpy import *

def check_mask(mask):
    return alltrue(mask)
"""
    line, col = find_pos(code, "alltrue")
    res = resolver.resolve(code, line=line, column=col)
    assert res is not None, "Failed to resolve wildcard imported alltrue"
    assert res.name == "alltrue"
    assert "numpy" in res.qualified_name
    assert res.is_deprecated is True
    assert res.matched_catalog_symbol == "numpy.alltrue"


def test_benign_call_site_not_flagged(resolver: JediResolver):
    """Negative control: Non-deprecated API is resolved but not matched as deprecated."""
    code = """import numpy as np

def make_array():
    return np.zeros((3, 3))
"""
    line, col = find_pos(code, "zeros")
    res = resolver.resolve(code, line=line, column=col)
    assert res is not None, "Failed to resolve np.zeros"
    assert res.name == "zeros"
    assert res.is_deprecated is False
    assert res.matched_catalog_symbol is None


def test_scipy_submodule_import(resolver: JediResolver):
    """Submodule import: scipy.misc.comb."""
    code = """import scipy.misc as sm

def combinations(n, k):
    return sm.comb(n, k)
"""
    line, col = find_pos(code, "sm.comb")
    col += len("sm.")
    res = resolver.resolve(code, line=line, column=col)
    assert res is not None, "Failed to resolve sm.comb"
    assert "comb" in res.name
    assert res.is_deprecated is True
    assert res.matched_catalog_symbol == "scipy.misc.comb"
