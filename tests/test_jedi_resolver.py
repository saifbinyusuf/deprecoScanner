"""
tests/test_jedi_resolver.py - Verification for Jedi Call-Site Resolution.

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


# =========================================================================
# Low-Confidence Bucket & Failure Logging Tests
# =========================================================================


def test_low_confidence_unresolved_receiver(resolver: JediResolver):
    """Untyped receiver yields unresolved_receiver in low_confidence."""
    code = """import pandas as pd

def iterate(df):
    for k, v in df.iteritems():
        pass
"""
    result = resolver.analyze_client_snippet(code)
    assert len(result.resolved_deprecated) == 0
    assert len(result.low_confidence) == 1

    lc = result.low_confidence[0]
    assert lc.callee_name == "iteritems"
    assert lc.failure_reason == "unresolved_receiver"
    assert lc.matched_catalog_symbol == "pandas.DataFrame.iteritems"
    assert "df.iteritems()" in lc.call_site_snippet


def test_low_confidence_empty_goto_unbound_name(resolver: JediResolver):
    """Unbound name without import yields empty_goto in low_confidence."""
    code = """def compute(x):
    return alltrue(x)
"""
    result = resolver.analyze_client_snippet(code)
    assert len(result.resolved_deprecated) == 0
    assert len(result.low_confidence) == 1

    lc = result.low_confidence[0]
    assert lc.callee_name == "alltrue"
    assert lc.failure_reason == "empty_goto"
    assert lc.matched_catalog_symbol == "numpy.alltrue"


def test_low_confidence_dynamic_dispatch(resolver: JediResolver):
    """Dynamic getattr invocation yields dynamic_dispatch in low_confidence."""
    code = """import numpy as np

def compute(arr):
    fn = getattr(np, "alltrue")
    return fn(arr)
"""
    # Also test direct getattr invocation
    code_direct = """import numpy as np

def compute(arr):
    return getattr(np, "alltrue")(arr)
"""
    result = resolver.analyze_client_snippet(code_direct)
    assert len(result.low_confidence) >= 1
    lc = [c for c in result.low_confidence if c.failure_reason == "dynamic_dispatch"][0]
    assert lc.callee_name == "alltrue"
    assert lc.matched_catalog_symbol == "numpy.alltrue"


def test_low_confidence_syntax_error(resolver: JediResolver):
    """Malformed syntax yields syntax_error in low_confidence."""
    code = """def broken(:
    return 1
"""
    result = resolver.analyze_client_snippet(code)
    assert len(result.low_confidence) == 1
    assert result.low_confidence[0].failure_reason == "syntax_error"


def test_stage2_resolves_deprecated_and_benign_in_single_snippet(resolver: JediResolver):
    """Cleanly separates resolved deprecated, resolved benign, and low confidence."""
    code = """import numpy as np

def process(arr):
    a = np.alltrue(arr)
    b = np.zeros((2, 2))
    return a, b
"""
    # Add np.zeros to catalog as non-deprecated check: zeros is NOT in BENCHMARK_TARGETS
    result = resolver.analyze_client_snippet(code)
    assert len(result.resolved_deprecated) == 1
    assert result.resolved_deprecated[0].matched_catalog_symbol == "numpy.alltrue"
    assert len(result.low_confidence) == 0


def test_normalize_snippet_indentation_mixed_tabs_spaces(resolver: JediResolver):
    """
    Verify normalize_snippet_indentation recovers method snippets with
    leading space on line 1 and tabs in body (the upstream scraping artifact).
    """
    from src.resolution.jedi_resolver import normalize_snippet_indentation

    # The exact pattern from numpy_84 and scipy_2069
    snippet = " def compute(*args):\n\t\timport numpy as np\n\t\treturn np.alltrue(*args)\n"
    
    # Raw textwrap.dedent fails on this pattern due to no common prefix
    import textwrap, ast
    raw_dedent = textwrap.dedent(snippet)
    has_raw_syntax_error = False
    try:
        ast.parse(raw_dedent)
    except SyntaxError:
        has_raw_syntax_error = True
    assert has_raw_syntax_error is True

    # normalize_snippet_indentation recovers and parses cleanly
    cleaned = normalize_snippet_indentation(snippet)
    ast.parse(cleaned)  # must not raise SyntaxError

    # analyze_client_snippet resolves the call site rather than falling back to syntax_error
    result = resolver.analyze_client_snippet(snippet)
    assert len(result.resolved_deprecated) == 1
    assert result.resolved_deprecated[0].matched_catalog_symbol == "numpy.alltrue"
    assert len(result.low_confidence) == 0

