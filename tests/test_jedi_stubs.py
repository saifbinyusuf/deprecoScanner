"""
tests/test_jedi_stubs.py - Verification for Jedi Resolution via Historical Stubs & Paths.

Validates:
1. Jedi resolves top-level re-exports (e.g. np.alltrue) via __init__.pyi stubs
   to the historical implementation module (numpy.core.fromnumeric.alltrue).
2. Symbol compatibility engine correctly associates the resolved symbol with the benchmark target API.
3. Jedi resolves pure-Python methods (e.g. df.iteritems) directly from historical snapshots.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import jedi

from src.detectors.union_dedup import _is_symbol_compatible


BENCHMARK_LIBS_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmark_libs"


def test_jedi_resolves_numpy_alltrue_via_stub():
    stubs_dir = BENCHMARK_LIBS_DIR / "stubs"
    numpy_dir = BENCHMARK_LIBS_DIR / "numpy-1.26.4"

    if not (numpy_dir / "numpy").exists() or not (stubs_dir / "numpy" / "__init__.pyi").exists():
        pytest.skip("Historical NumPy 1.26.4 or stubs not yet extracted")

    code = """
import numpy as np

def check(x):
    return np.alltrue(x)
"""
    default_sys_path = jedi.get_default_environment().get_sys_path()
    clean_sys_path = [str(stubs_dir), str(numpy_dir)] + [p for p in default_sys_path if "site-packages" not in p]
    proj = jedi.Project(path=".", sys_path=clean_sys_path)
    script = jedi.Script(code, project=proj)

    defs = script.goto(5, 16)
    assert len(defs) > 0, "Jedi failed to resolve np.alltrue via stub"
    resolved_full_name = defs[0].full_name
    assert "alltrue" in resolved_full_name

    # Confirm compatibility with benchmark target name
    assert _is_symbol_compatible("numpy.alltrue", resolved_full_name)


def test_jedi_resolves_pandas_dataframe_iteritems():
    pandas_dir = BENCHMARK_LIBS_DIR / "pandas-1.5.3"

    if not (pandas_dir / "pandas").exists():
        pytest.skip("Historical Pandas 1.5.3 not yet extracted")

    code = """
import pandas as pd

df = pd.DataFrame({"a": [1, 2]})
for col, series in df.iteritems():
    pass
"""
    default_sys_path = jedi.get_default_environment().get_sys_path()
    clean_sys_path = [str(pandas_dir)] + [p for p in default_sys_path if "site-packages" not in p]
    proj = jedi.Project(path=".", sys_path=clean_sys_path)
    script = jedi.Script(code, project=proj)

    defs = script.goto(5, 23)
    assert len(defs) > 0, "Jedi failed to resolve df.iteritems from pandas-1.5.3"
    resolved_full_name = defs[0].full_name
    assert resolved_full_name == "pandas.core.frame.DataFrame.iteritems"

    # Confirm compatibility with benchmark target name
    assert _is_symbol_compatible("pandas.DataFrame.iteritems", resolved_full_name)
