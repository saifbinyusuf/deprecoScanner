"""
test_parameter_detector.py - Unit tests for Task 2.3: Hardening Fix #2 (parameter-level deprecation).

Verifies detection and (function, param)-level scoping for deprecation warnings
conditioned on function parameters or kwargs.
"""

from src.detectors.legacy_heuristics import detect_legacy_deprecations
from src.detectors.parameter_detector import detect_parameter_deprecations


def test_function_warning_only_when_param_true():
    """
    Verification test required by Task 2.3:
    A function that only warns when param=True must be scoped to (function, param).
    """
    code = """
import warnings

def compute(x, param=False):
    if param:
        warnings.warn("param=True is deprecated, use compute_v2()", DeprecationWarning)
    return x * 2
"""
    candidates = detect_legacy_deprecations(code, filename="test_compute.py")
    assert len(candidates) == 1, f"Expected 1 candidate, got {len(candidates)}: {candidates}"

    cand = candidates[0]
    # Check scoping
    assert cand.scope == "parameter", f"Expected scope 'parameter', got '{cand.scope}'"
    assert cand.param_name == "param", f"Expected param_name 'param', got '{cand.param_name}'"
    assert cand.function_name == "compute", f"Expected function_name 'compute', got '{cand.function_name}'"
    assert cand.qualified_name == "compute::param"
    assert cand.scoped_key == ("compute", "param")
    assert "conditioned on param 'param'" in cand.raw_evidence
    assert "(if param)" in cand.raw_evidence


def test_whole_function_warning_remains_function_scoped():
    """Verify unconditional warnings remain function-scoped."""
    code = """
import warnings

def whole_func(x):
    warnings.warn("whole_func is deprecated", DeprecationWarning)
    return x
"""
    candidates = detect_legacy_deprecations(code, filename="test_whole.py")
    assert len(candidates) == 1

    cand = candidates[0]
    assert cand.scope == "function"
    assert cand.param_name is None
    assert cand.function_name == "whole_func"
    assert cand.qualified_name == "whole_func"
    assert cand.scoped_key == ("whole_func", None)


def test_kwarg_membership_condition():
    """Verify warnings conditioned on kwargs membership are scoped to the specific kwarg."""
    code = """
import warnings

def process_data(data, **kwargs):
    if "legacy_flag" in kwargs:
        warnings.warn("legacy_flag kwarg is deprecated", FutureWarning)
    return data
"""
    candidates = detect_legacy_deprecations(code, filename="test_kwargs.py")
    assert len(candidates) == 1

    cand = candidates[0]
    assert cand.scope == "parameter"
    assert cand.param_name == "legacy_flag"
    assert cand.qualified_name == "process_data::legacy_flag"
    assert cand.scoped_key == ("process_data", "legacy_flag")


def test_decorator_kwarg_scoping():
    """Verify kwarg deprecation decorators are scoped to the parameter."""
    code = """
@deprecate_kwarg("old_arg", "new_arg")
def transform(data, old_arg=None, new_arg=None):
    return data
"""
    candidates = detect_legacy_deprecations(code, filename="test_dec.py")
    assert len(candidates) == 1

    cand = candidates[0]
    assert cand.scope == "parameter"
    assert cand.param_name == "old_arg"
    assert cand.qualified_name == "transform::old_arg"
    assert cand.scoped_key == ("transform", "old_arg")


def test_unrelated_if_condition_not_parameter_scoped():
    """Verify warnings inside an `if` not referencing a parameter remain function-scoped."""
    code = """
import warnings

def check_env(x):
    if GLOBAL_DEBUG_FLAG:
        warnings.warn("check_env is deprecated", DeprecationWarning)
    return x
"""
    candidates = detect_legacy_deprecations(code, filename="test_env.py")
    assert len(candidates) == 1

    cand = candidates[0]
    assert cand.scope == "function"
    assert cand.param_name is None
    assert cand.qualified_name == "check_env"


def test_standalone_parameter_detector_filter():
    """Verify detect_parameter_deprecations only returns parameter-scoped candidates."""
    code = """
import warnings

def mixed_module(x, old_kw=None):
    if old_kw is not None:
        warnings.warn("old_kw is deprecated", DeprecationWarning)
    return x

def regular_dep(y):
    warnings.warn("regular_dep is deprecated", DeprecationWarning)
    return y
"""
    param_cands = detect_parameter_deprecations(code, filename="mixed.py")
    assert len(param_cands) == 1
    assert param_cands[0].param_name == "old_kw"
    assert param_cands[0].scope == "parameter"
