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


def test_deprecate_kwarg_does_not_fire_whole_function_heuristic():
    """
    Interaction verification required by human review:
    @deprecate_kwarg on a function taking **kwargs must NOT trigger Task 2.1's
    whole-function decorator heuristic. It must emit ONLY a parameter-scoped
    candidate and keep the whole-function matcher silent.
    """
    code = """
@deprecate_kwarg("old_arg", "new_arg")
def transform(data, **kwargs):
    return data
"""
    all_candidates = detect_legacy_deprecations(code, filename="test_kwarg_dec.py")
    assert len(all_candidates) == 1, f"Expected exactly 1 candidate, got {len(all_candidates)}"

    cand = all_candidates[0]
    assert cand.scope == "parameter", f"Expected 'parameter' scope, got '{cand.scope}'"
    assert cand.param_name == "old_arg"
    assert cand.function_name == "transform"
    assert cand.qualified_name == "transform::old_arg"

    # Confirm Task 2.1 whole-function decorator matcher stayed completely silent
    function_cands = [c for c in all_candidates if c.scope == "function"]
    assert len(function_cands) == 0, f"Expected 0 function-scoped candidates, got {function_cands}"


def test_deprecate_kwarg_with_keyword_arguments():
    """Verify @deprecate_kwarg with keyword arguments (old_arg_name=...) is scoped to parameter."""
    code = """
@deprecate_kwarg(FutureWarning, old_arg_name="cols", new_arg_name="columns")
def select(columns=None, **kwargs):
    return columns
"""
    all_candidates = detect_legacy_deprecations(code, filename="test_kw_arg.py")
    assert len(all_candidates) == 1

    cand = all_candidates[0]
    assert cand.scope == "parameter"
    assert cand.param_name == "cols"
    assert cand.qualified_name == "select::cols"

    function_cands = [c for c in all_candidates if c.scope == "function"]
    assert len(function_cands) == 0


def test_warning_in_else_branch_is_parameter_scoped():
    """
    Verify warnings inside an `else:` branch of an if-statement checking a parameter
    are correctly scoped to that parameter and do not misattribute to the whole function.
    """
    code = """
import warnings

def query(table, fast=False):
    if fast:
        return table
    else:
        warnings.warn("fast=False is deprecated; use fast=True", FutureWarning)
        return table
"""
    all_candidates = detect_legacy_deprecations(code, filename="test_else.py")
    assert len(all_candidates) == 1

    cand = all_candidates[0]
    assert cand.scope == "parameter"
    assert cand.param_name == "fast"
    assert cand.function_name == "query"
    assert cand.qualified_name == "query::fast"
    assert "else: not (fast)" in cand.raw_evidence

    function_cands = [c for c in all_candidates if c.scope == "function"]
    assert len(function_cands) == 0


def test_multi_parameter_condition_and_vs_or():
    """
    Verify AND vs. OR condition differentiation:
    - `if flag1 and flag2:` requires both arguments, emitting a joint candidate
      `combine_and::flag1+flag2` so client calls passing only one flag are not falsely flagged.
    - `if old_a or old_b:` triggers if either argument is supplied, emitting independent candidates
      for `old_a` and `old_b`.
    """
    code_and = """
import warnings

def combine_and(a, b, flag1=False, flag2=False):
    if flag1 and flag2:
        warnings.warn("Combining flag1 and flag2 is deprecated", DeprecationWarning)
    return a + b
"""
    cands_and = detect_legacy_deprecations(code_and, filename="test_and.py")
    assert len(cands_and) == 1, f"Expected 1 joint candidate for AND, got {len(cands_and)}: {cands_and}"
    assert cands_and[0].scope == "parameter"
    assert cands_and[0].param_name == "flag1+flag2"
    assert cands_and[0].qualified_name == "combine_and::flag1+flag2"

    code_or = """
import warnings

def combine_or(a, b, old_a=None, old_b=None):
    if old_a or old_b:
        warnings.warn("Either old_a or old_b is deprecated", DeprecationWarning)
    return a + b
"""
    cands_or = detect_legacy_deprecations(code_or, filename="test_or.py")
    assert len(cands_or) == 2, f"Expected 2 independent candidates for OR, got {len(cands_or)}"
    params = {c.param_name for c in cands_or}
    assert params == {"old_a", "old_b"}
    for c in cands_or:
        assert c.scope == "parameter"
        assert c.function_name == "combine_or"


def test_deprecated_message_with_argument_words_remains_function_scoped():
    """
    Verification required by human review:
    A whole-function @deprecated(...) decorator whose message text happens to mention
    'argument', 'param', or 'kwarg' must NEVER be misclassified as a parameter-scoped deprecation.
    It must stay strictly whole-function scoped.
    """
    code = """
@deprecated("the argument handling has changed, use new_func() instead")
def whole_func_arg_message(x):
    return x
"""
    candidates = detect_legacy_deprecations(code, filename="test_msg.py")
    assert len(candidates) == 1

    cand = candidates[0]
    assert cand.scope == "function", f"Expected 'function' scope, got '{cand.scope}'"
    assert cand.param_name is None, f"Expected None param_name, got '{cand.param_name}'"
    assert cand.qualified_name == "whole_func_arg_message"
    assert cand.origin == "decorator"


def test_unconditional_pandas4warning_detected_by_warning_heuristic():
    """
    Verification required by human review:
    Subclasses of DeprecationWarning (such as Pandas4Warning) used unconditionally
    in warnings.warn must be detected by Task 2.1's whole-function warning heuristic,
    even when the warning message does not contain the word 'deprecat'.
    """
    code = """
import warnings

def legacy_pandas_api(df):
    warnings.warn("legacy_pandas_api is obsolete", Pandas4Warning)
    return df
"""
    candidates = detect_legacy_deprecations(code, filename="test_pandas4.py")
    assert len(candidates) == 1, f"Expected 1 candidate, got {len(candidates)}: {candidates}"

    cand = candidates[0]
    assert cand.scope == "function"
    assert cand.origin == "warning"
    assert cand.qualified_name == "legacy_pandas_api"
    assert "Pandas4Warning" in cand.raw_evidence


def test_real_pandas_file_spot_check():
    """
    Real-world spot-check on installed pandas source:
    pandas.core.resample.Resampler.interpolate deprecates the 'inplace' parameter
    via `if "inplace" in kwargs: warnings.warn(...)`.
    """
    from pathlib import Path
    import pandas
    from src.detectors.parameter_detector import detect_parameter_deprecations_from_file

    resample_path = Path(pandas.__file__).parent / "core" / "resample.py"
    candidates = detect_parameter_deprecations_from_file(resample_path, package_prefix="pandas")

    inplace_cands = [c for c in candidates if c.param_name == "inplace" and "interpolate" in c.qualified_name]
    assert len(inplace_cands) >= 1, f"Expected to find pandas.Resampler.interpolate::inplace, got: {candidates}"
    cand = inplace_cands[0]
    assert cand.scope == "parameter"
    assert cand.param_name == "inplace"
    assert "pandas.Resampler.interpolate" in cand.qualified_name


def test_real_scipy_file_spot_check():
    """
    Real-world spot-check on installed scipy source:
    scipy.linalg._decomp_qr.qr deprecates the 'lwork' keyword argument
    via `if lwork is not _NoValue: ... else: warnings.warn(...)`.
    """
    from pathlib import Path
    import scipy
    from src.detectors.parameter_detector import detect_parameter_deprecations_from_file

    decomp_path = Path(scipy.__file__).parent / "linalg" / "_decomp_qr.py"
    candidates = detect_parameter_deprecations_from_file(decomp_path, package_prefix="scipy")

    lwork_cands = [c for c in candidates if c.param_name == "lwork"]
    assert len(lwork_cands) >= 1, f"Expected to find scipy.qr::lwork, got: {candidates}"
    cand = lwork_cands[0]
    assert cand.scope == "parameter"
    assert cand.param_name == "lwork"
    assert "scipy.qr" in cand.qualified_name

