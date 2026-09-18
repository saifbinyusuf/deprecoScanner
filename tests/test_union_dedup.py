"""
tests/test_union_dedup.py - Test suite for Task 2.5 Union and Deduplication Engine.

Verifies:
1. Multi-origin single function deduplication across all Stage 1 sources.
2. Line offset reconciliation (decorator header vs docstring vs warning body).
3. Isolation between parameter-scoped and function-scoped candidates.
4. Canonical parameter AND condition order-independent deduplication.
5. Merging of PEP 702 Griffe metadata with legacy heuristics and message preservation.
6. Acceptance Criterion 1: Client call-site reconciliation with symbol suffix normalization.
7. Acceptance Criterion 2: Formalized asymmetric cases (Case A both fire, Case B recovery, Case C baseline).
"""

from src.detectors.legacy_heuristics import DeprecationCandidate
from src.detectors.union_dedup import (
    ClientInvocation,
    union_stage1_candidates,
    reconcile_client_invocations,
)


def test_multi_origin_single_function_deduplication():
    """
    Verify that a single function flagged by decorator, warning, docstring,
    and comment heuristics collapses into exactly ONE candidate recording all origins.
    """
    file_loc = "/workspace/pkg/module.py"

    c_decorator = DeprecationCandidate(
        qualified_name="pkg.module.legacy_func",
        origin="decorator",
        location=f"{file_loc}:4",
        raw_evidence="@deprecated('old')",
        line=4,
        origins={"decorator"},
    )
    c_docstring = DeprecationCandidate(
        qualified_name="pkg.module.legacy_func",
        origin="docstring",
        location=f"{file_loc}:6",
        raw_evidence=".. deprecated:: 1.0",
        line=6,
        origins={"docstring"},
    )
    c_warning = DeprecationCandidate(
        qualified_name="pkg.module.legacy_func",
        origin="warning",
        location=f"{file_loc}:10",
        raw_evidence="warnings.warn('legacy_func is deprecated', DeprecationWarning)",
        line=10,
        origins={"warning"},
    )
    c_comment = DeprecationCandidate(
        qualified_name="pkg.module.legacy_func",
        origin="comment",
        location=f"{file_loc}:5",
        raw_evidence="# deprecated: use modern_func",
        line=5,
        origins={"comment"},
    )

    merged = union_stage1_candidates([c_decorator, c_docstring], [c_warning, c_comment])

    assert len(merged) == 1
    c = merged[0]
    assert c.qualified_name == "pkg.module.legacy_func"
    assert c.origins == {"decorator", "warning", "docstring", "comment"}
    # Line number points to earliest header line (line 4)
    assert c.line == 4
    assert c.location == f"{file_loc}:4"
    # Raw evidence contains distinct evidence from all origins
    assert "@deprecated" in c.raw_evidence
    assert "warnings.warn" in c.raw_evidence
    assert "deprecated:: 1.0" in c.raw_evidence
    assert "# deprecated: use modern_func" in c.raw_evidence


def test_parameter_vs_function_deduplication_isolation():
    """
    Verify parameter-scoped deprecations (foo::param) NEVER merge into
    whole-function deprecations (foo).
    """
    file_loc = "/workspace/pkg/module.py"

    c_func = DeprecationCandidate(
        qualified_name="pkg.module.process",
        origin="docstring",
        location=f"{file_loc}:20",
        raw_evidence="process is deprecated",
        line=20,
        scope="function",
    )
    c_param = DeprecationCandidate(
        qualified_name="pkg.module.process::fast_mode",
        origin="parameter",
        location=f"{file_loc}:25",
        raw_evidence="if fast_mode: warnings.warn('fast_mode deprecated')",
        line=25,
        scope="parameter",
        param_name="fast_mode",
        function_name="pkg.module.process",
    )

    merged = union_stage1_candidates([c_func], [c_param])

    assert len(merged) == 2
    by_scope = {c.scope: c for c in merged}
    assert "function" in by_scope
    assert "parameter" in by_scope
    assert by_scope["function"].qualified_name == "pkg.module.process"
    assert by_scope["parameter"].qualified_name == "pkg.module.process::fast_mode"


def test_parameter_and_condition_canonical_dedup():
    """
    Verify multi-parameter AND conditions merge order-independently
    (e.g., 'calc::a+b' and 'calc::b+a' collapse to one candidate).
    """
    file_loc = "/workspace/pkg/calc.py"

    c1 = DeprecationCandidate(
        qualified_name="pkg.calc.solve::alpha+beta",
        origin="parameter",
        location=f"{file_loc}:30",
        raw_evidence="if alpha and beta: warnings.warn(...)",
        line=30,
        scope="parameter",
        param_name="alpha+beta",
        function_name="pkg.calc.solve",
    )
    c2 = DeprecationCandidate(
        qualified_name="pkg.calc.solve::beta+alpha",
        origin="parameter",
        location=f"{file_loc}:32",
        raw_evidence="warnings.warn('alpha and beta together deprecated')",
        line=32,
        scope="parameter",
        param_name="beta+alpha",
        function_name="pkg.calc.solve",
    )

    merged = union_stage1_candidates([c1, c2])

    assert len(merged) == 1
    c = merged[0]
    assert c.param_tuple == ("alpha", "beta")
    assert c.line == 30


def test_pep702_griffe_merges_with_legacy_heuristics():
    """
    Verify that PEP 702 Griffe candidates merge cleanly with legacy heuristics
    and preserve the structured message.
    """
    file_loc = "/workspace/pkg/math_utils.py"

    c_griffe = DeprecationCandidate(
        qualified_name="pkg.math_utils.old_sum",
        origin="pep702:griffe",
        location=f"{file_loc}:15",
        raw_evidence="@warnings.deprecated('use new_sum() instead')",
        line=15,
        origins={"pep702", "pep702:griffe"},
        message="use new_sum() instead",
    )
    c_legacy = DeprecationCandidate(
        qualified_name="pkg.math_utils.old_sum",
        origin="docstring",
        location=f"{file_loc}:16",
        raw_evidence=".. deprecated:: 2.0",
        line=16,
        origins={"docstring"},
    )

    merged = union_stage1_candidates([c_griffe], [c_legacy])

    assert len(merged) == 1
    c = merged[0]
    assert c.qualified_name == "pkg.math_utils.old_sum"
    assert c.origins == {"pep702", "pep702:griffe", "docstring"}
    assert c.message == "use new_sum() instead"
    assert c.line == 15


def test_client_site_reconciliation_case_a_both_fire():
    """
    Acceptance Criterion 1 & 2 (Case A):
    Channel 1 (Jedi) and Channel 2 (Mypy) fire on the same call site with
    qualified vs partially-qualified symbol names (pandas.DataFrame.applymap vs DataFrame.applymap).
    Verify they collapse into 1 record and combine origins.
    """
    s1_invocations = [
        ClientInvocation(
            client_file="/client_project/analysis.py",
            line=42,
            target_symbol="pandas.DataFrame.applymap",
            origins={"matched:stage1"},
            raw_evidence="df.applymap(lambda x: x*2)",
        )
    ]
    tc_candidates = [
        DeprecationCandidate(
            qualified_name="DataFrame.applymap",
            origin="pep702:mypy",
            location="/client_project/analysis.py:42",
            raw_evidence="mypy [deprecated]: DataFrame.applymap is deprecated (use DataFrame.map instead)",
            line=42,
            origins={"pep702", "pep702:mypy"},
            message="use DataFrame.map instead",
        )
    ]

    reconciled = reconcile_client_invocations(s1_invocations, tc_candidates)

    assert len(reconciled) == 1
    rec = reconciled[0]
    assert rec.client_file == "/client_project/analysis.py"
    assert rec.line == 42
    assert rec.target_symbol == "pandas.DataFrame.applymap"
    assert rec.origins == {"matched:stage1", "pep702", "pep702:mypy"}
    assert rec.message == "use DataFrame.map instead"
    assert rec.is_typechecker_recovery is False
    assert "df.applymap" in rec.raw_evidence
    assert "mypy [deprecated]" in rec.raw_evidence


def test_client_site_reconciliation_case_b_channel_2_recovery():
    """
    Acceptance Criterion 2 (Case B):
    Channel 2 (Pyright) flags a deprecation call site that Stage 1/Jedi missed.
    Verify it is captured with is_typechecker_recovery = True.
    """
    s1_invocations = []  # Jedi missed this call site
    tc_candidates = [
        DeprecationCandidate(
            qualified_name="fastapi.FastAPI.on_event",
            origin="pep702:pyright",
            location="/client_project/main.py:15",
            raw_evidence="pyright [reportDeprecated]: on_event is deprecated, use lifespan event handlers",
            line=15,
            origins={"pep702", "pep702:pyright"},
            message="use lifespan event handlers",
        )
    ]

    reconciled = reconcile_client_invocations(s1_invocations, tc_candidates)

    assert len(reconciled) == 1
    rec = reconciled[0]
    assert rec.client_file == "/client_project/main.py"
    assert rec.line == 15
    assert rec.target_symbol == "fastapi.FastAPI.on_event"
    assert rec.is_typechecker_recovery is True
    assert rec.origins == {"pep702", "pep702:pyright"}
    assert rec.message == "use lifespan event handlers"


def test_client_site_reconciliation_case_c_channel_1_only():
    """
    Acceptance Criterion 2 (Case C):
    Channel 1 (Stage 1 / Jedi) fires on an un-typed legacy call site
    where typecheckers emit no diagnostics.
    Verify it is preserved cleanly with baseline origins.
    """
    s1_invocations = [
        ClientInvocation(
            client_file="/client_project/legacy_script.py",
            line=88,
            target_symbol="numpy.alltrue",
            origins={"matched:stage1"},
            raw_evidence="np.alltrue(arr)",
        )
    ]
    tc_candidates = []  # Typecheckers did not emit diagnostics

    reconciled = reconcile_client_invocations(s1_invocations, tc_candidates)

    assert len(reconciled) == 1
    rec = reconciled[0]
    assert rec.client_file == "/client_project/legacy_script.py"
    assert rec.line == 88
    assert rec.target_symbol == "numpy.alltrue"
    assert rec.origins == {"matched:stage1"}
    assert rec.is_typechecker_recovery is False


def test_disjunctive_parameter_conditions_remain_distinct():
    """
    Verify that independent parameter deprecations on different parameters
    of the same function do NOT collapse together.
    """
    file_loc = "/workspace/pkg/models.py"

    c1 = DeprecationCandidate(
        qualified_name="pkg.models.fit::learning_rate",
        origin="parameter",
        location=f"{file_loc}:50",
        raw_evidence="if learning_rate: warnings.warn(...)",
        line=50,
        scope="parameter",
        param_name="learning_rate",
        function_name="pkg.models.fit",
    )
    c2 = DeprecationCandidate(
        qualified_name="pkg.models.fit::optimizer",
        origin="parameter",
        location=f"{file_loc}:55",
        raw_evidence="if optimizer: warnings.warn(...)",
        line=55,
        scope="parameter",
        param_name="optimizer",
        function_name="pkg.models.fit",
    )

    merged = union_stage1_candidates([c1], [c2])

    assert len(merged) == 2
    params = {c.param_name for c in merged}
    assert params == {"learning_rate", "optimizer"}


def test_class_method_collision_guard_dataframe_vs_series_iteritems():
    """
    Directly tests collision protection for the real Pandas benchmark pair:
    'pandas.DataFrame.iteritems' vs 'pandas.Series.iteritems'.
    Verifies that suffix matching cannot cross-collide between distinct classes,
    and bare method names ('iteritems') cannot falsely match class methods.
    """
    from src.detectors.union_dedup import _is_symbol_compatible

    # 1. Matching class-qualified partials is permitted
    assert _is_symbol_compatible("pandas.DataFrame.iteritems", "DataFrame.iteritems") is True
    assert _is_symbol_compatible("pandas.Series.iteritems", "Series.iteritems") is True

    # 2. Cross-class methods MUST NOT match
    assert _is_symbol_compatible("pandas.DataFrame.iteritems", "pandas.Series.iteritems") is False
    assert _is_symbol_compatible("pandas.DataFrame.iteritems", "Series.iteritems") is False
    assert _is_symbol_compatible("pandas.Series.iteritems", "DataFrame.iteritems") is False

    # 3. Bare method names MUST NOT match class methods (collision guard)
    assert _is_symbol_compatible("pandas.DataFrame.iteritems", "iteritems") is False
    assert _is_symbol_compatible("pandas.Series.iteritems", "iteritems") is False
    assert _is_symbol_compatible("pandas.DataFrame.pad", "pad") is False
    assert _is_symbol_compatible("pandas.Series.pad", "pad") is False

    # 4. In reconcile_client_invocations: two calls on the same line for DataFrame vs Series
    # must NEVER cross-collapse into one record
    df_inv = ClientInvocation(
        client_file="/client/script.py",
        line=10,
        target_symbol="pandas.DataFrame.iteritems",
        origins={"matched:stage1"},
        raw_evidence="df.iteritems()",
    )
    series_cand = DeprecationCandidate(
        qualified_name="Series.iteritems",
        origin="pep702:mypy",
        location="/client/script.py:10",
        raw_evidence="mypy [deprecated]: Series.iteritems is deprecated",
        line=10,
        origins={"pep702", "pep702:mypy"},
    )

    reconciled = reconcile_client_invocations([df_inv], [series_cand])
    # Must produce 2 records (no false merge between DataFrame and Series)
    assert len(reconciled) == 2
    symbols = {r.target_symbol for r in reconciled}
    assert "pandas.DataFrame.iteritems" in symbols
    assert "Series.iteritems" in symbols

