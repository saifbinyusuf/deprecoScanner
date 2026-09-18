"""
union_dedup.py - Task 2.5: Union and Deduplication Engine.

Implements recall-preserving candidate merging across all Stage 1 sources
and dual-channel reconciliation at client call sites:

1. `union_stage1_candidates(*candidate_lists)`:
   - Unions candidates from all Stage 1 detectors:
     - legacy heuristics (decorator, warning, docstring)
     - comment detector (# deprecated)
     - parameter detector ((function, param) conditions)
     - PEP 702 detector (griffe declaration metadata)
   - Collapses duplicate multi-origin hits pointing to the same API declaration,
     reconciling line offsets (def header vs decorator vs warning inside body).
   - Retains the full `origins: Set[str]` set for per-heuristic ablation breakdowns.
   - Isolates parameter deprecations so they never collapse into whole-function deprecations.
   - Preserves canonical parameter ordering (AND conditions).

2. `reconcile_client_invocations(stage1_matches, typechecker_cands)`:
   - Reconciles Channel 1 (Jedi-matched Stage 1 catalog hits) with Channel 2
     (Type-checker native diagnostics from mypy/pyright).
   - Handles symbol qualification differences (e.g. `pandas.DataFrame.applymap` vs `applymap`).
   - Formalizes all three scenarios:
     * Case A (Both fire): Collapses to single invocation record, combining origins.
     * Case B (Channel 2 only): Flagged as `is_typechecker_recovery = True` (adds recall).
     * Case C (Channel 1 only): Standard baseline path.

Caveat Note:
Dual-channel reconciliation applies to modern codebases with PEP 702 adoption
(such as the parked Pydantic/FastAPI study) and is empirically unexercised on the
core 0/32 historical APIScanner benchmark where PEP 702 is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from src.detectors.legacy_heuristics import DeprecationCandidate


@dataclass
class ClientInvocation:
    """
    Represents a client call-site invocation of a deprecated API.
    Used during Stage 2 client matching and evaluation reconciliation.
    """

    client_file: str
    line: int
    target_symbol: str
    origins: Set[str] = field(default_factory=set)
    raw_evidence: str = ""
    message: Optional[str] = None
    is_typechecker_recovery: bool = False
    param_name: Optional[str] = None
    scope: str = "function"


def _extract_file_path(location: str) -> str:
    """Extracts the file path from a 'filepath:line' location string."""
    if ":" in location:
        return location.rsplit(":", 1)[0]
    return location


def _is_symbol_compatible(sym1: str, sym2: str) -> bool:
    """
    Determines if two symbol strings refer to the same target API,
    accommodating fully-qualified names vs partially-qualified names,
    base-class inheritance (NDFrame for DataFrame/Series), and
    intermediate internal submodules (e.g. numpy.alltrue vs numpy.core.fromnumeric.alltrue,
    or scipy.integrate.cumtrapz vs scipy.integrate._quadrature.cumtrapz).

    Guards against cross-class method collisions (e.g. DataFrame.iteritems vs Series.iteritems)
    by requiring class qualification for class methods.
    """
    if sym1 == sym2:
        return True

    # Suffix matching (e.g. DataFrame.iteritems and pandas.DataFrame.iteritems)
    if sym1.endswith(f".{sym2}") or sym2.endswith(f".{sym1}"):
        longer, shorter = (sym1, sym2) if len(sym1) > len(sym2) else (sym2, sym1)
        if "." not in shorter:
            parts = longer.split(".")
            # If the immediate parent is a Class (starts with uppercase), reject bare method match
            if len(parts) >= 2 and parts[-2][0].isupper():
                return False
        return True

    p1 = sym1.split(".")
    p2 = sym2.split(".")

    # Same root library (e.g. both pandas.* or both numpy.* or both scipy.*)
    if p1[0] == p2[0]:
        # Case 1: Class methods (e.g. pandas.DataFrame.iteritems vs pandas.core.frame.DataFrame.iteritems)
        if len(p1) >= 2 and p1[-2][0].isupper() and len(p2) >= 2 and p2[-2][0].isupper():
            cls1, meth1 = p1[-2], p1[-1]
            cls2, meth2 = p2[-2], p2[-1]
            if meth1 == meth2:
                if cls1 == cls2:
                    return True
                # Pandas base class inheritance: NDFrame is the canonical base for DataFrame and Series
                if p1[0] == "pandas":
                    if (cls1 in ("DataFrame", "Series") and cls2 == "NDFrame") or (cls2 in ("DataFrame", "Series") and cls1 == "NDFrame"):
                        return True
            return False

        # Case 2: Submodule re-exports / internal paths
        # (e.g. scipy.integrate.cumtrapz vs scipy.integrate._quadrature.cumtrapz)
        if p1[-1] == p2[-1]:
            # Same terminal function name
            if len(p1) >= 3 and len(p2) >= 3:
                sub1 = p1[1]
                sub2 = p2[1]
                if sub1 == sub2 or (sub1 in ("misc", "special") and sub2 in ("misc", "special")):
                    return True
            # Top-level module function re-export (e.g. numpy.alltrue vs numpy.core.fromnumeric.alltrue)
            if len(p1) == 2 or len(p2) == 2:
                if not (len(p1) >= 2 and p1[-2][0].isupper()) and not (len(p2) >= 2 and p2[-2][0].isupper()):
                    return True

    return False


def union_stage1_candidates(
    *candidate_lists: List[DeprecationCandidate],
) -> List[DeprecationCandidate]:
    """
    Unions multiple candidate lists from Stage 1 sources into a deduplicated inventory.

    Deduplication rules:
    1. Scope isolation: Parameter-scoped candidates ('foo::param') NEVER merge into
       whole-function candidates ('foo').
    2. Canonical parameter grouping: Multi-parameter AND conditions use canonically sorted
       tuples so 'foo::a+b' and 'foo::b+a' collapse to the same surviving candidate.
    3. Line reconciliation: Multiple heuristics on the same function/class declaration
       (e.g., decorator on line 5, docstring on line 6, warning on line 10) collapse to a
       single candidate using the earliest header line.
    4. Origin tracking: The surviving candidate retains the union of all `origins` that fired.
    5. Evidence aggregation: Distinct raw evidence strings are concatenated with ' | '.
    6. Message preservation: Clean, structured messages (from PEP 702 or warnings) are preserved.
    """
    groups: Dict[Tuple, List[DeprecationCandidate]] = {}

    for cand_list in candidate_lists:
        for c in cand_list:
            file_path = _extract_file_path(c.location)
            canonical_file = str(Path(file_path).resolve()) if file_path != "<string>" else "<string>"

            if c.scope == "parameter":
                # Parameter deprecations group by function_name, file, and canonically sorted param_tuple
                func_name = c.function_name or c.qualified_name.split("::")[0]
                ptup = c.param_tuple
                key = (func_name, canonical_file, "parameter", ptup)
            else:
                # Function/class deprecations group by qualified_name, file, and scope
                key = (c.qualified_name, canonical_file, c.scope)

            if key not in groups:
                groups[key] = []
            groups[key].append(c)

    surviving: List[DeprecationCandidate] = []

    for key, cluster in groups.items():
        # Earliest line number represents the declaration header
        lines = [c.line for c in cluster if c.line > 0]
        min_line = min(lines) if lines else cluster[0].line

        # Union of all origins that fired
        all_origins: Set[str] = set()
        for c in cluster:
            if c.origins:
                all_origins.update(c.origins)
            elif c.origin:
                all_origins.add(c.origin)

        # Primary origin label: prioritize pep702:griffe, then decorator, then warning, docstring, comment
        origin_priority = ["pep702:griffe", "decorator", "parameter", "warning", "docstring", "comment"]
        chosen_origin = cluster[0].origin
        for p in origin_priority:
            if any(p in c.origins or c.origin == p for c in cluster):
                chosen_origin = p
                break

        # Collect distinct raw evidence snippets
        distinct_evidence = []
        seen_evidence = set()
        for c in cluster:
            ev = c.raw_evidence.strip()
            if ev and ev not in seen_evidence:
                seen_evidence.add(ev)
                distinct_evidence.append(ev)
        combined_evidence = " | ".join(distinct_evidence)

        # Preserve the cleanest message (PEP 702 prioritized)
        chosen_message: Optional[str] = None
        for c in cluster:
            if c.message:
                chosen_message = c.message
                if "pep702" in c.origins or "pep702" in c.origin:
                    break

        rep = cluster[0]
        file_path = _extract_file_path(rep.location)
        surviving_location = f"{file_path}:{min_line}"

        surviving.append(
            DeprecationCandidate(
                qualified_name=rep.qualified_name,
                origin=chosen_origin,
                location=surviving_location,
                raw_evidence=combined_evidence,
                line=min_line,
                origins=all_origins,
                scope=rep.scope,
                param_name=rep.param_name,
                function_name=rep.function_name,
                message=chosen_message,
            )
        )

    return surviving


def reconcile_client_invocations(
    stage1_matches: List[ClientInvocation],
    typechecker_cands: List[DeprecationCandidate],
) -> List[ClientInvocation]:
    """
    Reconciles Channel 1 (Jedi-matched Stage 1 catalog hits) with Channel 2
    (Type-checker native diagnostics from Mypy/Pyright) at client call sites.

    Handles symbol normalization (e.g. 'pandas.DataFrame.applymap' vs 'applymap')
    and formalizes three distinct cases:
    - Case A (Both fire): Merges into 1 record; combines origins.
    - Case B (Channel 2 only): Flagged as `is_typechecker_recovery = True` (adds recall).
    - Case C (Channel 1 only): Standard baseline path.

    Returns the unified, deduplicated list of ClientInvocation records.
    """
    reconciled: List[ClientInvocation] = []
    matched_tc_indices: Set[int] = set()

    for s1_inv in stage1_matches:
        s1_file = str(Path(s1_inv.client_file).resolve()) if s1_inv.client_file != "<string>" else "<string>"
        collapsed = False

        for idx, tc in enumerate(typechecker_cands):
            if idx in matched_tc_indices:
                continue

            tc_file_str = _extract_file_path(tc.location)
            tc_file = str(Path(tc_file_str).resolve()) if tc_file_str != "<string>" else "<string>"

            # Check file, line, and symbol compatibility
            if s1_file == tc_file and s1_inv.line == tc.line:
                if _is_symbol_compatible(s1_inv.target_symbol, tc.qualified_name):
                    # Case A: Both channels fired on the same call site! Collapse into 1 record.
                    s1_inv.origins.update(tc.origins if tc.origins else {tc.origin})
                    if not s1_inv.message and tc.message:
                        s1_inv.message = tc.message
                    if tc.raw_evidence and tc.raw_evidence not in s1_inv.raw_evidence:
                        s1_inv.raw_evidence = f"{s1_inv.raw_evidence} | {tc.raw_evidence}"
                    matched_tc_indices.add(idx)
                    collapsed = True
                    break

        # Case C: Channel 1 only (or Case A after collapsing)
        reconciled.append(s1_inv)

    # Case B: Channel 2 only (Type checker flagged a call site that Stage 1/Jedi missed)
    for idx, tc in enumerate(typechecker_cands):
        if idx not in matched_tc_indices:
            tc_file_str = _extract_file_path(tc.location)
            all_tc_origins = set(tc.origins) if tc.origins else {tc.origin}

            reconciled.append(
                ClientInvocation(
                    client_file=tc_file_str,
                    line=tc.line,
                    target_symbol=tc.qualified_name,
                    origins=all_tc_origins,
                    raw_evidence=tc.raw_evidence,
                    message=tc.message,
                    is_typechecker_recovery=True,
                    param_name=tc.param_name,
                    scope=tc.scope,
                )
            )

    return reconciled
