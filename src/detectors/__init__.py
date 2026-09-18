"""Detection modules for legacy heuristics, hardening fixes, PEP 702, and union/dedup."""

from src.detectors.legacy_heuristics import (
    DeprecationCandidate,
    detect_legacy_deprecations,
    detect_legacy_deprecations_from_file,
)
from src.detectors.comment_detector import (
    detect_comment_deprecations,
    detect_comment_deprecations_from_file,
)
from src.detectors.parameter_detector import (
    detect_parameter_deprecations,
    detect_parameter_deprecations_from_file,
)
from src.detectors.pep702_detector import (
    detect_pep702_via_griffe,
    detect_pep702_via_mypy,
    detect_pep702_via_pyright,
    detect_pep702_via_typechecker,
    detect_pep702_deprecations,
)
from src.detectors.union_dedup import (
    ClientInvocation,
    union_stage1_candidates,
    reconcile_client_invocations,
)

__all__ = [
    "DeprecationCandidate",
    "ClientInvocation",
    "detect_legacy_deprecations",
    "detect_legacy_deprecations_from_file",
    "detect_comment_deprecations",
    "detect_comment_deprecations_from_file",
    "detect_parameter_deprecations",
    "detect_parameter_deprecations_from_file",
    "detect_pep702_via_griffe",
    "detect_pep702_via_mypy",
    "detect_pep702_via_pyright",
    "detect_pep702_via_typechecker",
    "detect_pep702_deprecations",
    "union_stage1_candidates",
    "reconcile_client_invocations",
]
