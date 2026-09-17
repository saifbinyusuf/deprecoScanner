"""Detection modules for legacy heuristics, hardening fixes, and PEP 702."""

from src.detectors.legacy_heuristics import (
    DeprecationCandidate,
    detect_legacy_deprecations,
    detect_legacy_deprecations_from_file,
)
from src.detectors.comment_detector import (
    detect_comment_deprecations,
    detect_comment_deprecations_from_file,
)

__all__ = [
    "DeprecationCandidate",
    "detect_legacy_deprecations",
    "detect_legacy_deprecations_from_file",
    "detect_comment_deprecations",
    "detect_comment_deprecations_from_file",
]

