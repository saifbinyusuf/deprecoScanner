"""Detection modules for legacy heuristics, hardening fixes, and PEP 702."""

from src.detectors.legacy_heuristics import (
    DeprecationCandidate,
    detect_legacy_deprecations,
    detect_legacy_deprecations_from_file,
)

__all__ = [
    "DeprecationCandidate",
    "detect_legacy_deprecations",
    "detect_legacy_deprecations_from_file",
]
