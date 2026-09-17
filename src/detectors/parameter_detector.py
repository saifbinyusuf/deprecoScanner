"""
parameter_detector.py - Hardening Fix #2: Parameter-level deprecation detection.

Addresses APIScanner's failure mode where warnings or decorators conditioned on
a specific parameter were misattributed to the entire function.

Walks each function's AST body to detect `warnings.warn(...)` nested inside
an `if` statement referencing a parameter name, and decorators targeting kwargs.
Stores candidates with (function, param)-scoping.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from src.detectors.legacy_heuristics import (
    DeprecationCandidate,
    detect_legacy_deprecations,
    detect_legacy_deprecations_from_file,
)


def detect_parameter_deprecations(
    source_code: str,
    filename: str = "<string>",
    package_prefix: str = "",
) -> List[DeprecationCandidate]:
    """
    Detects parameter-scoped deprecations in source code.
    Returns only candidates whose scope is 'parameter'.
    """
    all_candidates = detect_legacy_deprecations(
        source_code=source_code,
        filename=filename,
        package_prefix=package_prefix,
    )
    return [c for c in all_candidates if c.scope == "parameter"]


def detect_parameter_deprecations_from_file(
    filepath: Union[str, Path],
    package_prefix: str = "",
) -> List[DeprecationCandidate]:
    """Detects parameter-scoped deprecations from a file on disk."""
    all_candidates = detect_legacy_deprecations_from_file(
        filepath=filepath,
        package_prefix=package_prefix,
    )
    return [c for c in all_candidates if c.scope == "parameter"]
