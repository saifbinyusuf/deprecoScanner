"""
src/resolution/benchmark_targets.py - Canonical Benchmark Target Definitions & Matcher.

Establishes the single repository source of truth for the 31 canonical benchmark target APIs
annotated in the figshare / llm-dep-api benchmark dataset.

Guarantees:
1. Strictly maps symbols using _is_symbol_compatible.
2. Contains ZERO callee stem fallbacks (.split(".")[-1]).
3. Rejects modern replacement APIs (e.g. scipy.special.comb, scipy.integrate.simpson).
4. Rejects out-of-scope historical deprecations (e.g. numpy.expand_dims, scipy.linalg.pinv).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from src.detectors.union_dedup import _is_symbol_compatible

BENCHMARK_TARGET_APIS: Dict[str, List[str]] = {
    "numpy": [
        "numpy.alltrue",
        "numpy.product",
        "numpy.cumproduct",
    ],
    "scipy": [
        "scipy.misc.logsumexp",
        "scipy.misc.comb",
        "scipy.integrate.cumtrapz",
        "scipy.integrate.simps",
        "scipy.integrate.trapz",
        "scipy.interpolate.interp2d",
        "scipy.linalg.pinv2",
        "scipy.misc.factorial",
        "scipy.stats.itemfreq",
        "scipy.signal.hanning",
        "scipy.special.sph_jn",
        "scipy.stats.betai",
        "scipy.stats.chisqprob",
        "scipy.misc.face",
        "scipy.misc.factorial2",
        "scipy.special.sph_yn",
        "scipy.special.errprint",
        "scipy.stats.rvs_ratio_uniforms",
    ],
    "pandas": [
        "pandas.io.formats.style.Styler.render",
        "pandas.DataFrame.swapaxes",
        "pandas.DataFrame.applymap",
        "pandas.DataFrame.pad",
        "pandas.Series.iteritems",
        "pandas.DataFrame.iteritems",
        "pandas.DataFrame.select",
        "pandas.DataFrame.first",
        "pandas.DataFrame.last",
        "pandas.Series.pad",
    ],
}

ALL_BENCHMARK_TARGETS: Set[str] = {
    target
    for targets in BENCHMARK_TARGET_APIS.values()
    for target in targets
}

TARGET_SHORT_NAMES: Set[str] = {
    t.split(".")[-1]
    for t in ALL_BENCHMARK_TARGETS
}


def match_benchmark_target(symbol: Optional[str]) -> Optional[str]:
    """
    Matches a resolved symbol against the 31 canonical benchmark target APIs.

    Returns the canonical target API name if compatible via _is_symbol_compatible,
    or None if the symbol is not one of the 31 benchmark targets.
    """
    if not symbol:
        return None

    clean_sym = symbol.strip()
    if clean_sym in ALL_BENCHMARK_TARGETS:
        return clean_sym

    for target in ALL_BENCHMARK_TARGETS:
        if _is_symbol_compatible(target, clean_sym) or _is_symbol_compatible(clean_sym, target):
            return target

    return None
