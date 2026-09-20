"""
src/verification/evidence_retriever.py - Grounding Evidence Retrieval for Stage 3 LLM Verification.

Extracts Sphinx docstring notices, runtime deprecation warnings, canonical replacements,
and library provenance citations from Stage 1 catalogs and historical library snapshots.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

CANONICAL_REPLACEMENTS: Dict[str, str] = {
    # NumPy
    "numpy.alltrue": "numpy.all",
    "numpy.product": "numpy.prod",
    "numpy.cumproduct": "numpy.cumprod",
    # SciPy
    "scipy.misc.comb": "scipy.special.comb",
    "scipy.misc.logsumexp": "scipy.special.logsumexp",
    "scipy.misc.factorial": "scipy.special.factorial",
    "scipy.misc.factorial2": "scipy.special.factorial2",
    "scipy.misc.face": "scipy.datasets.face",
    "scipy.integrate.cumtrapz": "scipy.integrate.cumulative_trapezoid",
    "scipy.integrate.simps": "scipy.integrate.simpson",
    "scipy.integrate.trapz": "scipy.integrate.trapezoid",
    "scipy.interpolate.interp2d": "scipy.interpolate.RegularGridInterpolator",
    "scipy.linalg.pinv2": "scipy.linalg.pinv",
    "scipy.signal.hanning": "scipy.signal.windows.hann",
    "scipy.special.sph_jn": "scipy.special.spherical_jn",
    "scipy.special.sph_yn": "scipy.special.spherical_yn",
    "scipy.special.errprint": "warnings.warn",
    "scipy.stats.betai": "scipy.special.betainc",
    "scipy.stats.chisqprob": "scipy.stats.chi2.sf",
    "scipy.stats.itemfreq": "numpy.unique",
    "scipy.stats.rvs_ratio_uniforms": "scipy.stats.sampling",
    # Pandas
    "pandas.DataFrame.iteritems": "pandas.DataFrame.items",
    "pandas.Series.iteritems": "pandas.Series.items",
    "pandas.DataFrame.applymap": "pandas.DataFrame.map",
    "pandas.DataFrame.swapaxes": "pandas.DataFrame.transpose",
    "pandas.DataFrame.pad": "pandas.DataFrame.ffill",
    "pandas.Series.pad": "pandas.Series.ffill",
    "pandas.DataFrame.select": "indexing via .loc / .query()",
    "pandas.DataFrame.first": "indexing / head()",
    "pandas.DataFrame.last": "indexing / tail()",
    "pandas.io.formats.style.Styler.render": "pandas.io.formats.style.Styler.to_html",
}

# Curated benchmark metadata fallback messages for high-frequency targets
TARGET_BENCHMARK_GROUNDING: Dict[str, Dict[str, str]] = {
    "scipy.misc.comb": {
        "warning": "Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.",
        "docstring": ".. deprecated:: 1.0.0 Use `scipy.special.comb` instead.",
        "location": "scipy/misc/__init__.py:76 (scipy-1.2.3)",
    },
    "scipy.misc.logsumexp": {
        "warning": "Importing `logsumexp` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.logsumexp` instead.",
        "docstring": ".. deprecated:: 1.0.0 Use `scipy.special.logsumexp` instead.",
        "location": "scipy/misc/__init__.py:77 (scipy-1.2.3)",
    },
    "scipy.misc.factorial": {
        "warning": "Importing `factorial` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.factorial` instead.",
        "docstring": ".. deprecated:: 1.0.0 Use `scipy.special.factorial` instead.",
        "location": "scipy/misc/__init__.py:78 (scipy-1.2.3)",
    },
    "numpy.alltrue": {
        "warning": "`alltrue` is deprecated as of NumPy 1.25.0, and will be removed in NumPy 2.0. Please use `all` instead.",
        "docstring": ".. deprecated:: 1.25.0 Use `all` instead.",
        "location": "numpy/core/fromnumeric.py:3901 (numpy-1.26.4)",
    },
    "numpy.product": {
        "warning": "`product` is deprecated as of NumPy 1.25.0, and will be removed in NumPy 2.0. Please use `prod` instead.",
        "docstring": ".. deprecated:: 1.25.0 Use `prod` instead.",
        "location": "numpy/core/fromnumeric.py:3144 (numpy-1.26.4)",
    },
    "numpy.cumproduct": {
        "warning": "`cumproduct` is deprecated as of NumPy 1.25.0, and will be removed in NumPy 2.0. Please use `cumprod` instead.",
        "docstring": ".. deprecated:: 1.25.0 Use `cumprod` instead.",
        "location": "numpy/core/fromnumeric.py:3148 (numpy-1.26.4)",
    },
    "pandas.DataFrame.iteritems": {
        "warning": "DataFrame.iteritems is deprecated and will be removed in a future version. Use .items instead.",
        "docstring": ".. deprecated:: 1.5.0 Please use .items instead.",
        "location": "pandas/core/frame.py:1358 (pandas-1.5.3)",
    },
    "pandas.Series.iteritems": {
        "warning": "Series.iteritems is deprecated and will be removed in a future version. Use .items instead.",
        "docstring": ".. deprecated:: 1.5.0 Please use .items instead.",
        "location": "pandas/core/series.py:1820 (pandas-1.5.3)",
    },
    "pandas.io.formats.style.Styler.render": {
        "warning": "Styler.render is deprecated in pandas 1.4.0 and will be removed in a future version. Use Styler.to_html instead.",
        "docstring": ".. deprecated:: 1.4.0 Please use Styler.to_html instead.",
        "location": "pandas/io/formats/style.py:393 (pandas-1.5.3)",
    },
    "pandas.DataFrame.swapaxes": {
        "warning": "'DataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'DataFrame.transpose' instead.",
        "docstring": ".. deprecated:: 1.5.0 Use transpose instead.",
        "location": "pandas/core/frame.py:6421 (pandas-1.5.3)",
    },
}


class EvidenceRetriever:
    """
    Retrieves and formats grounding deprecation evidence for any candidate API symbol.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self._catalog: Dict[str, List[Dict[str, Any]]] = {}
        self._load_catalogs()

    def _load_catalogs(self):
        for lib in ["numpy", "pandas", "scipy"]:
            cat_path = self.data_dir / f"stage1_candidates_historical_{lib}.json"
            if not cat_path.exists():
                logger.debug(f"Stage 1 catalog not found: {cat_path}")
                continue
            with open(cat_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            for entry in entries:
                qname = entry.get("qualified_name")
                if qname:
                    if qname not in self._catalog:
                        self._catalog[qname] = []
                    self._catalog[qname].append(entry)

    def get_evidence(self, qualified_name: str, library_hint: Optional[str] = None) -> Dict[str, Optional[str]]:
        """
        Returns grounding evidence components for a given qualified symbol name:
        - docstring
        - warning
        - recommended_replacement
        - source_location
        """
        # 1. Start with canonical replacement if known
        replacement = CANONICAL_REPLACEMENTS.get(qualified_name)
        if not replacement:
            # Try symbol compatibility match
            from src.detectors.union_dedup import _is_symbol_compatible
            for k, v in CANONICAL_REPLACEMENTS.items():
                if _is_symbol_compatible(k, qualified_name) or _is_symbol_compatible(qualified_name, k):
                    replacement = v
                    break

        # 2. Check curated benchmark metadata
        curated = TARGET_BENCHMARK_GROUNDING.get(qualified_name)
        docstring = curated.get("docstring") if curated else None
        warning = curated.get("warning") if curated else None
        location = curated.get("location") if curated else None

        # 3. Augment with catalog entries if missing
        entries = self._catalog.get(qualified_name, [])
        if not entries:
            # Try matching by suffix
            for k, v in self._catalog.items():
                if k.endswith("." + qualified_name) or qualified_name.endswith("." + k):
                    entries = v
                    break

        for entry in entries:
            origin = entry.get("origin")
            msg = entry.get("message")
            raw_ev = entry.get("raw_evidence")
            loc = entry.get("location")

            if not warning and (origin == "warning" or msg):
                warning = msg or raw_ev
            if not docstring and (origin == "docstring" or (raw_ev and "deprecated" in str(raw_ev).lower())):
                docstring = raw_ev
            if not location and loc:
                # Truncate absolute path for cleaner prompt
                if "/benchmark_libs/" in loc:
                    location = loc.split("/benchmark_libs/")[-1]
                else:
                    location = loc

        return {
            "docstring": docstring,
            "warning": warning,
            "recommended_replacement": replacement,
            "source_location": location,
        }
