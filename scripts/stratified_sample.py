#!/usr/bin/env python3
"""
scripts/stratified_sample.py - Extracts the authoritative, frozen 150-call-site benchmark.

Methodology:
- Stratum 1: 100 Candidate Positives drawn deterministically (seed=42) from data/stage3_candidates_manifest.json
  across 28 canonical targets matching the Phase 5 quota table (NumPy: 12, Pandas: 27, SciPy: 61).
- Stratum 2: 8 Genuine Pipeline Misses quota-capped at k=4 for SciPy (seed=42):
  - 4 scipy.special.errprint call sites (from 23 available sites across 10 samples)
  - 4 scipy.stats.rvs_ratio_uniforms call sites (from 13 available sites across 4 samples)
- Stratum 3: 2 Ground-Truth Label Anomalies (GT=False):
  - scipy_1560 (L40: call to mpmath.factorial)
  - scipy_1500 (L40: call to _sp_cumtrapz alias declaration)
- Stratum 4: 40 Fresh Held-Out Hard Negatives (GT=False):
  - 16 Modern Replacement Overloads
  - 8 Submodule / Non-Benchmark Overlaps
  - 6 Standard Library Collisions
  - 10 Third-Party Wrapper Lookalikes
  Strictly excluding the 10 Phase 4 calibration/audit sample IDs.

Outputs:
- data/benchmark/phase5_benchmark_unannotated.jsonl (exact N = 150)
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
import random
import sys
import textwrap
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.resolution.jedi_resolver import normalize_snippet_indentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stratified_sample")

def safe_ast_parse(code: str) -> Optional[ast.AST]:
    """Safely normalizes and parses snippet AST, returning None on failure."""
    try:
        norm = normalize_snippet_indentation(code)
        return ast.parse(norm)
    except Exception:
        return None

EXCLUDED_CALIBRATION_IDS: Set[str] = {
    "numpy_0", "numpy_3", "scipy_0", "scipy_58", "scipy_577",
    "pandas_0", "pandas_70", "scipy_1560", "pandas_32", "pandas_33"
}

# Quotas for candidate positives (sum = 100)
CANDIDATE_QUOTAS: Dict[str, int] = {
    # NumPy (12)
    "numpy.alltrue": 4,
    "numpy.product": 4,
    "numpy.cumproduct": 4,
    # Pandas (27)
    "pandas.io.formats.style.Styler.render": 3,
    "pandas.DataFrame.swapaxes": 3,
    "pandas.DataFrame.applymap": 3,
    "pandas.DataFrame.pad": 0,  # 0 available
    "pandas.Series.iteritems": 3,
    "pandas.DataFrame.iteritems": 3,
    "pandas.DataFrame.select": 3,
    "pandas.DataFrame.first": 3,
    "pandas.DataFrame.last": 3,
    "pandas.Series.pad": 3,
    # SciPy (61)
    "scipy.misc.logsumexp": 4,
    "scipy.misc.comb": 4,
    "scipy.integrate.cumtrapz": 4,
    "scipy.integrate.simps": 4,
    "scipy.integrate.trapz": 4,
    "scipy.interpolate.interp2d": 4,
    "scipy.linalg.pinv2": 1,  # capped by availability (1 available)
    "scipy.misc.factorial": 4,
    "scipy.stats.itemfreq": 4,
    "scipy.signal.hanning": 4,
    "scipy.special.sph_jn": 4,
    "scipy.stats.betai": 4,
    "scipy.stats.chisqprob": 4,
    "scipy.misc.face": 4,
    "scipy.misc.factorial2": 4,
    "scipy.special.sph_yn": 4,
    "scipy.special.errprint": 0,  # pipeline miss stratum below
    "scipy.stats.rvs_ratio_uniforms": 0,  # pipeline miss stratum below
}


def load_raw_samples(library: str) -> List[Dict[str, Any]]:
    path = REPO_ROOT / "data" / "raw" / "llm-dep-api" / "probing-inputs" / library / "samples.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_candidate_manifest() -> List[Dict[str, Any]]:
    path = REPO_ROOT / "data" / "stage3_candidates_manifest.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_candidate_positives(manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Samples exactly 100 candidate positives across 28 targets using fixed seed 42."""
    by_target: Dict[str, List[Dict[str, Any]]] = {}
    for c in manifest:
        t = c["target_api"]
        by_target.setdefault(t, []).append(c)

    selected: List[Dict[str, Any]] = []
    rng = random.Random(42)

    for target, quota in CANDIDATE_QUOTAS.items():
        if quota == 0:
            continue
        cands = by_target.get(target, [])
        if len(cands) < quota:
            raise ValueError(f"Target {target} has {len(cands)} cands, expected at least quota {quota}")
        # Sort deterministically by candidate_id before sampling
        cands_sorted = sorted(cands, key=lambda x: x["candidate_id"])
        chosen = rng.sample(cands_sorted, quota)
        for c in chosen:
            selected.append({
                "candidate_id": c["candidate_id"],
                "sample_id": c["sample_id"],
                "sample_idx": c["sample_idx"],
                "library": c["target_api"].split(".")[0],
                "sample_library": c["library"],
                "cohort": c["cohort"],
                "stratum": "candidate_positive",
                "sub_stratum": "manifest_candidate",
                "target_api": c["target_api"],
                "client_line": c["client_line"],
                "column": c["column"],
                "call_site_snippet": c["call_site_snippet"],
                "enclosing_code": c["enclosing_code"],
                "stage3_preview_decision": True,
                "stage3_preview_rationale": "Directly invokes target deprecated API symbol.",
                "expected_ground_truth": True,
                "annotation_status": "unannotated",
                "is_deprecated_call": None,
            })

    logger.info(f"Sampled {len(selected)} candidate positives (expected 100).")
    return selected


def sample_pipeline_misses(scipy_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Samples exactly 8 genuine pipeline misses (k=4 errprint, k=4 rvs_ratio_uniforms, seed=42)."""
    # 1. errprint (scipy_267 to scipy_276)
    errprint_sites = []
    for idx in range(267, 277):
        s = scipy_samples[idx]
        sid = f"scipy_{idx}"
        fn = s["function"]
        tree = safe_ast_parse(fn)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if name == "errprint":
                    line_text = fn.splitlines()[node.lineno - 1].strip()
                    errprint_sites.append({
                        "candidate_id": f"{sid}_errprint_miss_{node.lineno}_{node.col_offset}",
                        "sample_id": sid,
                        "sample_idx": idx,
                        "library": "scipy",
                        "cohort": "outdated",
                        "stratum": "pipeline_miss",
                        "sub_stratum": "cython_native_ufunc",
                        "target_api": "scipy.special.errprint",
                        "client_line": node.lineno,
                        "column": node.col_offset,
                        "call_site_snippet": line_text,
                        "enclosing_code": fn,
                        "stage3_preview_decision": False,
                        "stage3_preview_rationale": "Pipeline missed: compiled Cython native ufunc unparseable by AST.",
                        "expected_ground_truth": True,
                        "annotation_status": "pre_labeled",
                        "is_deprecated_call": True,
                    })

    # 2. rvs_ratio_uniforms (scipy_458 to scipy_461)
    rvs_sites = []
    for idx in [458, 459, 460, 461]:
        s = scipy_samples[idx]
        sid = f"scipy_{idx}"
        fn = s["function"]
        tree = safe_ast_parse(fn)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if name == "rvs_ratio_uniforms":
                    line_text = fn.splitlines()[node.lineno - 1].strip()
                    rvs_sites.append({
                        "candidate_id": f"{sid}_rvs_miss_{node.lineno}_{node.col_offset}",
                        "sample_id": sid,
                        "sample_idx": idx,
                        "library": "scipy",
                        "cohort": "outdated",
                        "stratum": "pipeline_miss",
                        "sub_stratum": "preamble_filter_omission",
                        "target_api": "scipy.stats.rvs_ratio_uniforms",
                        "client_line": node.lineno,
                        "column": node.col_offset,
                        "call_site_snippet": line_text,
                        "enclosing_code": fn,
                        "stage3_preview_decision": False,
                        "stage3_preview_rationale": "Pipeline missed: receiver 'stats' omitted from preamble, callee dropped by manifest short-name filter.",
                        "expected_ground_truth": True,
                        "annotation_status": "pre_labeled",
                        "is_deprecated_call": True,
                    })

    errprint_sites.sort(key=lambda x: (x["sample_id"], x["client_line"], x["column"]))
    rvs_sites.sort(key=lambda x: (x["sample_id"], x["client_line"], x["column"]))

    rng = random.Random(42)
    selected_errprint = rng.sample(errprint_sites, 4)
    rng = random.Random(42)
    selected_rvs = rng.sample(rvs_sites, 4)

    misses = selected_errprint + selected_rvs
    logger.info(f"Sampled {len(misses)} genuine pipeline misses (expected 8).")
    return misses


def get_label_anomalies(scipy_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Returns the 2 documented ground-truth label anomalies (GT=False)."""
    anomalies = []
    # 1. scipy_1560 (calls mpmath.factorial at line 40)
    s1560 = scipy_samples[1560]
    fn1560 = s1560["function"]
    anomalies.append({
        "candidate_id": "scipy_1560_label_anomaly_40_18",
        "sample_id": "scipy_1560",
        "sample_idx": 1560,
        "library": "scipy",
        "cohort": "outdated",
        "stratum": "label_anomaly",
        "sub_stratum": "external_library_mislabel",
        "target_api": "scipy.misc.factorial",
        "client_line": 40,
        "column": 18,
        "call_site_snippet": "factorial_N = factorial(N)",
        "enclosing_code": fn1560,
        "stage3_preview_decision": False,
        "stage3_preview_rationale": "Documented anomaly: client code imports and invokes mpmath.factorial, not scipy.misc.factorial.",
        "expected_ground_truth": False,
        "annotation_status": "pre_labeled",
        "is_deprecated_call": False,
    })

    # 2. scipy_1500 (calls _sp_cumtrapz alias at line 40)
    s1500 = scipy_samples[1500]
    fn1500 = s1500["function"]
    anomalies.append({
        "candidate_id": "scipy_1500_label_anomaly_40_19",
        "sample_id": "scipy_1500",
        "sample_idx": 1500,
        "library": "scipy",
        "cohort": "outdated",
        "stratum": "label_anomaly",
        "sub_stratum": "alias_declaration_anomaly",
        "target_api": "scipy.integrate.cumtrapz",
        "client_line": 40,
        "column": 19,
        "call_site_snippet": "ret.data = _sp_cumtrapz(arr, crd_arr.reshape(-1), axis=fld_axis, initial=initial)",
        "enclosing_code": fn1500,
        "stage3_preview_decision": False,
        "stage3_preview_rationale": "Documented anomaly: internal alias declaration inside method body.",
        "expected_ground_truth": False,
        "annotation_status": "pre_labeled",
        "is_deprecated_call": False,
    })

    logger.info(f"Loaded {len(anomalies)} documented label anomalies (expected 2).")
    return anomalies


def sample_hard_negatives(
    numpy_samples: List[Dict[str, Any]],
    pandas_samples: List[Dict[str, Any]],
    scipy_samples: List[Dict[str, Any]],
    pos_sample_ids: Set[str],
) -> List[Dict[str, Any]]:
    """Samples exactly 40 fresh hard negatives across 4 categories (seed=42)."""
    blocked_ids = EXCLUDED_CALIBRATION_IDS | pos_sample_ids
    negatives: List[Dict[str, Any]] = []

    # Category 1: Modern Replacement Overloads (N = 16)
    # 1.A: scipy.special.comb (N = 8) from up-to-date cohort
    comb_pool = []
    for idx, s in enumerate(scipy_samples):
        sid = f"scipy_{idx}"
        if sid in blocked_ids or s.get("category") != "up-to-dated":
            continue
        fn = s.get("function", "")
        if "special.comb" in fn or "sps.comb" in fn:
            tree = safe_ast_parse(fn)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                    if name == "comb":
                        line_text = fn.splitlines()[node.lineno - 1].strip()
                        comb_pool.append({
                            "candidate_id": f"{sid}_neg_comb_{node.lineno}_{node.col_offset}",
                            "sample_id": sid,
                            "sample_idx": idx,
                            "library": "scipy",
                            "cohort": "up-to-dated",
                            "stratum": "hard_negative",
                            "sub_stratum": "replacement_overload_comb",
                            "target_api": "scipy.misc.comb",
                            "client_line": node.lineno,
                            "column": node.col_offset,
                            "call_site_snippet": line_text,
                            "enclosing_code": fn,
                            "stage3_preview_decision": False,
                            "stage3_preview_rationale": "Negative: modern replacement scipy.special.comb.",
                            "expected_ground_truth": False,
                            "annotation_status": "unannotated",
                            "is_deprecated_call": None,
                        })
                        break
    comb_pool.sort(key=lambda x: x["candidate_id"])
    rng = random.Random(42)
    selected_comb = rng.sample(comb_pool, 8)
    negatives.extend(selected_comb)

    # 1.B: scipy.integrate replacements (cumulative_trapezoid, simpson) (N = 4)
    integrate_pool = []
    for idx, s in enumerate(scipy_samples):
        sid = f"scipy_{idx}"
        if sid in blocked_ids or s.get("category") != "up-to-dated":
            continue
        fn = s.get("function", "")
        if "cumulative_trapezoid" in fn or "simpson" in fn:
            tree = safe_ast_parse(fn)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                    if name in ("cumulative_trapezoid", "simpson"):
                        line_text = fn.splitlines()[node.lineno - 1].strip()
                        target = "scipy.integrate.cumtrapz" if name == "cumulative_trapezoid" else "scipy.integrate.simps"
                        integrate_pool.append({
                            "candidate_id": f"{sid}_neg_integrate_{node.lineno}_{node.col_offset}",
                            "sample_id": sid,
                            "sample_idx": idx,
                            "library": "scipy",
                            "cohort": "up-to-dated",
                            "stratum": "hard_negative",
                            "sub_stratum": "replacement_overload_integrate",
                            "target_api": target,
                            "client_line": node.lineno,
                            "column": node.col_offset,
                            "call_site_snippet": line_text,
                            "enclosing_code": fn,
                            "stage3_preview_decision": False,
                            "stage3_preview_rationale": f"Negative: modern replacement {name}.",
                            "expected_ground_truth": False,
                            "annotation_status": "unannotated",
                            "is_deprecated_call": None,
                        })
                        break
    integrate_pool.sort(key=lambda x: x["candidate_id"])
    rng = random.Random(42)
    selected_integrate = rng.sample(integrate_pool, 4)
    negatives.extend(selected_integrate)

    # 1.C: pandas modern replacements (map, ffill, bfill) (N = 4)
    pandas_repl_pool = []
    for idx, s in enumerate(pandas_samples):
        sid = f"pandas_{idx}"
        if sid in blocked_ids or s.get("category") != "up-to-dated":
            continue
        fn = s.get("function", "")
        if any(m in fn for m in [".map(", ".ffill(", ".bfill("]):
            tree = safe_ast_parse(fn)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                    if name in ("map", "ffill", "bfill"):
                        line_text = fn.splitlines()[node.lineno - 1].strip()
                        target = "pandas.DataFrame.applymap" if name == "map" else "pandas.Series.pad"
                        pandas_repl_pool.append({
                            "candidate_id": f"{sid}_neg_pandas_repl_{node.lineno}_{node.col_offset}",
                            "sample_id": sid,
                            "sample_idx": idx,
                            "library": "pandas",
                            "cohort": "up-to-dated",
                            "stratum": "hard_negative",
                            "sub_stratum": "replacement_overload_pandas",
                            "target_api": target,
                            "client_line": node.lineno,
                            "column": node.col_offset,
                            "call_site_snippet": line_text,
                            "enclosing_code": fn,
                            "stage3_preview_decision": False,
                            "stage3_preview_rationale": f"Negative: modern replacement {name}.",
                            "expected_ground_truth": False,
                            "annotation_status": "unannotated",
                            "is_deprecated_call": None,
                        })
                        break
    pandas_repl_pool.sort(key=lambda x: x["candidate_id"])
    rng = random.Random(42)
    selected_pandas_repl = rng.sample(pandas_repl_pool, 4)
    negatives.extend(selected_pandas_repl)

    # Category 2: Submodule / Non-Benchmark Overlaps (N = 8)
    # 2.A: scipy.linalg.pinv (N = 5)
    pinv_pool = []
    for idx, s in enumerate(scipy_samples):
        sid = f"scipy_{idx}"
        if sid in blocked_ids:
            continue
        fn = s.get("function", "")
        if "linalg.pinv(" in fn or "scipy.linalg.pinv(" in fn:
            tree = safe_ast_parse(fn)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                    if name == "pinv":
                        line_text = fn.splitlines()[node.lineno - 1].strip()
                        pinv_pool.append({
                            "candidate_id": f"{sid}_neg_pinv_{node.lineno}_{node.col_offset}",
                            "sample_id": sid,
                            "sample_idx": idx,
                            "library": "scipy",
                            "cohort": s.get("category", "unknown"),
                            "stratum": "hard_negative",
                            "sub_stratum": "submodule_non_deprecated_pinv",
                            "target_api": "scipy.linalg.pinv2",
                            "client_line": node.lineno,
                            "column": node.col_offset,
                            "call_site_snippet": line_text,
                            "enclosing_code": fn,
                            "stage3_preview_decision": False,
                            "stage3_preview_rationale": "Negative: non-deprecated standard pinv, not pinv2.",
                            "expected_ground_truth": False,
                            "annotation_status": "unannotated",
                            "is_deprecated_call": None,
                        })
                        break
    pinv_pool.sort(key=lambda x: x["candidate_id"])
    rng = random.Random(42)
    selected_pinv = rng.sample(pinv_pool, 5)
    negatives.extend(selected_pinv)

    # 2.B: scipy.misc non-benchmark images (ascent, imresize, electrocardiogram) (N = 3)
    misc_pool = []
    for idx, s in enumerate(scipy_samples):
        sid = f"scipy_{idx}"
        if sid in blocked_ids:
            continue
        fn = s.get("function", "")
        if any(f in fn for f in ["misc.ascent", "misc.electrocardiogram", "misc.imresize"]):
            tree = safe_ast_parse(fn)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                    if name in ("ascent", "electrocardiogram", "imresize"):
                        line_text = fn.splitlines()[node.lineno - 1].strip()
                        misc_pool.append({
                            "candidate_id": f"{sid}_neg_misc_other_{node.lineno}_{node.col_offset}",
                            "sample_id": sid,
                            "sample_idx": idx,
                            "library": "scipy",
                            "cohort": s.get("category", "unknown"),
                            "stratum": "hard_negative",
                            "sub_stratum": "misc_non_benchmark_symbol",
                            "target_api": "scipy.misc.face",
                            "client_line": node.lineno,
                            "column": node.col_offset,
                            "call_site_snippet": line_text,
                            "enclosing_code": fn,
                            "stage3_preview_decision": False,
                            "stage3_preview_rationale": f"Negative: non-benchmark function {name}.",
                            "expected_ground_truth": False,
                            "annotation_status": "unannotated",
                            "is_deprecated_call": None,
                        })
                        break
    misc_pool.sort(key=lambda x: x["candidate_id"])
    rng = random.Random(42)
    selected_misc = rng.sample(misc_pool, 3)
    negatives.extend(selected_misc)

    # Category 3: Standard Library Name Collisions (itertools.product) (N = 8)
    product_pool = []
    for lib_name, smp in [("numpy", numpy_samples), ("scipy", scipy_samples), ("pandas", pandas_samples)]:
        for idx, s in enumerate(smp):
            sid = f"{lib_name}_{idx}"
            if sid in blocked_ids:
                continue
            fn = s.get("function", "")
            if "itertools.product" in fn or "it.product" in fn:
                tree = safe_ast_parse(fn)
                if tree is None:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                        if name == "product":
                            line_text = fn.splitlines()[node.lineno - 1].strip()
                            product_pool.append({
                                "candidate_id": f"{sid}_neg_it_product_{node.lineno}_{node.col_offset}",
                                "sample_id": sid,
                                "sample_idx": idx,
                                "library": lib_name,
                                "cohort": s.get("category", "unknown"),
                                "stratum": "hard_negative",
                                "sub_stratum": "stdlib_name_collision_product",
                                "target_api": "numpy.product",
                                "client_line": node.lineno,
                                "column": node.col_offset,
                                "call_site_snippet": line_text,
                                "enclosing_code": fn,
                                "stage3_preview_decision": False,
                                "stage3_preview_rationale": "Negative: itertools.product standard library collision.",
                                "expected_ground_truth": False,
                                "annotation_status": "unannotated",
                                "is_deprecated_call": None,
                            })
                            break
    product_pool.sort(key=lambda x: x["candidate_id"])
    rng = random.Random(42)
    selected_product = rng.sample(product_pool, 8)
    negatives.extend(selected_product)

    # Category 4: Third-Party Wrapper Lookalikes (PySpark / Koalas) (N = 8)
    wrapper_pool = []
    for idx, s in enumerate(pandas_samples):
        sid = f"pandas_{idx}"
        if sid in blocked_ids:
            continue
        fn = s.get("function", "")
        if any(k in fn for k in ["pyspark", "ps.", "psdf", "koalas", "dask"]):
            tree = safe_ast_parse(fn)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                    if name in ("iteritems", "items", "pad", "last", "first", "ffill", "transpose"):
                        line_text = fn.splitlines()[node.lineno - 1].strip()
                        wrapper_pool.append({
                            "candidate_id": f"{sid}_neg_wrapper_{node.lineno}_{node.col_offset}",
                            "sample_id": sid,
                            "sample_idx": idx,
                            "library": "pandas",
                            "cohort": s.get("category", "unknown"),
                            "stratum": "hard_negative",
                            "sub_stratum": "wrapper_lookalike_pyspark",
                            "target_api": "pandas.DataFrame.iteritems",
                            "client_line": node.lineno,
                            "column": node.col_offset,
                            "call_site_snippet": line_text,
                            "enclosing_code": fn,
                            "stage3_preview_decision": False,
                            "stage3_preview_rationale": "Negative: third-party PySpark/Koalas DataFrame/Series method.",
                            "expected_ground_truth": False,
                            "annotation_status": "unannotated",
                            "is_deprecated_call": None,
                        })
                        break
    wrapper_pool.sort(key=lambda x: x["candidate_id"])
    rng = random.Random(42)
    selected_wrapper = rng.sample(wrapper_pool, 8)
    negatives.extend(selected_wrapper)

    logger.info(f"Sampled {len(negatives)} fresh hard negatives (expected 40).")
    return negatives


def main():
    logger.info("Starting Phase 5 stratified sampling (Option B: N = 150)...")

    # 1. Load data sources
    manifest = load_candidate_manifest()
    numpy_samples = load_raw_samples("numpy")
    pandas_samples = load_raw_samples("pandas")
    scipy_samples = load_raw_samples("scipy")

    # 2. Extract strata
    positives = sample_candidate_positives(manifest)
    misses = sample_pipeline_misses(scipy_samples)
    anomalies = get_label_anomalies(scipy_samples)

    pos_sample_ids = {c["sample_id"] for c in positives} | {c["sample_id"] for c in misses} | {c["sample_id"] for c in anomalies}
    negatives = sample_hard_negatives(numpy_samples, pandas_samples, scipy_samples, pos_sample_ids)

    # 3. Combine into frozen evaluation benchmark
    benchmark: List[Dict[str, Any]] = positives + misses + anomalies + negatives

    # Deterministic overall ordering (seed=42 shuffle for presentation)
    rng = random.Random(42)
    rng.shuffle(benchmark)

    # Assign benchmark sequence numbers
    for idx, item in enumerate(benchmark, 1):
        item["benchmark_id"] = f"bench_{idx:03d}"

    # 4. Strict assertions
    assert len(benchmark) == 150, f"Benchmark total {len(benchmark)} != 150"
    pos_count = sum(1 for b in benchmark if b["expected_ground_truth"] is True)
    neg_count = sum(1 for b in benchmark if b["expected_ground_truth"] is False)
    assert pos_count == 108, f"True deprecations {pos_count} != 108"
    assert neg_count == 42, f"True benign {neg_count} != 42"

    # Calibration contamination guard assertion
    hard_neg_sample_ids = {b["sample_id"] for b in benchmark if b["stratum"] == "hard_negative"}
    contamination = hard_neg_sample_ids & EXCLUDED_CALIBRATION_IDS
    assert not contamination, f"Contamination guard failed! Hard negatives contain calibration IDs: {contamination}"

    # Output to JSONL
    out_dir = REPO_ROOT / "data" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase5_benchmark_unannotated.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for item in benchmark:
            f.write(json.dumps(item) + "\n")

    logger.info(f"Successfully generated {out_path} with {len(benchmark)} items.")
    logger.info(f"Class balance: {pos_count} True Deprecations (72.0%), {neg_count} True Benign (28.0%).")


if __name__ == "__main__":
    main()
