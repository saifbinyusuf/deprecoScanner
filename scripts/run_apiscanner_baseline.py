#!/usr/bin/env python3
"""
scripts/run_apiscanner_baseline.py - Controlled Single-Snapshot APIScanner Baseline Comparison.

Downloads and extracts the exact versions evaluated in the APIScanner paper (ICSE 2021, arXiv:2102.09251):
- NumPy: 1.20.0 (Jan 2021)
- Pandas: 1.2.0 (Dec 2020)
- SciPy: 1.6.0 (Dec 2020) & 1.5.4 (Nov 2020)

Runs DeprecoScanner Stage 1 multi-detector stack against these exact single-version snapshots,
and compares candidates directly against APIScanner's reported numbers and bundled outputs.
"""

from __future__ import annotations

import json
import re
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detectors.comment_detector import detect_comment_deprecations
from src.detectors.legacy_heuristics import (
    DeprecationCandidate,
    detect_legacy_deprecations,
)
from src.detectors.parameter_detector import detect_parameter_deprecations
from src.detectors.pep702_detector import detect_pep702_via_griffe
from src.detectors.union_dedup import union_stage1_candidates

BASELINE_SNAPSHOTS = [
    ("numpy", "1.20.0"),
    ("pandas", "1.2.0"),
    ("scipy", "1.6.0"),
    ("scipy", "1.5.4"),
]

APISCANNER_REPORTED = {
    "numpy": {"paper_table1_detected": 39, "paper_table1_manual": 36, "bundled_file_elements": 40},
    "pandas": {"paper_table1_detected": 66, "paper_table1_manual": 59, "bundled_file_elements": 67},
    "scipy": {"paper_table1_detected": 46, "paper_table1_manual": 49, "bundled_file_elements": 49},
}


def download_and_extract(pkg: str, version: str, base_dir: Path) -> Path:
    out_dir = base_dir / f"{pkg}-{version}"
    pkg_dir = out_dir / pkg
    if pkg_dir.exists():
        return pkg_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://pypi.org/pypi/{pkg}/{version}/json"
    req = urllib.request.Request(url, headers={"User-Agent": "deprecoScanner-pilot"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    urls = data.get("urls", [])

    # Prefer sdist .tar.gz for complete source tree
    sdist = next((u for u in urls if u["packagetype"] == "sdist" and u["filename"].endswith(".tar.gz")), None)
    if sdist:
        tar_path = out_dir / sdist["filename"]
        if not tar_path.exists():
            print(f"Downloading {sdist['filename']}...")
            urllib.request.urlretrieve(sdist["url"], tar_path)
        print(f"Extracting {tar_path.name}...")
        prefix = f"{pkg}-{version}/{pkg}/"
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf.getmembers():
                if m.name.startswith(prefix):
                    rel = m.name[len(f"{pkg}-{version}/"):]
                    m.name = rel
                    tf.extract(m, out_dir)
        return pkg_dir

    # Fallback to wheel
    whl = next((u for u in urls if u["packagetype"] == "bdist_wheel"), None)
    if whl:
        whl_path = out_dir / whl["filename"]
        if not whl_path.exists():
            print(f"Downloading {whl['filename']}...")
            urllib.request.urlretrieve(whl["url"], whl_path)
        print(f"Extracting {whl_path.name}...")
        with zipfile.ZipFile(whl_path, "r") as zf:
            members = [m for m in zf.namelist() if m.startswith(f"{pkg}/") and not m.endswith(".so")]
            for m in members:
                zf.extract(m, out_dir)
        return pkg_dir

    raise RuntimeError(f"Could not find package for {pkg}=={version}")


def compute_package_prefix(file_path: Path, lib_root: Path, lib_name: str) -> str:
    rel = file_path.relative_to(lib_root)
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts.pop()
    if not parts:
        return lib_name
    return f"{lib_name}.{'.'.join(parts)}"


def scan_single_tree(pkg_dir: Path, lib_name: str) -> List[DeprecationCandidate]:
    py_files = sorted(list(pkg_dir.rglob("*.py")))
    raw_candidates: List[DeprecationCandidate] = []

    for f in py_files:
        pkg_prefix = compute_package_prefix(f, pkg_dir, lib_name)
        try:
            source_code = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # 1. Legacy heuristics
        try:
            cands = detect_legacy_deprecations(source_code, str(f), pkg_prefix)
            raw_candidates.extend(cands)
        except (SyntaxError, UnicodeDecodeError):
            pass

        # 2. Comment detector
        try:
            cands = detect_comment_deprecations(source_code, str(f), pkg_prefix)
            raw_candidates.extend(cands)
        except (SyntaxError, UnicodeDecodeError):
            pass

        # 3. Parameter detector
        try:
            cands = detect_parameter_deprecations(source_code, str(f), pkg_prefix)
            raw_candidates.extend(cands)
        except (SyntaxError, UnicodeDecodeError):
            pass

        # 4. PEP 702 Griffe detector
        try:
            cands = detect_pep702_via_griffe(source_code, str(f), pkg_prefix)
            raw_candidates.extend(cands)
        except (SyntaxError, UnicodeDecodeError):
            pass

    return raw_candidates


def main():
    base_dir = Path("data/apiscanner_baseline").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CONTROLLED APISCANNER SINGLE-SNAPSHOT COMPARISON")
    print("=" * 70)

    results = {}

    for pkg, version in BASELINE_SNAPSHOTS:
        print(f"\nProcessing {pkg}=={version}...")
        pkg_dir = download_and_extract(pkg, version, base_dir)
        py_files = list(pkg_dir.rglob("*.py"))
        
        t0 = time.time()
        raw_cands = scan_single_tree(pkg_dir, pkg)
        scan_time = time.time() - t0
        
        unique_cands = union_stage1_candidates(raw_cands)
        param_cands = [c for c in unique_cands if c.scope == "parameter"]
        func_cands = [c for c in unique_cands if c.scope == "function"]

        origins_freq = {}
        for c in unique_cands:
            for o in c.origins:
                if o != "pep702":
                    origins_freq[o] = origins_freq.get(o, 0) + 1

        baseline_info = APISCANNER_REPORTED.get(pkg, {})

        results[f"{pkg}-{version}"] = {
            "library": pkg,
            "version": version,
            "files_scanned": len(py_files),
            "scan_time_sec": round(scan_time, 2),
            "raw_candidates": len(raw_cands),
            "unique_candidates": len(unique_cands),
            "function_scoped": len(func_cands),
            "parameter_scoped": len(param_cands),
            "origin_frequencies": origins_freq,
            "apiscanner_paper_table1_detected": baseline_info.get("paper_table1_detected"),
            "apiscanner_paper_table1_manual": baseline_info.get("paper_table1_manual"),
            "apiscanner_bundled_elements": baseline_info.get("bundled_file_elements"),
        }

        print(f"[{pkg}=={version}] Scanned {len(py_files)} files in {scan_time:.2f}s.")
        print(f"  -> DeprecoScanner: {len(raw_cands)} raw -> {len(unique_cands)} unique ({len(func_cands)} function, {len(param_cands)} parameter)")
        print(f"  -> APIScanner Paper Table 1: {baseline_info.get('paper_table1_detected')} detected (manual actual: {baseline_info.get('paper_table1_manual')})")
        print(f"  -> APIScanner Bundled Output: {baseline_info.get('bundled_file_elements')} elements")

    out_file = Path("data/apiscanner_controlled_comparison.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved controlled comparison results to {out_file}")


if __name__ == "__main__":
    main()
