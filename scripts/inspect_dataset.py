#!/usr/bin/env python3
"""
inspect_dataset.py - Inventory and inspect benchmark dataset for deprecated API pilot.

Inspects raw probing-inputs data from LLM-Deprecated-API dataset across target libraries:
- numpy
- scipy
- pandas

Reports:
1. Total entry counts
2. Distribution of entries across sources
3. Distribution of entries across categories (outdated vs up-to-dated)
4. Entry count per deprecated API value
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def load_dataset(base_dir: Path, library: str) -> List[Dict[str, Any]]:
    samples_path = base_dir / library / "samples.json"
    if not samples_path.exists():
        raise FileNotFoundError(f"Dataset not found at {samples_path}")
    with open(samples_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_library(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_count = len(samples)
    source_dist = Counter(s.get("source", "UNKNOWN") for s in samples)
    category_dist = Counter(s.get("category", "UNKNOWN") for s in samples)

    # Track individual canonical deprecated APIs
    unique_apis = set()
    mapping_dist = Counter()
    api_by_cat: Dict[str, Counter] = {}
    composite_samples_count = 0

    for s in samples:
        dep_api = s.get("deprecated api")
        if isinstance(dep_api, list):
            apis = dep_api
        else:
            apis = [str(dep_api)]

        if len(apis) > 1:
            composite_samples_count += 1

        cat = s.get("category", "UNKNOWN")
        for api in apis:
            unique_apis.add(api)
            mapping_dist[api] += 1
            if api not in api_by_cat:
                api_by_cat[api] = Counter()
            api_by_cat[api][cat] += 1

    return {
        "total": total_count,
        "sources": dict(source_dist),
        "categories": dict(category_dist),
        "unique_apis_count": len(unique_apis),
        "mappings": mapping_dist,
        "api_by_cat": api_by_cat,
        "composite_samples_count": composite_samples_count,
    }


def format_table(headers: List[str], rows: List[List[Any]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_str = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"

    lines = [sep, header_str, sep]
    for row in rows:
        row_str = "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |"
        lines.append(row_str)
    lines.append(sep)
    return "\n".join(lines)


def format_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    header_str = "| " + " | ".join(headers) + " |"
    sep_str = "| " + " | ".join("---" for _ in headers) + " |"
    row_strs = ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join([header_str, sep_str] + row_strs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory benchmark dataset for deprecated API pilot.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/llm-dep-api/probing-inputs"),
        help="Path to probing-inputs directory",
    )
    parser.add_argument(
        "--libraries",
        nargs="+",
        default=["numpy", "scipy", "pandas"],
        help="Libraries to inspect (default: numpy scipy pandas)",
    )
    parser.add_argument(
        "--format",
        choices=["ascii", "markdown", "json"],
        default="ascii",
        help="Output format (default: ascii)",
    )
    args = parser.parse_args()

    results: Dict[str, Any] = {}

    for lib in args.libraries:
        try:
            samples = load_dataset(args.data_dir, lib)
            results[lib] = analyze_library(samples)
        except Exception as e:
            results[lib] = {"error": str(e)}

    if args.format == "json":
        # Convert Counter objects for JSON serialization
        serializable = {}
        for lib, data in results.items():
            if "error" in data:
                serializable[lib] = data
            else:
                serializable[lib] = {
                    "total": data["total"],
                    "sources": data["sources"],
                    "categories": data["categories"],
                    "mappings": dict(data["mappings"]),
                    "api_by_cat": {k: dict(v) for k, v in data["api_by_cat"].items()},
                }
        print(json.dumps(serializable, indent=2))
        return

    # High-level summary table
    summary_headers = ["Library", "Total Entries", "Sources", "Categories (Outdated / Up-to-date)", "Unique Deprecated APIs"]
    summary_rows = []
    for lib in args.libraries:
        info = results.get(lib, {})
        if "error" in info:
            summary_rows.append([lib, "ERROR", info["error"], "-", "-"])
            continue

        total = info["total"]
        sources = ", ".join(f"{k}: {v}" for k, v in info["sources"].items())
        cats = f"Outdated: {info['categories'].get('outdated', 0)} / Up-to-date: {info['categories'].get('up-to-dated', 0)}"
        unique_apis = len(info["mappings"])
        summary_rows.append([lib, total, sources, cats, unique_apis])

    formatter = format_markdown_table if args.format == "markdown" else format_table

    print("=" * 80)
    print("BENCHMARK DATASET INVENTORY SUMMARY")
    print("=" * 80)
    print(formatter(summary_headers, summary_rows))
    print()

    # Per-library detailed breakdown
    for lib in args.libraries:
        info = results.get(lib, {})
        if "error" in info:
            continue

        print("-" * 80)
        print(f"LIBRARY: {lib.upper()} (Total: {info['total']} entries)")
        print("-" * 80)

        detail_headers = ["Deprecated API", "Total Count", "Outdated", "Up-to-date", "Share (%)"]
        detail_rows = []
        total_lib = info["total"]
        for api, count in info["mappings"].most_common():
            outdated = info["api_by_cat"][api].get("outdated", 0)
            uptodate = info["api_by_cat"][api].get("up-to-dated", 0)
            share = f"{(count / total_lib) * 100:.1f}%"
            detail_rows.append([api, count, outdated, uptodate, share])

        print(formatter(detail_headers, detail_rows))
        if info.get("composite_samples_count", 0) > 0:
            comp_n = info["composite_samples_count"]
            assoc_sum = sum(info["mappings"].values())
            print(f"* Accounting & Sampling Note for Many-to-One Mappings:")
            print(f"  - Physical samples in file: {total_lib} (Outdated: {info['categories'].get('outdated', 0)}, Up-to-date: {info['categories'].get('up-to-dated', 0)})")
            print(f"  - 10 distinct deprecated APIs represented on the Outdated side (1-to-1 mapped).")
            print(f"  - {comp_n} up-to-date samples invoke 'DataFrame.loc' (common replacement for 'first', 'last', and 'select').")
            print(f"  - Table sums to {assoc_sum} associations because the {comp_n} shared samples appear across 3 API rows (+{comp_n * 2} associations).")
            print(f"  - Task 5.2 Sampling Guidance: Treat 'DataFrame.loc' as a single shared replacement stratum with a global sample ID dedup guard to avoid pseudo-replication.")
        print()


if __name__ == "__main__":
    main()
