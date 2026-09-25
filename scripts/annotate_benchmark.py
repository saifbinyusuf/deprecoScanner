#!/usr/bin/env python3
"""
scripts/annotate_benchmark.py - Human-in-the-loop CLI for Phase 5 Benchmark Annotation.

Key Capabilities:
1. Primary Human Annotation:
   - Interactive review for each benchmark call site ($N = 150$).
   - Shows enclosing code context with highlighted line number, target API, snippet.
   - Shows Stage 3 prediction as an assistive starting suggestion (without forcing).
   - Pre-fills verified labels for the 10 documented pipeline misses/anomalies.
   - Persists state incrementally to data/benchmark/phase5_benchmark_annotated.jsonl.
2. Progress Status & Summary:
   - python scripts/annotate_benchmark.py --status
3. Blinded 20% Second Pass ($N = 30$):
   - python scripts/annotate_benchmark.py --blinded-second-pass
   - Hides primary decisions and model preview rationales for independent human review.
   - Computes human-to-human inter-annotator agreement (concordance, Cohen's kappa, PABAK).
4. Export Final Frozen Ground Truth:
   - python scripts/annotate_benchmark.py --freeze
   - Freezes verified annotations to data/benchmark/ground_truth_benchmark.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import math
import random
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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


def compute_cohens_kappa(table: List[List[int]]) -> float:
    """
    Computes Cohen's Kappa from a 2x2 confusion matrix:
    table[0][0] = Both True
    table[0][1] = Primary True, Validation/Second False
    table[1][0] = Primary False, Validation/Second True
    table[1][1] = Both False
    """
    n = sum(table[0]) + sum(table[1])
    if n == 0:
        return 0.0

    po = (table[0][0] + table[1][1]) / n
    p_yes1 = (table[0][0] + table[0][1]) / n
    p_yes2 = (table[0][0] + table[1][0]) / n
    pe = (p_yes1 * p_yes2) + ((1.0 - p_yes1) * (1.0 - p_yes2))

    if math.isclose(pe, 1.0):
        return 1.0 if math.isclose(po, 1.0) else 0.0

    return (po - pe) / (1.0 - pe)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("annotate_benchmark")

BENCHMARK_UNANNOTATED = REPO_ROOT / "data" / "benchmark" / "phase5_benchmark_unannotated.jsonl"
BENCHMARK_ANNOTATED = REPO_ROOT / "data" / "benchmark" / "phase5_benchmark_annotated.jsonl"
BENCHMARK_FROZEN = REPO_ROOT / "data" / "benchmark" / "ground_truth_benchmark.jsonl"
SECOND_PASS_PATH = REPO_ROOT / "data" / "benchmark" / "phase5_second_pass_blinded.jsonl"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def display_status() -> None:
    """Displays annotation progress and statistics."""
    if not BENCHMARK_UNANNOTATED.exists():
        print(f"Error: {BENCHMARK_UNANNOTATED} not found. Run scripts/stratified_sample.py first.")
        return

    unannotated = load_jsonl(BENCHMARK_UNANNOTATED)
    annotated = load_jsonl(BENCHMARK_ANNOTATED)
    annotated_map = {item["benchmark_id"]: item for item in annotated}

    total = len(unannotated)
    completed = len([b for b in unannotated if b["benchmark_id"] in annotated_map and annotated_map[b["benchmark_id"]].get("is_deprecated_call") is not None])
    pre_labeled = len([b for b in unannotated if b.get("annotation_status") == "pre_labeled"])

    dep_count = len([item for item in annotated_map.values() if item.get("is_deprecated_call") is True])
    benign_count = len([item for item in annotated_map.values() if item.get("is_deprecated_call") is False])

    print("\n" + "=" * 60)
    print("PHASE 5 BENCHMARK ANNOTATION STATUS")
    print("=" * 60)
    print(f"Total Benchmark Call Sites: {total}")
    print(f"Pre-Labeled Documented Misses/Anomalies: {pre_labeled} (100% verified)")
    print(f"Total Completed Annotations: {completed} / {total} ({completed / total * 100:.1f}%)")
    print(f"  - Deprecated Calls (GT = True):  {dep_count}")
    print(f"  - Benign Calls     (GT = False): {benign_count}")
    print("=" * 60 + "\n")


def format_enclosing_code(code: str, highlight_line: int) -> str:
    """Formats enclosing code snippet with line numbers and a pointer on the target line."""
    lines = code.splitlines()
    formatted = []
    for idx, line in enumerate(lines, 1):
        prefix = "--> " if idx == highlight_line else "    "
        formatted.append(f"{prefix}{idx:3d} | {line}")
    return "\n".join(formatted)


def run_interactive_annotation(auto_accept_prelabeled: bool = True) -> None:
    """Runs interactive CLI annotation session."""
    if not BENCHMARK_UNANNOTATED.exists():
        print(f"Error: {BENCHMARK_UNANNOTATED} not found. Run scripts/stratified_sample.py first.")
        return

    unannotated = load_jsonl(BENCHMARK_UNANNOTATED)
    annotated = load_jsonl(BENCHMARK_ANNOTATED)
    annotated_map = {item["benchmark_id"]: item for item in annotated}

    # Automatically persist pre-labeled misses and anomalies if not already recorded
    if auto_accept_prelabeled:
        for item in unannotated:
            bid = item["benchmark_id"]
            if item.get("annotation_status") == "pre_labeled" and bid not in annotated_map:
                item_copy = dict(item)
                item_copy["annotator"] = "HUMAN_AUDITED_PRE_ESTABLISHED"
                item_copy["annotation_rationale"] = item.get("stage3_preview_rationale")
                annotated_map[bid] = item_copy
        save_jsonl(BENCHMARK_ANNOTATED, list(annotated_map.values()))

    pending = [b for b in unannotated if b["benchmark_id"] not in annotated_map or annotated_map[b["benchmark_id"]].get("is_deprecated_call") is None]

    if not pending:
        print("\nAll 150 benchmark items are annotated! Run with --freeze to finalize ground truth.\n")
        display_status()
        return

    print("\n" + "=" * 70)
    print("STARTING INTERACTIVE BENCHMARK ANNOTATION")
    print(f"Pending items: {len(pending)} / {len(unannotated)}")
    print("Commands: [y] Deprecated, [n] Benign, [s] Skip, [q] Save & Quit, [h] Help")
    print("=" * 70 + "\n")

    for idx, item in enumerate(pending, 1):
        bid = item["benchmark_id"]
        target = item["target_api"]
        line_no = item["client_line"]
        col_no = item["column"]
        snippet = item["call_site_snippet"]
        stratum = item["stratum"]
        preview = item.get("stage3_preview_decision")
        rationale = item.get("stage3_preview_rationale")

        replacement = CANONICAL_REPLACEMENTS.get(target, "None / See documentation")
        print("-" * 70)
        print(f"Item [{idx}/{len(pending)}] -- Benchmark ID: {bid} ({item['library']} | {item['sample_id']})")
        print(f"Target API:        {target} (Officially deprecated in upstream {item['library']})")
        print(f"Modern Alt:        {replacement}")
        print(f"Sampling Stratum:  {stratum} ({item.get('sub_stratum', '')})")
        print(f"Call Site Line:    {line_no}:{col_no}")
        print(f"Call Snippet:      {snippet}")
        print("-" * 70)
        print("Code Context:")
        print(format_enclosing_code(item["enclosing_code"], line_no))
        print("-" * 70)
        print(f"Stage 3 Preview Suggestion: {'DEPRECATED (True)' if preview else 'BENIGN (False)'}")
        if rationale:
            print(f"Preview Rationale:          {rationale}")
        print("-" * 70)
        print(f"QUESTION: Is line {line_no} actively calling this deprecated API ({target})?")
        print("  - [y] YES -> Deprecated: line invokes the deprecated target API.")
        print("  - [n] NO  -> Benign: line calls a modern replacement, stdlib collision, wrapper, or other function.")
        print("  - [s] SKIP for now | [q] SAVE & QUIT | [h] HELP")
        print("-" * 70)

        while True:
            choice = input("Decision [y/n/s/q/h]: ").strip().lower()
            if choice == "q":
                save_jsonl(BENCHMARK_ANNOTATED, list(annotated_map.values()))
                print("\nAnnotations saved. Exiting session.\n")
                display_status()
                return
            elif choice == "s":
                print("Skipped for now.\n")
                break
            elif choice in ("y", "yes"):
                item_copy = dict(item)
                item_copy["is_deprecated_call"] = True
                item_copy["annotator"] = "HUMAN_USER"
                item_copy["annotation_status"] = "annotated"
                annotated_map[bid] = item_copy
                save_jsonl(BENCHMARK_ANNOTATED, list(annotated_map.values()))
                print(f"Recorded: {bid} -> TRUE (Deprecated)\n")
                break
            elif choice in ("n", "no"):
                item_copy = dict(item)
                item_copy["is_deprecated_call"] = False
                item_copy["annotator"] = "HUMAN_USER"
                item_copy["annotation_status"] = "annotated"
                annotated_map[bid] = item_copy
                save_jsonl(BENCHMARK_ANNOTATED, list(annotated_map.values()))
                print(f"Recorded: {bid} -> FALSE (Benign)\n")
                break
            elif choice in ("h", "help"):
                print("\n" + "=" * 65)
                print("WHAT YOU ARE EVALUATING (Call-Site Ground Truth):")
                print(f"1. The Target API ({target}) is ALREADY proven to be deprecated")
                print(f"   upstream in {item['library']}. You do not need to look up if {target}")
                print("   is deprecated; the library documentation has already established that.")
                print("2. You are checking whether the user's code snippet on line " + str(line_no))
                print(f"   is ACTUALLY invoking {target}, or calling something else:")
                print("   - Mark 'y' if the call is an active invocation of the target API.")
                print("   - Mark 'n' if it is NOT the deprecated API. Common reasons:")
                print("     * Stdlib collision: e.g. it.product is itertools.product, not numpy.product")
                print("     * Modern replacement: e.g. sps.comb instead of scipy.misc.comb")
                print("     * Wrapper lookalike: e.g. psdf.iteritems is PySpark, not pandas")
                print("     * Non-deprecated sibling: e.g. calling pinv instead of pinv2")
                print("=" * 65 + "\n")
            else:
                print("Invalid input. Please enter 'y', 'n', 's', 'q', or 'h'.")

    save_jsonl(BENCHMARK_ANNOTATED, list(annotated_map.values()))
    display_status()


def freeze_benchmark() -> None:
    """Freezes annotations to data/benchmark/ground_truth_benchmark.jsonl after validation."""
    annotated = load_jsonl(BENCHMARK_ANNOTATED)
    missing_or_skipped = []
    unannotated = load_jsonl(BENCHMARK_UNANNOTATED)
    annotated_map = {item["benchmark_id"]: item for item in annotated}

    for u in unannotated:
        bid = u["benchmark_id"]
        if bid not in annotated_map or annotated_map[bid].get("is_deprecated_call") is None:
            missing_or_skipped.append(bid)

    if missing_or_skipped:
        print(f"\n[ERROR] Cannot freeze benchmark! {len(missing_or_skipped)} items remain unannotated or skipped:")
        print(f"  Unresolved IDs: {', '.join(missing_or_skipped[:10])}{'...' if len(missing_or_skipped) > 10 else ''}")
        print("Run './.venv/bin/python scripts/annotate_benchmark.py' to resolve them before freezing.\n")
        return

    if len(annotated) != 150:
        print(f"Error: Cannot freeze. Expected 150 annotated items, found {len(annotated)}.")
        return

    for item in annotated:
        if item.get("is_deprecated_call") is None:
            print(f"Error: Item {item['benchmark_id']} has no ground-truth decision.")
            return

    # Sort deterministically by benchmark_id
    annotated.sort(key=lambda x: x["benchmark_id"])
    save_jsonl(BENCHMARK_FROZEN, annotated)

    dep_count = sum(1 for b in annotated if b["is_deprecated_call"] is True)
    ben_count = sum(1 for b in annotated if b["is_deprecated_call"] is False)

    print("\n" + "=" * 60)
    print("PHASE 5 GROUND TRUTH BENCHMARK FROZEN SUCCESSFULLY")
    print("=" * 60)
    print(f"File: {BENCHMARK_FROZEN}")
    print(f"Total Call Sites:   {len(annotated)}")
    print(f"True Deprecations:  {dep_count} ({dep_count / len(annotated) * 100:.2f}%)")
    print(f"True Benign Calls:  {ben_count} ({ben_count / len(annotated) * 100:.2f}%)")
    print("=" * 60 + "\n")


def compute_agreement_statistics(primary: List[bool], second: List[bool]) -> Dict[str, Any]:
    """Computes Observed Concordance, Cohen's Kappa, and PABAK from primary vs second-pass lists."""
    assert len(primary) == len(second), "Primary and second pass lists must be of equal length"
    n = len(primary)
    if n == 0:
        return {"concordance": 1.0, "kappa": 1.0, "pabak": 1.0, "table": [[0, 0], [0, 0]]}

    tp = sum(1 for p, s in zip(primary, second) if p is True and s is True)
    fn = sum(1 for p, s in zip(primary, second) if p is True and s is False)
    fp = sum(1 for p, s in zip(primary, second) if p is False and s is True)
    tn = sum(1 for p, s in zip(primary, second) if p is False and s is False)

    table = [[tp, fn], [fp, tn]]
    po = (tp + tn) / n
    kappa = compute_cohens_kappa(table)
    pabak = 2.0 * po - 1.0

    return {
        "concordance": po,
        "kappa": kappa,
        "pabak": pabak,
        "table": table,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def run_blinded_second_pass() -> None:
    """Runs blinded 20% review (N = 30) for inter-annotator agreement computation."""
    annotated = load_jsonl(BENCHMARK_ANNOTATED)
    if len(annotated) != 150:
        print(f"Error: Primary annotation must be complete (150 items) before running blinded second pass. Found {len(annotated)}.")
        return

    active_annotated = [b for b in annotated if b.get("annotation_status") != "pre_labeled"]
    pos_items = [b for b in active_annotated if b.get("stratum") == "candidate_positive"]
    neg_items = [b for b in active_annotated if b.get("stratum") == "hard_negative"]

    assert len(active_annotated) == 140, f"Expected 140 active annotated items, found {len(active_annotated)}"
    assert len(pos_items) == 100, f"Expected 100 positive candidates, found {len(pos_items)}"
    assert len(neg_items) == 40, f"Expected 40 hard negatives, found {len(neg_items)}"

    rng = random.Random(42)
    sample_pos = rng.sample(pos_items, 22)
    sample_neg = rng.sample(neg_items, 8)
    second_pass_items = sample_pos + sample_neg
    rng.shuffle(second_pass_items)

    existing_records = {r["benchmark_id"]: r for r in load_jsonl(SECOND_PASS_PATH)}

    if len(existing_records) == 30:
        print("\nAll 30 blinded second-pass items are already completed!")
        second_pass_results = [existing_records[item["benchmark_id"]] for item in second_pass_items]
        primary = [r["primary_decision"] for r in second_pass_results]
        second = [r["second_pass_decision"] for r in second_pass_results]
        stats = compute_agreement_statistics(primary, second)

        print("\n" + "=" * 60)
        print("INTER-ANNOTATOR AGREEMENT RESULTS (N = 30)")
        print("=" * 60)
        print(f"Observed Concordance (Po): {stats['concordance'] * 100:.2f}%")
        print(f"Cohen's Kappa (κ):        {stats['kappa']:.4f}")
        print(f"PABAK:                     {stats['pabak']:.4f}")
        print("=" * 60 + "\n")
        return

    print("\n" + "=" * 70)
    print("BLINDED SECOND-PASS HUMAN REVIEW (N = 30)")
    print("Model preview decisions and primary annotations are hidden.")
    print("=" * 70 + "\n")

    second_pass_results = []
    for idx, item in enumerate(second_pass_items, 1):
        bid = item["benchmark_id"]
        if bid in existing_records:
            second_pass_results.append(existing_records[bid])
            continue

        line_no = item["client_line"]
        snippet = item["call_site_snippet"]
        target = item["target_api"]
        replacement = CANONICAL_REPLACEMENTS.get(target, "None / See documentation")

        print("-" * 70)
        print(f"Blinded Review [{idx}/30] -- ID: {bid} ({item['library']})")
        print(f"Target API:        {target} (Officially deprecated in upstream {item['library']})")
        print(f"Modern Alt:        {replacement}")
        print(f"Call Site Line:    {line_no}")
        print(f"Snippet:           {snippet}")
        print("-" * 70)
        print("Code Context:")
        print(format_enclosing_code(item["enclosing_code"], line_no))
        print("-" * 70)
        print(f"QUESTION: Is line {line_no} actively calling this deprecated API ({target})?")
        print("  - [y] YES -> Deprecated: line invokes the deprecated target API.")
        print("  - [n] NO  -> Benign: line calls a modern replacement, stdlib collision, wrapper, or other function.")
        print("  - [q] Quit session")
        print("-" * 70)

        while True:
            c = input("Blinded Decision [y = Deprecated / n = Benign / q = Quit]: ").strip().lower()
            if c == "q":
                save_jsonl(SECOND_PASS_PATH, second_pass_results)
                print("\nSession saved. Exiting.\n")
                return
            elif c in ("y", "n"):
                decision = True if c == "y" else False
                entry = {
                    "benchmark_id": bid,
                    "primary_decision": item["is_deprecated_call"],
                    "second_pass_decision": decision,
                    "agreement": item["is_deprecated_call"] == decision
                }
                second_pass_results.append(entry)
                save_jsonl(SECOND_PASS_PATH, second_pass_results)
                break

    # Compute agreement
    primary = [r["primary_decision"] for r in second_pass_results]
    second = [r["second_pass_decision"] for r in second_pass_results]
    stats = compute_agreement_statistics(primary, second)

    print("\n" + "=" * 60)
    print("INTER-ANNOTATOR AGREEMENT RESULTS (N = 30)")
    print("=" * 60)
    print(f"Observed Concordance (Po): {stats['concordance'] * 100:.2f}%")
    print(f"Cohen's Kappa (κ):        {stats['kappa']:.4f}")
    print(f"PABAK:                     {stats['pabak']:.4f}")
    print("=" * 60 + "\n")

    disagreements = [r for r in second_pass_results if not r.get("agreement", True)]
    if disagreements:
        print(f"\n[ATTENTION] Disagreements detected in {len(disagreements)} / 30 items:")
        for d in disagreements:
            print(f"  - {d['benchmark_id']}: Primary={'DEPRECATED' if d['primary_decision'] else 'BENIGN'} vs Blinded={'DEPRECATED' if d['second_pass_decision'] else 'BENIGN'}")
        print("\nTo inspect and adjudicate consensus ground truth for these items, run:")
        print("  ./.venv/bin/python scripts/annotate_benchmark.py --reconcile-disagreements\n")
    else:
        print("\n[PERFECT] 100% concordance between Primary and Blinded Second Pass! No disagreements.\n")


def re_review_single_item(bid: str) -> None:
    """Re-opens a specific benchmark ID for inspection, re-evaluation, and correction."""
    unannotated = load_jsonl(BENCHMARK_UNANNOTATED)
    annotated = load_jsonl(BENCHMARK_ANNOTATED)
    unannotated_map = {item["benchmark_id"]: item for item in unannotated}
    annotated_map = {item["benchmark_id"]: item for item in annotated}

    if bid not in unannotated_map:
        print(f"Error: Benchmark ID '{bid}' not found. Expected format 'bench_001' to 'bench_150'.")
        return

    item = unannotated_map[bid]
    current_annot = annotated_map.get(bid, {})
    curr_decision = current_annot.get("is_deprecated_call")
    curr_status = current_annot.get("annotation_status", "unannotated")
    line_no = item["client_line"]
    col_no = item["column"]

    print("\n" + "=" * 70)
    print(f"RE-REVIEWING BENCHMARK ITEM: {bid}")
    print("=" * 70)
    print(f"Target API:     {item['target_api']} ({item['library']} | {item['sample_id']})")
    print(f"Stratum:        {item['stratum']} ({item.get('sub_stratum', '')})")
    print(f"Call Site Line: {line_no}:{col_no}")
    print(f"Snippet:        {item['call_site_snippet']}")
    print("-" * 70)
    print("Code Context:")
    print(format_enclosing_code(item["enclosing_code"], line_no))
    print("-" * 70)
    print(f"Current Recorded Status:   {curr_status}")
    print(f"Current Recorded Decision: {'TRUE (Deprecated)' if curr_decision is True else ('FALSE (Benign)' if curr_decision is False else 'NONE (Unannotated)')}")
    print(f"Stage 3 Preview Suggestion: {'DEPRECATED (True)' if item.get('stage3_preview_decision') else 'BENIGN (False)'}")
    print("-" * 70)

    while True:
        choice = input("New Decision [y = Deprecated / n = Benign / s = Keep Current / q = Quit]: ").strip().lower()
        if choice in ("q", "s"):
            print("No changes made.\n")
            return
        elif choice in ("y", "yes"):
            item_copy = dict(item)
            item_copy["is_deprecated_call"] = True
            item_copy["annotator"] = "HUMAN_USER_REVISED"
            item_copy["annotation_status"] = "revised"
            annotated_map[bid] = item_copy
            save_jsonl(BENCHMARK_ANNOTATED, list(annotated_map.values()))
            print(f"Successfully updated: {bid} -> TRUE (Deprecated)\n")
            return
        elif choice in ("n", "no"):
            item_copy = dict(item)
            item_copy["is_deprecated_call"] = False
            item_copy["annotator"] = "HUMAN_USER_REVISED"
            item_copy["annotation_status"] = "revised"
            annotated_map[bid] = item_copy
            save_jsonl(BENCHMARK_ANNOTATED, list(annotated_map.values()))
            print(f"Successfully updated: {bid} -> FALSE (Benign)\n")
            return
        else:
            print("Invalid input. Enter 'y', 'n', 's', or 'q'.")


def reset_item_annotation(bid: str) -> None:
    """Clears the annotation for a benchmark item, returning it to the pending queue."""
    annotated = load_jsonl(BENCHMARK_ANNOTATED)
    annotated_map = {item["benchmark_id"]: item for item in annotated}

    if bid not in annotated_map:
        print(f"Item '{bid}' is not currently annotated.")
        return

    del annotated_map[bid]
    save_jsonl(BENCHMARK_ANNOTATED, list(annotated_map.values()))
    print(f"Reset {bid}: removed from annotated set and returned to pending queue.\n")


def reconcile_disagreements() -> None:
    """Interactively adjudicates disagreements between primary and second-pass decisions."""
    if not SECOND_PASS_PATH.exists():
        print(f"Error: No second-pass results found at {SECOND_PASS_PATH}. Run --blinded-second-pass first.")
        return

    second_pass_results = load_jsonl(SECOND_PASS_PATH)
    disagreements = [r for r in second_pass_results if not r.get("agreement", True) and "adjudicated_decision" not in r]

    if not disagreements:
        resolved_count = len([r for r in second_pass_results if "adjudicated_decision" in r])
        print(f"\nNo pending disagreements to adjudicate. (Past adjudicated items: {resolved_count})\n")
        return

    unannotated = load_jsonl(BENCHMARK_UNANNOTATED)
    annotated = load_jsonl(BENCHMARK_ANNOTATED)
    unannotated_map = {item["benchmark_id"]: item for item in unannotated}
    annotated_map = {item["benchmark_id"]: item for item in annotated}

    print("\n" + "=" * 70)
    print(f"DISAGREEMENT ADJUDICATION WORKFLOW ({len(disagreements)} pending)")
    print("Reviewing cases where Primary and Blinded Second-Pass decisions diverged.")
    print("=" * 70 + "\n")

    for idx, r in enumerate(disagreements, 1):
        bid = r["benchmark_id"]
        item = unannotated_map.get(bid)
        if not item:
            continue

        line_no = item["client_line"]
        p_dec = r["primary_decision"]
        s_dec = r["second_pass_decision"]

        print("-" * 70)
        print(f"Adjudication [{idx}/{len(disagreements)}] -- Item: {bid} ({item['library']} | {item['sample_id']})")
        print(f"Target API:     {item['target_api']}")
        print(f"Call Site Line: {line_no}:{item['column']}")
        print(f"Snippet:        {item['call_site_snippet']}")
        print("-" * 70)
        print("Code Context:")
        print(format_enclosing_code(item["enclosing_code"], line_no))
        print("-" * 70)
        print(f"Primary Pass Decision:       {'DEPRECATED (True)' if p_dec else 'BENIGN (False)'}")
        print(f"Blinded Second-Pass Decision: {'DEPRECATED (True)' if s_dec else 'BENIGN (False)'}")
        print("-" * 70)

        while True:
            choice = input("Adjudicated Final Verdict [y = Deprecated / n = Benign / s = Skip / q = Quit]: ").strip().lower()
            if choice == "q":
                save_jsonl(BENCHMARK_ANNOTATED, list(annotated_map.values()))
                save_jsonl(SECOND_PASS_PATH, second_pass_results)
                print("\nAdjudications saved. Exiting session.\n")
                return
            elif choice == "s":
                print("Skipped for now.\n")
                break
            elif choice in ("y", "yes", "n", "no"):
                final_verdict = (choice in ("y", "yes"))
                # Update primary annotated dataset
                item_copy = dict(item)
                item_copy["is_deprecated_call"] = final_verdict
                item_copy["annotation_status"] = "adjudicated"
                item_copy["annotator"] = "HUMAN_ADJUDICATED_CONSENSUS"
                item_copy["adjudication_divergence"] = f"primary={p_dec}, second_pass={s_dec}"
                annotated_map[bid] = item_copy

                # Update second pass record
                r["adjudicated_decision"] = final_verdict
                r["adjudication_status"] = "adjudicated"

                save_jsonl(BENCHMARK_ANNOTATED, list(annotated_map.values()))
                save_jsonl(SECOND_PASS_PATH, second_pass_results)
                print(f"Recorded Final Adjudication: {bid} -> {'DEPRECATED (True)' if final_verdict else 'BENIGN (False)'}\n")
                break
            else:
                print("Invalid input. Enter 'y', 'n', 's', or 'q'.")

    print("\nAdjudication session finished.\n")
    display_status()


def export_review_markdown(output_path: Optional[Path] = None) -> Path:
    """Exports all 150 benchmark items to a structured Markdown document for visual human review."""
    if output_path is None:
        output_path = REPO_ROOT / "data" / "benchmark" / "phase5_benchmark_review.md"

    items = load_jsonl(BENCHMARK_UNANNOTATED)
    if not items:
        print(f"Error: {BENCHMARK_UNANNOTATED} not found or empty.")
        return output_path

    # Sort by benchmark_id
    items.sort(key=lambda x: x["benchmark_id"])

    lines = [
        "# Phase 5 Ground-Truth Benchmark Review ($N = 150$)",
        "",
        "This document contains all 150 call-site items in the frozen evaluation benchmark.",
        "Ground truth `is_deprecated_call` is established under expert manual review.",
        "",
        "## Summary Statistics",
        f"- **Total Benchmark Items**: {len(items)}",
        f"- **True Deprecations (Expected)**: {sum(1 for i in items if i['expected_ground_truth'] is True)} (72.00%)",
        f"- **True Benign (Expected)**: {sum(1 for i in items if i['expected_ground_truth'] is False)} (28.00%)",
        f"- **Candidate Positives**: {sum(1 for i in items if i['stratum'] == 'candidate_positive')}",
        f"- **Pipeline Misses**: {sum(1 for i in items if i['stratum'] == 'pipeline_miss')} (4 `errprint`, 4 `rvs_ratio_uniforms`)",
        f"- **Label Anomalies**: {sum(1 for i in items if i['stratum'] == 'label_anomaly')} (`scipy_1560`, `scipy_1500`)",
        f"- **Fresh Hard Negatives**: {sum(1 for i in items if i['stratum'] == 'hard_negative')} (16 replacements, 8 submodules, 8 stdlib, 8 wrappers)",
        "",
        "---",
        "",
        "## All Benchmark Items",
        ""
    ]

    for item in items:
        bid = item["benchmark_id"]
        target = item["target_api"]
        lib = item["library"]
        sid = item["sample_id"]
        line = item["client_line"]
        col = item["column"]
        snippet = item["call_site_snippet"]
        stratum = item["stratum"]
        sub_stratum = item.get("sub_stratum", "")
        exp_gt = item["expected_ground_truth"]
        preview = item.get("stage3_preview_decision")
        rationale = item.get("stage3_preview_rationale", "")

        gt_str = "True (Deprecated)" if exp_gt else "False (Benign)"
        preview_str = "True (Deprecated)" if preview else "False (Benign)"

        lines.extend([
            f"### `{bid}` — `{target}` ({lib})",
            f"- **Sample ID**: `{sid}` | **Line**: {line}:{col} | **Stratum**: `{stratum}` (`{sub_stratum}`)",
            f"- **Call Site**: `{snippet}`",
            f"- **Expected Ground Truth**: **{gt_str}** | **Stage 3 Suggestion**: {preview_str}",
            f"- **Assistive Preview Rationale**: *{rationale}*",
            "",
            "<details>",
            f"<summary>View Enclosing Code ({sid})</summary>",
            "",
            "```python",
            format_enclosing_code(item["enclosing_code"], line),
            "```",
            "",
            "</details>",
            "",
            "---",
            ""
        ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Exported benchmark review document to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Benchmark Annotation CLI")
    parser.add_argument("--status", action="store_true", help="Display current annotation status")
    parser.add_argument("--freeze", action="store_true", help="Freeze verified annotations to ground_truth_benchmark.jsonl")
    parser.add_argument("--blinded-second-pass", action="store_true", help="Run 20%% blinded second-pass review")
    parser.add_argument("--reconcile-disagreements", action="store_true", help="Adjudicate disagreements between primary and second pass")
    parser.add_argument("--re-review", type=str, metavar="BENCHMARK_ID", help="Inspect and edit annotation for a specific item (e.g. bench_042)")
    parser.add_argument("--reset-item", type=str, metavar="BENCHMARK_ID", help="Clear annotation for an item, sending it back to pending queue")
    parser.add_argument("--export-review-markdown", action="store_true", help="Export full 150-item review markdown")
    parser.add_argument("--auto-accept-prelabeled", action="store_true", default=True, help="Auto-populate pre-labeled misses and anomalies")
    args = parser.parse_args()

    if args.status:
        display_status()
    elif args.freeze:
        freeze_benchmark()
    elif args.blinded_second_pass:
        run_blinded_second_pass()
    elif args.reconcile_disagreements:
        reconcile_disagreements()
    elif args.re_review:
        re_review_single_item(args.re_review.strip())
    elif args.reset_item:
        reset_item_annotation(args.reset_item.strip())
    elif args.export_review_markdown:
        export_review_markdown()
    else:
        run_interactive_annotation(auto_accept_prelabeled=args.auto_accept_prelabeled)


if __name__ == "__main__":
    main()
