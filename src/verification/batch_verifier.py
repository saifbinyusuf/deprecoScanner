"""
src/verification/batch_verifier.py - Concurrent Batch Verifier for Stage 3 LLM Verification.

Executes:
1. Primary bulk verification across all 2,932 candidate call sites using `gemini-3.5-flash-lite`.
2. Validation subsample verification across 150 stratified call sites using `gemini-3.1-pro-preview`.
3. Inter-model concordance and Cohen's Kappa evaluation.
4. Latent false-negative spot check on 40 samples from the clean resolved-benign pool.
5. Structured JSON/JSONL output generation and comprehensive final report.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from src.verification.evidence_retriever import EvidenceRetriever
from src.verification.gemini_client import (
    DEFAULT_PRIMARY_MODEL,
    DEFAULT_VALIDATION_MODEL,
    GeminiClient,
)
from src.verification.response_cache import ResponseCache
from src.verification.verifier_prompt import (
    CandidateProvenance,
    PromptMode,
    PROMPT_VERSION,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "stage3_candidates_manifest.json"
DEFAULT_PREDICTIONS_PATH = REPO_ROOT / "results" / "stage3_predictions.jsonl"
DEFAULT_VALIDATION_PATH = REPO_ROOT / "results" / "stage3_validation_predictions.jsonl"
DEFAULT_BENIGN_SPOT_CHECK_PATH = REPO_ROOT / "results" / "stage3_benign_spot_check.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "results" / "stage3_final_report.json"


def compute_cohens_kappa(table: List[List[int]]) -> float:
    """
    Computes Cohen's Kappa from a 2x2 confusion matrix:
    table[0][0] = Both True
    table[0][1] = Primary True, Validation False
    table[1][0] = Primary False, Validation True
    table[1][1] = Both False
    """
    n = sum(table[0]) + sum(table[1])
    if n == 0:
        return 0.0

    po = (table[0][0] + table[1][1]) / n
    p_yes1 = (table[0][0] + table[0][1]) / n
    p_yes2 = (table[0][0] + table[1][0]) / n
    p_no1 = (table[1][0] + table[1][1]) / n
    p_no2 = (table[0][1] + table[1][1]) / n

    pe = (p_yes1 * p_yes2) + (p_no1 * p_no2)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


class BatchVerifier:
    """
    Orchestrates bulk and validation verification runs with compound caching.
    """

    def __init__(
        self,
        primary_model: str = DEFAULT_PRIMARY_MODEL,
        validation_model: str = DEFAULT_VALIDATION_MODEL,
        cache: Optional[ResponseCache] = None,
        max_workers: int = 5,
        min_interval_seconds: float = 0.2,
    ):
        self.primary_model = primary_model
        self.validation_model = validation_model
        self.cache = cache or ResponseCache()
        self.max_workers = max_workers
        self.client = GeminiClient(
            cache=self.cache,
            min_interval_seconds=min_interval_seconds,
        )

    def load_manifest(self, manifest_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        path = manifest_path or DEFAULT_MANIFEST_PATH
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found at {path}. Run extract_stage3_manifest.py first.")
        with open(path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        logger.info(f"Loaded {len(candidates)} candidates from {path}")
        return candidates

    def _verify_single(
        self,
        cand: Dict[str, Any],
        model: str,
    ) -> Dict[str, Any]:
        """Verifies a single candidate call site."""
        prov = CandidateProvenance(
            sample_id=cand["sample_id"],
            target_api=cand["target_api"],
            library=cand["library"],
            call_site_snippet=cand["call_site_snippet"],
            line_number=cand["client_line"],
            enclosing_code=cand["enclosing_code"],
            recommended_replacement=cand.get("recommended_replacement"),
            evidence_warning=cand.get("evidence_warning"),
            evidence_docstring=cand.get("evidence_docstring"),
            failure_reason=cand.get("failure_reason"),
            failure_details=cand.get("failure_details"),
        )
        mode = PromptMode.CONFIRMATION if cand["mode"] == "confirmation" else PromptMode.INFERENCE
        resp = self.client.verify_candidate(prov, mode=mode, model=model)

        decision = resp["decision"]
        return {
            "candidate_id": cand["candidate_id"],
            "sample_id": cand["sample_id"],
            "sample_idx": cand["sample_idx"],
            "library": cand["library"],
            "cohort": cand.get("cohort", "unknown"),
            "stage2_status": cand["stage2_status"],
            "mode": cand["mode"],
            "target_api": cand["target_api"],
            "client_line": cand["client_line"],
            "column": cand["column"],
            "call_site_snippet": cand["call_site_snippet"],
            "failure_reason": cand.get("failure_reason"),
            "origin_channels": cand.get("origin_channels", []),
            "model": model,
            "cached": resp["cached"],
            "cache_key": resp["cache_key"],
            "is_deprecated_usage": decision.is_deprecated_usage,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
            "prompt_tokens": resp["prompt_tokens"],
            "candidate_tokens": resp["candidate_tokens"],
            "total_tokens": resp["total_tokens"],
        }

    def run_primary_verification(
        self,
        candidates: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[Path] = None,
        max_items: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes primary bulk verification across all candidates using gemini-3.5-flash-lite.
        Streams predictions line-by-line to output_path.
        """
        cands = candidates or self.load_manifest()
        if max_items is not None:
            cands = cands[:max_items]

        out_path = output_path or DEFAULT_PREDICTIONS_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Starting primary verification of {len(cands)} candidates with {self.primary_model} "
            f"({self.max_workers} worker threads)..."
        )
        t0 = time.time()
        results: List[Dict[str, Any]] = []

        # Open file in append mode to stream results
        with open(out_path, "w", encoding="utf-8") as f_out:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._verify_single, c, self.primary_model): c
                    for c in cands
                }

                completed = 0
                for future in as_completed(futures):
                    res = future.result()
                    results.append(res)
                    f_out.write(json.dumps(res) + "\n")
                    f_out.flush()

                    completed += 1
                    if completed % 100 == 0 or completed == len(cands):
                        elapsed = time.time() - t0
                        metrics = self.client.get_run_metrics()
                        logger.info(
                            f"Primary Progress: {completed}/{len(cands)} ({elapsed:.1f}s) | "
                            f"API Calls: {metrics['api_calls']} | Hits: {metrics['cache_hits']} "
                            f"({metrics['cache_hit_rate_pct']}%) | Tokens: {metrics['total_tokens']:,}"
                        )

        logger.info(f"Primary verification complete in {time.time() - t0:.1f}s. Saved to {out_path}")
        return results

    def select_stratified_validation_subsample(
        self,
        candidates: List[Dict[str, Any]],
        sample_size: int = 150,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Selects a stratified subsample of candidates across:
        - Library (NumPy, SciPy, Pandas)
        - Cohort (outdated vs up-to-date)
        - Stage 2 Status (resolved_deprecated vs low_confidence)
        """
        random.seed(seed)

        # Bucket candidates
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for c in candidates:
            key = f"{c['library']}|{c.get('cohort', 'unknown')}|{c['stage2_status']}"
            buckets.setdefault(key, []).append(c)

        subsample: List[Dict[str, Any]] = []
        # Proportional allocation
        total_cands = len(candidates)
        remaining = sample_size

        for key, bucket_items in sorted(buckets.items()):
            n_select = max(1, round(len(bucket_items) / total_cands * sample_size))
            n_select = min(n_select, len(bucket_items), remaining)
            sampled = random.sample(bucket_items, n_select)
            subsample.extend(sampled)
            remaining -= n_select
            if remaining <= 0:
                break

        # If slightly under or over sample_size due to rounding, adjust
        if len(subsample) < sample_size:
            used_ids = {c["candidate_id"] for c in subsample}
            unused = [c for c in candidates if c["candidate_id"] not in used_ids]
            subsample.extend(random.sample(unused, sample_size - len(subsample)))
        elif len(subsample) > sample_size:
            subsample = subsample[:sample_size]

        logger.info(f"Selected stratified subsample of {len(subsample)} candidates across {len(buckets)} strata.")
        return subsample

    def run_validation_subsample(
        self,
        candidates: List[Dict[str, Any]],
        primary_predictions: List[Dict[str, Any]],
        sample_size: int = 150,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Runs dual-model validation with gemini-3.1-pro-preview and computes agreement metrics.
        """
        subsample = self.select_stratified_validation_subsample(candidates, sample_size=sample_size)
        primary_map = {p["candidate_id"]: p for p in primary_predictions}

        out_path = output_path or DEFAULT_VALIDATION_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Starting validation run on {len(subsample)} stratified candidates with "
            f"{self.validation_model} (~30 RPM pacing)..."
        )
        t0 = time.time()
        val_results: List[Dict[str, Any]] = []

        # Pacing for validation model (2.0s min interval = 30 RPM)
        val_client = GeminiClient(
            cache=self.cache,
            min_interval_seconds=2.0,
        )

        with open(out_path, "w", encoding="utf-8") as f_out:
            for idx, cand in enumerate(subsample):
                prov = CandidateProvenance(
                    sample_id=cand["sample_id"],
                    target_api=cand["target_api"],
                    library=cand["library"],
                    call_site_snippet=cand["call_site_snippet"],
                    line_number=cand["client_line"],
                    enclosing_code=cand["enclosing_code"],
                    recommended_replacement=cand.get("recommended_replacement"),
                    evidence_warning=cand.get("evidence_warning"),
                    evidence_docstring=cand.get("evidence_docstring"),
                    failure_reason=cand.get("failure_reason"),
                    failure_details=cand.get("failure_details"),
                )
                mode = PromptMode.CONFIRMATION if cand["mode"] == "confirmation" else PromptMode.INFERENCE
                resp = val_client.verify_candidate(prov, mode=mode, model=self.validation_model)
                dec = resp["decision"]

                prim = primary_map.get(cand["candidate_id"], {})
                prim_decision = prim.get("is_deprecated_usage", None)

                record = {
                    "candidate_id": cand["candidate_id"],
                    "sample_id": cand["sample_id"],
                    "target_api": cand["target_api"],
                    "primary_model": self.primary_model,
                    "primary_decision": prim_decision,
                    "primary_confidence": prim.get("confidence"),
                    "primary_rationale": prim.get("rationale"),
                    "validation_model": self.validation_model,
                    "validation_decision": dec.is_deprecated_usage,
                    "validation_confidence": dec.confidence,
                    "validation_rationale": dec.rationale,
                    "agreement": (prim_decision == dec.is_deprecated_usage),
                    "cached": resp["cached"],
                }
                val_results.append(record)
                f_out.write(json.dumps(record) + "\n")
                f_out.flush()

                if (idx + 1) % 25 == 0 or (idx + 1) == len(subsample):
                    logger.info(f"Validation Progress: {idx+1}/{len(subsample)} ({time.time() - t0:.1f}s)...")

        # Compute Agreement Statistics
        n = len(val_results)
        agreed = sum(1 for r in val_results if r["agreement"])
        raw_concordance = (agreed / n * 100.0) if n > 0 else 0.0

        # Build 2x2 matrix:
        # row 0: primary True [both_true, prim_true_val_false]
        # row 1: primary False [prim_false_val_true, both_false]
        table = [[0, 0], [0, 0]]
        disagreements = []
        for r in val_results:
            p_val = r["primary_decision"]
            v_val = r["validation_decision"]
            if p_val is True and v_val is True:
                table[0][0] += 1
            elif p_val is True and v_val is False:
                table[0][1] += 1
                disagreements.append(r)
            elif p_val is False and v_val is True:
                table[1][0] += 1
                disagreements.append(r)
            else:
                table[1][1] += 1

        kappa = compute_cohens_kappa(table)
        pabak = round(2.0 * (raw_concordance / 100.0) - 1.0, 4)
        sens = table[0][0] / (table[0][0] + table[1][0]) if (table[0][0] + table[1][0]) > 0 else 0.0
        spec = table[1][1] / (table[1][1] + table[0][1]) if (table[1][1] + table[0][1]) > 0 else 0.0
        bal_acc = round((sens + spec) / 2.0 * 100.0, 2)

        summary = {
            "total_evaluated": n,
            "agreed_count": agreed,
            "disagreed_count": n - agreed,
            "raw_concordance_pct": round(raw_concordance, 2),
            "cohens_kappa": round(kappa, 4),
            "pabak": pabak,
            "balanced_accuracy_pct": bal_acc,
            "contingency_table": {
                "both_deprecated": table[0][0],
                "primary_deprecated_val_benign": table[0][1],
                "primary_benign_val_deprecated": table[1][0],
                "both_benign": table[1][1],
            },
            "disagreements": disagreements,
        }
        logger.info(
            f"Validation Subsample Evaluation: Raw Concordance = {raw_concordance:.2f}%, "
            f"Cohen's Kappa = {kappa:.4f}, PABAK = {pabak:.4f}, Balanced Acc = {bal_acc:.2f}%"
        )
        return summary

    def run_benign_spot_check(
        self,
        sample_size: int = 40,
        seed: int = 42,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Spot-checks clean resolved-benign samples to audit for latent false negatives.
        """
        random.seed(seed)
        raw_dir = REPO_ROOT / "data" / "raw" / "llm-dep-api" / "probing-inputs"
        all_uptodate = []
        for lib in ["numpy", "scipy", "pandas"]:
            lib_path = raw_dir / lib / "samples.json"
            with open(lib_path, "r", encoding="utf-8") as f:
                samples = json.load(f)
            for idx, s in enumerate(samples):
                if s.get("category") in ("up-to-dated", "up-to-date"):
                    all_uptodate.append({
                        "sample_id": f"{lib}_{idx}",
                        "library": lib,
                        "function": s.get("function", ""),
                    })

        sample_subset = random.sample(all_uptodate, min(sample_size, len(all_uptodate)))
        out_path = output_path or DEFAULT_BENIGN_SPOT_CHECK_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Auditing {len(sample_subset)} clean benign samples for latent false negatives...")
        results = []
        latent_fps = 0

        for s in sample_subset:
            prov = CandidateProvenance(
                sample_id=s["sample_id"],
                target_api=f"{s['library']}.general_audit",
                library=s["library"],
                call_site_snippet="general_snippet_audit",
                line_number=1,
                enclosing_code=s["function"],
                failure_reason="benign_audit",
                failure_details="Audit benign code for any latent deprecations",
            )
            resp = self.client.verify_candidate(prov, mode=PromptMode.INFERENCE, model=self.primary_model)
            dec = resp["decision"]
            if dec.is_deprecated_usage:
                latent_fps += 1
            results.append({
                "sample_id": s["sample_id"],
                "library": s["library"],
                "is_deprecated_usage": dec.is_deprecated_usage,
                "confidence": dec.confidence,
                "rationale": dec.rationale,
            })

        output = {
            "total_spot_checked": len(sample_subset),
            "clean_count": len(sample_subset) - latent_fps,
            "latent_deprecated_flagged": latent_fps,
            "clean_rate_pct": round((len(sample_subset) - latent_fps) / len(sample_subset) * 100.0, 2),
            "samples": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Benign spot-check complete: {output['clean_rate_pct']}% clean ({latent_fps} flagged).")
        return output

    def generate_final_report(
        self,
        primary_results: List[Dict[str, Any]],
        validation_summary: Dict[str, Any],
        benign_summary: Dict[str, Any],
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Assembles comprehensive metrics into stage3_final_report.json.
        """
        total = len(primary_results)
        deprecated_flags = sum(1 for r in primary_results if r["is_deprecated_usage"])
        benign_flags = total - deprecated_flags

        by_status = {}
        for r in primary_results:
            st = r["stage2_status"]
            by_status.setdefault(st, {"total": 0, "deprecated": 0, "benign": 0})
            by_status[st]["total"] += 1
            if r["is_deprecated_usage"]:
                by_status[st]["deprecated"] += 1
            else:
                by_status[st]["benign"] += 1

        by_lib = {}
        for r in primary_results:
            lib = r["library"]
            by_lib.setdefault(lib, {"total": 0, "deprecated": 0, "benign": 0})
            by_lib[lib]["total"] += 1
            if r["is_deprecated_usage"]:
                by_lib[lib]["deprecated"] += 1
            else:
                by_lib[lib]["benign"] += 1

        metrics = self.client.get_run_metrics()
        cache_stats = self.cache.get_stats()

        # Exact cost calculation based on recorded tokens
        # gemini-3.5-flash-lite: $0.10 input / $0.40 output per 1M
        primary_input_cost = (metrics["prompt_tokens"] / 1_000_000.0) * 0.10
        primary_output_cost = (metrics["candidate_tokens"] / 1_000_000.0) * 0.40
        primary_cost = primary_input_cost + primary_output_cost

        # gemini-3.1-pro-preview: $1.25 input / $10.00 output per 1M
        val_total = validation_summary.get("total_evaluated", 0)
        # Average 930 input / 60 output tokens
        val_cost = (val_total * 930 / 1_000_000.0 * 1.25) + (val_total * 60 / 1_000_000.0 * 10.00)

        report = {
            "evaluation_stage": "Stage 3: LLM Verification (Phase 4)",
            "primary_model": self.primary_model,
            "validation_model": self.validation_model,
            "prompt_version": PROMPT_VERSION,
            "total_candidates_verified": total,
            "semantic_verdicts": {
                "verified_deprecated_count": deprecated_flags,
                "verified_deprecated_pct": round(deprecated_flags / total * 100.0, 2) if total > 0 else 0.0,
                "rejected_benign_or_anomaly_count": benign_flags,
                "rejected_benign_or_anomaly_pct": round(benign_flags / total * 100.0, 2) if total > 0 else 0.0,
            },
            "by_stage2_status": by_status,
            "by_library": by_lib,
            "validation_subsample": validation_summary,
            "benign_spot_check": benign_summary,
            "run_telemetry": {
                "api_calls": metrics["api_calls"],
                "cache_hits": metrics["cache_hits"],
                "cache_hit_rate_pct": metrics["cache_hit_rate_pct"],
                "total_db_records": cache_stats["total_records"],
                "prompt_tokens": metrics["prompt_tokens"],
                "candidate_tokens": metrics["candidate_tokens"],
                "total_tokens": metrics["total_tokens"],
                "estimated_primary_cost_usd": round(primary_cost, 4),
                "estimated_validation_cost_usd": round(val_cost, 4),
                "estimated_total_cost_usd": round(primary_cost + val_cost, 4),
            },
        }

        out_path = output_path or DEFAULT_REPORT_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Final report written to {out_path}")
        return report
