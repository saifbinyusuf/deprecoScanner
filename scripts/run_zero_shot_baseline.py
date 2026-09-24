#!/usr/bin/env python3
"""
scripts/run_zero_shot_baseline.py - Zero-Shot LLM Deprecation Detection Baseline.

Evaluates gemini-3.5-flash-lite directly on the frozen Phase 5 benchmark (N = 150)
without static candidate extraction, without retrieved documentation/warnings,
and without Jedi type resolution.

Strict anti-leakage:
- Code and target line number only.
- Gold standard labels are withheld and used only downstream to score predictions.
- Results cached in data/zero_shot_response_cache.db for deterministic, zero-cost replay.
- Per-request token counts recorded and reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=REPO_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zero_shot_baseline")

BENCHMARK_PATH = REPO_ROOT / "data" / "benchmark" / "ground_truth_benchmark.jsonl"
CACHE_DB_PATH = REPO_ROOT / "data" / "zero_shot_response_cache.db"
PREDICTIONS_PATH = REPO_ROOT / "results" / "zero_shot_predictions.jsonl"
SUMMARY_PATH = REPO_ROOT / "results" / "zero_shot_evaluation_summary.json"

MODEL_NAME = "gemini-3.5-flash-lite"
API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

ZERO_SHOT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_deprecated_usage": {
            "type": "BOOLEAN",
            "description": "True if the call site on the target line invokes a deprecated API, False otherwise.",
        },
        "deprecated_api": {
            "type": "STRING",
            "description": "The qualified name of the deprecated API if deprecated, else null.",
        },
        "replacement_api": {
            "type": "STRING",
            "description": "The recommended modern replacement API if deprecated, else null.",
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence score between 0.0 and 1.0.",
        },
        "rationale": {
            "type": "STRING",
            "description": "Explanation for the decision.",
        },
    },
    "required": ["is_deprecated_usage", "confidence", "rationale"],
}


class ZeroShotCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    sample_id TEXT,
                    client_line INTEGER,
                    prompt_hash TEXT,
                    response_json TEXT,
                    prompt_tokens INTEGER,
                    candidate_tokens INTEGER,
                    total_tokens INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response_json, prompt_tokens, candidate_tokens, total_tokens FROM cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "response": json.loads(row[0]),
                    "prompt_tokens": row[1] or 0,
                    "candidate_tokens": row[2] or 0,
                    "total_tokens": row[3] or 0,
                }
        return None

    def put(
        self,
        cache_key: str,
        sample_id: str,
        client_line: int,
        prompt_hash: str,
        response_json: str,
        prompt_tokens: int,
        candidate_tokens: int,
        total_tokens: int,
    ):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache 
                (cache_key, sample_id, client_line, prompt_hash, response_json, prompt_tokens, candidate_tokens, total_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    sample_id,
                    client_line,
                    prompt_hash,
                    response_json,
                    prompt_tokens,
                    candidate_tokens,
                    total_tokens,
                ),
            )
            conn.commit()


def format_zero_shot_prompt(enclosing_code: str, client_line: int, call_site_snippet: str, library: str) -> str:
    return f"""You are an expert Python static analysis and API migration assistant.
Analyze the following Python code snippet. The user wants to know if the function call on line {client_line} invokes a deprecated API in Python (e.g., from {library} or standard library).

Enclosing code:
```python
{enclosing_code}
```

Target call site on line {client_line}:
`{call_site_snippet}`

Instructions:
1. Determine whether line {client_line} invokes a deprecated Python API in active code execution.
2. If it is a benign call (e.g., modern replacement API, method on an unrelated object, standard library function, or inactive fallback), set is_deprecated_usage to false.
3. If it is genuinely calling a deprecated function or method, set is_deprecated_usage to true.
4. Output your decision as valid JSON matching the schema.
"""


def call_gemini(api_key: str, prompt: str) -> Tuple[Dict[str, Any], int, int, int]:
    url = f"{API_ENDPOINT}?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": ZERO_SHOT_SCHEMA,
            "temperature": 0.0,
        },
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    max_retries = 5
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))

            candidates = resp_data.get("candidates", [])
            if not candidates:
                raise RuntimeError("No candidates returned from Gemini API")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            usage = resp_data.get("usageMetadata", {})

            prompt_tokens = usage.get("promptTokenCount", 0)
            cand_tokens = usage.get("candidatesTokenCount", 0)
            total_tokens = usage.get("totalTokenCount", 0)

            parsed = json.loads(text.strip())
            return parsed, prompt_tokens, cand_tokens, total_tokens

        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < max_retries - 1:
                sleep_sec = (2 ** attempt) + 1.0
                time.sleep(sleep_sec)
            else:
                raise


def compute_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, Any]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt is True and yp is True)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt is False and yp is True)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt is True and yp is False)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt is False and yp is False)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Run zero-shot LLM baseline on frozen benchmark.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items for test run.")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment or .env!")
        sys.exit(1)

    logger.info("Loading frozen benchmark from %s...", BENCHMARK_PATH)
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        items = items[: args.limit]

    logger.info("Evaluating %d benchmark items...", len(items))
    cache = ZeroShotCache(CACHE_DB_PATH)

    predictions = []
    ground_truth = []
    results_records = []

    total_prompt_tokens = 0
    total_candidate_tokens = 0
    cache_hits = 0
    api_calls = 0

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(PREDICTIONS_PATH, "w", encoding="utf-8")

    t0 = time.time()
    for idx, item in enumerate(items, start=1):
        bid = item["benchmark_id"]
        cid = item["candidate_id"]
        sid = item["sample_id"]
        line_num = item["client_line"]
        snippet = item["call_site_snippet"]
        code = item["enclosing_code"]
        lib = item["library"]
        gt = item["is_deprecated_call"]

        prompt_str = format_zero_shot_prompt(code, line_num, snippet, lib)
        prompt_hash = hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()
        cache_key = hashlib.sha256(f"{MODEL_NAME}:{line_num}:{prompt_hash}".encode("utf-8")).hexdigest()

        cached_entry = cache.get(cache_key)
        if cached_entry:
            parsed = cached_entry["response"]
            p_tok = cached_entry["prompt_tokens"]
            c_tok = cached_entry["candidate_tokens"]
            t_tok = cached_entry["total_tokens"]
            cache_hits += 1
        else:
            parsed, p_tok, c_tok, t_tok = call_gemini(api_key, prompt_str)
            cache.put(
                cache_key=cache_key,
                sample_id=sid,
                client_line=line_num,
                prompt_hash=prompt_hash,
                response_json=json.dumps(parsed),
                prompt_tokens=p_tok,
                candidate_tokens=c_tok,
                total_tokens=t_tok,
            )
            api_calls += 1
            # Rate pacing: ~0.15s between calls
            time.sleep(0.15)

        total_prompt_tokens += p_tok
        total_candidate_tokens += c_tok

        pred_bool = bool(parsed.get("is_deprecated_usage", False))
        predictions.append(pred_bool)
        ground_truth.append(gt)

        record = {
            "benchmark_id": bid,
            "candidate_id": cid,
            "sample_id": sid,
            "library": lib,
            "stratum": item["stratum"],
            "target_api": item["target_api"],
            "client_line": line_num,
            "ground_truth": gt,
            "zero_shot_pred": pred_bool,
            "confidence": parsed.get("confidence", 0.0),
            "rationale": parsed.get("rationale", ""),
            "detected_api": parsed.get("deprecated_api"),
            "replacement_api": parsed.get("replacement_api"),
            "prompt_tokens": p_tok,
            "candidate_tokens": c_tok,
            "total_tokens": t_tok,
            "correct": pred_bool == gt,
        }
        results_records.append(record)
        out_file.write(json.dumps(record) + "\n")
        out_file.flush()

        if idx % 25 == 0 or idx == len(items):
            logger.info("Progress: %d/%d (%.1fs) | API: %d, Hits: %d", idx, len(items), time.time() - t0, api_calls, cache_hits)

    out_file.close()
    elapsed = time.time() - t0

    # Metrics
    overall_metrics = compute_metrics(ground_truth, predictions)

    # By library
    by_lib = {}
    for lib in ["numpy", "scipy", "pandas"]:
        lib_idxs = [i for i, r in enumerate(results_records) if r["library"] == lib]
        lib_gt = [ground_truth[i] for i in lib_idxs]
        lib_preds = [predictions[i] for i in lib_idxs]
        by_lib[lib] = compute_metrics(lib_gt, lib_preds)
        by_lib[lib]["total"] = len(lib_idxs)

    # Token usage stats
    n_items = len(items)
    token_stats = {
        "total_prompt_tokens": total_prompt_tokens,
        "total_candidate_tokens": total_candidate_tokens,
        "total_tokens": total_prompt_tokens + total_candidate_tokens,
        "avg_prompt_tokens_per_request": round(total_prompt_tokens / n_items, 1) if n_items else 0,
        "avg_candidate_tokens_per_request": round(total_candidate_tokens / n_items, 1) if n_items else 0,
        "avg_total_tokens_per_request": round((total_prompt_tokens + total_candidate_tokens) / n_items, 1) if n_items else 0,
    }

    summary = {
        "model": MODEL_NAME,
        "total_items": n_items,
        "elapsed_seconds": round(elapsed, 2),
        "api_calls": api_calls,
        "cache_hits": cache_hits,
        "token_usage": token_stats,
        "overall_metrics": overall_metrics,
        "by_library": by_lib,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("======================================================================")
    logger.info("ZERO-SHOT BASELINE EVALUATION COMPLETE")
    logger.info("======================================================================")
    logger.info("TP=%d, FP=%d, TN=%d, FN=%d", overall_metrics["tp"], overall_metrics["fp"], overall_metrics["tn"], overall_metrics["fn"])
    logger.info("Precision: %.2f%%, Recall: %.2f%%, F1: %.4f", overall_metrics["precision"] * 100, overall_metrics["recall"] * 100, overall_metrics["f1"])
    logger.info("Tokens per request: prompt=%.1f, candidate=%.1f, total=%.1f", token_stats["avg_prompt_tokens_per_request"], token_stats["avg_candidate_tokens_per_request"], token_stats["avg_total_tokens_per_request"])
    logger.info("Saved to %s", SUMMARY_PATH)


if __name__ == "__main__":
    main()
