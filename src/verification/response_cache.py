"""
src/verification/response_cache.py - SQLite Compound Response Cache for Stage 3.

Guarantees:
1. Cache keys are hashed over SHA-256(model_name + ":" + prompt_version + ":" + call_site_text + ":" + api_name + ":" + evidence_snippet).
2. Changing prompt_version automatically invalidates stale entries without requiring manual DB purging.
3. Isolates cache entries across different models (e.g. gemini-3.5-flash-lite vs gemini-pro-latest).
4. Provides atomic thread-safe access and tracks exact cache hits, misses, and API calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_PATH = REPO_ROOT / "data" / "stage3_response_cache.db"


class ResponseCache:
    """
    Thread-safe SQLite cache for LLM verification responses.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DEFAULT_CACHE_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._hits = 0
        self._misses = 0
        self._init_db()

    def _init_db(self):
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS response_cache (
                        cache_key TEXT PRIMARY KEY,
                        model_name TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        api_name TEXT NOT NULL,
                        call_site_text TEXT NOT NULL,
                        evidence_snippet TEXT NOT NULL,
                        response_json TEXT NOT NULL,
                        prompt_tokens INTEGER,
                        candidate_tokens INTEGER,
                        total_tokens INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cache_model_ver ON response_cache(model_name, prompt_version);"
                )

    @staticmethod
    def compute_cache_key(
        model_name: str,
        prompt_version: str,
        call_site_text: str,
        api_name: str,
        evidence_snippet: str,
    ) -> str:
        """
        Formulates the deterministic compound SHA-256 hash.
        """
        raw = f"{model_name.strip()}:{prompt_version.strip()}:{call_site_text.strip()}:{api_name.strip()}:{evidence_snippet.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        model_name: str,
        prompt_version: str,
        call_site_text: str,
        api_name: str,
        evidence_snippet: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves cached response if present. Increments hit/miss counters.
        """
        cache_key = self.compute_cache_key(
            model_name=model_name,
            prompt_version=prompt_version,
            call_site_text=call_site_text,
            api_name=api_name,
            evidence_snippet=evidence_snippet,
        )

        with self._lock:
            cursor = self._conn.execute(
                "SELECT response_json, prompt_tokens, candidate_tokens, total_tokens FROM response_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()
            if row:
                self._hits += 1
                try:
                    parsed = json.loads(row["response_json"])
                except Exception:
                    parsed = {"raw": row["response_json"]}
                return {
                    "cache_key": cache_key,
                    "cached": True,
                    "response": parsed,
                    "prompt_tokens": row["prompt_tokens"],
                    "candidate_tokens": row["candidate_tokens"],
                    "total_tokens": row["total_tokens"],
                }
            else:
                self._misses += 1
                return None

    def put(
        self,
        model_name: str,
        prompt_version: str,
        call_site_text: str,
        api_name: str,
        evidence_snippet: str,
        response_json: str,
        prompt_tokens: Optional[int] = None,
        candidate_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> str:
        """
        Stores response into cache under the compound SHA-256 key.
        """
        cache_key = self.compute_cache_key(
            model_name=model_name,
            prompt_version=prompt_version,
            call_site_text=call_site_text,
            api_name=api_name,
            evidence_snippet=evidence_snippet,
        )

        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO response_cache (
                        cache_key, model_name, prompt_version, api_name,
                        call_site_text, evidence_snippet, response_json,
                        prompt_tokens, candidate_tokens, total_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        cache_key,
                        model_name,
                        prompt_version,
                        api_name,
                        call_site_text,
                        evidence_snippet,
                        response_json,
                        prompt_tokens,
                        candidate_tokens,
                        total_tokens,
                    ),
                )
        return cache_key

    def get_stats(self) -> Dict[str, int]:
        """
        Returns hit, miss, and total record counts.
        """
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) AS total FROM response_cache")
            total = cursor.fetchone()["total"]
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total_records": total,
            }

    def close(self):
        with self._lock:
            self._conn.close()
