"""
src/verification/gemini_client.py - Gemini REST API Client for Stage 3 LLM Verification.

Features:
1. Direct Google AI Studio REST API integration (v1beta endpoint).
2. Structured JSON generation mode with Pydantic / OpenAPI schema enforcement.
3. Configurable default models: 'gemini-3.5-flash-lite' (primary) and 'gemini-pro-latest' (validation).
4. Exponential backoff with jitter on HTTP 429 (Resource Exhausted) and 503.
5. Integrated SQLite compound caching with SHA-256 keys to avoid duplicate API calls.
6. Rate pacing and token accounting.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional, Tuple
import urllib.error
import urllib.request

from dotenv import load_dotenv
from pydantic import ValidationError

from src.verification.verifier_prompt import (
    CandidateProvenance,
    PromptMode,
    VerificationDecision,
    format_verification_prompt,
    PROMPT_VERSION,
)
from src.verification.response_cache import ResponseCache

logger = logging.getLogger(__name__)

# Load environment variables from .env if present
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=REPO_ROOT / ".env")

DEFAULT_PRIMARY_MODEL = "gemini-3.5-flash-lite"
DEFAULT_VALIDATION_MODEL = "gemini-3.1-pro-preview"

GEMINI_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

VERIFICATION_JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_deprecated_usage": {
            "type": "BOOLEAN",
            "description": "True if the call site invokes a deprecated API in active code, False if benign, inactive fallback, or false match.",
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence score between 0.0 and 1.0.",
        },
        "rationale": {
            "type": "STRING",
            "description": "Concise one-line explanation of the verification decision.",
        },
    },
    "required": ["is_deprecated_usage", "confidence", "rationale"],
}


class GeminiClient:
    """
    Client for interacting with Google AI Studio Gemini models for candidate verification.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_PRIMARY_MODEL,
        cache: Optional[ResponseCache] = None,
        max_retries: int = 5,
        min_interval_seconds: float = 0.2,  # ~300 RPM rate pacing
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add it to .env or environment variables."
            )
        self.default_model = default_model
        self.cache = cache or ResponseCache()
        self.max_retries = max_retries
        self.min_interval_seconds = min_interval_seconds
        self._last_call_time = 0.0

        # Run-level tracking metrics
        self.total_api_calls = 0
        self.total_cache_hits = 0
        self.total_prompt_tokens = 0
        self.total_candidate_tokens = 0

    def verify_candidate(
        self,
        provenance: CandidateProvenance,
        mode: PromptMode,
        model: Optional[str] = None,
        prompt_version: str = PROMPT_VERSION,
    ) -> Dict[str, Any]:
        """
        Verifies a candidate call site using the specified prompt mode and model.
        Checks the compound SQLite cache first.
        """
        model_name = model or self.default_model
        prompt = format_verification_prompt(provenance, mode)

        # Build evidence snippet string for compound cache key
        evidence_snippet = f"{provenance.evidence_docstring or ''}|{provenance.evidence_warning or ''}|{provenance.recommended_replacement or ''}"

        # 1. Check compound cache
        cached_entry = self.cache.get(
            model_name=model_name,
            prompt_version=prompt_version,
            call_site_text=provenance.call_site_snippet,
            api_name=provenance.target_api,
            evidence_snippet=evidence_snippet,
        )

        if cached_entry is not None:
            self.total_cache_hits += 1
            resp_data = cached_entry["response"]
            return {
                "decision": VerificationDecision(**resp_data),
                "model": model_name,
                "cached": True,
                "cache_key": cached_entry["cache_key"],
                "prompt_tokens": cached_entry["prompt_tokens"] or 0,
                "candidate_tokens": cached_entry["candidate_tokens"] or 0,
                "total_tokens": cached_entry["total_tokens"] or 0,
            }

        # 2. Cache miss -> Call Gemini REST API
        raw_response, usage = self._call_gemini_api(prompt=prompt, model=model_name)
        self.total_api_calls += 1

        prompt_tokens = usage.get("promptTokenCount", 0)
        candidate_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get("totalTokenCount", 0)

        self.total_prompt_tokens += prompt_tokens
        self.total_candidate_tokens += candidate_tokens

        # Validate structured JSON output with Pydantic
        try:
            parsed_json = json.loads(raw_response)
            decision = VerificationDecision(**parsed_json)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse model JSON: {raw_response} - Error: {e}")
            # Fallback safe decision
            decision = VerificationDecision(
                is_deprecated_usage=False,
                confidence=0.0,
                rationale=f"Failed to parse JSON response: {str(e)[:100]}",
            )
            parsed_json = decision.model_dump()

        # 3. Store in SQLite cache
        cache_key = self.cache.put(
            model_name=model_name,
            prompt_version=prompt_version,
            call_site_text=provenance.call_site_snippet,
            api_name=provenance.target_api,
            evidence_snippet=evidence_snippet,
            response_json=json.dumps(decision.model_dump()),
            prompt_tokens=prompt_tokens,
            candidate_tokens=candidate_tokens,
            total_tokens=total_tokens,
        )

        return {
            "decision": decision,
            "model": model_name,
            "cached": False,
            "cache_key": cache_key,
            "prompt_tokens": prompt_tokens,
            "candidate_tokens": candidate_tokens,
            "total_tokens": total_tokens,
        }

    def _call_gemini_api(self, prompt: str, model: str) -> Tuple[str, Dict[str, Any]]:
        """
        Executes HTTP POST request with exponential backoff and rate pacing.
        """
        url = f"{GEMINI_API_ENDPOINT.format(model=model)}?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "response_schema": VERIFICATION_JSON_SCHEMA,
                "temperature": 0.0,
            },
        }
        body_bytes = json.dumps(payload).encode("utf-8")

        for attempt in range(self.max_retries):
            # Enforce minimum interval between calls
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - elapsed)

            req = urllib.request.Request(
                url=url,
                data=body_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                self._last_call_time = time.time()
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))

                # Extract text parts
                candidates = resp_data.get("candidates", [])
                if not candidates:
                    raise RuntimeError("No candidates returned in Gemini API response")

                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts)
                usage = resp_data.get("usageMetadata", {})
                return text.strip(), usage

            except urllib.error.HTTPError as e:
                status_code = e.code
                error_body = e.read().decode("utf-8") if e.fp else str(e)
                logger.warning(f"HTTP {status_code} on attempt {attempt+1}/{self.max_retries}: {error_body}")

                if status_code in (429, 500, 503) and attempt < self.max_retries - 1:
                    sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(f"Gemini API error (HTTP {status_code}): {error_body}") from e
            except Exception as e:
                logger.warning(f"Exception on attempt {attempt+1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                    time.sleep(sleep_time)
                else:
                    raise

        raise RuntimeError("Exceeded maximum retries for Gemini API request")

    def get_run_metrics(self) -> Dict[str, Any]:
        """
        Returns total API calls, cache hits, token usage, and cache hit rate.
        """
        total = self.total_api_calls + self.total_cache_hits
        hit_rate = (self.total_cache_hits / total * 100.0) if total > 0 else 0.0
        return {
            "total_requests": total,
            "api_calls": self.total_api_calls,
            "cache_hits": self.total_cache_hits,
            "cache_hit_rate_pct": round(hit_rate, 2),
            "prompt_tokens": self.total_prompt_tokens,
            "candidate_tokens": self.total_candidate_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_candidate_tokens,
        }
