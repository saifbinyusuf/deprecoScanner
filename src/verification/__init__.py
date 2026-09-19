"""
src/verification - Stage 3: LLM Semantic Verification Stack.

Provides prompt formatting, grounding evidence retrieval, Gemini API client with rate pacing,
and SQLite compound response caching for deprecated API candidate verification.
"""

from src.verification.gemini_client import GeminiClient
from src.verification.verifier_prompt import (
    CandidateProvenance,
    PromptMode,
    VerificationDecision,
    format_verification_prompt,
)
from src.verification.evidence_retriever import EvidenceRetriever
from src.verification.response_cache import ResponseCache

__all__ = [
    "GeminiClient",
    "CandidateProvenance",
    "PromptMode",
    "VerificationDecision",
    "format_verification_prompt",
    "EvidenceRetriever",
    "ResponseCache",
]
