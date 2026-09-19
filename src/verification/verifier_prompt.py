"""
src/verification/verifier_prompt.py - Provenance-Branching Prompt Formatter & Schema.

Formats verification prompts for Stage 3 LLM verification, branching based on Stage 2 candidate status:
- Branch A (Confirmation Mode): For resolved_deprecated candidates, with explicit checks for
  known ground-truth anomalies (fallback branches, composite-row artifacts, third-party lookalikes).
- Branch B (Inference Mode): For low_confidence candidates (unresolved_receiver, empty_goto),
  reasoning over surrounding code context to infer receiver types.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

PROMPT_VERSION = "v1.0"


class PromptMode(str, Enum):
    CONFIRMATION = "confirmation"  # Branch A: Resolved candidates
    INFERENCE = "inference"        # Branch B: Low-confidence candidates


class CandidateProvenance(BaseModel):
    """Metadata and context describing the candidate call site."""
    sample_id: str
    target_api: str
    library: str
    call_site_snippet: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    enclosing_code: str
    failure_reason: Optional[str] = None  # e.g., "unresolved_receiver", "empty_goto"
    failure_details: Optional[str] = None
    evidence_docstring: Optional[str] = None
    evidence_warning: Optional[str] = None
    recommended_replacement: Optional[str] = None
    source_location: Optional[str] = None


class VerificationDecision(BaseModel):
    """Structured JSON schema returned by Gemini for candidate verification."""
    is_deprecated_usage: bool = Field(
        description="True if the call site invokes a deprecated API in active code, False if benign, inactive fallback, or false match."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )
    rationale: str = Field(
        description="Concise one-line explanation of the semantic verification decision."
    )


def format_verification_prompt(provenance: CandidateProvenance, mode: PromptMode) -> str:
    """
    Constructs the prompt for Gemini using provenance-guided branching.
    """
    evidence_parts = []
    if provenance.recommended_replacement:
        evidence_parts.append(f"- Recommended Replacement: `{provenance.recommended_replacement}`")
    if provenance.evidence_warning:
        evidence_parts.append(f"- Deprecation Warning Message: {provenance.evidence_warning}")
    if provenance.evidence_docstring:
        evidence_parts.append(f"- Docstring Notice: {provenance.evidence_docstring}")
    if provenance.source_location:
        evidence_parts.append(f"- Library Provenance: {provenance.source_location}")

    evidence_text = "\n".join(evidence_parts) if evidence_parts else "- Deprecation verified in historical API catalog."

    if mode == PromptMode.CONFIRMATION:
        prompt = f"""You are an expert static analysis and API migration auditor verifying whether a Python code snippet invokes a deprecated library API.

PROMPT VERSION: {PROMPT_VERSION}
TARGET API: `{provenance.target_api}` (Library: {provenance.library})
CALL SITE SNIPPET (Line {provenance.line_number or '?'}):
```python
{provenance.call_site_snippet}
```

GROUND TRUTH DEPRECATION EVIDENCE:
{evidence_text}

ENCLOSING CODE CONTEXT:
```python
{provenance.enclosing_code}
```

VERIFICATION TASK (Branch A: Confirmation Mode):
A static analysis tool flagged the call site as invoking deprecated API `{provenance.target_api}`.
Determine whether this call site represents a genuine, active invocation of the deprecated API or a false positive.

CRITICAL INSTRUCTIONS & ANOMALY CHECKS:
1. Fallback & Compatibility Guards: Check if this call site is inside an inactive fallback branch (e.g. `try...except ImportError` or `if hasattr(...)` or version check) where the primary code branch uses modern APIs. If the call is merely a legacy fallback or safety guard, verify if the code as written uses the deprecated API when executed in its target environment.
2. Third-Party Lookalikes: Verify whether the symbol belongs to the target library (`{provenance.library}`) rather than a lookalike module from another package (e.g., `mpmath.factorial` vs `scipy.misc.factorial`, or standard library `math`).
3. Companion / Composite Artifacts: Ensure that the flagged call site actually corresponds to `{provenance.target_api}` and not an adjacent, non-deprecated function call on the same line or in the same expression.

Respond ONLY with a JSON object matching this schema:
{{
  "is_deprecated_usage": <boolean: true if genuine deprecated usage in active code, false otherwise>,
  "confidence": <float: 0.0 to 1.0>,
  "rationale": <string: one-sentence explanation justifying your decision>
}}"""
    else:  # PromptMode.INFERENCE
        prompt = f"""You are an expert Python static analysis and type inference auditor investigating an unresolved API candidate call site.

PROMPT VERSION: {PROMPT_VERSION}
TARGET API CANDIDATE: `{provenance.target_api}` (Library: {provenance.library})
STATIC RESOLUTION FAILURE REASON: {provenance.failure_reason or 'unresolved_receiver'}
FAILURE DETAILS: {provenance.failure_details or 'Static analyzer could not resolve the receiver object type.'}

CALL SITE SNIPPET (Line {provenance.line_number or '?'}):
```python
{provenance.call_site_snippet}
```

GROUND TRUTH DEPRECATION EVIDENCE:
{evidence_text}

ENCLOSING CODE CONTEXT:
```python
{provenance.enclosing_code}
```

VERIFICATION TASK (Branch B: Type Inference Mode):
Static analysis could not resolve the receiver variable's type (e.g., due to dynamic typing, missing imports, or chained calls) and preserved this candidate as low-confidence.
Inspect the enclosing code context, variable naming, method invocations, docstrings, and parameter semantics to infer the receiver's type.

CRITICAL INSTRUCTIONS:
1. Type & Receiver Inference: Determine whether the receiver object of `{provenance.call_site_snippet}` is an instance of the class defining `{provenance.target_api}` (e.g., is `df` a `pandas.DataFrame` or `styler` a `pandas.io.formats.style.Styler`?).
2. Disambiguation: If the receiver is an unrelated object or standard Python collection (e.g. dict, list, string) having a method with the same name, flag as false.
3. If the evidence supports that the call invokes `{provenance.target_api}` in active code, return `is_deprecated_usage: true`. Otherwise return `false`.

Respond ONLY with a JSON object matching this schema:
{{
  "is_deprecated_usage": <boolean: true if genuine deprecated usage, false otherwise>,
  "confidence": <float: 0.0 to 1.0>,
  "rationale": <string: one-sentence explanation justifying the inferred type and decision>
}}"""

    return prompt.strip()
