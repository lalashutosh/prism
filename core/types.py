"""
core/types.py
─────────────
All shared data structures for the Prism EU AI Act compliance pipeline.

Pure type definitions only — no logic, no I/O, no imports beyond the
standard library.  Every inter-agent hand-off is expressed through one of
these types.  Raw dicts and strings are never allowed to cross an agent
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ── Enumerations ────────────────────────────────────────────────────────────

class Label(str, Enum):
    """Epistemological label attached to every claim in the pipeline.

    FACT       – stated explicitly in the uploaded document.
    RETRIEVED  – drawn from legislation / official_guidance corpus chunks.
    ASSUMPTION – inferred without direct evidence; must be flagged.
    UNCERTAIN  – evidence is absent, contradictory, or unresolvable.
    """
    FACT       = "FACT"
    RETRIEVED  = "RETRIEVED"
    ASSUMPTION = "ASSUMPTION"
    UNCERTAIN  = "UNCERTAIN"


class Confidence(str, Enum):
    """Confidence level for claims, dimension findings, and summaries.

    Order (strongest → weakest): HIGH > MEDIUM > LOW > INSUFFICIENT.
    INSUFFICIENT means a finding cannot be made at all.
    """
    HIGH         = "HIGH"
    MEDIUM       = "MEDIUM"
    LOW          = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


class ClaimStatus(str, Enum):
    """Result of the validation agent's re-assessment of a weak claim."""
    CONFIRMED  = "CONFIRMED"   # new evidence supports the original claim
    OVERTURNED = "OVERTURNED"  # new evidence contradicts the original claim
    UNRESOLVED = "UNRESOLVED"  # still insufficient evidence after retry


class RiskLevel(str, Enum):
    """EU AI Act risk tier (Articles 5, 6, Annex III)."""
    UNACCEPTABLE = "unacceptable"
    HIGH         = "high"
    LIMITED      = "limited"
    MINIMAL      = "minimal"
    UNKNOWN      = "unknown"    # cannot be determined from available evidence


# ── Retrieval primitive ──────────────────────────────────────────────────────

@dataclass
class Chunk:
    """Single retrieval unit returned by the RAG layer.

    source_type is the authoritative field for label-consistency checks:
      "legislation"       → corpus; must not be cited as FACT
      "official_guidance" → corpus; must not be cited as FACT
      "uploaded_doc"      → user document; must not be cited as RETRIEVED
      anything else       → treated as corpus for label purposes
    """
    chunk_id:    str
    text:        str
    source_type: str             # see docstring above
    article_id:  Optional[str]   # e.g. "Article 3", "Annex I para 2"
    metadata:    dict = field(default_factory=dict)  # arbitrary key-value pairs from the retrieval layer (e.g. page_number, section_heading)


# ── Atomic evidence unit ─────────────────────────────────────────────────────

@dataclass
class Claim:
    """A single legal assertion with full provenance.

    chunk_ids references Chunk.chunk_id values already in the session's
    retrieved_chunk_ids set.  Every Claim that crosses a proxy write-method
    will have its chunk_ids verified against that set.
    """
    claim_id:    str             # unique within the session; format: "{dim}_{n}"
    text:        str
    label:       Label
    confidence:  Confidence
    chunk_ids:   list[str]       # citations into the retrieved corpus
    is_weak:     bool = False    # set by the validation agent when any weak criterion matches
    weak_reason: Optional[str] = None  # one of "LOW_CONFIDENCE", "ASSUMPTION", "UNSUPPORTED"


# ── Dimension findings (base + six subtypes) ─────────────────────────────────

@dataclass
class DimensionFinding:
    """Base for all six legal-dimension assessment sections.

    Subclasses carry additional dimension-specific scalar fields while
    sharing the common claim/confidence/summary structure.
    """
    dimension_id: str
    claims:       list[Claim]  = field(default_factory=list)
    confidence:   Confidence   = Confidence.INSUFFICIENT
    summary:      str          = ""


@dataclass
class DefinitionSection(DimensionFinding):
    """Article 3 + Annex I: does the use case constitute an AI system?"""
    is_ai_system: Optional[bool] = None   # None = undetermined


@dataclass
class RiskSection(DimensionFinding):
    """Article 6 + Annex III: risk-tier classification."""
    risk_level: RiskLevel = RiskLevel.UNKNOWN


@dataclass
class ProhibitedSection(DimensionFinding):
    """Article 5: prohibited AI practices check."""
    triggered_articles: list[str]    = field(default_factory=list)
    prohibited:         Optional[bool] = None  # None = undetermined


@dataclass
class TransparencySection(DimensionFinding):
    """Article 50 + GPAI chapter: transparency and disclosure obligations."""
    applies_to_gpai:       bool = False
    labelling_required:    bool = False
    notification_required: bool = False


@dataclass
class RolesSection(DimensionFinding):
    """Provider vs. deployer determination (Articles 3, 25, 26)."""
    is_provider: bool = False
    is_deployer: bool = False
    is_both:     bool = False


@dataclass
class GovernanceSection(DimensionFinding):
    """Documentation, conformity, oversight obligations (Articles 9–17)."""
    documentation_required: bool = False
    oversight_required:     bool = False
    monitoring_required:    bool = False


# ── Extraction output ────────────────────────────────────────────────────────

@dataclass
class FactSection:
    """Structured output from the extraction agent.

    Populated before any retrieve() call.  All other agents treat this as
    read-only input.  source_chunk_ids references uploaded-doc chunk IDs
    that the orchestrator registers into retrieved_chunk_ids before writing.
    """
    use_case_name:      str
    description:        str
    industry:           Optional[str]      = None
    ai_capabilities:    list[str]          = field(default_factory=list)
    data_inputs:        list[str]          = field(default_factory=list)
    outputs:            list[str]          = field(default_factory=list)
    deployment_context: Optional[str]      = None
    affected_persons:   list[str]          = field(default_factory=list)
    existing_oversight: Optional[str]      = None
    vendor_or_developer: Optional[str]     = None
    additional_facts:   dict               = field(default_factory=dict)
    source_chunk_ids:   list[str]          = field(default_factory=list)


# ── Validation structures ────────────────────────────────────────────────────

@dataclass
class WeakClaim:
    """A claim flagged for independent re-assessment by the validation agent."""
    claim_id:            str
    dimension_id:        str
    claim_text:          str
    reason:              str        # why it was flagged (e.g. "LOW_CONFIDENCE")
    original_confidence: Confidence
    original_label:      Label


@dataclass
class OverturnedClaim:
    """A claim whose original finding was contradicted by new evidence."""
    claim_id:             str
    dimension_id:         str
    original_claim_text:  str
    new_finding:          str
    new_confidence:       Confidence
    new_label:            Label
    new_chunk_ids:        list[str]
    status:               ClaimStatus = ClaimStatus.OVERTURNED


@dataclass
class ValidationFlag:
    """Per-claim verdict produced by the validation agent."""
    claim_id:     str
    dimension_id: str
    status:       ClaimStatus
    notes:        str
    new_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class ValidationSection:
    """Summary of the full validation pass written by the validation agent."""
    flags:               list[ValidationFlag] = field(default_factory=list)
    overall_confidence:  Confidence           = Confidence.INSUFFICIENT
    summary:             str                  = ""


# ── Synthesis outputs ────────────────────────────────────────────────────────

@dataclass
class ReportSection:
    """Ten-section compliance report produced by the synthesis agent.

    Each section is a structured dict (keys defined by the synthesis agent)
    rather than a flat string so that the UI layer can render sections
    independently.
    """
    # Section 1
    use_case_summary:            str  = ""
    # Section 2
    extracted_facts:             dict = field(default_factory=dict)
    # Section 3
    ai_definition_check:         dict = field(default_factory=dict)
    # Section 4
    risk_classification:         dict = field(default_factory=dict)
    # Section 5
    prohibited_practices_check:  dict = field(default_factory=dict)
    # Section 6
    transparency_gpai_obligations: dict = field(default_factory=dict)
    # Section 7
    roles:                       dict = field(default_factory=dict)
    # Section 8
    governance_observations:     dict = field(default_factory=dict)
    # Section 9
    missing_information:         dict = field(default_factory=dict)
    # Section 10
    citations_by_source:         dict = field(default_factory=dict)


@dataclass
class FollowUpSection:
    """Questions the UI layer should surface to the user for clarification."""
    questions:        list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)


@dataclass
class ConfidenceSection:
    """Per-dimension confidence summary written by the synthesis agent.

    Used by the orchestrator to detect quality regression and loop conditions.
    """
    definition_check:    Confidence = Confidence.INSUFFICIENT
    risk_classification: Confidence = Confidence.INSUFFICIENT
    prohibited_practices: Confidence = Confidence.INSUFFICIENT
    transparency:        Confidence = Confidence.INSUFFICIENT
    roles:               Confidence = Confidence.INSUFFICIENT
    governance:          Confidence = Confidence.INSUFFICIENT
    overall:             Confidence = Confidence.INSUFFICIENT


# ── Orchestrator private state ───────────────────────────────────────────────

@dataclass
class OrchestratorState:
    """Internal state owned exclusively by the orchestrator.

    Never exposed to any agent proxy view.  Accessing it through a proxy
    raises AttributeError.
    """
    retrieval_cache:    dict[str, list[Chunk]] = field(default_factory=dict)
    # checkpoints stores named deep copies of SessionMemory (without _orchestrator)
    checkpoints:        dict[str, Any]         = field(default_factory=dict)
    retry_counts:       dict[str, int]         = field(default_factory=dict)  # keys: "analysis_{dim_id}" | "validation_{claim_id}"
    loop_count:         int                    = 0  # incremented once per analysis+validation+synthesis cycle; capped at MAX_LOOP_COUNT
    retrieved_chunk_ids: set[str]              = field(default_factory=set)  # union of all chunk_ids ever returned by retrieve_fn this session


# ── Signals ──────────────────────────────────────────────────────────────────

@dataclass
class RetrievalSignal:
    """Agent → Orchestrator: evidence is insufficient; please retrieve more.

    The orchestrator performs a cache-check, calls retrieve() if needed,
    extends the chunk list, and re-invokes the agent.
    """
    query:     str
    filters:   dict
    dimension: str   # dimension_id for analysis; claim_id for validation


@dataclass
class LoopSignal:
    """Synthesis → Orchestrator: critical dimensions are INSUFFICIENT.

    Emitted BEFORE any write to memory.  The orchestrator rolls back to
    the after_analysis checkpoint and re-runs analysis + validation with
    refined_context injected into the analysis agent's context dict.
    """
    reason:          str
    refined_context: str


@dataclass
class CompletionSignal:
    """Agent → Orchestrator: all writes committed successfully; agent is done."""
    agent:   str = ""
    message: str = ""


# ── Errors ───────────────────────────────────────────────────────────────────

class MemoryWriteError(Exception):
    """Raised by a proxy write method when any validation check fails.

    The orchestrator catches this, logs the reason, and decides whether to
    retry, flag corruption, or roll back to a checkpoint.

    Attributes
    ----------
    check_name : str
        Which validation check failed: "schema" | "citation_integrity" |
        "label_consistency" | "confidence_bounds".
    detail : str
        Human-readable explanation of the exact failure.
    """
    def __init__(self, check_name: str, detail: str) -> None:
        self.check_name = check_name
        self.detail = detail
        super().__init__(f"[{check_name}] {detail}")
