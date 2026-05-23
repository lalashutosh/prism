"""
core/memory.py
──────────────
SessionMemory: single shared state for the entire pipeline.

Per-agent proxy views enforce write isolation and validate every write
across four checks (schema, citation integrity, label consistency,
confidence bounds).  Reads return deep copies so agents cannot mutate
shared state by modifying a returned object.  Sections an agent does not
own are not exposed at all — accessing them raises AttributeError, not a
permission error.

Nothing in this file triggers I/O, LLM calls, or retrieval.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

from core.types import (
    Chunk,
    Claim,
    Confidence,
    ConfidenceSection,
    DefinitionSection,
    DimensionFinding,
    FactSection,
    FollowUpSection,
    GovernanceSection,
    Label,
    MemoryWriteError,
    OrchestratorState,
    OverturnedClaim,
    ProhibitedSection,
    ReportSection,
    RiskSection,
    RolesSection,
    TransparencySection,
    ValidationSection,
    WeakClaim,
)

# ── Source-type sets used by label-consistency validation ────────────────────

# Chunks from the built-in corpus — these may NOT be labeled FACT.
CORPUS_SOURCE_TYPES: frozenset[str] = frozenset({
    "legislation",
    "official_guidance",
    "supporting_document",
    "case_law",
})

# Chunks from the user's uploaded document — these may NOT be labeled RETRIEVED.
UPLOADED_SOURCE_TYPES: frozenset[str] = frozenset({
    "uploaded_doc",
})


# ── SessionMemory ─────────────────────────────────────────────────────────────

@dataclass
class SessionMemory:
    """Single source of truth for the pipeline.

    Field ownership:
      facts                 → extraction agent writes, all agents read
      risk_classification … governance → analysis agent writes
      validation_flags … overturned_claims → validation agent writes
      final_report … confidence_summary  → synthesis agent writes
      _orchestrator         → orchestrator only; never exposed via proxies
    """

    # Owned by extraction agent
    facts:                Optional[FactSection]       = None

    # Owned by analysis agent
    risk_classification:  Optional[RiskSection]       = None
    definition_check:     Optional[DefinitionSection] = None
    prohibited_practices: Optional[ProhibitedSection] = None
    transparency:         Optional[TransparencySection] = None
    roles:                Optional[RolesSection]       = None
    governance:           Optional[GovernanceSection]  = None

    # Owned by validation agent
    validation_flags:     Optional[ValidationSection]  = None
    weak_claims:          list[WeakClaim]              = field(default_factory=list)
    overturned_claims:    list[OverturnedClaim]        = field(default_factory=list)

    # Owned by synthesis agent
    final_report:         Optional[ReportSection]      = None
    follow_up_questions:  Optional[FollowUpSection]    = None
    confidence_summary:   Optional[ConfidenceSection]  = None

    # Private: orchestrator only — never surfaced through any proxy
    _orchestrator:        OrchestratorState            = field(
                              default_factory=OrchestratorState
                          )


# ── Shared validation helpers ─────────────────────────────────────────────────

def _check_schema(value: Any, expected_type: type, field_name: str) -> None:
    if not isinstance(value, expected_type):
        raise MemoryWriteError(
            "schema",
            f"{field_name}: expected {expected_type.__name__}, "
            f"got {type(value).__name__}",
        )


def _check_confidence(conf: Any, field_name: str) -> None:
    if not isinstance(conf, Confidence):
        raise MemoryWriteError(
            "confidence_bounds",
            f"{field_name}: {conf!r} is not a Confidence enum member",
        )


def _check_label(label: Any, field_name: str) -> None:
    if not isinstance(label, Label):
        raise MemoryWriteError(
            "confidence_bounds",        # grouped under same category for simplicity
            f"{field_name}: {label!r} is not a Label enum member",
        )


def _check_citations(
    chunk_ids: list[str],
    retrieved_chunk_ids: set[str],
    field_name: str,
) -> None:
    missing = [cid for cid in chunk_ids if cid not in retrieved_chunk_ids]
    if missing:
        raise MemoryWriteError(
            "citation_integrity",
            f"{field_name}: chunk_ids not in retrieved set: {missing}",
        )


def _check_label_consistency(
    chunk_ids: list[str],
    label: Label,
    chunk_lookup: dict[str, Chunk],
    field_name: str,
) -> None:
    """Corpus chunks → must not be FACT.  Uploaded-doc chunks → must not be RETRIEVED."""
    for cid in chunk_ids:
        chunk = chunk_lookup.get(cid)
        if chunk is None:
            continue  # citation-integrity check already handles truly missing IDs
        if chunk.source_type in CORPUS_SOURCE_TYPES and label == Label.FACT:
            raise MemoryWriteError(
                "label_consistency",
                f"{field_name}: corpus chunk '{cid}' "
                f"(source_type={chunk.source_type!r}) cannot be labeled FACT",
            )
        if chunk.source_type in UPLOADED_SOURCE_TYPES and label == Label.RETRIEVED:
            raise MemoryWriteError(
                "label_consistency",
                f"{field_name}: uploaded_doc chunk '{cid}' "
                f"cannot be labeled RETRIEVED",
            )


def _validate_claims(
    claims: list[Claim],
    retrieved_chunk_ids: set[str],
    chunk_lookup: dict[str, Chunk],
    parent_name: str,
) -> None:
    for i, claim in enumerate(claims):
        loc = f"{parent_name}.claims[{i}]"
        _check_label(claim.label, f"{loc}.label")
        _check_confidence(claim.confidence, f"{loc}.confidence")
        _check_citations(claim.chunk_ids, retrieved_chunk_ids, loc)
        _check_label_consistency(claim.chunk_ids, claim.label, chunk_lookup, loc)


def _validate_dimension_finding(
    finding: DimensionFinding,
    retrieved_chunk_ids: set[str],
    chunk_lookup: dict[str, Chunk],
    field_name: str,
) -> None:
    _check_confidence(finding.confidence, f"{field_name}.confidence")
    _validate_claims(finding.claims, retrieved_chunk_ids, chunk_lookup, field_name)


# ── Proxy classes ─────────────────────────────────────────────────────────────
#
# Each proxy is constructed by the orchestrator and passed to the agent.
# The orchestrator supplies:
#   - a reference to the live SessionMemory object
#   - a reference to the live retrieved_chunk_ids set (mutated as the
#     orchestrator adds new chunks)
#   - a snapshot of chunk_lookup at construction time (rebuilt each re-invoke)
#
# Agents only interact with memory through these proxy objects.

class ExtractionAgentMemoryView:
    """
    READ  : nothing (first agent; memory is empty at this point)
    WRITE : facts
    """

    def __init__(
        self,
        memory: SessionMemory,
        retrieved_chunk_ids: set[str],
        chunk_lookup: dict[str, Chunk],
    ) -> None:
        object.__setattr__(self, "_memory", memory)
        object.__setattr__(self, "_retrieved_chunk_ids", retrieved_chunk_ids)
        object.__setattr__(self, "_chunk_lookup", chunk_lookup)

    def write_facts(self, facts: FactSection) -> None:
        """Write the extraction output.

        Citation integrity for source_chunk_ids is checked only when the
        list is non-empty; the orchestrator must register document chunk IDs
        into retrieved_chunk_ids before calling this method.
        """
        _check_schema(facts, FactSection, "facts")
        if facts.source_chunk_ids:
            _check_citations(
                facts.source_chunk_ids,
                self._retrieved_chunk_ids,
                "facts.source_chunk_ids",
            )
        self._memory.facts = copy.deepcopy(facts)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"ExtractionAgentMemoryView does not expose attribute '{name}'"
        )


class AnalysisAgentMemoryView:
    """
    READ  : facts, validation_flags (loop-context on retry)
            + own section reads (for resume detection)
    WRITE : definition_check, risk_classification, prohibited_practices,
            transparency, roles, governance
    NOT EXPOSED: follow_up_questions, final_report, _orchestrator
    """

    def __init__(
        self,
        memory: SessionMemory,
        retrieved_chunk_ids: set[str],
        chunk_lookup: dict[str, Chunk],
    ) -> None:
        object.__setattr__(self, "_memory", memory)
        object.__setattr__(self, "_retrieved_chunk_ids", retrieved_chunk_ids)
        object.__setattr__(self, "_chunk_lookup", chunk_lookup)

    # ── Readable (deepcopy) ──────────────────────────────────────────────────

    @property
    def facts(self) -> Optional[FactSection]:
        return copy.deepcopy(self._memory.facts)

    @property
    def validation_flags(self) -> Optional[ValidationSection]:
        return copy.deepcopy(self._memory.validation_flags)

    # Own sections readable so the agent can detect which are already done.
    @property
    def definition_check(self) -> Optional[DefinitionSection]:
        return copy.deepcopy(self._memory.definition_check)

    @property
    def risk_classification(self) -> Optional[RiskSection]:
        return copy.deepcopy(self._memory.risk_classification)

    @property
    def prohibited_practices(self) -> Optional[ProhibitedSection]:
        return copy.deepcopy(self._memory.prohibited_practices)

    @property
    def transparency(self) -> Optional[TransparencySection]:
        return copy.deepcopy(self._memory.transparency)

    @property
    def roles(self) -> Optional[RolesSection]:
        return copy.deepcopy(self._memory.roles)

    @property
    def governance(self) -> Optional[GovernanceSection]:
        return copy.deepcopy(self._memory.governance)

    # ── Writable (validated) ─────────────────────────────────────────────────

    def write_definition(self, section: DefinitionSection) -> None:
        _check_schema(section, DefinitionSection, "definition_check")
        _validate_dimension_finding(
            section, self._retrieved_chunk_ids, self._chunk_lookup, "definition_check"
        )
        self._memory.definition_check = copy.deepcopy(section)

    def write_risk(self, section: RiskSection) -> None:
        _check_schema(section, RiskSection, "risk_classification")
        _validate_dimension_finding(
            section, self._retrieved_chunk_ids, self._chunk_lookup, "risk_classification"
        )
        self._memory.risk_classification = copy.deepcopy(section)

    def write_prohibited(self, section: ProhibitedSection) -> None:
        _check_schema(section, ProhibitedSection, "prohibited_practices")
        _validate_dimension_finding(
            section, self._retrieved_chunk_ids, self._chunk_lookup, "prohibited_practices"
        )
        self._memory.prohibited_practices = copy.deepcopy(section)

    def write_transparency(self, section: TransparencySection) -> None:
        _check_schema(section, TransparencySection, "transparency")
        _validate_dimension_finding(
            section, self._retrieved_chunk_ids, self._chunk_lookup, "transparency"
        )
        self._memory.transparency = copy.deepcopy(section)

    def write_roles(self, section: RolesSection) -> None:
        _check_schema(section, RolesSection, "roles")
        _validate_dimension_finding(
            section, self._retrieved_chunk_ids, self._chunk_lookup, "roles"
        )
        self._memory.roles = copy.deepcopy(section)

    def write_governance(self, section: GovernanceSection) -> None:
        _check_schema(section, GovernanceSection, "governance")
        _validate_dimension_finding(
            section, self._retrieved_chunk_ids, self._chunk_lookup, "governance"
        )
        self._memory.governance = copy.deepcopy(section)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"AnalysisAgentMemoryView does not expose attribute '{name}'"
        )


class ValidationAgentMemoryView:
    """
    READ  : facts, all six analysis sections
    WRITE : validation_flags, weak_claims, overturned_claims
    NOT EXPOSED: final_report, follow_up_questions, _orchestrator
    """

    def __init__(
        self,
        memory: SessionMemory,
        retrieved_chunk_ids: set[str],
        chunk_lookup: dict[str, Chunk],
    ) -> None:
        object.__setattr__(self, "_memory", memory)
        object.__setattr__(self, "_retrieved_chunk_ids", retrieved_chunk_ids)
        object.__setattr__(self, "_chunk_lookup", chunk_lookup)

    # ── Readable ─────────────────────────────────────────────────────────────

    @property
    def facts(self) -> Optional[FactSection]:
        return copy.deepcopy(self._memory.facts)

    @property
    def definition_check(self) -> Optional[DefinitionSection]:
        return copy.deepcopy(self._memory.definition_check)

    @property
    def risk_classification(self) -> Optional[RiskSection]:
        return copy.deepcopy(self._memory.risk_classification)

    @property
    def prohibited_practices(self) -> Optional[ProhibitedSection]:
        return copy.deepcopy(self._memory.prohibited_practices)

    @property
    def transparency(self) -> Optional[TransparencySection]:
        return copy.deepcopy(self._memory.transparency)

    @property
    def roles(self) -> Optional[RolesSection]:
        return copy.deepcopy(self._memory.roles)

    @property
    def governance(self) -> Optional[GovernanceSection]:
        return copy.deepcopy(self._memory.governance)

    # ── Writable ─────────────────────────────────────────────────────────────

    def write_validation_flags(self, section: ValidationSection) -> None:
        _check_schema(section, ValidationSection, "validation_flags")
        _check_confidence(section.overall_confidence, "validation_flags.overall_confidence")
        self._memory.validation_flags = copy.deepcopy(section)

    def write_weak_claims(self, claims: list[WeakClaim]) -> None:
        if not isinstance(claims, list):
            raise MemoryWriteError("schema", "weak_claims must be a list")
        for i, wc in enumerate(claims):
            _check_schema(wc, WeakClaim, f"weak_claims[{i}]")
            _check_confidence(wc.original_confidence, f"weak_claims[{i}].original_confidence")
            _check_label(wc.original_label, f"weak_claims[{i}].original_label")
        self._memory.weak_claims = copy.deepcopy(claims)

    def write_overturned_claims(self, claims: list[OverturnedClaim]) -> None:
        if not isinstance(claims, list):
            raise MemoryWriteError("schema", "overturned_claims must be a list")
        for i, oc in enumerate(claims):
            _check_schema(oc, OverturnedClaim, f"overturned_claims[{i}]")
            _check_confidence(oc.new_confidence, f"overturned_claims[{i}].new_confidence")
            _check_label(oc.new_label, f"overturned_claims[{i}].new_label")
            _check_citations(
                oc.new_chunk_ids,
                self._retrieved_chunk_ids,
                f"overturned_claims[{i}]",
            )
            _check_label_consistency(
                oc.new_chunk_ids,
                oc.new_label,
                self._chunk_lookup,
                f"overturned_claims[{i}]",
            )
        self._memory.overturned_claims = copy.deepcopy(claims)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"ValidationAgentMemoryView does not expose attribute '{name}'"
        )


class SynthesisAgentMemoryView:
    """
    READ  : facts, all six analysis sections, validation_flags,
            weak_claims, overturned_claims
    WRITE : final_report, follow_up_questions, confidence_summary
    NOT EXPOSED: _orchestrator
    """

    def __init__(
        self,
        memory: SessionMemory,
        retrieved_chunk_ids: set[str],
        chunk_lookup: dict[str, Chunk],
    ) -> None:
        object.__setattr__(self, "_memory", memory)
        object.__setattr__(self, "_retrieved_chunk_ids", retrieved_chunk_ids)
        object.__setattr__(self, "_chunk_lookup", chunk_lookup)

    # ── Readable ─────────────────────────────────────────────────────────────

    @property
    def facts(self) -> Optional[FactSection]:
        return copy.deepcopy(self._memory.facts)

    @property
    def definition_check(self) -> Optional[DefinitionSection]:
        return copy.deepcopy(self._memory.definition_check)

    @property
    def risk_classification(self) -> Optional[RiskSection]:
        return copy.deepcopy(self._memory.risk_classification)

    @property
    def prohibited_practices(self) -> Optional[ProhibitedSection]:
        return copy.deepcopy(self._memory.prohibited_practices)

    @property
    def transparency(self) -> Optional[TransparencySection]:
        return copy.deepcopy(self._memory.transparency)

    @property
    def roles(self) -> Optional[RolesSection]:
        return copy.deepcopy(self._memory.roles)

    @property
    def governance(self) -> Optional[GovernanceSection]:
        return copy.deepcopy(self._memory.governance)

    @property
    def validation_flags(self) -> Optional[ValidationSection]:
        return copy.deepcopy(self._memory.validation_flags)

    @property
    def weak_claims(self) -> list[WeakClaim]:
        return copy.deepcopy(self._memory.weak_claims)

    @property
    def overturned_claims(self) -> list[OverturnedClaim]:
        return copy.deepcopy(self._memory.overturned_claims)

    # ── Writable ─────────────────────────────────────────────────────────────

    def write_final_report(self, report: ReportSection) -> None:
        _check_schema(report, ReportSection, "final_report")
        self._memory.final_report = copy.deepcopy(report)

    def write_follow_up_questions(self, section: FollowUpSection) -> None:
        _check_schema(section, FollowUpSection, "follow_up_questions")
        self._memory.follow_up_questions = copy.deepcopy(section)

    def write_confidence_summary(self, section: ConfidenceSection) -> None:
        _check_schema(section, ConfidenceSection, "confidence_summary")
        for fname in (
            "definition_check", "risk_classification", "prohibited_practices",
            "transparency", "roles", "governance", "overall",
        ):
            _check_confidence(
                getattr(section, fname),
                f"confidence_summary.{fname}",
            )
        self._memory.confidence_summary = copy.deepcopy(section)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"SynthesisAgentMemoryView does not expose attribute '{name}'"
        )
