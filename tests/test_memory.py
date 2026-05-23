"""
tests/test_memory.py
─────────────────────
Tests for SessionMemory, all four proxy views, write validation,
read isolation, and checkpoint save/restore.

All tests are pure unit tests — no LLM calls, no retrieval calls.
"""

import copy

import pytest

from core.types import (
    Chunk,
    Claim,
    ClaimStatus,
    Confidence,
    ConfidenceSection,
    DefinitionSection,
    FactSection,
    FollowUpSection,
    GovernanceSection,
    Label,
    MemoryWriteError,
    OverturnedClaim,
    ProhibitedSection,
    ReportSection,
    RiskLevel,
    RiskSection,
    RolesSection,
    TransparencySection,
    ValidationFlag,
    ValidationSection,
    WeakClaim,
)
from core.memory import (
    AnalysisAgentMemoryView,
    ExtractionAgentMemoryView,
    SessionMemory,
    SynthesisAgentMemoryView,
    ValidationAgentMemoryView,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────
# These helpers build minimal valid objects for test setup.
# Default values are chosen to pass all four validation checks so tests can
# selectively break one check at a time by overriding a single argument.

def _make_chunk(
    chunk_id: str = "leg_001",
    source_type: str = "legislation",
    text: str = "EU AI Act Article 3 definition.",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_type=source_type,
        article_id="Article 3",
    )


def _make_claim(
    chunk_ids: list[str] = None,
    label: Label = Label.RETRIEVED,
    confidence: Confidence = Confidence.HIGH,
    claim_id: str = "def_0",
) -> Claim:
    return Claim(
        claim_id=claim_id,
        text="The system meets the Article 3 definition.",
        label=label,
        confidence=confidence,
        chunk_ids=chunk_ids or ["leg_001"],
    )


def _make_definition_section(
    claims: list[Claim] = None,
    confidence: Confidence = Confidence.HIGH,
) -> DefinitionSection:
    return DefinitionSection(
        dimension_id="definition_check",
        claims=claims or [_make_claim()],
        confidence=confidence,
        summary="System is an AI system.",
        is_ai_system=True,
    )


def _setup_memory_with_chunk(
    source_type: str = "legislation",
) -> tuple[SessionMemory, set[str], dict[str, Chunk]]:
    """Return (memory, retrieved_chunk_ids, chunk_lookup) with one pre-registered chunk.

    Pre-registering the chunk simulates the orchestrator having called retrieve()
    before the agent writes — all citation checks will pass for chunk_id="leg_001".
    """
    chunk = _make_chunk(source_type=source_type)
    retrieved = {chunk.chunk_id}
    lookup = {chunk.chunk_id: chunk}
    memory = SessionMemory()
    return memory, retrieved, lookup


def _make_facts() -> FactSection:
    return FactSection(
        use_case_name="Medical Triage AI",
        description="Automated patient prioritisation system.",
    )


# ── Read isolation ─────────────────────────────────────────────────────────
# These tests verify the deepcopy contract: mutating an object returned by a
# proxy read must NOT affect the underlying SessionMemory.

class TestReadIsolation:
    def test_analysis_proxy_facts_is_deepcopy(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        memory.facts = _make_facts()
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)

        returned_facts = view.facts
        returned_facts.use_case_name = "MUTATED"

        # The mutation must not propagate into SessionMemory.
        assert memory.facts.use_case_name == "Medical Triage AI"

    def test_validation_proxy_definition_check_is_deepcopy(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        memory.definition_check = _make_definition_section()
        view = ValidationAgentMemoryView(memory, retrieved, lookup)

        returned = view.definition_check
        returned.summary = "MUTATED"

        assert memory.definition_check.summary == "System is an AI system."

    def test_synthesis_proxy_weak_claims_is_deepcopy(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        wc = WeakClaim(
            claim_id="def_0",
            dimension_id="definition_check",
            claim_text="Original text.",
            reason="LOW_CONFIDENCE",
            original_confidence=Confidence.LOW,
            original_label=Label.ASSUMPTION,
        )
        memory.weak_claims = [wc]
        view = SynthesisAgentMemoryView(memory, retrieved, lookup)

        returned = view.weak_claims
        returned[0].claim_text = "MUTATED"

        assert memory.weak_claims[0].claim_text == "Original text."


# ── Unexposed attribute access ─────────────────────────────────────────────
# These tests verify that each proxy raises AttributeError for attributes it
# does not own.  This is the primary enforcement mechanism for the
# intelligence-layer's write-isolation contract.

class TestAttributeAccess:
    def test_analysis_view_blocks_final_report(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(AttributeError):
            _ = view.final_report  # noqa: F841

    def test_analysis_view_blocks_follow_up_questions(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(AttributeError):
            _ = view.follow_up_questions

    def test_analysis_view_blocks_orchestrator(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(AttributeError):
            _ = view._orchestrator

    def test_validation_view_blocks_final_report(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        view = ValidationAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(AttributeError):
            _ = view.final_report

    def test_synthesis_view_blocks_orchestrator(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        view = SynthesisAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(AttributeError):
            _ = view._orchestrator

    def test_extraction_view_blocks_everything(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        view = ExtractionAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(AttributeError):
            _ = view.facts  # cannot read; can only write


# ── Schema validation ──────────────────────────────────────────────────────

class TestSchemaValidation:
    def test_wrong_type_raises_memory_write_error(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(MemoryWriteError) as exc_info:
            view.write_definition(RiskSection(  # wrong type: RiskSection instead of DefinitionSection
                dimension_id="definition_check",
                claims=[],
                confidence=Confidence.HIGH,
                summary="",
            ))
        assert exc_info.value.check_name == "schema"

    def test_risk_section_rejects_definition_section(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(MemoryWriteError) as exc_info:
            view.write_risk(DefinitionSection(
                dimension_id="risk_classification",
                claims=[],
                confidence=Confidence.HIGH,
                summary="",
            ))
        assert exc_info.value.check_name == "schema"


# ── Citation integrity ─────────────────────────────────────────────────────

class TestCitationIntegrity:
    def test_unknown_chunk_id_raises_error(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        # chunk_id "unknown_999" is not in retrieved set
        bad_claim = _make_claim(chunk_ids=["unknown_999"])
        section = _make_definition_section(claims=[bad_claim])
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(MemoryWriteError) as exc_info:
            view.write_definition(section)
        assert exc_info.value.check_name == "citation_integrity"
        assert "unknown_999" in exc_info.value.detail

    def test_registered_chunk_id_passes(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        # "leg_001" is in retrieved set
        good_claim = _make_claim(chunk_ids=["leg_001"])
        section = _make_definition_section(claims=[good_claim])
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        view.write_definition(section)   # must not raise
        assert memory.definition_check is not None

    def test_empty_chunk_ids_always_passes_citation_check(self):
        """A claim with no citations is allowed through citation check
        (weakness detection catches it separately).

        Rationale: the citation check only validates that *listed* chunk_ids
        are in the retrieved set.  An empty list has nothing to check, so it
        passes — the UNSUPPORTED weakness reason handles the policy concern.
        """
        memory, retrieved, lookup = _setup_memory_with_chunk()
        claim = _make_claim(chunk_ids=[])
        section = _make_definition_section(claims=[claim])
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        view.write_definition(section)   # should not raise on citations
        assert memory.definition_check is not None


# ── Label consistency ──────────────────────────────────────────────────────

class TestLabelConsistency:
    def test_corpus_chunk_labeled_fact_raises(self):
        """A legislation chunk must NOT be labeled FACT."""
        memory, retrieved, lookup = _setup_memory_with_chunk(source_type="legislation")
        claim = _make_claim(chunk_ids=["leg_001"], label=Label.FACT)
        section = _make_definition_section(claims=[claim])
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(MemoryWriteError) as exc_info:
            view.write_definition(section)
        assert exc_info.value.check_name == "label_consistency"

    def test_official_guidance_chunk_labeled_fact_raises(self):
        """An official_guidance chunk must NOT be labeled FACT."""
        chunk = _make_chunk(chunk_id="guid_001", source_type="official_guidance")
        retrieved = {"guid_001"}
        lookup = {"guid_001": chunk}
        memory = SessionMemory()
        claim = _make_claim(chunk_ids=["guid_001"], label=Label.FACT)
        section = _make_definition_section(claims=[claim])
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(MemoryWriteError) as exc_info:
            view.write_definition(section)
        assert exc_info.value.check_name == "label_consistency"

    def test_uploaded_doc_chunk_labeled_retrieved_raises(self):
        """An uploaded_doc chunk must NOT be labeled RETRIEVED."""
        chunk = _make_chunk(chunk_id="doc_001", source_type="uploaded_doc")
        retrieved = {"doc_001"}
        lookup = {"doc_001": chunk}
        memory = SessionMemory()
        claim = _make_claim(chunk_ids=["doc_001"], label=Label.RETRIEVED)
        section = _make_definition_section(claims=[claim])
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(MemoryWriteError) as exc_info:
            view.write_definition(section)
        assert exc_info.value.check_name == "label_consistency"

    def test_uploaded_doc_labeled_fact_passes(self):
        """An uploaded_doc chunk CAN be labeled FACT."""
        chunk = _make_chunk(chunk_id="doc_001", source_type="uploaded_doc")
        retrieved = {"doc_001"}
        lookup = {"doc_001": chunk}
        memory = SessionMemory()
        claim = _make_claim(chunk_ids=["doc_001"], label=Label.FACT)
        section = _make_definition_section(claims=[claim])
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        view.write_definition(section)   # must not raise

    def test_corpus_chunk_labeled_retrieved_passes(self):
        """A legislation chunk CAN be labeled RETRIEVED."""
        memory, retrieved, lookup = _setup_memory_with_chunk(source_type="legislation")
        claim = _make_claim(chunk_ids=["leg_001"], label=Label.RETRIEVED)
        section = _make_definition_section(claims=[claim])
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        view.write_definition(section)  # must not raise


# ── Confidence bounds ──────────────────────────────────────────────────────

# ── Confidence bounds ──────────────────────────────────────────────────────
# "confidence_bounds" is the check_name for both Confidence enum validation
# AND Label enum validation.  Both are grouped under one check_name to
# simplify orchestrator error-handling (one branch catches both).

class TestConfidenceBounds:
    def test_invalid_confidence_string_on_section_raises(self):
        """Passing a raw string as confidence must fail."""
        memory, retrieved, lookup = _setup_memory_with_chunk()
        section = DefinitionSection(
            dimension_id="definition_check",
            claims=[],
            confidence="VERY_HIGH",   # not a Confidence enum
            summary="",
        )
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(MemoryWriteError) as exc_info:
            view.write_definition(section)
        assert exc_info.value.check_name == "confidence_bounds"

    def test_invalid_confidence_on_claim_raises(self):
        """A Claim with a raw string confidence must fail validation."""
        memory, retrieved, lookup = _setup_memory_with_chunk()
        bad_claim = Claim(
            claim_id="def_0",
            text="Some claim.",
            label=Label.RETRIEVED,
            confidence="bad_value",   # not a Confidence enum
            chunk_ids=["leg_001"],
        )
        section = _make_definition_section(claims=[bad_claim])
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(MemoryWriteError) as exc_info:
            view.write_definition(section)
        assert exc_info.value.check_name == "confidence_bounds"

    def test_invalid_label_on_claim_raises(self):
        """A Claim with a raw string label must fail validation."""
        memory, retrieved, lookup = _setup_memory_with_chunk()
        bad_claim = Claim(
            claim_id="def_0",
            text="Some claim.",
            label="DEFINITELY",   # not a Label enum
            confidence=Confidence.HIGH,
            chunk_ids=["leg_001"],
        )
        section = _make_definition_section(claims=[bad_claim])
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(MemoryWriteError) as exc_info:
            view.write_definition(section)
        # The label check is grouped under "confidence_bounds" per implementation
        assert exc_info.value.check_name == "confidence_bounds"


# ── MemoryWriteError message format ───────────────────────────────────────

class TestMemoryWriteErrorMessage:
    def test_message_format(self):
        err = MemoryWriteError("schema", "expected DefinitionSection, got str")
        assert str(err) == "[schema] expected DefinitionSection, got str"
        assert err.check_name == "schema"
        assert err.detail == "expected DefinitionSection, got str"

    def test_citation_integrity_error_format(self):
        err = MemoryWriteError("citation_integrity", "chunk_ids not found: ['x']")
        assert "citation_integrity" in str(err)
        assert "x" in str(err)


# ── Checkpoint save and restore ────────────────────────────────────────────

# ── Checkpoint save and restore ────────────────────────────────────────────
# These tests exercise checkpoint behaviour directly without the orchestrator.
# They simulate the orchestrator's _save_checkpoint / _restore_checkpoint
# pattern: deepcopy → clear nested checkpoints → store → restore individual fields.

class TestCheckpoints:
    """Tests for the orchestrator's checkpoint mechanism via direct memory manipulation."""

    def test_checkpoint_saves_current_state(self):
        memory = SessionMemory()
        chunk = _make_chunk()
        retrieved = {chunk.chunk_id}
        lookup = {chunk.chunk_id: chunk}

        # Write a section
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        section = _make_definition_section()
        view.write_definition(section)

        # Save checkpoint (simulate orchestrator behaviour)
        snapshot = copy.deepcopy(memory)
        snapshot._orchestrator.checkpoints = {}
        memory._orchestrator.checkpoints["test"] = snapshot

        # Write a second section that changes things
        view2 = AnalysisAgentMemoryView(memory, retrieved, lookup)
        risk_section = RiskSection(
            dimension_id="risk_classification",
            claims=[],
            confidence=Confidence.LOW,
            summary="Low risk.",
        )
        view2.write_risk(risk_section)

        # Both sections exist now
        assert memory.definition_check is not None
        assert memory.risk_classification is not None

        # Restore checkpoint
        snap = memory._orchestrator.checkpoints["test"]
        memory.risk_classification = copy.deepcopy(snap.risk_classification)

        # After restore, risk_classification should be None again
        assert memory.risk_classification is None
        assert memory.definition_check is not None

    def test_checkpoint_is_independent_of_original(self):
        """Modifying memory after save must not alter the checkpoint."""
        memory = SessionMemory()
        chunk = _make_chunk()
        retrieved = {chunk.chunk_id}
        lookup = {chunk.chunk_id: chunk}

        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        view.write_definition(_make_definition_section())

        snapshot = copy.deepcopy(memory)
        snapshot._orchestrator.checkpoints = {}
        memory._orchestrator.checkpoints["snap"] = snapshot

        # Mutate original
        memory.definition_check.summary = "MUTATED"

        # Checkpoint must be unchanged
        saved = memory._orchestrator.checkpoints["snap"]
        assert saved.definition_check.summary == "System is an AI system."

    def test_restore_identical_state(self):
        """Restoring a checkpoint should produce a state equal to when it was saved."""
        memory = SessionMemory()
        chunk = _make_chunk()
        retrieved = {chunk.chunk_id}
        lookup = {chunk.chunk_id: chunk}

        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        section = _make_definition_section()
        view.write_definition(section)

        # Save
        snapshot = copy.deepcopy(memory)
        snapshot._orchestrator.checkpoints = {}
        memory._orchestrator.checkpoints["after_analysis"] = snapshot

        # Overwrite with different data
        memory.definition_check = None

        # Restore
        saved = memory._orchestrator.checkpoints["after_analysis"]
        memory.definition_check = copy.deepcopy(saved.definition_check)

        assert memory.definition_check is not None
        assert memory.definition_check.is_ai_system is True


# ── Validation proxy write methods ────────────────────────────────────────

class TestValidationProxyWrites:
    def test_write_validation_flags_success(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        view = ValidationAgentMemoryView(memory, retrieved, lookup)
        vf = ValidationSection(
            flags=[],
            overall_confidence=Confidence.HIGH,
            summary="No weak claims.",
        )
        view.write_validation_flags(vf)
        assert memory.validation_flags is not None

    def test_write_overturned_claims_validates_citations(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        view = ValidationAgentMemoryView(memory, retrieved, lookup)
        oc = OverturnedClaim(
            claim_id="def_0",
            dimension_id="definition_check",
            original_claim_text="Original.",
            new_finding="Overturned.",
            new_confidence=Confidence.HIGH,
            new_label=Label.RETRIEVED,
            new_chunk_ids=["missing_chunk_xyz"],
        )
        with pytest.raises(MemoryWriteError) as exc_info:
            view.write_overturned_claims([oc])
        assert exc_info.value.check_name == "citation_integrity"

    def test_write_weak_claims_validates_schema(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        view = ValidationAgentMemoryView(memory, retrieved, lookup)
        with pytest.raises(MemoryWriteError):
            view.write_weak_claims("not a list")  # type: ignore

    def test_write_weak_claims_validates_confidence(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        view = ValidationAgentMemoryView(memory, retrieved, lookup)
        wc = WeakClaim(
            claim_id="def_0",
            dimension_id="definition_check",
            claim_text="A claim.",
            reason="LOW_CONFIDENCE",
            original_confidence="bad",  # not a Confidence enum
            original_label=Label.ASSUMPTION,
        )
        with pytest.raises(MemoryWriteError) as exc_info:
            view.write_weak_claims([wc])
        assert exc_info.value.check_name == "confidence_bounds"


# ── Synthesis proxy write methods ─────────────────────────────────────────

class TestSynthesisProxyWrites:
    def test_write_confidence_summary_validates_fields(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        view = SynthesisAgentMemoryView(memory, retrieved, lookup)
        cs = ConfidenceSection(
            definition_check="NOT_VALID",   # invalid
            risk_classification=Confidence.HIGH,
            prohibited_practices=Confidence.HIGH,
            transparency=Confidence.HIGH,
            roles=Confidence.HIGH,
            governance=Confidence.HIGH,
            overall=Confidence.HIGH,
        )
        with pytest.raises(MemoryWriteError) as exc_info:
            view.write_confidence_summary(cs)
        assert exc_info.value.check_name == "confidence_bounds"

    def test_write_final_report_success(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        view = SynthesisAgentMemoryView(memory, retrieved, lookup)
        report = ReportSection(use_case_summary="Test summary.")
        view.write_final_report(report)
        assert memory.final_report is not None
        assert memory.final_report.use_case_summary == "Test summary."

    def test_write_final_report_is_deepcopy(self):
        memory, retrieved, lookup = _setup_memory_with_chunk()
        view = SynthesisAgentMemoryView(memory, retrieved, lookup)
        report = ReportSection(use_case_summary="Original.")
        view.write_final_report(report)
        report.use_case_summary = "MUTATED"
        assert memory.final_report.use_case_summary == "Original."
