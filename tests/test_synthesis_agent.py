"""
tests/test_synthesis_agent.py
──────────────────────────────
Unit tests for synthesis_agent.py.

Intelligence functions tested with mock inputs only.
Orchestration tested with a mock LLM client.
No real API calls.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.types import (
    Chunk,
    Claim,
    ClaimStatus,
    CompletionSignal,
    Confidence,
    ConfidenceSection,
    DefinitionSection,
    FactSection,
    FollowUpSection,
    GovernanceSection,
    Label,
    LoopSignal,
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
from core.memory import SessionMemory, SynthesisAgentMemoryView
from agents.synthesis_agent import (
    build_synthesis_prompt,
    determine_loop_condition,
    merge_analysis_with_validation,
    parse_synthesis_response,
    run_synthesis_agent,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_facts() -> FactSection:
    return FactSection(
        use_case_name="Hiring Algorithm",
        description="Screens job applicants using ML.",
        industry="HR Technology",
        ai_capabilities=["classification"],
        data_inputs=["CVs", "assessment scores"],
        outputs=["ranking score"],
        deployment_context="Corporate HR department",
        affected_persons=["job applicants"],
    )


def _make_claim(
    claim_id: str = "def_0",
    label: Label = Label.RETRIEVED,
    confidence: Confidence = Confidence.HIGH,
    chunk_ids: list[str] = None,
    text: str = "A claim.",
) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        label=label,
        confidence=confidence,
        chunk_ids=chunk_ids or ["leg_001"],
    )


def _make_definition_section(confidence: Confidence = Confidence.HIGH) -> DefinitionSection:
    return DefinitionSection(
        dimension_id="definition_check",
        claims=[_make_claim()],
        confidence=confidence,
        summary="System is an AI system.",
        is_ai_system=True,
    )


def _make_risk_section(confidence: Confidence = Confidence.HIGH) -> RiskSection:
    return RiskSection(
        dimension_id="risk_classification",
        claims=[_make_claim(claim_id="risk_0")],
        confidence=confidence,
        summary="High risk.",
        risk_level=RiskLevel.HIGH,
    )


def _populate_all_analysis_sections(memory: SessionMemory) -> None:
    """Write all six analysis sections into memory directly (bypass proxy)."""
    memory.definition_check = _make_definition_section()
    memory.risk_classification = _make_risk_section()
    memory.prohibited_practices = ProhibitedSection(
        dimension_id="prohibited_practices",
        claims=[],
        confidence=Confidence.HIGH,
        summary="No prohibited practices.",
    )
    memory.transparency = TransparencySection(
        dimension_id="transparency",
        claims=[],
        confidence=Confidence.MEDIUM,
        summary="Notification required.",
    )
    memory.roles = RolesSection(
        dimension_id="roles",
        claims=[],
        confidence=Confidence.HIGH,
        summary="Deployer only.",
        is_deployer=True,
    )
    memory.governance = GovernanceSection(
        dimension_id="governance",
        claims=[],
        confidence=Confidence.MEDIUM,
        summary="Documentation required.",
    )
    memory.validation_flags = ValidationSection(
        flags=[],
        overall_confidence=Confidence.HIGH,
        summary="No weak claims.",
    )
    memory.weak_claims = []
    memory.overturned_claims = []


def _make_synthesis_view(memory: SessionMemory) -> SynthesisAgentMemoryView:
    chunk = Chunk(
        chunk_id="leg_001",
        text="EU AI Act Article 3 definition.",
        source_type="legislation",
        article_id="Article 3",
    )
    retrieved = {"leg_001"}
    lookup = {"leg_001": chunk}
    return SynthesisAgentMemoryView(memory, retrieved, lookup)


def _make_mock_llm(response_json: dict) -> Any:
    content = json.dumps(response_json)
    mock_cb = MagicMock()
    mock_cb.text = content
    mock_resp = MagicMock()
    mock_resp.content = [mock_cb]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_resp
    return mock_client


def _make_valid_synthesis_response(
    def_confidence: str = "HIGH",
    risk_confidence: str = "HIGH",
) -> dict:
    """Build a plausible synthesis response dict."""
    return {
        "report": {
            "use_case_summary": "The hiring algorithm screens applicants.",
            "extracted_facts": {"use_case_name": "Hiring Algorithm"},
            "ai_definition_check": {
                "finding": "System qualifies as an AI system.",
                "confidence": def_confidence,
            },
            "risk_classification": {
                "finding": "High-risk system per Annex III.",
                "risk_level": "high",
                "confidence": risk_confidence,
            },
            "prohibited_practices_check": {"finding": "No prohibited practices.", "confidence": "HIGH"},
            "transparency_gpai_obligations": {"finding": "Disclosure required.", "confidence": "MEDIUM"},
            "roles": {"finding": "Deployer only.", "confidence": "HIGH"},
            "governance_observations": {"finding": "Documentation required.", "confidence": "MEDIUM"},
            "missing_information": {"gaps": [], "uncertain_claims": [], "unresolved_dimensions": []},
            "citations_by_source": {"legislation": [], "official_guidance": [], "uploaded_doc": []},
        },
        "follow_up": {
            "questions": ["Does the system operate without human review?"],
            "missing_evidence": ["Technical documentation"],
        },
        "confidence": {
            "definition_check": def_confidence,
            "risk_classification": risk_confidence,
            "prohibited_practices": "HIGH",
            "transparency": "MEDIUM",
            "roles": "HIGH",
            "governance": "MEDIUM",
            "overall": "MEDIUM",
        },
    }


# ── merge_analysis_with_validation ────────────────────────────────────────

class TestMergeAnalysisWithValidation:
    def test_overturned_claim_overrides_analysis(self):
        memory = SessionMemory()
        memory.facts = _make_facts()
        _populate_all_analysis_sections(memory)

        # Override the definition claim
        memory.overturned_claims = [
            OverturnedClaim(
                claim_id="def_0",
                dimension_id="definition_check",
                original_claim_text="A claim.",
                new_finding="Overturned: new evidence shows different conclusion.",
                new_confidence=Confidence.HIGH,
                new_label=Label.RETRIEVED,
                new_chunk_ids=["leg_001"],
                status=ClaimStatus.OVERTURNED,
            )
        ]

        view = _make_synthesis_view(memory)
        merged = merge_analysis_with_validation(view)

        def_claims = merged["definition_check"]["claims"]
        overturned = [c for c in def_claims if c["source"] == "validation_override"]
        assert len(overturned) == 1
        assert "new evidence" in overturned[0]["text"]

    def test_unresolved_claim_marked_uncertain(self):
        memory = SessionMemory()
        memory.facts = _make_facts()
        _populate_all_analysis_sections(memory)

        # Mark def_0 as unresolved
        memory.validation_flags = ValidationSection(
            flags=[
                ValidationFlag(
                    claim_id="def_0",
                    dimension_id="definition_check",
                    status=ClaimStatus.UNRESOLVED,
                    notes="Could not resolve.",
                )
            ],
            overall_confidence=Confidence.LOW,
            summary="1 unresolved.",
        )

        view = _make_synthesis_view(memory)
        merged = merge_analysis_with_validation(view)

        def_claims = merged["definition_check"]["claims"]
        unresolved = [c for c in def_claims if c["source"] == "unresolved"]
        assert len(unresolved) == 1
        assert unresolved[0]["label"] == Label.UNCERTAIN.value

    def test_regular_claims_pass_through_unchanged(self):
        memory = SessionMemory()
        memory.facts = _make_facts()
        _populate_all_analysis_sections(memory)

        view = _make_synthesis_view(memory)
        merged = merge_analysis_with_validation(view)

        def_claims = merged["definition_check"]["claims"]
        analysis_claims = [c for c in def_claims if c["source"] == "analysis"]
        assert len(analysis_claims) == 1
        assert analysis_claims[0]["label"] == Label.RETRIEVED.value

    def test_none_section_produces_insufficient_entry(self):
        memory = SessionMemory()
        memory.facts = _make_facts()
        _populate_all_analysis_sections(memory)
        memory.governance = None   # simulate missing section

        view = _make_synthesis_view(memory)
        merged = merge_analysis_with_validation(view)

        assert merged["governance"]["confidence"] == Confidence.INSUFFICIENT.value

    def test_merged_contains_all_six_dimensions(self):
        memory = SessionMemory()
        memory.facts = _make_facts()
        _populate_all_analysis_sections(memory)

        view = _make_synthesis_view(memory)
        merged = merge_analysis_with_validation(view)

        expected = {
            "definition_check", "risk_classification", "prohibited_practices",
            "transparency", "roles", "governance",
        }
        assert set(merged.keys()) == expected


# ── determine_loop_condition ──────────────────────────────────────────────

class TestDetermineLoopCondition:
    def _make_conf(
        self,
        definition: Confidence = Confidence.HIGH,
        risk: Confidence = Confidence.HIGH,
    ) -> ConfidenceSection:
        return ConfidenceSection(
            definition_check=definition,
            risk_classification=risk,
            prohibited_practices=Confidence.HIGH,
            transparency=Confidence.MEDIUM,
            roles=Confidence.HIGH,
            governance=Confidence.MEDIUM,
            overall=Confidence.MEDIUM,
        )

    def test_returns_true_when_both_critical_insufficient_and_loop_count_0(self):
        conf = self._make_conf(
            definition=Confidence.INSUFFICIENT,
            risk=Confidence.INSUFFICIENT,
        )
        assert determine_loop_condition(conf, loop_count=0) is True

    def test_returns_false_when_loop_count_is_1(self):
        conf = self._make_conf(
            definition=Confidence.INSUFFICIENT,
            risk=Confidence.INSUFFICIENT,
        )
        assert determine_loop_condition(conf, loop_count=1) is False

    def test_returns_false_when_only_one_is_insufficient(self):
        conf = self._make_conf(
            definition=Confidence.INSUFFICIENT,
            risk=Confidence.HIGH,
        )
        assert determine_loop_condition(conf, loop_count=0) is False

    def test_returns_false_when_neither_is_insufficient(self):
        conf = self._make_conf(
            definition=Confidence.MEDIUM,
            risk=Confidence.MEDIUM,
        )
        assert determine_loop_condition(conf, loop_count=0) is False

    def test_returns_false_when_loop_count_greater_than_max(self):
        conf = self._make_conf(
            definition=Confidence.INSUFFICIENT,
            risk=Confidence.INSUFFICIENT,
        )
        assert determine_loop_condition(conf, loop_count=2) is False


# ── build_synthesis_prompt ────────────────────────────────────────────────

class TestBuildSynthesisPrompt:
    def test_prompt_contains_use_case_name(self):
        memory = SessionMemory()
        memory.facts = _make_facts()
        _populate_all_analysis_sections(memory)
        view = _make_synthesis_view(memory)
        merged = merge_analysis_with_validation(view)

        prompt = build_synthesis_prompt(
            facts=_make_facts(),
            merged=merged,
            chunks=[],
            loop_count=0,
        )
        assert "Hiring Algorithm" in prompt

    def test_prompt_contains_merged_json(self):
        memory = SessionMemory()
        memory.facts = _make_facts()
        _populate_all_analysis_sections(memory)
        view = _make_synthesis_view(memory)
        merged = merge_analysis_with_validation(view)

        prompt = build_synthesis_prompt(
            facts=_make_facts(),
            merged=merged,
            chunks=[],
            loop_count=0,
        )
        assert "definition_check" in prompt

    def test_prompt_includes_loop_context_on_second_pass(self):
        memory = SessionMemory()
        memory.facts = _make_facts()
        _populate_all_analysis_sections(memory)
        view = _make_synthesis_view(memory)
        merged = merge_analysis_with_validation(view)

        prompt = build_synthesis_prompt(
            facts=_make_facts(),
            merged=merged,
            chunks=[],
            loop_count=1,
        )
        assert "loop pass" in prompt.lower() or "pass 2" in prompt

    def test_prompt_no_loop_context_on_first_pass(self):
        memory = SessionMemory()
        memory.facts = _make_facts()
        _populate_all_analysis_sections(memory)
        view = _make_synthesis_view(memory)
        merged = merge_analysis_with_validation(view)

        prompt = build_synthesis_prompt(
            facts=_make_facts(),
            merged=merged,
            chunks=[],
            loop_count=0,
        )
        # No loop context message on first pass
        assert "loop pass" not in prompt.lower()


# ── parse_synthesis_response ──────────────────────────────────────────────

class TestParseSynthesisResponse:
    def test_parses_valid_response(self):
        raw = _make_valid_synthesis_response()
        report, follow_up, confidence = parse_synthesis_response(json.dumps(raw))

        assert isinstance(report, ReportSection)
        assert isinstance(follow_up, FollowUpSection)
        assert isinstance(confidence, ConfidenceSection)

        assert "hiring algorithm" in report.use_case_summary.lower()
        assert confidence.definition_check == Confidence.HIGH
        assert confidence.risk_classification == Confidence.HIGH
        assert len(follow_up.questions) == 1

    def test_malformed_response_returns_defaults(self):
        report, follow_up, confidence = parse_synthesis_response("not json")
        assert isinstance(report, ReportSection)
        assert confidence.definition_check == Confidence.INSUFFICIENT
        assert confidence.overall == Confidence.INSUFFICIENT

    def test_parses_all_confidence_fields(self):
        raw = _make_valid_synthesis_response(def_confidence="MEDIUM", risk_confidence="LOW")
        _, _, confidence = parse_synthesis_response(json.dumps(raw))
        assert confidence.definition_check == Confidence.MEDIUM
        assert confidence.risk_classification == Confidence.LOW

    def test_parses_missing_evidence_list(self):
        raw = _make_valid_synthesis_response()
        _, follow_up, _ = parse_synthesis_response(json.dumps(raw))
        assert "Technical documentation" in follow_up.missing_evidence

    def test_parses_extracted_facts(self):
        raw = _make_valid_synthesis_response()
        report, _, _ = parse_synthesis_response(json.dumps(raw))
        assert report.extracted_facts.get("use_case_name") == "Hiring Algorithm"


# ── run_synthesis_agent (orchestration) ───────────────────────────────────

class TestRunSynthesisAgent:
    def test_emits_loop_signal_when_critical_sections_insufficient(self):
        """When both critical dimensions are INSUFFICIENT, emit LoopSignal."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        _populate_all_analysis_sections(memory)
        # Override critical sections to INSUFFICIENT
        memory.definition_check.confidence = Confidence.INSUFFICIENT
        memory.risk_classification.confidence = Confidence.INSUFFICIENT

        view = _make_synthesis_view(memory)
        context = {"loop_count": 0}
        raw = _make_valid_synthesis_response(
            def_confidence="INSUFFICIENT",
            risk_confidence="INSUFFICIENT",
        )
        mock_client = _make_mock_llm(raw)

        signal = run_synthesis_agent(view, [], context, llm_client=mock_client)
        assert isinstance(signal, LoopSignal)

    def test_does_not_write_to_memory_when_loop_signal_emitted(self):
        """Memory must stay clean when LoopSignal is returned."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        _populate_all_analysis_sections(memory)
        memory.definition_check.confidence = Confidence.INSUFFICIENT
        memory.risk_classification.confidence = Confidence.INSUFFICIENT

        view = _make_synthesis_view(memory)
        context = {"loop_count": 0}
        raw = _make_valid_synthesis_response(
            def_confidence="INSUFFICIENT",
            risk_confidence="INSUFFICIENT",
        )
        mock_client = _make_mock_llm(raw)

        run_synthesis_agent(view, [], context, llm_client=mock_client)

        # No writes should have happened
        assert memory.final_report is None
        assert memory.confidence_summary is None

    def test_emits_completion_signal_when_not_looping(self):
        """When confidence is adequate, emit CompletionSignal and write report."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        _populate_all_analysis_sections(memory)

        view = _make_synthesis_view(memory)
        context = {"loop_count": 0}
        raw = _make_valid_synthesis_response()
        mock_client = _make_mock_llm(raw)

        signal = run_synthesis_agent(view, [], context, llm_client=mock_client)
        assert isinstance(signal, CompletionSignal)

    def test_writes_all_three_sections_on_completion(self):
        """After CompletionSignal, all three sections must be in memory."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        _populate_all_analysis_sections(memory)

        view = _make_synthesis_view(memory)
        context = {"loop_count": 0}
        raw = _make_valid_synthesis_response()
        mock_client = _make_mock_llm(raw)

        run_synthesis_agent(view, [], context, llm_client=mock_client)

        assert memory.final_report is not None
        assert memory.follow_up_questions is not None
        assert memory.confidence_summary is not None

    def test_loop_signal_not_emitted_when_loop_count_at_max(self):
        """With loop_count=1, even INSUFFICIENT confidence → CompletionSignal."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        _populate_all_analysis_sections(memory)
        memory.definition_check.confidence = Confidence.INSUFFICIENT
        memory.risk_classification.confidence = Confidence.INSUFFICIENT

        view = _make_synthesis_view(memory)
        context = {"loop_count": 1}   # already at max
        raw = _make_valid_synthesis_response(
            def_confidence="INSUFFICIENT",
            risk_confidence="INSUFFICIENT",
        )
        # Update overall to something non-INSUFFICIENT so confidence_summary write passes
        raw["confidence"]["overall"] = "LOW"
        mock_client = _make_mock_llm(raw)

        signal = run_synthesis_agent(view, [], context, llm_client=mock_client)
        assert isinstance(signal, CompletionSignal)

    def test_loop_signal_contains_refined_context(self):
        """LoopSignal.refined_context should be non-empty."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        _populate_all_analysis_sections(memory)
        memory.definition_check.confidence = Confidence.INSUFFICIENT
        memory.risk_classification.confidence = Confidence.INSUFFICIENT

        view = _make_synthesis_view(memory)
        context = {"loop_count": 0}
        raw = _make_valid_synthesis_response(
            def_confidence="INSUFFICIENT",
            risk_confidence="INSUFFICIENT",
        )
        raw["follow_up"]["missing_evidence"] = ["Need Article 3 evidence."]
        mock_client = _make_mock_llm(raw)

        signal = run_synthesis_agent(view, [], context, llm_client=mock_client)
        assert isinstance(signal, LoopSignal)
        assert signal.refined_context  # must be non-empty
