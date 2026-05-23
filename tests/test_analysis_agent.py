"""
tests/test_analysis_agent.py
─────────────────────────────
Unit tests for analysis_agent.py.

All intelligence functions tested with mock inputs.
Orchestration functions tested with a mock LLM client.
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
    CompletionSignal,
    Confidence,
    DefinitionSection,
    FactSection,
    GovernanceSection,
    Label,
    ProhibitedSection,
    RetrievalSignal,
    RiskLevel,
    RiskSection,
    RolesSection,
    TransparencySection,
)
from core.memory import AnalysisAgentMemoryView, SessionMemory
from agents.analysis_agent import (
    DIMENSION_ORDER,
    _extract_json,
    _make_insufficient_finding,
    _make_claim,
    build_dimension_prompt,
    check_evidence_sufficiency,
    formulate_retrieval_query,
    parse_dimension_response,
    run_analysis_agent,
    score_overall_confidence,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _legislation_chunk(
    chunk_id: str = "leg_001",
    text: str = "ai system definition machine learning algorithm",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_type="legislation",
        article_id="Article 3",
    )


def _uploaded_chunk(
    chunk_id: str = "doc_001",
    text: str = "our product uses a machine learning model",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_type="uploaded_doc",
        article_id=None,
    )


def _make_facts(
    name: str = "Medical Triage AI",
    description: str = "Automated prioritisation using machine learning.",
    industry: str = "Healthcare",
) -> FactSection:
    return FactSection(
        use_case_name=name,
        description=description,
        industry=industry,
        ai_capabilities=["classification", "prediction"],
        data_inputs=["patient records", "vital signs"],
        outputs=["priority score"],
        deployment_context="Hospital emergency department",
        affected_persons=["patients"],
    )


def _make_mock_llm(response_json: dict | str) -> Any:
    """Return a mock client whose messages.create() returns a fixed string."""
    if isinstance(response_json, dict):
        content = json.dumps(response_json)
    else:
        content = response_json

    mock_content_block = MagicMock()
    mock_content_block.text = content
    mock_response = MagicMock()
    mock_response.content = [mock_content_block]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client


def _make_analysis_view(
    memory: SessionMemory = None,
    extra_chunks: list[Chunk] = None,
) -> tuple[AnalysisAgentMemoryView, set[str], dict[str, Chunk]]:
    if memory is None:
        memory = SessionMemory()
    memory.facts = _make_facts()
    chunks = [_legislation_chunk(), _uploaded_chunk()]
    if extra_chunks:
        chunks.extend(extra_chunks)
    retrieved = {c.chunk_id for c in chunks}
    lookup = {c.chunk_id: c for c in chunks}
    view = AnalysisAgentMemoryView(memory, retrieved, lookup)
    return view, retrieved, lookup


# ── check_evidence_sufficiency ────────────────────────────────────────────

class TestCheckEvidenceSufficiency:
    def test_returns_false_for_empty_chunks(self):
        sufficient, reason = check_evidence_sufficiency([], "definition_check")
        assert sufficient is False
        assert reason is not None

    def test_returns_false_when_no_keyword_match(self):
        # Governance keywords won't match in a definition-check context
        chunk = Chunk(
            chunk_id="gov_001", text="technical documentation oversight",
            source_type="legislation", article_id=None,
        )
        # No definition-check keywords present
        sufficient, reason = check_evidence_sufficiency(
            [chunk], "definition_check"
        )
        assert sufficient is False

    def test_returns_false_when_no_authoritative_source(self):
        """Keyword match but all chunks are uploaded_doc — not sufficient."""
        chunk = _uploaded_chunk(text="ai system machine learning algorithm model")
        sufficient, reason = check_evidence_sufficiency([chunk], "definition_check")
        assert sufficient is False
        assert "authoritative" in reason.lower()

    def test_returns_true_with_keyword_and_legislation(self):
        chunk = _legislation_chunk(text="ai system definition machine learning annex i")
        sufficient, reason = check_evidence_sufficiency([chunk], "definition_check")
        assert sufficient is True
        assert reason is None

    def test_returns_true_for_risk_with_relevant_chunk(self):
        chunk = _legislation_chunk(
            chunk_id="leg_risk",
            text="high-risk ai system annex iii risk classification employment",
        )
        sufficient, reason = check_evidence_sufficiency([chunk], "risk_classification")
        assert sufficient is True

    def test_returns_false_for_prohibited_without_match(self):
        chunk = _legislation_chunk(text="ai system definition article 3")
        sufficient, reason = check_evidence_sufficiency([chunk], "prohibited_practices")
        assert sufficient is False


# ── formulate_retrieval_query ─────────────────────────────────────────────

class TestFormulateRetrievalQuery:
    def test_query_contains_dimension_context(self):
        facts = _make_facts()
        query, filters = formulate_retrieval_query("definition_check", facts, [])
        assert "AI Act" in query or "article" in query.lower() or "definition" in query.lower()

    def test_filters_contain_source_types(self):
        facts = _make_facts()
        _, filters = formulate_retrieval_query("risk_classification", facts, [])
        assert "source_types" in filters
        assert "legislation" in filters["source_types"]

    def test_filters_contain_dimension(self):
        facts = _make_facts()
        _, filters = formulate_retrieval_query("transparency", facts, [])
        assert filters.get("dimension") == "transparency"

    def test_query_incorporates_use_case_name(self):
        facts = _make_facts(name="Credit Scoring Engine")
        query, _ = formulate_retrieval_query("definition_check", facts, [])
        assert "Credit Scoring Engine" in query


# ── build_dimension_prompt ────────────────────────────────────────────────

class TestBuildDimensionPrompt:
    def test_prompt_contains_use_case_name(self):
        facts = _make_facts(name="Fraud Detection System")
        chunks = [_legislation_chunk()]
        prompt = build_dimension_prompt(facts, chunks, "definition_check")
        assert "Fraud Detection System" in prompt

    def test_prompt_contains_chunk_text(self):
        facts = _make_facts()
        chunk = _legislation_chunk(text="Unique chunk content XYZ987")
        prompt = build_dimension_prompt(facts, [chunk], "definition_check")
        assert "Unique chunk content XYZ987" in prompt

    def test_prompt_contains_chunk_id(self):
        facts = _make_facts()
        chunk = _legislation_chunk(chunk_id="unique_id_ABC")
        prompt = build_dimension_prompt(facts, [chunk], "definition_check")
        assert "unique_id_ABC" in prompt

    def test_refined_context_included_when_provided(self):
        facts = _make_facts()
        prompt = build_dimension_prompt(
            facts, [], "definition_check",
            refined_context="Please look for Annex I techniques."
        )
        assert "Annex I techniques" in prompt

    def test_refined_context_absent_when_empty(self):
        facts = _make_facts()
        prompt = build_dimension_prompt(facts, [], "definition_check", refined_context="")
        assert "ADDITIONAL CONTEXT" not in prompt

    def test_raises_for_unknown_dimension(self):
        facts = _make_facts()
        with pytest.raises(ValueError, match="No prompt template"):
            build_dimension_prompt(facts, [], "nonexistent_dimension")


# ── parse_dimension_response ──────────────────────────────────────────────

class TestParseDimensionResponse:
    def _valid_definition_json(self, chunk_id: str = "leg_001") -> str:
        return json.dumps({
            "dimension_id": "definition_check",
            "claims": [{
                "claim_id": "def_0",
                "text": "The system uses ML algorithms.",
                "label": "RETRIEVED",
                "confidence": "HIGH",
                "chunk_ids": [chunk_id],
                "is_weak": False,
                "weak_reason": None,
            }],
            "confidence": "HIGH",
            "summary": "System qualifies as an AI system.",
            "is_ai_system": True,
        })

    def test_parses_well_formed_definition_response(self):
        result = parse_dimension_response(
            self._valid_definition_json(), "definition_check"
        )
        assert isinstance(result, DefinitionSection)
        assert result.confidence == Confidence.HIGH
        assert result.is_ai_system is True
        assert len(result.claims) == 1
        assert result.claims[0].label == Label.RETRIEVED

    def test_parses_well_formed_risk_response(self):
        raw = json.dumps({
            "dimension_id": "risk_classification",
            "claims": [],
            "confidence": "MEDIUM",
            "summary": "High-risk system.",
            "risk_level": "high",
        })
        result = parse_dimension_response(raw, "risk_classification")
        assert isinstance(result, RiskSection)
        assert result.risk_level == RiskLevel.HIGH
        assert result.confidence == Confidence.MEDIUM

    def test_parses_well_formed_prohibited_response(self):
        raw = json.dumps({
            "dimension_id": "prohibited_practices",
            "claims": [],
            "confidence": "HIGH",
            "summary": "No prohibited practices.",
            "triggered_articles": [],
            "prohibited": False,
        })
        result = parse_dimension_response(raw, "prohibited_practices")
        assert isinstance(result, ProhibitedSection)
        assert result.prohibited is False

    def test_parses_transparency_section(self):
        raw = json.dumps({
            "dimension_id": "transparency",
            "claims": [],
            "confidence": "MEDIUM",
            "summary": "Notification required.",
            "applies_to_gpai": False,
            "labelling_required": True,
            "notification_required": True,
        })
        result = parse_dimension_response(raw, "transparency")
        assert isinstance(result, TransparencySection)
        assert result.labelling_required is True
        assert result.notification_required is True

    def test_parses_roles_section(self):
        raw = json.dumps({
            "dimension_id": "roles",
            "claims": [],
            "confidence": "HIGH",
            "summary": "Both provider and deployer.",
            "is_provider": True,
            "is_deployer": True,
            "is_both": True,
        })
        result = parse_dimension_response(raw, "roles")
        assert isinstance(result, RolesSection)
        assert result.is_both is True

    def test_parses_governance_section(self):
        raw = json.dumps({
            "dimension_id": "governance",
            "claims": [],
            "confidence": "HIGH",
            "summary": "Documentation required.",
            "documentation_required": True,
            "oversight_required": True,
            "monitoring_required": False,
        })
        result = parse_dimension_response(raw, "governance")
        assert isinstance(result, GovernanceSection)
        assert result.documentation_required is True

    def test_malformed_response_returns_insufficient(self):
        result = parse_dimension_response("this is not json at all", "definition_check")
        assert isinstance(result, DefinitionSection)
        assert result.confidence == Confidence.INSUFFICIENT
        assert result.claims == []

    def test_empty_response_returns_insufficient(self):
        result = parse_dimension_response("", "risk_classification")
        assert isinstance(result, RiskSection)
        assert result.confidence == Confidence.INSUFFICIENT

    def test_json_in_markdown_block_parsed(self):
        raw = json.dumps({
            "dimension_id": "definition_check",
            "claims": [],
            "confidence": "LOW",
            "summary": "Insufficient evidence.",
            "is_ai_system": None,
        })
        wrapped = f"Here is my analysis:\n```json\n{raw}\n```\nEnd."
        result = parse_dimension_response(wrapped, "definition_check")
        assert isinstance(result, DefinitionSection)
        assert result.confidence == Confidence.LOW


# ── _extract_json ─────────────────────────────────────────────────────────

class TestExtractJson:
    def test_direct_json(self):
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_markdown_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_json_embedded_in_text(self):
        text = 'Some text before {"key": "value"} some text after'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_returns_empty_dict_on_failure(self):
        result = _extract_json("completely invalid text }{")
        assert result == {}

    def test_returns_empty_dict_for_empty_string(self):
        result = _extract_json("")
        assert result == {}


# ── score_overall_confidence ──────────────────────────────────────────────

class TestScoreOverallConfidence:
    def _claim(self, confidence: Confidence) -> Claim:
        return Claim(
            claim_id="x",
            text="test",
            label=Label.RETRIEVED,
            confidence=confidence,
            chunk_ids=[],
        )

    def test_all_high_returns_high(self):
        claims = [self._claim(Confidence.HIGH)] * 5
        assert score_overall_confidence(claims) == Confidence.HIGH

    def test_any_insufficient_returns_insufficient(self):
        claims = [
            self._claim(Confidence.HIGH),
            self._claim(Confidence.HIGH),
            self._claim(Confidence.INSUFFICIENT),
        ]
        assert score_overall_confidence(claims) == Confidence.INSUFFICIENT

    def test_all_insufficient_returns_insufficient(self):
        claims = [self._claim(Confidence.INSUFFICIENT)] * 3
        assert score_overall_confidence(claims) == Confidence.INSUFFICIENT

    def test_mostly_high_returns_high(self):
        claims = [self._claim(Confidence.HIGH)] * 4 + [self._claim(Confidence.MEDIUM)]
        assert score_overall_confidence(claims) == Confidence.HIGH

    def test_mixed_high_medium_returns_medium(self):
        claims = [self._claim(Confidence.HIGH)] * 2 + [self._claim(Confidence.LOW)] * 3
        assert score_overall_confidence(claims) == Confidence.LOW

    def test_empty_claims_returns_insufficient(self):
        assert score_overall_confidence([]) == Confidence.INSUFFICIENT


# ── _make_insufficient_finding ────────────────────────────────────────────

class TestMakeInsufficientFinding:
    def test_returns_definition_section_for_definition_check(self):
        result = _make_insufficient_finding("definition_check")
        assert isinstance(result, DefinitionSection)
        assert result.confidence == Confidence.INSUFFICIENT

    def test_returns_risk_section_for_risk_classification(self):
        result = _make_insufficient_finding("risk_classification")
        assert isinstance(result, RiskSection)
        assert result.risk_level == RiskLevel.UNKNOWN

    def test_returns_prohibited_section(self):
        result = _make_insufficient_finding("prohibited_practices")
        assert isinstance(result, ProhibitedSection)

    def test_returns_transparency_section(self):
        result = _make_insufficient_finding("transparency")
        assert isinstance(result, TransparencySection)

    def test_returns_roles_section(self):
        result = _make_insufficient_finding("roles")
        assert isinstance(result, RolesSection)

    def test_returns_governance_section(self):
        result = _make_insufficient_finding("governance")
        assert isinstance(result, GovernanceSection)


# ── run_analysis_agent (orchestration) ───────────────────────────────────

class TestRunAnalysisAgent:
    """Orchestration tests using a mock LLM client."""

    def _valid_llm_response(self, dimension_id: str, chunk_id: str = "leg_001") -> dict:
        """Produce a valid dimension JSON response for the mock LLM."""
        base = {
            "dimension_id": dimension_id,
            "claims": [{
                "claim_id": f"{dimension_id}_0",
                "text": f"Valid claim for {dimension_id}.",
                "label": "RETRIEVED",
                "confidence": "HIGH",
                "chunk_ids": [chunk_id],
                "is_weak": False,
                "weak_reason": None,
            }],
            "confidence": "HIGH",
            "summary": f"Assessment for {dimension_id}.",
        }
        # Add dimension-specific fields
        extras = {
            "definition_check":    {"is_ai_system": True},
            "risk_classification": {"risk_level": "high"},
            "prohibited_practices": {"triggered_articles": [], "prohibited": False},
            "transparency":        {"applies_to_gpai": False, "labelling_required": False, "notification_required": True},
            "roles":               {"is_provider": False, "is_deployer": True, "is_both": False},
            "governance":          {"documentation_required": True, "oversight_required": True, "monitoring_required": False},
        }
        base.update(extras.get(dimension_id, {}))
        return base

    def _make_multi_response_client(self, chunk_id: str = "leg_001") -> Any:
        """Mock client that returns a valid response for each dimension call."""
        responses = [
            self._valid_llm_response(dim, chunk_id) for dim in DIMENSION_ORDER
        ]
        call_count = [0]

        mock_content = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client = MagicMock()

        def side_effect(*args, **kwargs):
            idx = call_count[0]
            if idx < len(responses):
                mock_content.text = json.dumps(responses[idx])
            call_count[0] += 1
            return mock_response

        mock_client.messages.create.side_effect = side_effect
        return mock_client, call_count

    def test_emits_retrieval_signal_when_no_authoritative_chunks(self):
        """With only uploaded_doc chunks, the agent should request more."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        chunk = _uploaded_chunk(text="ai system machine learning model algorithm")
        retrieved = {chunk.chunk_id}
        lookup = {chunk.chunk_id: chunk}
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        context = {"max_retrievals_reached": set(), "refined_context": ""}

        signal = run_analysis_agent(view, [chunk], context, llm_client=MagicMock())
        assert isinstance(signal, RetrievalSignal)

    def test_emits_completion_signal_when_all_dimensions_done(self):
        """With sufficient chunks and a mock LLM, agent completes all dimensions."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        leg_chunk = _legislation_chunk(
            text="ai system risk annex iii prohibited transparency provider deployer "
                 "documentation oversight monitoring governance article",
        )
        retrieved = {leg_chunk.chunk_id}
        lookup = {leg_chunk.chunk_id: leg_chunk}
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        context = {"max_retrievals_reached": set(), "refined_context": ""}
        mock_client, call_count = self._make_multi_response_client(leg_chunk.chunk_id)

        signal = run_analysis_agent(
            view, [leg_chunk], context, llm_client=mock_client
        )
        assert isinstance(signal, CompletionSignal)
        assert call_count[0] == 6   # one call per dimension

    def test_resumes_from_partially_completed_dimensions(self):
        """Pre-writing definition_check should result in only 5 LLM calls."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        leg_chunk = _legislation_chunk(
            chunk_id="leg_001",
            text="ai system risk annex iii prohibited transparency provider deployer "
                 "documentation oversight monitoring governance",
        )
        retrieved = {leg_chunk.chunk_id}
        lookup = {leg_chunk.chunk_id: leg_chunk}

        # Pre-write definition_check so the agent skips it
        memory.definition_check = DefinitionSection(
            dimension_id="definition_check",
            claims=[],
            confidence=Confidence.HIGH,
            summary="Already done.",
            is_ai_system=True,
        )

        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        context = {"max_retrievals_reached": set(), "refined_context": ""}
        mock_client, call_count = self._make_multi_response_client(leg_chunk.chunk_id)

        signal = run_analysis_agent(
            view, [leg_chunk], context, llm_client=mock_client
        )
        assert isinstance(signal, CompletionSignal)
        assert call_count[0] == 5   # skipped 1 dimension

    def test_proceeds_with_insufficient_when_max_retries_reached(self):
        """When a dimension is in max_retrievals_reached, agent must not signal."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        # Provide legislation chunks for all dims except definition_check
        leg_chunk = _legislation_chunk(
            text="risk annex iii prohibited transparency provider deployer "
                 "documentation oversight monitoring",
        )
        retrieved = {leg_chunk.chunk_id}
        lookup = {leg_chunk.chunk_id: leg_chunk}
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        # Mark definition_check as max reached
        context = {
            "max_retrievals_reached": {"definition_check"},
            "refined_context": "",
        }
        mock_client, call_count = self._make_multi_response_client(leg_chunk.chunk_id)

        signal = run_analysis_agent(
            view, [leg_chunk], context, llm_client=mock_client
        )
        # Should complete (not loop forever signalling for definition_check)
        assert isinstance(signal, CompletionSignal)
