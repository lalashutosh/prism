"""
tests/test_validation_agent.py
───────────────────────────────
Unit tests for validation_agent.py.

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
    DefinitionSection,
    FactSection,
    Label,
    OverturnedClaim,
    RetrievalSignal,
    RiskSection,
    RiskLevel,
    ValidationSection,
    WeakClaim,
)
from core.memory import SessionMemory, ValidationAgentMemoryView
from agents.validation_agent import (
    build_claim_validation_prompt,
    build_overturned_claim,
    check_weak_claim_criteria,
    identify_weak_claims,
    parse_claim_validation_response,
    run_validation_agent,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _leg_chunk(chunk_id: str = "leg_001", text: str = "EU AI Act Article 3") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_type="legislation",
        article_id="Article 3",
    )


def _make_facts() -> FactSection:
    return FactSection(
        use_case_name="Fraud Detection",
        description="Detects fraudulent transactions using ML.",
    )


def _make_claim(
    claim_id: str = "def_0",
    label: Label = Label.RETRIEVED,
    confidence: Confidence = Confidence.HIGH,
    chunk_ids: list[str] = None,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        text="The system uses ML for fraud detection.",
        label=label,
        confidence=confidence,
        # Use None sentinel so callers can explicitly pass an empty list.
        chunk_ids=chunk_ids if chunk_ids is not None else ["leg_001"],
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


def _make_risk_section(
    claims: list[Claim] = None,
    confidence: Confidence = Confidence.HIGH,
) -> RiskSection:
    return RiskSection(
        dimension_id="risk_classification",
        claims=claims or [_make_claim(claim_id="risk_0")],
        confidence=confidence,
        summary="High risk.",
        risk_level=RiskLevel.HIGH,
    )


def _setup_validation_view(
    memory: SessionMemory = None,
    chunks: list[Chunk] = None,
) -> tuple[ValidationAgentMemoryView, list[Chunk]]:
    if memory is None:
        memory = SessionMemory()
    memory.facts = _make_facts()
    if chunks is None:
        chunks = [_leg_chunk()]
    retrieved = {c.chunk_id for c in chunks}
    lookup = {c.chunk_id: c for c in chunks}
    view = ValidationAgentMemoryView(memory, retrieved, lookup)
    return view, chunks


def _make_mock_llm(response_json: dict | str) -> Any:
    if isinstance(response_json, dict):
        content = json.dumps(response_json)
    else:
        content = str(response_json)
    mock_cb = MagicMock()
    mock_cb.text = content
    mock_resp = MagicMock()
    mock_resp.content = [mock_cb]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_resp
    return mock_client


# ── check_weak_claim_criteria ─────────────────────────────────────────────

class TestCheckWeakClaimCriteria:
    def test_low_confidence_is_weak(self):
        claim = _make_claim(confidence=Confidence.LOW)
        is_weak, reason = check_weak_claim_criteria(claim, "definition_check")
        assert is_weak is True
        assert reason == "LOW_CONFIDENCE"

    def test_insufficient_confidence_is_weak(self):
        claim = _make_claim(confidence=Confidence.INSUFFICIENT)
        is_weak, reason = check_weak_claim_criteria(claim, "definition_check")
        assert is_weak is True
        assert reason == "LOW_CONFIDENCE"

    def test_assumption_label_on_critical_dim_is_weak(self):
        claim = _make_claim(label=Label.ASSUMPTION, confidence=Confidence.HIGH)
        is_weak, reason = check_weak_claim_criteria(claim, "definition_check")
        assert is_weak is True
        assert reason == "ASSUMPTION"

    def test_assumption_label_on_noncritical_dim_with_high_conf_is_not_weak(self):
        claim = _make_claim(label=Label.ASSUMPTION, confidence=Confidence.HIGH)
        is_weak, reason = check_weak_claim_criteria(claim, "governance")
        assert is_weak is False

    def test_assumption_on_noncritical_dim_with_low_conf_is_weak(self):
        claim = _make_claim(label=Label.ASSUMPTION, confidence=Confidence.MEDIUM)
        is_weak, reason = check_weak_claim_criteria(claim, "governance")
        assert is_weak is True
        assert reason == "ASSUMPTION"

    def test_empty_chunk_ids_is_weak(self):
        claim = _make_claim(confidence=Confidence.HIGH, chunk_ids=[])
        is_weak, reason = check_weak_claim_criteria(claim, "transparency")
        assert is_weak is True
        assert reason == "UNSUPPORTED"

    def test_strong_claim_is_not_weak(self):
        claim = _make_claim(
            confidence=Confidence.HIGH,
            label=Label.RETRIEVED,
            chunk_ids=["leg_001"],
        )
        is_weak, reason = check_weak_claim_criteria(claim, "roles")
        assert is_weak is False
        assert reason == ""


# ── identify_weak_claims ──────────────────────────────────────────────────

class TestIdentifyWeakClaims:
    def test_finds_low_confidence_claim(self):
        sections = {
            "definition_check": _make_definition_section(
                claims=[_make_claim(confidence=Confidence.LOW)]
            )
        }
        result = identify_weak_claims(sections)
        assert len(result) == 1
        assert result[0].reason == "LOW_CONFIDENCE"

    def test_finds_assumption_label_in_critical_dim(self):
        sections = {
            "risk_classification": _make_risk_section(
                claims=[_make_claim(claim_id="risk_0", label=Label.ASSUMPTION)]
            )
        }
        result = identify_weak_claims(sections)
        assert len(result) == 1
        assert result[0].reason == "ASSUMPTION"

    def test_finds_empty_chunk_ids(self):
        sections = {
            "governance": _make_definition_section(
                claims=[
                    Claim(
                        claim_id="gov_0",
                        text="Governance claim.",
                        label=Label.RETRIEVED,
                        confidence=Confidence.HIGH,
                        chunk_ids=[],  # no citations
                    )
                ]
            )
        }
        result = identify_weak_claims(sections)
        assert len(result) == 1
        assert result[0].reason == "UNSUPPORTED"

    def test_returns_empty_for_all_strong_claims(self):
        sections = {
            "definition_check": _make_definition_section(
                claims=[
                    _make_claim(
                        confidence=Confidence.HIGH,
                        label=Label.RETRIEVED,
                        chunk_ids=["leg_001"],
                    )
                ]
            )
        }
        result = identify_weak_claims(sections)
        assert result == []

    def test_deduplicates_by_claim_id(self):
        """Same claim_id appearing in two sections should only appear once."""
        same_claim = _make_claim(claim_id="shared_0", confidence=Confidence.LOW)
        sections = {
            "definition_check": _make_definition_section(claims=[same_claim]),
            "risk_classification": _make_risk_section(claims=[same_claim]),
        }
        result = identify_weak_claims(sections)
        ids = [wc.claim_id for wc in result]
        assert ids.count("shared_0") == 1

    def test_skips_none_sections(self):
        sections = {
            "definition_check": None,
            "risk_classification": _make_risk_section(claims=[]),
        }
        result = identify_weak_claims(sections)
        assert result == []


# ── build_claim_validation_prompt ─────────────────────────────────────────

class TestBuildClaimValidationPrompt:
    def test_prompt_contains_claim_text(self):
        wc = WeakClaim(
            claim_id="def_0",
            dimension_id="definition_check",
            claim_text="Unique claim text ABC123",
            reason="LOW_CONFIDENCE",
            original_confidence=Confidence.LOW,
            original_label=Label.ASSUMPTION,
        )
        chunks = [_leg_chunk(text="Relevant legislation text")]
        prompt = build_claim_validation_prompt(wc, _make_facts(), chunks)
        assert "Unique claim text ABC123" in prompt

    def test_prompt_contains_dimension_id(self):
        wc = WeakClaim(
            claim_id="def_0",
            dimension_id="definition_check",
            claim_text="A claim.",
            reason="UNSUPPORTED",
            original_confidence=Confidence.HIGH,
            original_label=Label.RETRIEVED,
        )
        prompt = build_claim_validation_prompt(wc, _make_facts(), [])
        assert "definition_check" in prompt

    def test_prompt_contains_chunk_id(self):
        wc = WeakClaim(
            claim_id="def_0",
            dimension_id="definition_check",
            claim_text="A claim.",
            reason="LOW_CONFIDENCE",
            original_confidence=Confidence.LOW,
            original_label=Label.ASSUMPTION,
        )
        chunk = _leg_chunk(chunk_id="unique_chunk_XYZ")
        prompt = build_claim_validation_prompt(wc, _make_facts(), [chunk])
        assert "unique_chunk_XYZ" in prompt

    def test_prompt_contains_use_case_name(self):
        wc = WeakClaim(
            claim_id="def_0",
            dimension_id="definition_check",
            claim_text="A claim.",
            reason="LOW_CONFIDENCE",
            original_confidence=Confidence.LOW,
            original_label=Label.ASSUMPTION,
        )
        facts = FactSection(
            use_case_name="Unique Use Case Name QWERTY",
            description="Test.",
        )
        prompt = build_claim_validation_prompt(wc, facts, [])
        assert "Unique Use Case Name QWERTY" in prompt


# ── parse_claim_validation_response ──────────────────────────────────────

class TestParseClaimValidationResponse:
    def _make_weak_claim(self) -> WeakClaim:
        return WeakClaim(
            claim_id="def_0",
            dimension_id="definition_check",
            claim_text="The claim.",
            reason="LOW_CONFIDENCE",
            original_confidence=Confidence.LOW,
            original_label=Label.ASSUMPTION,
        )

    def test_parses_confirmed_status(self):
        raw = json.dumps({
            "claim_id": "def_0",
            "status": "CONFIRMED",
            "finding": "Evidence confirms the claim.",
            "supporting_chunk_ids": ["leg_001"],
            "new_confidence": "HIGH",
            "new_label": "RETRIEVED",
        })
        status, finding, chunk_ids, confidence, label = parse_claim_validation_response(
            raw, self._make_weak_claim()
        )
        assert status == ClaimStatus.CONFIRMED
        assert "confirms" in finding
        assert "leg_001" in chunk_ids
        assert confidence == Confidence.HIGH
        assert label == Label.RETRIEVED

    def test_parses_overturned_status(self):
        raw = json.dumps({
            "claim_id": "def_0",
            "status": "OVERTURNED",
            "finding": "New evidence contradicts.",
            "supporting_chunk_ids": ["leg_002"],
            "new_confidence": "MEDIUM",
            "new_label": "RETRIEVED",
        })
        status, _, _, _, _ = parse_claim_validation_response(
            raw, self._make_weak_claim()
        )
        assert status == ClaimStatus.OVERTURNED

    def test_parses_unresolved_status(self):
        raw = json.dumps({
            "claim_id": "def_0",
            "status": "UNRESOLVED",
            "finding": "Insufficient evidence.",
            "supporting_chunk_ids": [],
            "new_confidence": "INSUFFICIENT",
            "new_label": "UNCERTAIN",
        })
        status, _, _, _, _ = parse_claim_validation_response(
            raw, self._make_weak_claim()
        )
        assert status == ClaimStatus.UNRESOLVED

    def test_malformed_response_returns_unresolved(self):
        status, finding, _, _, _ = parse_claim_validation_response(
            "not json", self._make_weak_claim()
        )
        assert status == ClaimStatus.UNRESOLVED

    def test_unknown_status_maps_to_unresolved(self):
        raw = json.dumps({
            "status": "TOTALLY_MADE_UP",
            "finding": "?",
            "supporting_chunk_ids": [],
            "new_confidence": "LOW",
            "new_label": "UNCERTAIN",
        })
        status, _, _, _, _ = parse_claim_validation_response(
            raw, self._make_weak_claim()
        )
        assert status == ClaimStatus.UNRESOLVED


# ── build_overturned_claim ────────────────────────────────────────────────

class TestBuildOverturnedClaim:
    def test_builds_correctly(self):
        wc = WeakClaim(
            claim_id="def_0",
            dimension_id="definition_check",
            claim_text="Original claim.",
            reason="LOW_CONFIDENCE",
            original_confidence=Confidence.LOW,
            original_label=Label.ASSUMPTION,
        )
        oc = build_overturned_claim(
            wc,
            new_finding="New evidence shows X.",
            new_confidence=Confidence.HIGH,
            new_label=Label.RETRIEVED,
            new_chunk_ids=["leg_002"],
        )
        assert isinstance(oc, OverturnedClaim)
        assert oc.claim_id == "def_0"
        assert oc.dimension_id == "definition_check"
        assert oc.original_claim_text == "Original claim."
        assert oc.new_finding == "New evidence shows X."
        assert oc.new_confidence == Confidence.HIGH
        assert oc.new_label == Label.RETRIEVED
        assert oc.new_chunk_ids == ["leg_002"]
        assert oc.status == ClaimStatus.OVERTURNED


# ── run_validation_agent (orchestration) ─────────────────────────────────

class TestRunValidationAgent:
    """Orchestration tests with a mock LLM client."""

    def _confirmed_response(self, claim_id: str = "def_0", chunk_id: str = "leg_001") -> dict:
        return {
            "claim_id": claim_id,
            "status": "CONFIRMED",
            "finding": "Evidence confirms the claim.",
            "supporting_chunk_ids": [chunk_id],
            "new_confidence": "HIGH",
            "new_label": "RETRIEVED",
        }

    def test_emits_completion_signal_when_no_weak_claims(self):
        """When analysis sections have no weak claims, agent completes immediately."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        # All strong claims
        memory.definition_check = _make_definition_section(
            claims=[_make_claim(confidence=Confidence.HIGH, chunk_ids=["leg_001"])]
        )
        chunk = _leg_chunk()
        view, chunks = _setup_validation_view(memory=memory, chunks=[chunk])
        context: dict = {}
        mock_client = _make_mock_llm(self._confirmed_response())

        signal = run_validation_agent(view, chunks, context, llm_client=mock_client)
        assert isinstance(signal, CompletionSignal)

    def test_emits_retrieval_signal_for_weak_claim_without_authoritative_chunks(self):
        """When we have a weak claim but no legislation chunks, signal for more."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        # Weak claim (LOW confidence)
        memory.definition_check = _make_definition_section(
            claims=[_make_claim(confidence=Confidence.LOW, chunk_ids=["doc_001"])]
        )
        # Only uploaded doc chunks available
        doc_chunk = Chunk(
            chunk_id="doc_001",
            text="ai system machine learning",
            source_type="uploaded_doc",
            article_id=None,
        )
        retrieved = {"doc_001"}
        lookup = {"doc_001": doc_chunk}
        view = ValidationAgentMemoryView(memory, retrieved, lookup)
        context: dict = {}

        signal = run_validation_agent(view, [doc_chunk], context, llm_client=MagicMock())
        assert isinstance(signal, RetrievalSignal)

    def test_emits_completion_signal_after_processing_weak_claim(self):
        """With legislation chunks and a mock LLM, agent processes claim and completes."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        memory.definition_check = _make_definition_section(
            claims=[_make_claim(confidence=Confidence.LOW, chunk_ids=["leg_001"])]
        )
        chunk = _leg_chunk(text="ai system definition machine learning algorithm")
        view, chunks = _setup_validation_view(memory=memory, chunks=[chunk])
        context: dict = {}
        mock_client = _make_mock_llm(self._confirmed_response("def_0", "leg_001"))

        signal = run_validation_agent(view, chunks, context, llm_client=mock_client)
        assert isinstance(signal, CompletionSignal)
        # Validation flags must be written
        assert memory.validation_flags is not None

    def test_writes_overturned_claims_to_memory(self):
        """When LLM returns OVERTURNED, it should appear in memory.overturned_claims."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        memory.definition_check = _make_definition_section(
            claims=[_make_claim(confidence=Confidence.LOW, chunk_ids=["leg_001"])]
        )
        chunk = _leg_chunk(text="ai system definition algorithm machine learning")
        view, chunks = _setup_validation_view(memory=memory, chunks=[chunk])
        context: dict = {}
        overturned_resp = {
            "claim_id": "def_0",
            "status": "OVERTURNED",
            "finding": "Evidence contradicts.",
            "supporting_chunk_ids": ["leg_001"],
            "new_confidence": "HIGH",
            "new_label": "RETRIEVED",
        }
        mock_client = _make_mock_llm(overturned_resp)

        signal = run_validation_agent(view, chunks, context, llm_client=mock_client)
        assert isinstance(signal, CompletionSignal)
        assert len(memory.overturned_claims) == 1
        assert memory.overturned_claims[0].status == ClaimStatus.OVERTURNED

    def test_processes_each_weak_claim_once(self):
        """Agent should not call LLM multiple times for the same claim."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        # Two weak claims
        memory.definition_check = _make_definition_section(
            claims=[
                _make_claim(claim_id="def_0", confidence=Confidence.LOW, chunk_ids=["leg_001"]),
                _make_claim(claim_id="def_1", confidence=Confidence.LOW, chunk_ids=["leg_001"]),
            ]
        )
        chunk = _leg_chunk(text="ai system definition algorithm machine learning")
        view, chunks = _setup_validation_view(memory=memory, chunks=[chunk])
        context: dict = {}

        call_count = [0]
        resp = self._confirmed_response("def_0", "leg_001")

        mock_cb = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [mock_cb]
        mock_client = MagicMock()

        def side_effect(*args, **kwargs):
            if call_count[0] == 0:
                mock_cb.text = json.dumps(self._confirmed_response("def_0", "leg_001"))
            else:
                mock_cb.text = json.dumps(self._confirmed_response("def_1", "leg_001"))
            call_count[0] += 1
            return mock_resp

        mock_client.messages.create.side_effect = side_effect

        signal = run_validation_agent(view, chunks, context, llm_client=mock_client)
        assert isinstance(signal, CompletionSignal)
        assert call_count[0] == 2   # exactly one call per weak claim

    def test_writes_weak_claims_list_to_memory(self):
        """After completion, memory.weak_claims should contain the identified claims."""
        memory = SessionMemory()
        memory.facts = _make_facts()
        memory.definition_check = _make_definition_section(
            claims=[_make_claim(confidence=Confidence.LOW, chunk_ids=["leg_001"])]
        )
        chunk = _leg_chunk(text="ai system definition algorithm machine learning")
        view, chunks = _setup_validation_view(memory=memory, chunks=[chunk])
        context: dict = {}
        mock_client = _make_mock_llm(self._confirmed_response())

        run_validation_agent(view, chunks, context, llm_client=mock_client)
        assert len(memory.weak_claims) == 1
        assert memory.weak_claims[0].claim_id == "def_0"
