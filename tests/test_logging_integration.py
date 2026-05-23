"""
tests/test_logging_integration.py
───────────────────────────────────
Integration tests verifying that the three logging wiring points work
end-to-end:

  1. log_reasoning decorator on agent _call_llm functions
       → reasoning entries appear in the DB with the correct session_id,
         agent tag, dimension, prompt, and response.

  2. _log_write decorator on memory proxy write methods
       → StateChangeEntry rows appear in the DB for both successful and
         failed writes, with the correct section name, agent, and payload.

  3. Orchestrator with prism_logger=
       → pipeline events (SESSION_STARTED/COMPLETED, AGENT_STARTED/
         COMPLETED, CHECKPOINT_SAVED, CACHE_HIT, RETRIEVE_CALLED) appear
         in the DB, and RETRIEVAL_SIGNAL / MEMORY_WRITE_ERROR signals are
         recorded with resolve_signal called after handling.

All tests use an in-memory SQLite database (:memory:) and mock LLM clients
— no real API calls and no file I/O.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.log_schema import PipelineEvent, ReasoningEntry, StateChangeEntry, SignalEntry
from core.logger import PrismLogger, SESSION_ID_VAR, _LOGGER_VAR, log_reasoning
from core.memory import (
    AnalysisAgentMemoryView,
    SessionMemory,
    ValidationAgentMemoryView,
    SynthesisAgentMemoryView,
)
from core.types import (
    Chunk,
    Claim,
    ClaimStatus,
    Confidence,
    ConfidenceSection,
    DefinitionSection,
    FactSection,
    FollowUpSection,
    Label,
    MemoryWriteError,
    OrchestratorState,
    ReportSection,
    RiskLevel,
    RiskSection,
    ValidationSection,
)
from agents.analysis_agent import run_analysis_agent, DIMENSION_ORDER
from core.orchestrator import Orchestrator


# ── Shared helpers ────────────────────────────────────────────────────────────

SESSION = "integ-session-001"


@pytest.fixture
def lg():
    """Fresh in-memory PrismLogger for each test."""
    prism = PrismLogger(session_id=SESSION, db_path=":memory:")
    yield prism
    prism.close()


def _legislation_chunk(chunk_id: str = "leg_001") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        # Text spans all six dimension keyword sets so one chunk satisfies all
        # evidence-sufficiency checks in run_analysis_agent.
        text=(
            "ai system definition machine learning algorithm risk annex iii "
            "prohibited transparency provider deployer documentation oversight "
            "monitoring governance article"
        ),
        source_type="legislation",
        article_id="Article 3",
    )


def _make_facts() -> FactSection:
    return FactSection(
        use_case_name="Hiring Algorithm",
        description="Automated CV screening using ML classification.",
        industry="Human Resources",
        ai_capabilities=["classification"],
        data_inputs=["CVs"],
        outputs=["shortlist"],
        deployment_context="HR department",
        affected_persons=["job applicants"],
    )


def _make_mock_llm(response_text: str) -> Any:
    """Return a mock Anthropic client that always returns *response_text*."""
    block = MagicMock()
    block.text = response_text
    resp = MagicMock()
    resp.content = [block]
    client = MagicMock()
    client.messages.create.return_value = resp
    return client


def _valid_dim_response(dim: str, chunk_id: str = "leg_001") -> str:
    """Produce a JSON string that parse_dimension_response accepts."""
    base = {
        "dimension_id": dim,
        "claims": [{
            "claim_id": f"{dim}_0",
            "text": f"Claim for {dim}.",
            "label": "RETRIEVED",
            "confidence": "HIGH",
            "chunk_ids": [chunk_id],
            "is_weak": False,
        }],
        "confidence": "HIGH",
        "summary": f"Summary for {dim}.",
    }
    extras = {
        "definition_check":     {"is_ai_system": True},
        "risk_classification":  {"risk_level": "high"},
        "prohibited_practices": {"triggered_articles": [], "prohibited": False},
        "transparency":         {"applies_to_gpai": False, "labelling_required": False,
                                 "notification_required": True},
        "roles":                {"is_provider": False, "is_deployer": True, "is_both": False},
        "governance":           {"documentation_required": True, "oversight_required": True,
                                 "monitoring_required": False},
    }
    base.update(extras.get(dim, {}))
    return json.dumps(base)


def _make_multi_dim_client(chunk_id: str = "leg_001") -> tuple[Any, list]:
    """Mock client that serves one valid response per dimension in DIMENSION_ORDER."""
    responses = [_valid_dim_response(dim, chunk_id) for dim in DIMENSION_ORDER]
    call_count = [0]

    block = MagicMock()
    resp = MagicMock()
    resp.content = [block]
    client = MagicMock()

    def side_effect(*args, **kwargs):
        idx = call_count[0]
        block.text = responses[idx] if idx < len(responses) else "{}"
        call_count[0] += 1
        return resp

    client.messages.create.side_effect = side_effect
    return client, call_count


# ── 1. log_reasoning decorator wiring ────────────────────────────────────────

class TestReasoningDecoratorWiring:
    """Verify that the log_reasoning decorator on agent _call_llm functions
    correctly writes ReasoningEntry rows when a PrismLogger is active."""

    def test_analysis_agent_reasoning_entries_written(self, lg):
        """run_analysis_agent writes one ReasoningEntry per dimension (6 total)."""
        chunk = _legislation_chunk()
        memory = SessionMemory()
        memory.facts = _make_facts()
        retrieved = {chunk.chunk_id}
        lookup = {chunk.chunk_id: chunk}
        view = AnalysisAgentMemoryView(memory, retrieved, lookup)
        context = {"max_retrievals_reached": set(), "refined_context": ""}

        client, _ = _make_multi_dim_client(chunk.chunk_id)
        run_analysis_agent(view, [chunk], context, llm_client=client)

        entries = lg.get_dimension_reasoning(SESSION, "definition_check")
        assert len(entries) == 1
        # Six dimensions → six entries total
        full = lg.get_full_session_log(SESSION)
        assert len(full.reasoning_entries) == 6

    def test_analysis_dimension_tag_per_entry(self, lg):
        """Each ReasoningEntry is tagged with the dimension it assessed."""
        chunk = _legislation_chunk()
        memory = SessionMemory()
        memory.facts = _make_facts()
        view = AnalysisAgentMemoryView(
            memory, {chunk.chunk_id}, {chunk.chunk_id: chunk}
        )
        context = {"max_retrievals_reached": set(), "refined_context": ""}
        client, _ = _make_multi_dim_client(chunk.chunk_id)
        run_analysis_agent(view, [chunk], context, llm_client=client)

        for dim in DIMENSION_ORDER:
            entries = lg.get_dimension_reasoning(SESSION, dim)
            assert len(entries) == 1, f"Expected 1 entry for {dim}, got {len(entries)}"
            assert entries[0].agent == "analysis"
            assert entries[0].dimension == dim

    def test_reasoning_entry_prompt_and_response_captured(self, lg):
        """The prompt sent and raw LLM response are stored verbatim."""
        chunk = _legislation_chunk()
        memory = SessionMemory()
        memory.facts = _make_facts()
        view = AnalysisAgentMemoryView(
            memory, {chunk.chunk_id}, {chunk.chunk_id: chunk}
        )
        context = {"max_retrievals_reached": set(), "refined_context": ""}
        client, _ = _make_multi_dim_client(chunk.chunk_id)
        run_analysis_agent(view, [chunk], context, llm_client=client)

        entry = lg.get_dimension_reasoning(SESSION, "definition_check")[0]
        # Prompt must mention the use case name injected from FactSection.
        assert "Hiring Algorithm" in entry.prompt_sent
        # Response must be the raw JSON string returned by the mock LLM.
        assert "definition_check" in entry.llm_response

    def test_reasoning_duration_ms_non_negative(self, lg):
        """duration_ms is populated and is a non-negative integer."""
        chunk = _legislation_chunk()
        memory = SessionMemory()
        memory.facts = _make_facts()
        view = AnalysisAgentMemoryView(
            memory, {chunk.chunk_id}, {chunk.chunk_id: chunk}
        )
        context = {"max_retrievals_reached": set(), "refined_context": ""}
        client, _ = _make_multi_dim_client(chunk.chunk_id)
        run_analysis_agent(view, [chunk], context, llm_client=client)

        entry = lg.get_dimension_reasoning(SESSION, "risk_classification")[0]
        assert isinstance(entry.duration_ms, int)
        assert entry.duration_ms >= 0

    def test_decorator_noop_when_no_logger_active(self):
        """When SESSION_ID_VAR and _LOGGER_VAR hold their defaults, the decorator
        must not raise — agent tests that never set up a PrismLogger rely on this."""
        # Ensure ContextVars are at their defaults for this test.
        t1 = _LOGGER_VAR.set(None)
        t2 = SESSION_ID_VAR.set("")
        try:
            fn = MagicMock(return_value="raw response")
            decorated = log_reasoning(agent="analysis", dimension="definition_check")(fn)
            result = decorated("my prompt", "system text", MagicMock())
            assert result == "raw response"
            fn.assert_called_once()
        finally:
            _LOGGER_VAR.reset(t1)
            SESSION_ID_VAR.reset(t2)

    def test_synthesis_agent_reasoning_entry_has_no_dimension(self, lg):
        """Synthesis _call_llm is decorated with dimension=None."""
        from agents.synthesis_agent import _call_llm as synth_call_llm

        mock_client = _make_mock_llm(json.dumps({
            "report": {
                "use_case_summary": "test",
                "extracted_facts": {},
                "ai_definition_check": {},
                "risk_classification": {},
                "prohibited_practices_check": {},
                "transparency_gpai_obligations": {},
                "roles": {},
                "governance_observations": {},
                "missing_information": {},
                "citations_by_source": {},
            },
            "follow_up": {"questions": [], "missing_evidence": []},
            "confidence": {
                "definition_check": "HIGH", "risk_classification": "HIGH",
                "prohibited_practices": "HIGH", "transparency": "HIGH",
                "roles": "HIGH", "governance": "HIGH", "overall": "HIGH",
            },
        }))
        synth_call_llm("synthesis prompt", "system", mock_client)

        full = lg.get_full_session_log(SESSION)
        assert len(full.reasoning_entries) == 1
        entry = full.reasoning_entries[0]
        assert entry.agent == "synthesis"
        assert entry.dimension is None


# ── 2. _log_write proxy decorator wiring ─────────────────────────────────────

class TestStateChangeWiring:
    """Verify that _log_write on proxy write methods emits StateChangeEntry rows."""

    def _make_valid_definition(self, chunk_id: str = "leg_001") -> DefinitionSection:
        return DefinitionSection(
            dimension_id="definition_check",
            claims=[Claim(
                claim_id="def_0",
                text="This is an AI system.",
                label=Label.RETRIEVED,
                confidence=Confidence.HIGH,
                chunk_ids=[chunk_id],
            )],
            confidence=Confidence.HIGH,
            summary="AI system confirmed.",
            is_ai_system=True,
        )

    def test_successful_write_emits_state_change(self, lg):
        """A validated write produces a StateChangeEntry with write_validated=True."""
        chunk = _legislation_chunk()
        memory = SessionMemory()
        view = AnalysisAgentMemoryView(
            memory, {chunk.chunk_id}, {chunk.chunk_id: chunk}
        )
        section = self._make_valid_definition(chunk.chunk_id)
        view.write_definition(section)

        entries = lg.get_state_history(SESSION, "definition_check")
        assert len(entries) == 1
        entry = entries[0]
        assert entry.write_validated is True
        assert entry.agent == "analysis"
        assert entry.section == "definition_check"
        assert entry.validation_errors == []

    def test_failed_write_emits_state_change_with_errors(self, lg):
        """A rejected write produces a StateChangeEntry with write_validated=False."""
        chunk = _legislation_chunk()
        memory = SessionMemory()
        view = AnalysisAgentMemoryView(
            memory, {chunk.chunk_id}, {chunk.chunk_id: chunk}
        )
        # Pass a RiskSection where DefinitionSection is expected → schema error.
        wrong_type = RiskSection(
            dimension_id="risk_classification",
            claims=[],
            confidence=Confidence.HIGH,
            summary="wrong",
            risk_level=RiskLevel.HIGH,
        )
        with pytest.raises(MemoryWriteError):
            view.write_definition(wrong_type)  # type: ignore[arg-type]

        entries = lg.get_state_history(SESSION, "definition_check")
        assert len(entries) == 1
        entry = entries[0]
        assert entry.write_validated is False
        assert len(entry.validation_errors) == 1
        assert "schema" in entry.validation_errors[0]

    def test_previous_state_captured_on_overwrite(self, lg):
        """When a section is written twice, the second entry records the first as
        previous_state."""
        chunk = _legislation_chunk()
        memory = SessionMemory()
        view = AnalysisAgentMemoryView(
            memory, {chunk.chunk_id}, {chunk.chunk_id: chunk}
        )
        first = self._make_valid_definition(chunk.chunk_id)
        view.write_definition(first)

        second = self._make_valid_definition(chunk.chunk_id)
        second.summary = "Updated summary."
        view.write_definition(second)

        entries = lg.get_state_history(SESSION, "definition_check")
        assert len(entries) == 2
        # First write: no previous state.
        assert entries[0].previous_state is None
        # Second write: previous_state is the first write's new_state.
        assert entries[1].previous_state is not None

    def test_multiple_sections_recorded_independently(self, lg):
        """Each analysis section has its own state_change row."""
        chunk = _legislation_chunk()
        memory = SessionMemory()
        view = AnalysisAgentMemoryView(
            memory, {chunk.chunk_id}, {chunk.chunk_id: chunk}
        )
        # Write three different sections.
        view.write_definition(self._make_valid_definition(chunk.chunk_id))

        risk_section = RiskSection(
            dimension_id="risk_classification",
            claims=[],
            confidence=Confidence.INSUFFICIENT,
            summary="Risk TBD.",
            risk_level=RiskLevel.UNKNOWN,
        )
        view.write_risk(risk_section)

        entries_def  = lg.get_state_history(SESSION, "definition_check")
        entries_risk = lg.get_state_history(SESSION, "risk_classification")
        assert len(entries_def)  == 1
        assert len(entries_risk) == 1
        assert entries_def[0].section  == "definition_check"
        assert entries_risk[0].section == "risk_classification"

    def test_list_section_serialised_as_dict(self, lg):
        """weak_claims (a list) is stored as {"items": [...]} — always a dict."""
        memory = SessionMemory()
        view = ValidationAgentMemoryView(memory, set(), {})

        val_section = ValidationSection(
            flags=[],
            overall_confidence=Confidence.HIGH,
            summary="No weak claims.",
        )
        view.write_validation_flags(val_section)
        view.write_weak_claims([])
        view.write_overturned_claims([])

        wc_entries = lg.get_state_history(SESSION, "weak_claims")
        assert len(wc_entries) == 1
        # new_state must always be a dict (never a bare list).
        assert isinstance(wc_entries[0].new_state, dict)

    def test_no_state_change_written_when_no_logger_active(self):
        """Proxy writes are silent when no PrismLogger is active (e.g. unit tests)."""
        # Clear ContextVars so the helpers see defaults.
        t1 = _LOGGER_VAR.set(None)
        t2 = SESSION_ID_VAR.set("")
        try:
            chunk = _legislation_chunk()
            memory = SessionMemory()
            view = AnalysisAgentMemoryView(
                memory, {chunk.chunk_id}, {chunk.chunk_id: chunk}
            )
            section = DefinitionSection(
                dimension_id="definition_check",
                claims=[Claim(
                    claim_id="def_0", text="test", label=Label.RETRIEVED,
                    confidence=Confidence.HIGH, chunk_ids=[chunk.chunk_id],
                )],
                confidence=Confidence.HIGH,
                summary="ok",
                is_ai_system=True,
            )
            # Should complete without error even though no logger is active.
            view.write_definition(section)
            assert memory.definition_check is not None
        finally:
            _LOGGER_VAR.reset(t1)
            SESSION_ID_VAR.reset(t2)


# ── 3. Orchestrator pipeline / signal logging ─────────────────────────────────

class TestOrchestratorLogging:
    """Verify that Orchestrator emits pipeline events and signals to PrismLogger."""

    def _make_retrieve_fn(self, chunks: list[Chunk]):
        """Return a retrieve_fn that always returns *chunks*."""
        return lambda query, filters: chunks

    def _run_orchestrator(self, lg: PrismLogger, facts: FactSection) -> None:
        """Run the full pipeline with a mock LLM and supplied PrismLogger."""
        chunk = _legislation_chunk()

        # Build a multi-dim client + synthesis response.
        multi_client, _ = _make_multi_dim_client(chunk.chunk_id)
        synthesis_resp = json.dumps({
            "report": {
                "use_case_summary": "test",
                "extracted_facts": {}, "ai_definition_check": {},
                "risk_classification": {}, "prohibited_practices_check": {},
                "transparency_gpai_obligations": {}, "roles": {},
                "governance_observations": {}, "missing_information": {},
                "citations_by_source": {},
            },
            "follow_up": {"questions": [], "missing_evidence": []},
            "confidence": {
                "definition_check": "HIGH", "risk_classification": "HIGH",
                "prohibited_practices": "HIGH", "transparency": "HIGH",
                "roles": "HIGH", "governance": "HIGH", "overall": "HIGH",
            },
        })

        # Chain: 6 dimension responses, then 1 synthesis response, then 1
        # synthesis response for validation (validation has no weak claims
        # with HIGH confidence, so no extra calls).
        call_count = [0]
        dim_responses = [_valid_dim_response(d, chunk.chunk_id) for d in DIMENSION_ORDER]
        all_responses = dim_responses + [synthesis_resp]

        block = MagicMock()
        resp = MagicMock()
        resp.content = [block]
        combined_client = MagicMock()

        def side_effect(*args, **kwargs):
            idx = call_count[0]
            block.text = all_responses[idx] if idx < len(all_responses) else "{}"
            call_count[0] += 1
            return resp

        combined_client.messages.create.side_effect = side_effect

        orch = Orchestrator(
            retrieve_fn=self._make_retrieve_fn([chunk]),
            prism_logger=lg,
            llm_client=combined_client,
        )
        orch.run(facts=facts)

    def test_session_started_emitted(self, lg):
        """SESSION_STARTED is the first pipeline event."""
        self._run_orchestrator(lg, _make_facts())
        events = lg.get_session_trace(SESSION)
        assert events[0].event_type == "SESSION_STARTED"

    def test_session_completed_emitted(self, lg):
        """SESSION_COMPLETED is emitted after the pipeline finishes."""
        self._run_orchestrator(lg, _make_facts())
        types = [e.event_type for e in lg.get_session_trace(SESSION)]
        assert "SESSION_COMPLETED" in types

    def test_agent_started_completed_per_phase(self, lg):
        """AGENT_STARTED and AGENT_COMPLETED are emitted for analysis, validation,
        and synthesis."""
        self._run_orchestrator(lg, _make_facts())
        types = [e.event_type for e in lg.get_session_trace(SESSION)]
        for agent in ("analysis", "validation", "synthesis"):
            # Each phase must have both STARTED and COMPLETED.
            agents_started   = [e for e in lg.get_session_trace(SESSION)
                                 if e.event_type == "AGENT_STARTED" and e.agent == agent]
            agents_completed = [e for e in lg.get_session_trace(SESSION)
                                 if e.event_type == "AGENT_COMPLETED" and e.agent == agent]
            assert len(agents_started)   >= 1, f"AGENT_STARTED missing for {agent}"
            assert len(agents_completed) >= 1, f"AGENT_COMPLETED missing for {agent}"

    def test_checkpoint_saved_events(self, lg):
        """CHECKPOINT_SAVED events are emitted after extraction, analysis, validation,
        and synthesis."""
        self._run_orchestrator(lg, _make_facts())
        saved = [e for e in lg.get_session_trace(SESSION)
                 if e.event_type == "CHECKPOINT_SAVED"]
        # At minimum: after_extraction, after_analysis, after_validation, after_synthesis
        assert len(saved) >= 4
        names = {e.metadata.get("checkpoint") for e in saved}
        assert "after_extraction" in names
        assert "after_analysis"   in names
        assert "after_validation" in names
        assert "after_synthesis"  in names

    def test_retrieve_called_event_emitted(self, lg):
        """RETRIEVE_CALLED is emitted for the initial broad retrieval."""
        self._run_orchestrator(lg, _make_facts())
        retrieve_events = [e for e in lg.get_session_trace(SESSION)
                           if e.event_type == "RETRIEVE_CALLED"]
        assert len(retrieve_events) >= 1

    def test_session_completed_has_duration_ms(self, lg):
        """SESSION_COMPLETED carries a non-negative duration_ms."""
        self._run_orchestrator(lg, _make_facts())
        completed = [e for e in lg.get_session_trace(SESSION)
                     if e.event_type == "SESSION_COMPLETED"]
        assert len(completed) == 1
        assert completed[0].duration_ms is not None
        assert completed[0].duration_ms >= 0

    def test_no_logger_orchestrator_runs_without_error(self):
        """Orchestrator without prism_logger= still completes the pipeline."""
        chunk = _legislation_chunk()
        multi_client, _ = _make_multi_dim_client(chunk.chunk_id)
        synthesis_resp = json.dumps({
            "report": {k: {} for k in [
                "use_case_summary", "extracted_facts", "ai_definition_check",
                "risk_classification", "prohibited_practices_check",
                "transparency_gpai_obligations", "roles", "governance_observations",
                "missing_information", "citations_by_source",
            ]},
            "follow_up": {"questions": [], "missing_evidence": []},
            "confidence": {d: "HIGH" for d in [
                "definition_check", "risk_classification", "prohibited_practices",
                "transparency", "roles", "governance", "overall",
            ]},
        })

        call_count = [0]
        dim_responses = [_valid_dim_response(d, chunk.chunk_id) for d in DIMENSION_ORDER]
        all_responses = dim_responses + [synthesis_resp]

        block = MagicMock()
        resp = MagicMock()
        resp.content = [block]
        combined_client = MagicMock()

        def side_effect(*args, **kwargs):
            idx = call_count[0]
            block.text = all_responses[idx] if idx < len(all_responses) else "{}"
            call_count[0] += 1
            return resp

        combined_client.messages.create.side_effect = side_effect

        orch = Orchestrator(
            retrieve_fn=lambda q, f: [chunk],
            prism_logger=None,     # no logger
            llm_client=combined_client,
        )
        result = orch.run(facts=_make_facts())
        assert result is not None

    def test_memory_write_error_signal_emitted(self, lg):
        """When a MemoryWriteError occurs, a MEMORY_WRITE_ERROR signal is logged."""
        chunk = _legislation_chunk()

        # First call returns a response whose JSON will fail schema validation:
        # we return a RiskSection JSON when DefinitionSection is expected.
        wrong_response = json.dumps({
            "dimension_id": "definition_check",  # parse will succeed → write will validate
            # Force claim with citation to a chunk_id NOT in retrieved_chunk_ids
            "claims": [{
                "claim_id": "def_0",
                "text": "claim",
                "label": "RETRIEVED",
                "confidence": "HIGH",
                "chunk_ids": ["phantom_chunk_id_not_retrieved"],
                "is_weak": False,
            }],
            "confidence": "HIGH",
            "summary": "test",
            "is_ai_system": True,
        })

        block = MagicMock()
        block.text = wrong_response
        resp = MagicMock()
        resp.content = [block]
        client = MagicMock()
        client.messages.create.return_value = resp

        orch = Orchestrator(
            retrieve_fn=lambda q, f: [chunk],
            prism_logger=lg,
            llm_client=client,
        )
        # The MemoryWriteError (citation_integrity) is caught internally by the
        # orchestrator — run() should still complete without propagating.
        orch.run(facts=_make_facts())

        errors = lg.get_errors(SESSION)
        assert any(e.signal_type == "MEMORY_WRITE_ERROR" for e in errors), (
            "Expected MEMORY_WRITE_ERROR signal in error log"
        )
