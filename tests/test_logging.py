"""
tests/test_logging.py
──────────────────────
Unit tests for the Prism logging layer (log_schema.py, log_store.py, logger.py).

All tests use an in-memory SQLite database (:memory:) so there is no file I/O
for the database itself.  The one exception is TestFallbackLog, which needs a
real file path for the fallback log — it uses tempfile.TemporaryDirectory() for
that single test.

Coverage targets (from spec):
  - PipelineEvent written and retrieved correctly
  - ReasoningEntry written and retrieved correctly
  - StateChangeEntry written with validation_errors correctly
  - SignalEntry written and resolve_signal updates resolved_at
  - get_full_session_log returns all four streams for a session
  - get_dimension_reasoning filters correctly by dimension
  - get_errors returns only MEMORY_WRITE_ERROR and LLM_PARSE_ERROR entries
  - fallback.log written when SQLite write fails
  - session_id ContextVar correctly flows through reasoning decorator
  - two concurrent sessions do not mix log entries
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.log_schema import (
    PipelineEvent,
    ReasoningEntry,
    SessionLog,
    SignalEntry,
    StateChangeEntry,
    utc_now,
)
from core.logger import (
    SESSION_ID_VAR,
    _LOGGER_VAR,
    PrismLogger,
    log_reasoning,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────
# The logger fixture creates a fresh in-memory DB per test, preventing any
# state leakage between tests.  close() resets ContextVars so the next test
# starts with a clean slate.

@pytest.fixture
def session_id() -> str:
    return "test-session-001"


@pytest.fixture
def logger(session_id: str):
    """PrismLogger backed by an in-memory SQLite database."""
    lg = PrismLogger(session_id=session_id, db_path=":memory:")
    yield lg
    lg.close()


# ── Helper builders ───────────────────────────────────────────────────────────

def _pipeline_event(session_id: str, event_type: str = "SESSION_STARTED", **kwargs) -> PipelineEvent:
    return PipelineEvent(session_id=session_id, event_type=event_type, **kwargs)


def _reasoning_entry(session_id: str, agent: str = "analysis", **kwargs) -> ReasoningEntry:
    return ReasoningEntry(
        session_id=session_id,
        agent=agent,
        prompt_sent=kwargs.pop("prompt_sent", "What is an AI system?"),
        llm_response=kwargs.pop("llm_response", '{"confidence": "HIGH"}'),
        **kwargs,
    )


def _state_change(session_id: str, section: str = "definition_check", **kwargs) -> StateChangeEntry:
    return StateChangeEntry(
        session_id=session_id,
        section=section,
        agent=kwargs.pop("agent", "analysis"),
        new_state=kwargs.pop("new_state", {"dimension_id": section}),
        **kwargs,
    )


def _signal_entry(session_id: str, signal_type: str = "RETRIEVAL_SIGNAL", **kwargs) -> SignalEntry:
    return SignalEntry(
        session_id=session_id,
        signal_type=signal_type,
        agent=kwargs.pop("agent", "analysis"),
        payload=kwargs.pop("payload", {"query": "EU AI Act"}),
        resolution=kwargs.pop("resolution", "retrieved and re-invoked"),
        **kwargs,
    )


# ── PipelineEvent ─────────────────────────────────────────────────────────────

class TestPipelineEvent:
    """Verify that PipelineEvent is persisted and retrieved with all fields intact."""

    def test_written_and_retrieved_correctly(self, logger, session_id):
        event = PipelineEvent(
            session_id=session_id,
            event_type="SESSION_STARTED",
            metadata={"use_case": "Medical AI", "version": 2},
        )
        logger.pipeline(event)

        events = logger.get_session_trace(session_id)

        assert len(events) == 1
        e = events[0]
        assert e.session_id == session_id
        assert e.event_type == "SESSION_STARTED"
        assert e.metadata == {"use_case": "Medical AI", "version": 2}

    def test_agent_field_preserved(self, logger, session_id):
        logger.pipeline(_pipeline_event(session_id, "AGENT_STARTED", agent="analysis"))

        events = logger.get_session_trace(session_id)
        assert events[0].agent == "analysis"

    def test_duration_ms_preserved(self, logger, session_id):
        logger.pipeline(_pipeline_event(session_id, "AGENT_COMPLETED", agent="validation", duration_ms=2048))

        events = logger.get_session_trace(session_id)
        assert events[0].duration_ms == 2048

    def test_none_agent_for_session_level_events(self, logger, session_id):
        """Session-level events (SESSION_STARTED/COMPLETED) have no agent."""
        logger.pipeline(_pipeline_event(session_id, "SESSION_COMPLETED", agent=None))

        events = logger.get_session_trace(session_id)
        assert events[0].agent is None

    def test_empty_metadata_round_trips_as_empty_dict(self, logger, session_id):
        logger.pipeline(_pipeline_event(session_id, "CHECKPOINT_SAVED"))

        events = logger.get_session_trace(session_id)
        assert events[0].metadata == {}

    def test_multiple_events_returned_in_insertion_order(self, logger, session_id):
        for et in ("SESSION_STARTED", "AGENT_STARTED", "AGENT_COMPLETED"):
            logger.pipeline(_pipeline_event(session_id, et))

        events = logger.get_session_trace(session_id)
        assert [e.event_type for e in events] == [
            "SESSION_STARTED", "AGENT_STARTED", "AGENT_COMPLETED"
        ]


# ── ReasoningEntry ────────────────────────────────────────────────────────────

class TestReasoningEntry:
    """Verify that ReasoningEntry is persisted and retrieved with all fields intact."""

    def test_written_and_retrieved_correctly(self, logger, session_id):
        entry = ReasoningEntry(
            session_id=session_id,
            agent="analysis",
            dimension="definition_check",
            prompt_sent="What is an AI system under Article 3?",
            llm_response='{"confidence": "HIGH", "claims": []}',
            parsed_output={"confidence": "HIGH"},
            parse_succeeded=True,
            confidence="HIGH",
            claims_count=3,
            weak_claims_count=1,
            chunk_ids_used=["leg_001", "leg_002"],
            duration_ms=512,
        )
        logger.reasoning(entry)

        results = logger.get_dimension_reasoning(session_id, "definition_check")

        assert len(results) == 1
        r = results[0]
        assert r.session_id == session_id
        assert r.agent == "analysis"
        assert r.dimension == "definition_check"
        assert r.prompt_sent == "What is an AI system under Article 3?"
        assert r.parse_succeeded is True
        assert r.confidence == "HIGH"
        assert r.claims_count == 3
        assert r.weak_claims_count == 1
        assert r.chunk_ids_used == ["leg_001", "leg_002"]
        assert r.duration_ms == 512

    def test_parsed_output_serialized_and_deserialized(self, logger, session_id):
        parsed = {"dimension_id": "risk_classification", "risk_level": "high", "confidence": "MEDIUM"}
        entry = _reasoning_entry(
            session_id,
            dimension="risk_classification",
            parsed_output=parsed,
        )
        logger.reasoning(entry)

        # Inspect the raw store row to verify JSON round-trip.
        rows = logger._store.get_reasoning_entries(session_id)
        assert len(rows) == 1
        assert json.loads(rows[0]["parsed_output"]) == parsed

    def test_parse_failed_entry_stored_correctly(self, logger, session_id):
        entry = _reasoning_entry(
            session_id,
            dimension="governance",
            llm_response="not valid json at all",
            parsed_output=None,
            parse_succeeded=False,
            confidence=None,
        )
        logger.reasoning(entry)

        results = logger.get_dimension_reasoning(session_id, "governance")
        r = results[0]
        assert r.parse_succeeded is False
        assert r.parsed_output is None
        assert r.confidence is None

    def test_chunk_ids_used_round_trips(self, logger, session_id):
        ids = ["leg_001", "guid_002", "doc_003"]
        entry = _reasoning_entry(session_id, chunk_ids_used=ids)
        logger.reasoning(entry)

        rows = logger._store.get_reasoning_entries(session_id)
        assert json.loads(rows[0]["chunk_ids_used"]) == ids

    def test_empty_chunk_ids_round_trips_as_empty_list(self, logger, session_id):
        entry = _reasoning_entry(session_id, chunk_ids_used=[])
        logger.reasoning(entry)

        rows = logger._store.get_reasoning_entries(session_id)
        assert json.loads(rows[0]["chunk_ids_used"]) == []


# ── StateChangeEntry ──────────────────────────────────────────────────────────

class TestStateChangeEntry:
    """Verify write-attempt recording for both successful and failed proxy writes."""

    def test_successful_write_logged_correctly(self, logger, session_id):
        entry = _state_change(
            session_id,
            section="risk_classification",
            previous_state={"risk_level": "unknown"},
            new_state={"risk_level": "high"},
            write_validated=True,
        )
        logger.state_change(entry)

        history = logger.get_state_history(session_id, "risk_classification")
        assert len(history) == 1
        h = history[0]
        assert h.write_validated is True
        assert h.validation_errors == []
        assert h.previous_state == {"risk_level": "unknown"}
        assert h.new_state == {"risk_level": "high"}

    def test_written_with_validation_errors_correctly(self, logger, session_id):
        """Failed writes must record write_validated=False and the error messages."""
        entry = StateChangeEntry(
            session_id=session_id,
            section="definition_check",
            agent="analysis",
            new_state={"dimension_id": "definition_check", "confidence": "HIGH"},
            previous_state=None,
            write_validated=False,
            validation_errors=[
                "schema: expected DefinitionSection, got RiskSection",
                "citation_integrity: chunk_ids not in retrieved set: ['missing_001']",
            ],
        )
        logger.state_change(entry)

        history = logger.get_state_history(session_id, "definition_check")
        h = history[0]
        assert h.write_validated is False
        assert len(h.validation_errors) == 2
        assert "schema" in h.validation_errors[0]
        assert "citation_integrity" in h.validation_errors[1]

    def test_first_write_has_none_previous_state(self, logger, session_id):
        entry = _state_change(session_id, previous_state=None)
        logger.state_change(entry)

        history = logger.get_state_history(session_id, "definition_check")
        assert history[0].previous_state is None

    def test_section_filter_returns_only_matching_section(self, logger, session_id):
        logger.state_change(_state_change(session_id, section="definition_check"))
        logger.state_change(_state_change(session_id, section="risk_classification"))
        logger.state_change(_state_change(session_id, section="definition_check"))

        defn = logger.get_state_history(session_id, "definition_check")
        risk = logger.get_state_history(session_id, "risk_classification")

        assert len(defn) == 2
        assert all(h.section == "definition_check" for h in defn)
        assert len(risk) == 1
        assert risk[0].section == "risk_classification"

    def test_empty_validation_errors_round_trips_as_empty_list(self, logger, session_id):
        entry = _state_change(session_id, write_validated=True)
        logger.state_change(entry)

        history = logger.get_state_history(session_id, "definition_check")
        assert history[0].validation_errors == []


# ── SignalEntry ───────────────────────────────────────────────────────────────

class TestSignalEntry:
    """Verify signal persistence and the resolve_signal lifecycle."""

    def test_written_and_retrieved_correctly(self, logger, session_id):
        entry = SignalEntry(
            session_id=session_id,
            signal_type="RETRIEVAL_SIGNAL",
            agent="analysis",
            dimension="definition_check",
            payload={"query": "AI system definition", "filters": {"source_types": ["legislation"]}},
            resolution="retrieved 3 chunks and re-invoked analysis agent",
            retry_count=1,
        )
        logger.signal(entry)

        rows = logger._store.get_signals(session_id)
        assert len(rows) == 1
        s = rows[0]
        assert s["signal_type"] == "RETRIEVAL_SIGNAL"
        assert s["agent"] == "analysis"
        assert s["dimension"] == "definition_check"
        assert s["retry_count"] == 1
        # resolved_at is None until resolve_signal() is called.
        assert s["resolved_at"] is None

    def test_resolve_signal_sets_resolved_at(self, logger, session_id):
        """resolve_signal() must update resolved_at to a non-null ISO 8601 timestamp."""
        entry = _signal_entry(session_id, "LOOP_SIGNAL", agent="synthesis")
        sig_id = logger.signal(entry)
        assert sig_id is not None, "signal() must return the row ID for resolve_signal()"

        logger.resolve_signal(sig_id)

        rows = logger._store.get_signals(session_id)
        resolved_at = rows[0]["resolved_at"]
        assert resolved_at is not None
        # Verify it looks like an ISO 8601 timestamp (contains 'T' separator).
        assert "T" in resolved_at

    def test_signal_returns_row_id(self, logger, session_id):
        """signal() must return an integer row ID that is usable with resolve_signal."""
        id1 = logger.signal(_signal_entry(session_id, "RETRIEVAL_SIGNAL"))
        id2 = logger.signal(_signal_entry(session_id, "COMPLETION_SIGNAL"))

        assert isinstance(id1, int)
        assert isinstance(id2, int)
        # Each call should return a different (monotonically increasing) ID.
        assert id2 > id1

    def test_payload_serialization_preserves_nested_structure(self, logger, session_id):
        payload = {
            "query": "EU AI Act Article 3",
            "filters": {"source_types": ["legislation", "official_guidance"]},
            "dimension": "definition_check",
        }
        logger.signal(_signal_entry(session_id, payload=payload))

        rows = logger._store.get_signals(session_id)
        stored = json.loads(rows[0]["payload"])
        assert stored == payload

    def test_null_dimension_stored_and_retrieved(self, logger, session_id):
        """Synthesis signals have no specific dimension."""
        entry = _signal_entry(session_id, "LOOP_SIGNAL", agent="synthesis", dimension=None)
        logger.signal(entry)

        rows = logger._store.get_signals(session_id)
        assert rows[0]["dimension"] is None


# ── get_full_session_log ──────────────────────────────────────────────────────

class TestGetFullSessionLog:
    """Verify that get_full_session_log bundles all four streams correctly."""

    def test_returns_all_four_streams_populated(self, logger, session_id):
        logger.pipeline(_pipeline_event(session_id, "SESSION_STARTED"))
        logger.reasoning(_reasoning_entry(session_id, dimension="transparency"))
        logger.state_change(_state_change(session_id))
        logger.signal(_signal_entry(session_id))

        log = logger.get_full_session_log(session_id)

        assert isinstance(log, SessionLog)
        assert log.session_id == session_id
        assert len(log.pipeline_events) == 1
        assert len(log.reasoning_entries) == 1
        assert len(log.state_changes) == 1
        assert len(log.signals) == 1

    def test_returns_typed_dataclass_instances(self, logger, session_id):
        logger.pipeline(_pipeline_event(session_id))
        logger.reasoning(_reasoning_entry(session_id))
        logger.state_change(_state_change(session_id))
        logger.signal(_signal_entry(session_id))

        log = logger.get_full_session_log(session_id)

        assert all(isinstance(e, PipelineEvent) for e in log.pipeline_events)
        assert all(isinstance(e, ReasoningEntry) for e in log.reasoning_entries)
        assert all(isinstance(e, StateChangeEntry) for e in log.state_changes)
        assert all(isinstance(e, SignalEntry) for e in log.signals)

    def test_empty_session_returns_empty_streams(self, logger):
        log = logger.get_full_session_log("nonexistent-session-xyz")

        assert log.pipeline_events == []
        assert log.reasoning_entries == []
        assert log.state_changes == []
        assert log.signals == []

    def test_returns_entries_across_multiple_writes(self, logger, session_id):
        for i in range(3):
            logger.pipeline(_pipeline_event(session_id, f"EVENT_{i}"))

        log = logger.get_full_session_log(session_id)
        assert len(log.pipeline_events) == 3


# ── get_dimension_reasoning ───────────────────────────────────────────────────

class TestGetDimensionReasoning:
    """Verify that get_dimension_reasoning correctly filters by dimension."""

    def test_filters_by_dimension(self, logger, session_id):
        logger.reasoning(_reasoning_entry(session_id, dimension="definition_check"))
        logger.reasoning(_reasoning_entry(session_id, dimension="risk_classification"))
        logger.reasoning(_reasoning_entry(session_id, dimension="definition_check"))

        results = logger.get_dimension_reasoning(session_id, "definition_check")

        assert len(results) == 2
        assert all(r.dimension == "definition_check" for r in results)

    def test_does_not_return_other_dimensions(self, logger, session_id):
        logger.reasoning(_reasoning_entry(session_id, dimension="transparency"))
        logger.reasoning(_reasoning_entry(session_id, dimension="governance"))

        results = logger.get_dimension_reasoning(session_id, "definition_check")
        assert results == []

    def test_returns_empty_for_unknown_dimension(self, logger, session_id):
        results = logger.get_dimension_reasoning(session_id, "invented_dimension")
        assert results == []

    def test_returned_entries_are_reasoning_entry_instances(self, logger, session_id):
        logger.reasoning(_reasoning_entry(session_id, dimension="roles"))

        results = logger.get_dimension_reasoning(session_id, "roles")
        assert all(isinstance(r, ReasoningEntry) for r in results)


# ── get_errors ────────────────────────────────────────────────────────────────

class TestGetErrors:
    """Verify that get_errors returns only MEMORY_WRITE_ERROR and LLM_PARSE_ERROR."""

    def test_returns_only_error_signal_types(self, logger, session_id):
        # Write signals of every type including both error types.
        for sig_type in (
            "RETRIEVAL_SIGNAL",
            "LOOP_SIGNAL",
            "MEMORY_WRITE_ERROR",
            "COMPLETION_SIGNAL",
            "LLM_PARSE_ERROR",
            "RETRIEVE_TIMEOUT",
        ):
            logger.signal(_signal_entry(session_id, sig_type))

        errors = logger.get_errors(session_id)

        assert len(errors) == 2
        error_types = {e.signal_type for e in errors}
        assert error_types == {"MEMORY_WRITE_ERROR", "LLM_PARSE_ERROR"}

    def test_returns_empty_when_no_errors(self, logger, session_id):
        logger.signal(_signal_entry(session_id, "RETRIEVAL_SIGNAL"))
        logger.signal(_signal_entry(session_id, "COMPLETION_SIGNAL"))

        assert logger.get_errors(session_id) == []

    def test_returned_entries_are_signal_entry_instances(self, logger, session_id):
        logger.signal(_signal_entry(session_id, "MEMORY_WRITE_ERROR"))

        errors = logger.get_errors(session_id)
        assert all(isinstance(e, SignalEntry) for e in errors)

    def test_error_payload_preserved(self, logger, session_id):
        payload = {"check_name": "citation_integrity", "detail": "chunk not in retrieved set"}
        logger.signal(_signal_entry(session_id, "MEMORY_WRITE_ERROR", payload=payload))

        errors = logger.get_errors(session_id)
        assert errors[0].payload == payload


# ── fallback.log ──────────────────────────────────────────────────────────────

class TestFallbackLog:
    """Verify that SQLite write failures are recorded in the plaintext fallback log."""

    def test_fallback_written_when_sqlite_write_fails(self):
        """When _conn.execute raises, _fallback() must write to fallback_log_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_path = os.path.join(tmpdir, "fallback.log")
            lg = PrismLogger(
                session_id="fallback-test",
                db_path=":memory:",
                fallback_log_path=fallback_path,
            )
            try:
                # Replace the live SQLite connection with a mock whose execute()
                # always raises.  This simulates a corrupt or closed database.
                mock_conn = MagicMock()
                mock_conn.execute.side_effect = sqlite3.OperationalError("table locked")
                lg._store._conn = mock_conn

                # Attempt a write — must silently fail and write to the fallback log.
                lg.pipeline(_pipeline_event("fallback-test", "SESSION_STARTED"))

                assert os.path.exists(fallback_path), "fallback.log must be created"
                content = Path(fallback_path).read_text(encoding="utf-8")
                assert "WRITE_ERROR" in content
                assert "write_pipeline_event" in content
                assert "OperationalError" in content
            finally:
                lg.close()

    def test_fallback_written_for_each_failing_write_method(self):
        """Every write method (not just pipeline) must route failures to fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_path = os.path.join(tmpdir, "fallback.log")
            lg = PrismLogger(
                session_id="fallback-test2",
                db_path=":memory:",
                fallback_log_path=fallback_path,
            )
            try:
                mock_conn = MagicMock()
                mock_conn.execute.side_effect = sqlite3.OperationalError("disk full")
                lg._store._conn = mock_conn

                lg.reasoning(_reasoning_entry("fallback-test2"))
                lg.state_change(_state_change("fallback-test2"))
                lg.signal(_signal_entry("fallback-test2"))

                content = Path(fallback_path).read_text(encoding="utf-8")
                assert "write_reasoning_entry" in content
                assert "write_state_change" in content
                assert "write_signal" in content
            finally:
                lg.close()


# ── ContextVar flow through reasoning decorator ───────────────────────────────

class TestReasoningDecorator:
    """Verify that session_id flows from ContextVar into the log entry."""

    def test_session_id_flows_into_reasoning_entry(self, logger, session_id):
        """The decorator reads session_id from ContextVar, not from parameters."""
        # The logger fixture set SESSION_ID_VAR to session_id in __init__.

        @log_reasoning(agent="analysis", dimension="definition_check")
        def mock_llm_call(prompt: str) -> str:
            return '{"confidence": "HIGH"}'

        mock_llm_call("What is an AI system under Article 3?")

        entries = logger.get_dimension_reasoning(session_id, "definition_check")
        assert len(entries) == 1
        assert entries[0].session_id == session_id

    def test_decorator_captures_agent_and_dimension(self, logger, session_id):
        @log_reasoning(agent="validation", dimension="risk_classification")
        def mock_call(prompt: str) -> str:
            return "response"

        mock_call("validate risk claim")

        entries = logger.get_dimension_reasoning(session_id, "risk_classification")
        assert len(entries) == 1
        assert entries[0].agent == "validation"
        assert entries[0].dimension == "risk_classification"

    def test_decorator_captures_prompt_and_response(self, logger, session_id):
        @log_reasoning(agent="synthesis", dimension=None)
        def mock_synthesis_call(prompt: str) -> str:
            return "the final synthesis response text"

        mock_synthesis_call("Produce the final report.")

        rows = logger._store.get_reasoning_entries(session_id, agent="synthesis")
        assert len(rows) == 1
        assert rows[0]["prompt_sent"] == "Produce the final report."
        assert rows[0]["llm_response"] == "the final synthesis response text"

    def test_decorator_captures_duration_ms(self, logger, session_id):
        @log_reasoning(agent="analysis", dimension="governance")
        def mock_call(prompt: str) -> str:
            return "response"

        mock_call("assess governance obligations")

        rows = logger._store.get_reasoning_entries(session_id, agent="analysis")
        assert rows[0]["duration_ms"] is not None
        assert isinstance(rows[0]["duration_ms"], int)
        assert rows[0]["duration_ms"] >= 0

    def test_decorated_function_return_value_unchanged(self, logger, session_id):
        """The decorator must be transparent — it must not alter the return value."""
        expected = '{"dimension_id": "transparency", "confidence": "MEDIUM"}'

        @log_reasoning(agent="analysis", dimension="transparency")
        def mock_call(prompt: str) -> str:
            return expected

        result = mock_call("assess transparency obligations")
        assert result == expected

    def test_decorator_is_noop_without_active_logger(self):
        """When no PrismLogger is active, the decorator runs the function silently."""
        # Override ContextVars to simulate "no active session" state.
        t1 = _LOGGER_VAR.set(None)
        t2 = SESSION_ID_VAR.set("")
        try:
            @log_reasoning(agent="analysis", dimension="test")
            def mock_call(prompt: str) -> str:
                return "bare result"

            result = mock_call("some prompt")
            # Function must still execute and return correctly.
            assert result == "bare result"
        finally:
            # Restore ContextVars to their prior state.
            _LOGGER_VAR.reset(t1)
            SESSION_ID_VAR.reset(t2)

    def test_none_dimension_stored_as_null(self, logger, session_id):
        """Synthesis/extraction calls with dimension=None must store NULL."""
        @log_reasoning(agent="synthesis", dimension=None)
        def mock_synth(prompt: str) -> str:
            return "synthesis output"

        mock_synth("synthesise")

        rows = logger._store.get_reasoning_entries(session_id, agent="synthesis")
        assert rows[0]["dimension"] is None


# ── Session isolation ─────────────────────────────────────────────────────────

class TestSessionIsolation:
    """Verify that entries from different sessions are never mixed."""

    def test_pipeline_events_isolated_by_session_id(self, logger, session_id):
        """Querying session A must not return session B's entries."""
        other_session = "other-session-999"

        # Write to the primary session via the public API.
        logger.pipeline(_pipeline_event(session_id, metadata={"session": "primary"}))

        # Write to a different session directly through the store (same DB).
        logger._store.write_pipeline_event(
            _pipeline_event(other_session, metadata={"session": "other"})
        )

        primary = logger.get_session_trace(session_id)
        other = logger.get_session_trace(other_session)

        assert len(primary) == 1
        assert primary[0].metadata["session"] == "primary"
        assert all(e.session_id == session_id for e in primary)

        assert len(other) == 1
        assert other[0].metadata["session"] == "other"
        assert all(e.session_id == other_session for e in other)

    def test_reasoning_entries_isolated_by_session_id(self, logger, session_id):
        other_session = "other-reasoning-session"

        logger.reasoning(_reasoning_entry(
            session_id,
            dimension="definition_check",
            prompt_sent="primary prompt",
            llm_response="primary response",
        ))
        logger._store.write_reasoning_entry(_reasoning_entry(
            other_session,
            dimension="definition_check",
            prompt_sent="other prompt",
            llm_response="other response",
        ))

        primary = logger.get_dimension_reasoning(session_id, "definition_check")
        other = logger.get_dimension_reasoning(other_session, "definition_check")

        assert len(primary) == 1
        assert primary[0].prompt_sent == "primary prompt"

        assert len(other) == 1
        assert other[0].prompt_sent == "other prompt"

    def test_state_history_isolated_by_session_id(self, logger, session_id):
        other_session = "other-state-session"

        logger.state_change(_state_change(session_id, section="governance"))
        logger._store.write_state_change(_state_change(other_session, section="governance"))

        primary = logger.get_state_history(session_id, "governance")
        other = logger.get_state_history(other_session, "governance")

        assert len(primary) == 1
        assert primary[0].session_id == session_id

        assert len(other) == 1
        assert other[0].session_id == other_session

    def test_errors_isolated_by_session_id(self, logger, session_id):
        other_session = "other-error-session"

        logger.signal(_signal_entry(session_id, "MEMORY_WRITE_ERROR"))
        logger._store.write_signal(_signal_entry(other_session, "MEMORY_WRITE_ERROR"))

        primary_errors = logger.get_errors(session_id)
        other_errors = logger.get_errors(other_session)

        assert len(primary_errors) == 1
        assert primary_errors[0].session_id == session_id

        assert len(other_errors) == 1
        assert other_errors[0].session_id == other_session

    def test_get_all_sessions_returns_all_distinct_ids(self, logger, session_id):
        logger.pipeline(_pipeline_event(session_id, "SESSION_STARTED"))
        logger._store.write_pipeline_event(_pipeline_event("second-session", "SESSION_STARTED"))
        logger._store.write_pipeline_event(_pipeline_event("third-session", "SESSION_STARTED"))

        sessions = logger.get_all_sessions()

        assert session_id in sessions
        assert "second-session" in sessions
        assert "third-session" in sessions
        # Exactly three distinct sessions in the DB.
        assert len(sessions) == 3


# ── utc_now() ─────────────────────────────────────────────────────────────────

class TestUtcNow:
    """Verify the timestamp helper produces a valid ISO 8601 UTC string."""

    def test_returns_string(self):
        ts = utc_now()
        assert isinstance(ts, str)

    def test_contains_utc_offset_marker(self):
        ts = utc_now()
        # ISO 8601 UTC timestamps include '+00:00' offset.
        assert "+00:00" in ts

    def test_contains_date_time_separator(self):
        ts = utc_now()
        assert "T" in ts

    def test_two_calls_produce_distinct_or_equal_timestamps(self):
        t1 = utc_now()
        t2 = utc_now()
        # t2 must be at or after t1 (monotonic).
        assert t2 >= t1
