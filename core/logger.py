"""
core/logger.py
───────────────
PrismLogger: the single public interface for all Prism logging operations.

Architecture position
─────────────────────
  orchestrator.py → logger.py → log_store.py → SQLite
  agent files     → log_reasoning decorator (defined here) → logger.py
  memory proxies  → logger.py (state_change writes)

Nothing outside core/ imports from log_store.py directly.
Agents never import from logger.py — their LLM calls are wrapped transparently
by the log_reasoning decorator, and session_id flows via a ContextVar, not as
a function parameter.

Key design decisions
────────────────────
  1. ContextVar for session_id: set once by the orchestrator at session start;
     read automatically by the decorator and (optionally) by memory proxies.
     Agents never handle session_id.

  2. ContextVar for logger reference: the decorator needs to call logger.reasoning()
     without being handed the logger explicitly.  Storing the active logger in a
     ContextVar makes the decorator a zero-configuration wrapper.

  3. signal() returns the row ID: the orchestrator receives the ID so it can
     call resolve_signal(id) after handling the signal — completing the lifecycle.

  4. Row → dataclass conversion is PrismLogger's responsibility, not LogStore's.
     LogStore returns plain dicts; PrismLogger converts them to typed objects.
"""

from __future__ import annotations

import json
import time
from contextvars import ContextVar, Token
from functools import wraps
from pathlib import Path
from typing import Optional

from core.log_schema import (
    PipelineEvent,
    ReasoningEntry,
    SessionLog,
    SignalEntry,
    StateChangeEntry,
    utc_now,
)
from core.log_store import LogStore


# ── Session-scoped ContextVars ────────────────────────────────────────────────
#
# Both vars are set by PrismLogger.__init__() and reset by PrismLogger.close().
# The decorator and (optionally) memory proxy write methods read from them.
# Default values ("" / None) make the decorator a safe no-op when no logger
# is active — useful in tests that run agent logic without a PrismLogger.

SESSION_ID_VAR: ContextVar[str] = ContextVar("prism_session_id", default="")
# _LOGGER_VAR is private; external callers use PrismLogger directly.
_LOGGER_VAR: ContextVar[Optional["PrismLogger"]] = ContextVar("prism_logger", default=None)


# ── Reasoning decorator ───────────────────────────────────────────────────────

def log_reasoning(agent: str, dimension: Optional[str] = None):
    """Decorator factory that wraps an agent's LLM call to capture reasoning.

    Records the full prompt, raw response, and wall-clock duration as a
    ReasoningEntry without any changes to the wrapped function's signature
    or return value.

    The decorator reads session_id and the active PrismLogger from ContextVars,
    so agents do not need to receive or pass either.  If no logger is active
    (e.g. when running agent unit tests without a PrismLogger), the decorator
    is a transparent no-op — the wrapped function executes normally.

    Parameters
    ----------
    agent : str
        Agent name tag for the log entry: "analysis" | "validation" |
        "synthesis" | "extraction".
    dimension : str | None
        EU AI Act dimension being assessed; None for synthesis/extraction calls.

    Usage
    -----
    @log_reasoning(agent="analysis", dimension="risk_classification")
    def _call_llm(prompt: str, system: str, client) -> str:
        return client.messages.create(...).content[0].text
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(prompt: str, *args, **kwargs):
            # Read ContextVars at call time (not decoration time) so they
            # reflect the session that is active when the function is called.
            active_logger: Optional[PrismLogger] = _LOGGER_VAR.get()
            session_id: str = SESSION_ID_VAR.get()

            start = time.time()
            raw_response = fn(prompt, *args, **kwargs)
            duration_ms = int((time.time() - start) * 1000)

            # Only log when a PrismLogger has been activated for this session.
            # This keeps agent unit tests clean — they can call _call_llm
            # directly without setting up a logging context.
            if active_logger is not None and session_id:
                entry = ReasoningEntry(
                    session_id=session_id,
                    agent=agent,
                    dimension=dimension,
                    prompt_sent=prompt,
                    llm_response=raw_response,
                    duration_ms=duration_ms,
                    # parse_succeeded defaults True here; if parsing subsequently
                    # fails the orchestrator logs a separate LLM_PARSE_ERROR signal.
                )
                active_logger.reasoning(entry)

            return raw_response

        return wrapper
    return decorator


# ── PrismLogger ───────────────────────────────────────────────────────────────

class PrismLogger:
    """Single public interface for all Prism logging write and query operations.

    The orchestrator creates exactly one PrismLogger per session.  All other
    code that needs logging functionality imports from logger.py — never from
    log_store.py or log_schema.py directly (those are internal).

    Instantiating PrismLogger sets the two module-level ContextVars so the
    log_reasoning decorator and memory proxy write methods can access the
    session_id and logger reference without receiving them as parameters.
    Call close() at session end to release the database connection and reset
    the ContextVars (important for test isolation).

    Parameters
    ----------
    session_id : str
        UUID generated by the orchestrator before creating the logger.
    db_path : str
        SQLite database file path, or ":memory:" for in-process testing.
    fallback_log_path : str | None
        Explicit path for the plaintext fallback error log.  When None:
          - Real DB: defaults to <db_dir>/fallback.log.
          - :memory:: falls back to Python's logging system (no file I/O).
    """

    def __init__(
        self,
        session_id: str,
        db_path: str,
        fallback_log_path: Optional[str] = None,
    ) -> None:
        self.session_id = session_id

        # Resolve default fallback path for real database files.
        if fallback_log_path is None and db_path != ":memory:":
            fallback_log_path = str(Path(db_path).parent / "fallback.log")

        self._store = LogStore(db_path, fallback_log_path)
        self._store.initialize()

        # Set ContextVars and store the tokens for later reset in close().
        # Storing tokens (not just calling set()) is required to correctly
        # undo the set when multiple nested loggers exist in tests.
        self._session_token: Token = SESSION_ID_VAR.set(session_id)
        self._logger_token: Token = _LOGGER_VAR.set(self)

    def close(self) -> None:
        """Close the database connection and reset session ContextVars.

        Must be called at session end to prevent ContextVar leakage in tests
        and to release the SQLite connection.
        """
        self._store.close()
        SESSION_ID_VAR.reset(self._session_token)
        _LOGGER_VAR.reset(self._logger_token)

    # ── Write methods ─────────────────────────────────────────────────────────
    # Called by the orchestrator (pipeline, state_change, signal) and the
    # log_reasoning decorator (reasoning).  Agents never call these directly.

    def pipeline(self, event: PipelineEvent) -> None:
        """Record a pipeline lifecycle event (orchestrator-only)."""
        self._store.write_pipeline_event(event)

    def reasoning(self, entry: ReasoningEntry) -> None:
        """Record an LLM call reasoning trace (called by log_reasoning decorator)."""
        self._store.write_reasoning_entry(entry)

    def state_change(self, entry: StateChangeEntry) -> None:
        """Record a memory section write attempt (called by proxy write methods)."""
        self._store.write_state_change(entry)

    def signal(self, entry: SignalEntry) -> Optional[int]:
        """Record a signal or error entry.

        Returns the database row ID so the orchestrator can call
        resolve_signal(id) after handling the signal.  Returns None if the
        write fails (the failure is recorded in the fallback log).
        """
        return self._store.write_signal(entry)

    def resolve_signal(self, signal_id: int) -> None:
        """Close a signal's lifecycle by recording when it was resolved."""
        self._store.resolve_signal(signal_id, utc_now())

    # ── Query methods ─────────────────────────────────────────────────────────
    # Called by the report exporter and UI layer.  Each method converts raw
    # store dicts back to typed dataclasses before returning.

    def get_session_trace(self, session_id: str) -> list[PipelineEvent]:
        """Return all pipeline events for *session_id* in insertion order."""
        rows = self._store.get_pipeline_events(session_id)
        return [self._to_pipeline_event(r) for r in rows]

    def get_dimension_reasoning(
        self,
        session_id: str,
        dimension: str,
    ) -> list[ReasoningEntry]:
        """Return all reasoning entries for a specific EU AI Act dimension."""
        rows = self._store.get_reasoning_entries(session_id, dimension=dimension)
        return [self._to_reasoning_entry(r) for r in rows]

    def get_state_history(
        self,
        session_id: str,
        section: str,
    ) -> list[StateChangeEntry]:
        """Return all write attempts (validated and failed) for a memory section."""
        rows = self._store.get_state_history(session_id, section=section)
        return [self._to_state_change(r) for r in rows]

    def get_errors(self, session_id: str) -> list[SignalEntry]:
        """Return all MEMORY_WRITE_ERROR and LLM_PARSE_ERROR signals for a session."""
        rows = self._store.get_errors(session_id)
        return [self._to_signal(r) for r in rows]

    def get_full_session_log(self, session_id: str) -> SessionLog:
        """Return all four log streams for a session as a SessionLog.

        Used by the report exporter and judge-facing transparency outputs.
        All four queries are executed and bundled; the result is suitable
        for direct serialization to JSON for report section 10.
        """
        return SessionLog(
            session_id=session_id,
            pipeline_events=[
                self._to_pipeline_event(r)
                for r in self._store.get_pipeline_events(session_id)
            ],
            reasoning_entries=[
                self._to_reasoning_entry(r)
                for r in self._store.get_reasoning_entries(session_id)
            ],
            state_changes=[
                self._to_state_change(r)
                for r in self._store.get_state_history(session_id)
            ],
            signals=[
                self._to_signal(r)
                for r in self._store.get_signals(session_id)
            ],
        )

    def get_all_sessions(self) -> list[str]:
        """Return all distinct session_ids that have at least one pipeline event."""
        return self._store.get_all_sessions()

    # ── Row → dataclass converters ────────────────────────────────────────────
    # Static methods: no instance state needed.  JSON columns are decoded back
    # to Python objects; None SQL values are converted to appropriate defaults.

    @staticmethod
    def _to_pipeline_event(row: dict) -> PipelineEvent:
        return PipelineEvent(
            session_id=row["session_id"],
            event_type=row["event_type"],
            agent=row["agent"],
            # metadata is always stored as JSON; parse it back to a dict.
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            duration_ms=row["duration_ms"],
            timestamp=row["timestamp"],
        )

    @staticmethod
    def _to_reasoning_entry(row: dict) -> ReasoningEntry:
        return ReasoningEntry(
            session_id=row["session_id"],
            agent=row["agent"],
            prompt_sent=row["prompt_sent"],
            llm_response=row["llm_response"],
            dimension=row["dimension"],
            # parsed_output stored as JSON string or NULL; decode to dict or None.
            parsed_output=json.loads(row["parsed_output"]) if row["parsed_output"] else None,
            # SQLite stores booleans as 0/1; convert back to Python bool.
            parse_succeeded=bool(row["parse_succeeded"]),
            confidence=row["confidence"],
            claims_count=row["claims_count"] or 0,
            weak_claims_count=row["weak_claims_count"] or 0,
            chunk_ids_used=json.loads(row["chunk_ids_used"]) if row["chunk_ids_used"] else [],
            duration_ms=row["duration_ms"],
            timestamp=row["timestamp"],
        )

    @staticmethod
    def _to_state_change(row: dict) -> StateChangeEntry:
        return StateChangeEntry(
            session_id=row["session_id"],
            section=row["section"],
            agent=row["agent"],
            new_state=json.loads(row["new_state"]),
            previous_state=json.loads(row["previous_state"]) if row["previous_state"] else None,
            write_validated=bool(row["write_validated"]),
            # validation_errors stored as JSON or NULL; decode to list or [].
            validation_errors=json.loads(row["validation_errors"]) if row["validation_errors"] else [],
            timestamp=row["timestamp"],
        )

    @staticmethod
    def _to_signal(row: dict) -> SignalEntry:
        return SignalEntry(
            session_id=row["session_id"],
            signal_type=row["signal_type"],
            agent=row["agent"],
            payload=json.loads(row["payload"]),
            resolution=row["resolution"],
            dimension=row["dimension"],
            retry_count=row["retry_count"] or 0,
            resolved_at=row["resolved_at"],
            timestamp=row["timestamp"],
        )
