"""
core/log_schema.py
───────────────────
Typed dataclasses for every Prism log entry.

Pure type definitions only — no storage logic, no agent knowledge, no imports
beyond the standard library.  Every log entry is JSON-serializable and maps
directly to one row in one of the four SQLite tables.

The four entry types correspond to the four tables:
  PipelineEvent    → pipeline_events    (orchestrator lifecycle events)
  ReasoningEntry   → reasoning_entries  (LLM calls, captured by decorator)
  StateChangeEntry → state_changes      (memory proxy write attempts)
  SignalEntry      → signals            (inter-agent signals and errors)

SessionLog bundles all four streams for a session into a single object used
by the report exporter and judge-facing transparency output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ── Timestamp helper ──────────────────────────────────────────────────────────

def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Used as default_factory on timestamp fields so the timestamp is set at
    object-creation time, not at class-definition time.  This ensures each
    entry carries the moment it was constructed, not a stale module-load time.
    """
    return datetime.now(timezone.utc).isoformat()


# ── Log entry dataclasses ─────────────────────────────────────────────────────

@dataclass
class PipelineEvent:
    """Structured log entry for a pipeline lifecycle event.

    Written by the orchestrator at session start/end, agent boundaries,
    checkpoints, retrieval calls, and signal receipt.  Never written by agents.

    event_type values (from spec):
      SESSION_STARTED, SESSION_COMPLETED
      AGENT_STARTED, AGENT_COMPLETED
      CHECKPOINT_SAVED, CHECKPOINT_RESTORED
      RETRIEVE_CALLED, CACHE_HIT, CACHE_MISS
      LOOP_TRIGGERED, ROLLBACK_TRIGGERED
    """
    session_id:  str                        # UUID threaded through from orchestrator
    event_type:  str                        # one of the SESSION_* / AGENT_* / ... constants
    agent:       Optional[str]  = None      # None for session-level events (start, complete)
    metadata:    dict           = field(default_factory=dict)   # event-specific context
    duration_ms: Optional[int]  = None      # populated on AGENT_COMPLETED, SESSION_COMPLETED
    timestamp:   str            = field(default_factory=utc_now)


@dataclass
class ReasoningEntry:
    """Full record of a single LLM call: prompt, raw response, and parse outcome.

    Written by the log_reasoning decorator, which wraps each agent's _call_llm
    function.  Agents do not call logger.reasoning() directly — the decorator
    is the sole writer of this table.

    parse_succeeded and confidence are set optimistically to True / None by the
    decorator.  If parsing fails, the orchestrator writes a MEMORY_WRITE_ERROR
    or LLM_PARSE_ERROR signal rather than updating this entry in place.
    """
    session_id:        str             # from SESSION_ID_VAR ContextVar
    agent:             str             # "analysis" | "validation" | "synthesis" | "extraction"
    prompt_sent:       str             # full prompt text sent to the LLM
    llm_response:      str             # raw text before any parsing
    dimension:         Optional[str]  = None   # EU AI Act dimension; None for synthesis/extraction
    parsed_output:     Optional[dict] = None   # JSON-serializable parsed finding; None on parse failure
    parse_succeeded:   bool           = True   # False when parse_dimension_response falls back to INSUFFICIENT
    confidence:        Optional[str]  = None   # HIGH | MEDIUM | LOW | INSUFFICIENT; None if not yet parsed
    claims_count:      int            = 0      # number of claims in the parsed finding
    weak_claims_count: int            = 0      # claims flagged for validation
    chunk_ids_used:    list[str]      = field(default_factory=list)  # chunk_ids cited in the finding
    duration_ms:       Optional[int]  = None   # wall-clock time of the LLM call
    timestamp:         str            = field(default_factory=utc_now)


@dataclass
class StateChangeEntry:
    """Record of a single memory section write attempt by an agent proxy.

    Written by proxy write methods on BOTH successful and failed writes so
    the full history of write attempts is captured.  On failure:
      write_validated = False
      validation_errors = [...check_name + detail strings...]
    The MemoryWriteError is then re-raised to the orchestrator for handling.

    previous_state is None on the first write to a section (nothing existed before).
    """
    session_id:        str             # from SESSION_ID_VAR ContextVar
    section:           str             # memory section name: "definition_check", "final_report", etc.
    agent:             str             # which agent performed the write attempt
    new_state:         dict            # JSON-serializable representation of the attempted write value
    previous_state:    Optional[dict] = None   # None on first write to this section
    write_validated:   bool           = True   # False when any validation check failed
    validation_errors: list[str]      = field(default_factory=list)  # populated when write_validated=False
    timestamp:         str            = field(default_factory=utc_now)


@dataclass
class SignalEntry:
    """Record of a signal emitted by an agent or an error raised by the pipeline.

    Covers all inter-agent signals (RetrievalSignal, LoopSignal, CompletionSignal)
    and error conditions (MemoryWriteError → MEMORY_WRITE_ERROR,
    parse failures → LLM_PARSE_ERROR, timeouts → RETRIEVE_TIMEOUT).

    resolved_at is None until the orchestrator calls logger.resolve_signal(signal_id).
    Setting resolved_at closes the signal's lifecycle and enables latency analysis.

    signal_type values (from spec):
      RETRIEVAL_SIGNAL, LOOP_SIGNAL, COMPLETION_SIGNAL
      MEMORY_WRITE_ERROR, LLM_PARSE_ERROR, RETRIEVE_TIMEOUT
    """
    session_id:   str             # from SESSION_ID_VAR ContextVar
    signal_type:  str             # one of the signal_type constants above
    agent:        str             # agent that emitted the signal or triggered the error
    payload:      dict            # JSON-serializable signal content (query, filters, reason, etc.)
    resolution:   str             # what the orchestrator did in response
    dimension:    Optional[str]  = None   # dimension_id for analysis signals; claim_id for validation
    retry_count:  int            = 0      # how many prior retrieval attempts for this dimension/claim
    resolved_at:  Optional[str]  = None  # ISO 8601 UTC; set by resolve_signal()
    timestamp:    str            = field(default_factory=utc_now)


# ── Session bundle ────────────────────────────────────────────────────────────

@dataclass
class SessionLog:
    """Complete log record for a single analysis session.

    Bundles all four log streams in insertion order.  Used by the report
    exporter (section 10 citations) and judge-facing transparency outputs.
    Constructed by PrismLogger.get_full_session_log().
    """
    session_id:        str
    pipeline_events:   list[PipelineEvent]
    reasoning_entries: list[ReasoningEntry]
    state_changes:     list[StateChangeEntry]
    signals:           list[SignalEntry]
