"""
core/orchestrator.py
─────────────────────
The orchestrator owns the full pipeline shape.  It is the only file that
knows the sequence of agents, manages checkpoints and rollbacks, handles all
RetrievalSignals, drives retry logic, and caches retrieval results.

This file contains NO prompts, NO legal reasoning, NO confidence scoring,
and NO output parsing.  All intelligence lives in the agent files.

Pipeline sequence
─────────────────
  1.  Initialise SessionMemory
  2.  Invoke extraction agent → save checkpoint "after_extraction"
  3.  Initial broad retrieve()  → populate retrieved_chunk_ids
  4.  Invoke analysis agent (signal loop, max 2 retries / dimension)
      → save checkpoint "after_analysis"
  5.  Invoke validation agent (signal loop, max 2 retries / claim)
      → save checkpoint "after_validation"
  6.  Invoke synthesis agent
      → LoopSignal  → rollback to "after_analysis", re-run analysis + validation
                       (max 1 global loop)
      → CompletionSignal → save checkpoint "after_synthesis"
  7.  Return memory.final_report to caller
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from typing import Any, Callable, Optional

from core.types import (
    Chunk,
    CompletionSignal,
    LoopSignal,
    MemoryWriteError,
    OrchestratorState,
    ReportSection,
    RetrievalSignal,
)
from core.memory import (
    AnalysisAgentMemoryView,
    ExtractionAgentMemoryView,
    SessionMemory,
    SynthesisAgentMemoryView,
    ValidationAgentMemoryView,
)
from agents.analysis_agent   import run_analysis_agent
from agents.validation_agent import run_validation_agent
from agents.synthesis_agent  import run_synthesis_agent

logger = logging.getLogger(__name__)

# Maximum retrieval retries per dimension (analysis) or claim (validation).
MAX_RETRIEVAL_RETRIES = 2
# Maximum global analysis+validation+synthesis loops.
MAX_LOOP_COUNT = 1


class Orchestrator:
    """Drives the full Prism compliance-analysis pipeline.

    Parameters
    ----------
    retrieve_fn : Callable[[str, dict], list[Chunk]]
        The RAG retrieval function.  Treated as a black box.
    extraction_fn : Callable[[str], FactSection] | None
        The extraction agent entry-point.  If None, the orchestrator expects
        FactSection to be injected directly via `run(facts=...)`.
    llm_client : optional
        Anthropic client.  None → agents create their own real clients.
        Pass a mock in tests.
    """

    def __init__(
        self,
        retrieve_fn: Callable[[str, dict], list[Chunk]],
        extraction_fn: Optional[Callable] = None,
        llm_client: Any = None,
    ) -> None:
        self._retrieve_fn    = retrieve_fn
        self._extraction_fn  = extraction_fn
        self._llm_client     = llm_client

        self._memory: SessionMemory = SessionMemory()
        # Convenience alias — orchestrator accesses _orchestrator directly.
        self._state: OrchestratorState = self._memory._orchestrator

    # ── Public entry point ──────────────────────────────────────────────────

    def run(
        self,
        document_text: str = "",
        facts=None,  # FactSection | None — inject pre-extracted facts for testing
    ) -> ReportSection:
        """Execute the full pipeline and return the final compliance report.

        Parameters
        ----------
        document_text : str
            Raw text of the uploaded use-case document.  Passed to the
            extraction agent if one is configured.
        facts : FactSection | None
            If provided, skips the extraction phase and uses these facts
            directly.  Useful for testing and for external callers that
            already hold a FactSection.
        """
        # ── Step 1: extraction ──────────────────────────────────────────────
        if facts is not None:
            # Inject pre-extracted facts directly (bypass extraction proxy).
            # The orchestrator registers no doc chunk IDs, so source_chunk_ids
            # must be empty or pre-registered.
            self._memory.facts = copy.deepcopy(facts)
        elif self._extraction_fn is not None:
            self._run_extraction_phase(document_text)
        else:
            raise ValueError(
                "Either pass `facts=` directly or configure `extraction_fn`."
            )

        self._save_checkpoint("after_extraction")
        logger.info("Checkpoint saved: after_extraction")

        # ── Step 2: initial broad retrieval ─────────────────────────────────
        f = self._memory.facts
        initial_query = f"{f.use_case_name} {f.description}"[:300]
        self._retrieve_and_cache(initial_query, {})
        logger.info("Initial broad retrieval complete.")

        # ── Steps 3–5: analysis / validation / synthesis (with loop) ────────
        loop_context: str = ""
        for loop_iteration in range(MAX_LOOP_COUNT + 1):
            self._state.loop_count = loop_iteration

            self._run_analysis_phase(refined_context=loop_context)
            self._save_checkpoint("after_analysis")
            logger.info("Checkpoint saved: after_analysis (loop %d)", loop_iteration)

            self._run_validation_phase()
            self._save_checkpoint("after_validation")
            logger.info("Checkpoint saved: after_validation (loop %d)", loop_iteration)

            loop_signal = self._run_synthesis_phase()

            if loop_signal is None:
                # CompletionSignal — we are done.
                self._save_checkpoint("after_synthesis")
                logger.info("Checkpoint saved: after_synthesis")
                break

            # LoopSignal — roll back and re-run with refined context.
            if loop_iteration >= MAX_LOOP_COUNT:
                logger.warning(
                    "Max loops reached; proceeding with INSUFFICIENT markers."
                )
                # Force synthesis to complete on next attempt regardless.
                # The context flag is read by the synthesis agent via context dict.
                self._force_synthesis_complete()
                self._save_checkpoint("after_synthesis")
                break

            logger.info(
                "LoopSignal received: %s — rolling back to after_analysis.",
                loop_signal.reason,
            )
            loop_context = loop_signal.refined_context
            self._restore_checkpoint("after_analysis")
            # Clear analysis + validation sections so the agents can re-write.
            self._reset_analysis_and_validation()

        return self._memory.final_report

    # ── Extraction phase ─────────────────────────────────────────────────────

    def _run_extraction_phase(self, document_text: str) -> None:
        """Invoke the extraction function and write its output to memory.

        The extraction agent is external — its output (FactSection) is
        written directly to memory without going through a proxy, because
        extraction happens before retrieval so retrieved_chunk_ids is empty.
        """
        facts = self._extraction_fn(document_text)
        self._memory.facts = facts   # direct write; schema check omitted here
        # If the extraction function returned source_chunk_ids, register them.
        if facts.source_chunk_ids:
            for cid in facts.source_chunk_ids:
                self._state.retrieved_chunk_ids.add(cid)

    # ── Analysis phase ───────────────────────────────────────────────────────

    def _run_analysis_phase(self, refined_context: str = "") -> None:
        """Drive the analysis agent through its signal loop.

        Handles:
          - RetrievalSignal → cache-check → retrieve → re-invoke
          - MemoryWriteError → log and rollback if a critical section failed
          - retry_counts per dimension (max MAX_RETRIEVAL_RETRIES)
        """
        context: dict = {
            "max_retrievals_reached": set(),
            "refined_context": refined_context,
        }
        # Clear per-dimension retry counts from any previous loop.
        for key in list(self._state.retry_counts.keys()):
            if key.startswith("analysis_"):
                del self._state.retry_counts[key]

        while True:
            view = self._make_analysis_view()
            try:
                signal = run_analysis_agent(
                    view, self._get_all_chunks(), context, self._llm_client
                )
            except MemoryWriteError as exc:
                self._handle_memory_write_error(exc, "after_extraction", "analysis")
                break

            if isinstance(signal, CompletionSignal):
                break
            if isinstance(signal, RetrievalSignal):
                retry_key = f"analysis_{signal.dimension}"
                count = self._state.retry_counts.get(retry_key, 0)
                if count >= MAX_RETRIEVAL_RETRIES:
                    logger.warning(
                        "Max retrieval retries for analysis dimension '%s'.",
                        signal.dimension,
                    )
                    context["max_retrievals_reached"].add(signal.dimension)
                    # Continue loop — agent will proceed with INSUFFICIENT.
                else:
                    self._retrieve_and_cache(signal.query, signal.filters)
                    self._state.retry_counts[retry_key] = count + 1

    # ── Validation phase ─────────────────────────────────────────────────────

    def _run_validation_phase(self) -> None:
        """Drive the validation agent through its signal loop."""
        context: dict = {
            "processed_claim_ids":    set(),
            "retry_counts":           {},
            "max_retrievals_reached": set(),
            "overturned_claims":      [],
            "flags":                  [],
            "unresolved_ids":         [],
        }

        while True:
            view = self._make_validation_view()
            try:
                signal = run_validation_agent(
                    view, self._get_all_chunks(), context, self._llm_client
                )
            except MemoryWriteError as exc:
                self._handle_memory_write_error(exc, "after_analysis", "validation")
                break

            if isinstance(signal, CompletionSignal):
                break
            if isinstance(signal, RetrievalSignal):
                claim_id = signal.dimension  # per spec, claim_id used as dimension
                retry_key = f"validation_{claim_id}"
                count = self._state.retry_counts.get(retry_key, 0)
                if count >= MAX_RETRIEVAL_RETRIES:
                    logger.warning(
                        "Max retrieval retries for validation claim '%s'.", claim_id
                    )
                    context["max_retrievals_reached"].add(claim_id)
                    # Update per-claim retry in context too.
                    context["retry_counts"][claim_id] = MAX_RETRIEVAL_RETRIES
                else:
                    self._retrieve_and_cache(signal.query, signal.filters)
                    self._state.retry_counts[retry_key] = count + 1
                    context["retry_counts"][claim_id] = count + 1

    # ── Synthesis phase ──────────────────────────────────────────────────────

    def _run_synthesis_phase(self) -> Optional[LoopSignal]:
        """Invoke the synthesis agent once.

        Returns None (CompletionSignal received) or the LoopSignal so the
        caller can decide whether to roll back.
        """
        context: dict = {"loop_count": self._state.loop_count}
        view = self._make_synthesis_view()
        try:
            signal = run_synthesis_agent(
                view, self._get_all_chunks(), context, self._llm_client
            )
        except MemoryWriteError as exc:
            self._handle_memory_write_error(exc, "after_validation", "synthesis")
            return None

        if isinstance(signal, CompletionSignal):
            # Quality-regression safety net.
            if self._check_quality_regression():
                logger.warning(
                    "Quality regression detected after synthesis; "
                    "proceeding anyway (loop limit)."
                )
            return None

        if isinstance(signal, LoopSignal):
            return signal

        raise ValueError(f"Unexpected signal from synthesis agent: {type(signal)}")

    def _force_synthesis_complete(self) -> None:
        """Run synthesis one final time ignoring the loop condition.

        Called when loop_count has reached MAX_LOOP_COUNT so that we always
        produce a final report (potentially with INSUFFICIENT markers).
        """
        # Increment loop_count beyond max so determine_loop_condition returns False.
        self._state.loop_count = MAX_LOOP_COUNT + 1
        context: dict = {"loop_count": self._state.loop_count}
        view = self._make_synthesis_view()
        try:
            run_synthesis_agent(view, self._get_all_chunks(), context, self._llm_client)
        except MemoryWriteError as exc:
            logger.error("MemoryWriteError in forced synthesis: %s", exc)

    # ── Checkpointing ────────────────────────────────────────────────────────

    def _save_checkpoint(self, name: str) -> None:
        """Deep-copy the data sections of SessionMemory as a named checkpoint.

        Checkpoints exclude _orchestrator.checkpoints itself to prevent
        recursive deep copies.
        """
        snapshot = copy.deepcopy(self._memory)
        # Clear nested checkpoints in the snapshot to avoid infinite nesting.
        snapshot._orchestrator.checkpoints = {}
        self._state.checkpoints[name] = snapshot
        logger.debug("Checkpoint '%s' saved.", name)

    def _restore_checkpoint(self, name: str) -> None:
        """Restore SessionMemory data sections from a named checkpoint.

        Preserves the orchestrator's infrastructure (cache, chunk_ids, counts,
        checkpoints) across rollbacks so we don't re-retrieve already-fetched
        chunks.
        """
        snapshot = self._state.checkpoints.get(name)
        if snapshot is None:
            raise KeyError(f"No checkpoint named '{name}'")

        # Restore only data sections.
        self._memory.facts                = copy.deepcopy(snapshot.facts)
        self._memory.risk_classification  = copy.deepcopy(snapshot.risk_classification)
        self._memory.definition_check     = copy.deepcopy(snapshot.definition_check)
        self._memory.prohibited_practices = copy.deepcopy(snapshot.prohibited_practices)
        self._memory.transparency         = copy.deepcopy(snapshot.transparency)
        self._memory.roles                = copy.deepcopy(snapshot.roles)
        self._memory.governance           = copy.deepcopy(snapshot.governance)
        self._memory.validation_flags     = copy.deepcopy(snapshot.validation_flags)
        self._memory.weak_claims          = copy.deepcopy(snapshot.weak_claims)
        self._memory.overturned_claims    = copy.deepcopy(snapshot.overturned_claims)
        self._memory.final_report         = copy.deepcopy(snapshot.final_report)
        self._memory.follow_up_questions  = copy.deepcopy(snapshot.follow_up_questions)
        self._memory.confidence_summary   = copy.deepcopy(snapshot.confidence_summary)
        logger.debug("Checkpoint '%s' restored.", name)

    def _reset_analysis_and_validation(self) -> None:
        """Clear analysis + validation sections so agents can rewrite them."""
        self._memory.risk_classification  = None
        self._memory.definition_check     = None
        self._memory.prohibited_practices = None
        self._memory.transparency         = None
        self._memory.roles                = None
        self._memory.governance           = None
        self._memory.validation_flags     = None
        self._memory.weak_claims          = []
        self._memory.overturned_claims    = []

    # ── Retrieval + KV cache ─────────────────────────────────────────────────

    def _retrieve_and_cache(self, query: str, filters: dict) -> list[Chunk]:
        """Cache-check → retrieve → extend chunk pool.

        The cache key is a deterministic hash of (query, sorted filters).
        Returns cached result without calling retrieve() if present.
        """
        cache_key = _make_cache_key(query, filters)
        if cache_key in self._state.retrieval_cache:
            logger.debug("Cache hit for query: %s", query[:60])
            return self._state.retrieval_cache[cache_key]

        chunks = self._retrieve_fn(query, filters)
        self._state.retrieval_cache[cache_key] = chunks
        for chunk in chunks:
            self._state.retrieved_chunk_ids.add(chunk.chunk_id)
        logger.debug(
            "Retrieved %d chunk(s) for query: %s", len(chunks), query[:60]
        )
        return chunks

    # ── Proxy factory methods ────────────────────────────────────────────────

    def _make_analysis_view(self) -> AnalysisAgentMemoryView:
        return AnalysisAgentMemoryView(
            self._memory,
            self._state.retrieved_chunk_ids,
            self._build_chunk_lookup(),
        )

    def _make_validation_view(self) -> ValidationAgentMemoryView:
        return ValidationAgentMemoryView(
            self._memory,
            self._state.retrieved_chunk_ids,
            self._build_chunk_lookup(),
        )

    def _make_synthesis_view(self) -> SynthesisAgentMemoryView:
        return SynthesisAgentMemoryView(
            self._memory,
            self._state.retrieved_chunk_ids,
            self._build_chunk_lookup(),
        )

    # ── Chunk pool helpers ───────────────────────────────────────────────────

    def _build_chunk_lookup(self) -> dict[str, Chunk]:
        """Build {chunk_id: Chunk} from all cached retrieval results."""
        lookup: dict[str, Chunk] = {}
        for chunks in self._state.retrieval_cache.values():
            for chunk in chunks:
                lookup[chunk.chunk_id] = chunk
        return lookup

    def _get_all_chunks(self) -> list[Chunk]:
        """Return a flat deduplicated list of all retrieved chunks."""
        seen: set[str] = set()
        result: list[Chunk] = []
        for chunks in self._state.retrieval_cache.values():
            for chunk in chunks:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    result.append(chunk)
        return result

    # ── Quality regression check ─────────────────────────────────────────────

    def _check_quality_regression(self) -> bool:
        """Return True when synthesis confidence is suspiciously low.

        Triggers when BOTH definition_check AND risk_classification are
        INSUFFICIENT in the written confidence_summary, indicating the
        synthesis agent could not make the two most fundamental findings.
        """
        cs = self._memory.confidence_summary
        if cs is None:
            return False
        from core.types import Confidence
        return (
            cs.definition_check == Confidence.INSUFFICIENT
            and cs.risk_classification == Confidence.INSUFFICIENT
        )

    # ── Error handling ───────────────────────────────────────────────────────

    def _handle_memory_write_error(
        self,
        error: MemoryWriteError,
        fallback_checkpoint: str,
        agent_name: str,
    ) -> None:
        """Log and attempt rollback on MemoryWriteError."""
        logger.error(
            "MemoryWriteError in %s agent [%s]: %s",
            agent_name,
            error.check_name,
            error.detail,
        )
        try:
            self._restore_checkpoint(fallback_checkpoint)
            logger.warning("Rolled back to checkpoint '%s'.", fallback_checkpoint)
        except KeyError:
            logger.warning(
                "No checkpoint '%s' available for rollback.", fallback_checkpoint
            )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_cache_key(query: str, filters: dict) -> str:
    """Deterministic hash of (query, sorted filters) for the retrieval cache."""
    payload = query + json.dumps(sorted(filters.items()), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
