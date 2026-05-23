"""
agents/synthesis_agent.py
──────────────────────────
Synthesis agent: merges analysis and validation outputs into the ten-section
final compliance report.

══════════════════════════════════════════════════════════════════════════════
INTELLIGENCE — pure functions, no I/O, fully testable without API calls
══════════════════════════════════════════════════════════════════════════════
  merge_analysis_with_validation(memory) -> dict
  build_synthesis_prompt(facts, merged, chunks, loop_count) -> str
  parse_synthesis_response(response_text) -> (ReportSection, FollowUpSection, ConfidenceSection)
  determine_loop_condition(confidence_section, loop_count) -> bool
  _extract_json(text) -> dict
  _serialise_merged(merged_analysis) -> str
  _serialise_validation(memory) -> str
  _serialise_weak_claims(weak_claims) -> str
  _serialise_chunks_by_source(chunks) -> str

══════════════════════════════════════════════════════════════════════════════
ORCHESTRATION — coordinates calls, manages state, emits signals
══════════════════════════════════════════════════════════════════════════════
  _call_llm(prompt, system, client) -> str
  run_synthesis_agent(memory, chunks, context, llm_client) -> Signal
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Optional, Union

from core.types import (
    Chunk,
    ClaimStatus,
    CompletionSignal,
    Confidence,
    ConfidenceSection,
    DimensionFinding,
    FactSection,
    FollowUpSection,
    Label,
    LoopSignal,
    OverturnedClaim,
    ReportSection,
    ValidationSection,
    WeakClaim,
)
from core.memory import SynthesisAgentMemoryView
from prompts.synthesis_prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_PROMPT_TEMPLATE,
)
from agents.analysis_agent import _serialise_facts, _parse_confidence


# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def merge_analysis_with_validation(
    memory: SynthesisAgentMemoryView,
) -> dict[str, Any]:
    """Build a merged view of the analysis findings incorporating validation.

    Merge rules:
      - OVERTURNED claims → replace original claim text/confidence/label with
        the validation finding and mark as UNCERTAIN in relevant dimensions.
      - Unresolved claim IDs (ValidationFlag.status == UNRESOLVED) → mark
        those claims' labels as UNCERTAIN in the merged output.
      - All other claims → carried through unchanged.

    Returns a plain dict keyed by dimension_id so the synthesis prompt can
    serialise it without importing agent-specific types.
    """
    # Build lookup: claim_id → OverturnedClaim
    overturned_by_id: dict[str, OverturnedClaim] = {
        oc.claim_id: oc for oc in memory.overturned_claims
    }
    # Build set of unresolved claim IDs
    unresolved_ids: set[str] = set()
    vf = memory.validation_flags
    if vf:
        for flag in vf.flags:
            if flag.status == ClaimStatus.UNRESOLVED:
                unresolved_ids.add(flag.claim_id)

    sections = {
        "definition_check":    memory.definition_check,
        "risk_classification": memory.risk_classification,
        "prohibited_practices": memory.prohibited_practices,
        "transparency":        memory.transparency,
        "roles":               memory.roles,
        "governance":          memory.governance,
    }

    merged: dict[str, Any] = {}
    for dim_id, section in sections.items():
        if section is None:
            merged[dim_id] = {
                "dimension_id": dim_id,
                "confidence":   Confidence.INSUFFICIENT.value,
                "summary":      "Section not available.",
                "claims":       [],
            }
            continue

        merged_claims = []
        for claim in section.claims:
            if claim.claim_id in overturned_by_id:
                oc = overturned_by_id[claim.claim_id]
                merged_claims.append({
                    "claim_id":   claim.claim_id,
                    "text":       oc.new_finding,
                    "label":      oc.new_label.value,
                    "confidence": oc.new_confidence.value,
                    "chunk_ids":  oc.new_chunk_ids,
                    "source":     "validation_override",
                })
            elif claim.claim_id in unresolved_ids:
                merged_claims.append({
                    "claim_id":   claim.claim_id,
                    "text":       claim.text,
                    "label":      Label.UNCERTAIN.value,
                    "confidence": Confidence.LOW.value,
                    "chunk_ids":  claim.chunk_ids,
                    "source":     "unresolved",
                })
            else:
                merged_claims.append({
                    "claim_id":   claim.claim_id,
                    "text":       claim.text,
                    "label":      claim.label.value,
                    "confidence": claim.confidence.value,
                    "chunk_ids":  claim.chunk_ids,
                    "source":     "analysis",
                })

        # Carry dimension-specific scalar fields.
        extra: dict = {}
        for attr in (
            "is_ai_system", "risk_level",
            "triggered_articles", "prohibited",
            "applies_to_gpai", "labelling_required", "notification_required",
            "is_provider", "is_deployer", "is_both",
            "documentation_required", "oversight_required", "monitoring_required",
        ):
            val = getattr(section, attr, None)
            if val is not None:
                extra[attr] = val.value if hasattr(val, "value") else val

        merged[dim_id] = {
            "dimension_id": dim_id,
            "confidence":   section.confidence.value,
            "summary":      section.summary,
            "claims":       merged_claims,
            **extra,
        }

    return merged


def determine_loop_condition(
    confidence_section: ConfidenceSection,
    loop_count: int,
) -> bool:
    """Return True when the synthesis agent should trigger a re-analysis loop.

    Triggers when BOTH definition_check AND risk_classification are
    INSUFFICIENT and the global loop_count is still below the maximum (1).
    Critical dimensions failing simultaneously indicates the pipeline did not
    have enough evidence to make the most important findings.
    """
    if loop_count >= 1:
        return False  # max one loop; proceed with INSUFFICIENT markers
    return (
        confidence_section.definition_check == Confidence.INSUFFICIENT
        and confidence_section.risk_classification == Confidence.INSUFFICIENT
    )


def build_synthesis_prompt(
    facts: FactSection,
    merged: dict[str, Any],
    chunks: list[Chunk],
    loop_count: int,
    weak_claims: Optional[list[WeakClaim]] = None,
    validation_flags: Optional[ValidationSection] = None,
) -> str:
    """Render the full synthesis prompt.

    Pure function — no I/O.
    """
    loop_context = (
        f"This is loop pass {loop_count + 1}. "
        "Previous pass had insufficient evidence for critical dimensions. "
        "Be conservative and surface all remaining gaps explicitly in section 9."
        if loop_count > 0
        else ""
    )

    return SYNTHESIS_PROMPT_TEMPLATE.format(
        facts_text=_serialise_facts(facts),
        merged_analysis_json=json.dumps(merged, indent=2, default=str),
        validation_json=_serialise_validation_section(validation_flags),
        weak_claims_json=_serialise_weak_claims(weak_claims or []),
        loop_context=loop_context,
        chunks_by_source_json=_serialise_chunks_by_source(chunks),
    )


def parse_synthesis_response(
    response_text: str,
) -> tuple[ReportSection, FollowUpSection, ConfidenceSection]:
    """Parse an LLM synthesis response into three typed output objects.

    Falls back to INSUFFICIENT / empty structures on any parse failure.
    Never raises.
    """
    raw = _extract_json(response_text)
    if not raw:
        return (
            ReportSection(),
            FollowUpSection(),
            ConfidenceSection(),   # all fields default to INSUFFICIENT
        )

    try:
        report_raw   = raw.get("report", {})
        follow_raw   = raw.get("follow_up", {})
        conf_raw     = raw.get("confidence", {})

        report = ReportSection(
            use_case_summary=str(report_raw.get("use_case_summary", "")),
            extracted_facts=dict(report_raw.get("extracted_facts", {})),
            ai_definition_check=dict(report_raw.get("ai_definition_check", {})),
            risk_classification=dict(report_raw.get("risk_classification", {})),
            prohibited_practices_check=dict(
                report_raw.get("prohibited_practices_check", {})
            ),
            transparency_gpai_obligations=dict(
                report_raw.get("transparency_gpai_obligations", {})
            ),
            roles=dict(report_raw.get("roles", {})),
            governance_observations=dict(
                report_raw.get("governance_observations", {})
            ),
            missing_information=dict(report_raw.get("missing_information", {})),
            citations_by_source=dict(report_raw.get("citations_by_source", {})),
        )

        follow_up = FollowUpSection(
            questions=list(follow_raw.get("questions", [])),
            missing_evidence=list(follow_raw.get("missing_evidence", [])),
        )

        confidence = ConfidenceSection(
            definition_check=_parse_confidence(conf_raw.get("definition_check", "INSUFFICIENT")),
            risk_classification=_parse_confidence(conf_raw.get("risk_classification", "INSUFFICIENT")),
            prohibited_practices=_parse_confidence(conf_raw.get("prohibited_practices", "INSUFFICIENT")),
            transparency=_parse_confidence(conf_raw.get("transparency", "INSUFFICIENT")),
            roles=_parse_confidence(conf_raw.get("roles", "INSUFFICIENT")),
            governance=_parse_confidence(conf_raw.get("governance", "INSUFFICIENT")),
            overall=_parse_confidence(conf_raw.get("overall", "INSUFFICIENT")),
        )

        return report, follow_up, confidence

    except Exception:  # noqa: BLE001
        return ReportSection(), FollowUpSection(), ConfidenceSection()


# ── Private intelligence helpers ─────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Three-strategy JSON extractor (same pattern as other agents)."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    block_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if block_match:
        try:
            result = json.loads(block_match.group(1))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return {}


def _serialise_validation_section(vf: Optional[ValidationSection]) -> str:
    if vf is None:
        return "No validation data."
    return json.dumps(
        {
            "overall_confidence": vf.overall_confidence.value,
            "summary": vf.summary,
            "flags": [
                {
                    "claim_id":   f.claim_id,
                    "dimension":  f.dimension_id,
                    "status":     f.status.value,
                    "notes":      f.notes,
                }
                for f in vf.flags
            ],
        },
        indent=2,
    )


def _serialise_weak_claims(weak_claims: list[WeakClaim]) -> str:
    if not weak_claims:
        return "[]"
    return json.dumps(
        [
            {
                "claim_id":   wc.claim_id,
                "dimension":  wc.dimension_id,
                "text":       wc.claim_text,
                "reason":     wc.reason,
            }
            for wc in weak_claims
        ],
        indent=2,
    )


def _serialise_chunks_by_source(chunks: list[Chunk]) -> str:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source_type].append(
            {"chunk_id": chunk.chunk_id, "article_id": chunk.article_id}
        )
    return json.dumps(dict(by_source), indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def _call_llm(prompt: str, system: str, client: Any) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def run_synthesis_agent(
    memory: SynthesisAgentMemoryView,
    chunks: list[Chunk],
    context: dict,
    llm_client: Any = None,
) -> Union[LoopSignal, CompletionSignal]:
    """Drive the synthesis agent to produce the final compliance report.

    ORCHESTRATION ENTRY POINT.

    The agent:
      1. Merges analysis + validation into a consolidated view.
      2. Builds the synthesis prompt.
      3. Calls the LLM.
      4. Parses the response into (ReportSection, FollowUpSection, ConfidenceSection).
      5. Checks the loop condition on the PARSED confidence BEFORE writing.
         - If loop triggered: emits LoopSignal (no writes; memory stays clean).
         - If not: writes all three sections and emits CompletionSignal.

    context keys consumed:
      "loop_count" : int — current global loop count (max 1 before giving up)

    The loop condition check happens before any write so that if the synthesis
    determines critical dimensions are INSUFFICIENT the orchestrator can roll
    back to the after_analysis checkpoint without needing to undo any writes.
    """
    if llm_client is None:
        import anthropic
        llm_client = anthropic.Anthropic()

    facts = memory.facts
    if facts is None:
        raise ValueError("FactSection not available in memory.")

    loop_count: int = context.get("loop_count", 0)

    # -- INTELLIGENCE: merge and build prompt --
    merged = merge_analysis_with_validation(memory)

    prompt = build_synthesis_prompt(
        facts=facts,
        merged=merged,
        chunks=chunks,
        loop_count=loop_count,
        weak_claims=memory.weak_claims,
        validation_flags=memory.validation_flags,
    )

    # -- ORCHESTRATION: LLM call --
    response_text = _call_llm(prompt, SYNTHESIS_SYSTEM_PROMPT, llm_client)

    # -- INTELLIGENCE: parse --
    report, follow_up, confidence = parse_synthesis_response(response_text)

    # -- Loop condition check (BEFORE any write) --
    if determine_loop_condition(confidence, loop_count):
        refined = (
            follow_up.missing_evidence[0]
            if follow_up.missing_evidence
            else "Critical dimensions (AI system definition, risk classification) "
                 "lack sufficient legislative evidence. Retrieve more targeted chunks."
        )
        return LoopSignal(
            reason=(
                "Both definition_check and risk_classification are INSUFFICIENT. "
                "Looping back to analysis with refined context."
            ),
            refined_context=refined,
        )

    # -- Write (only if not looping) --
    memory.write_final_report(report)
    memory.write_follow_up_questions(follow_up)
    memory.write_confidence_summary(confidence)

    return CompletionSignal(agent="synthesis", message="Final report written.")
