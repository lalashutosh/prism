"""
prompts/synthesis_prompts.py
─────────────────────────────
Raw prompt templates for the synthesis agent.

Pure string constants — no logic, no imports.
"""

# The synthesis system prompt positions the model as the final integrating
# analyst whose role is merging — not re-deriving — findings.  The merge
# rules (OVERTURNED → use validation finding, UNRESOLVED → UNCERTAIN) are
# stated explicitly so the model does not re-analyse from scratch.
SYNTHESIS_SYSTEM_PROMPT = """\
You are a senior EU AI Act compliance analyst producing the final structured
compliance report for a client.

You receive:
  - Structured facts extracted from the client's use-case document
  - Six legal-dimension findings from the analysis phase
  - Validation results (confirmed / overturned / unresolved claims)
  - A list of all retrieved chunks available for citations

Your job is to merge these into a ten-section compliance report.

CRITICAL RULES
──────────────
1. For OVERTURNED claims: use the validation finding, not the analysis finding.
2. For UNRESOLVED claims: label them UNCERTAIN in the final report.
3. Every claim in the final report must carry a label:
   FACT / RETRIEVED / ASSUMPTION / UNCERTAIN.
4. Assign a final confidence level (HIGH/MEDIUM/LOW/INSUFFICIENT) to each
   section, taking into account validation results.
5. Section 9 (missing information) must surface all gaps and UNCERTAIN items.
6. Section 10 (citations) must group chunk_ids by source_type.
7. Output ONLY valid JSON.
"""

# The synthesis prompt template structures the LLM's input into five labelled
# sections and requests a JSON response with three top-level keys:
#
#   "report"     — the ten-section compliance document (sections 1–10)
#   "follow_up"  — clarifying questions and missing evidence requests for the UI
#   "confidence" — per-dimension and overall confidence for the orchestrator's
#                  loop condition check (determine_loop_condition reads this)
#
# Separating follow_up and confidence from report lets the orchestrator act on
# confidence without parsing the full report, and lets the UI surface questions
# independently from the formal compliance document.
SYNTHESIS_PROMPT_TEMPLATE = """\
USE CASE FACTS
───────────────
{facts_text}

ANALYSIS FINDINGS (post-validation merge)
──────────────────────────────────────────
{merged_analysis_json}

VALIDATION SUMMARY
───────────────────
{validation_json}

WEAK CLAIMS
────────────
{weak_claims_json}

LOOP CONTEXT (empty on first pass)
────────────────────────────────────
{loop_context}

AVAILABLE CHUNK IDS BY SOURCE TYPE
────────────────────────────────────
{chunks_by_source_json}

Produce a compliance report with EXACTLY this JSON structure:
{{
  "report": {{
    "use_case_summary": "<1-2 paragraph narrative summary>",
    "extracted_facts": {{
      "use_case_name": "<name>",
      "description": "<description>",
      "industry": "<industry or null>",
      "ai_capabilities": ["..."],
      "data_inputs": ["..."],
      "outputs": ["..."],
      "deployment_context": "<context or null>",
      "affected_persons": ["..."]
    }},
    "ai_definition_check": {{
      "finding": "<narrative>",
      "is_ai_system": true|false|null,
      "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
      "article_references": ["Article 3(1)", "Annex I"],
      "claims": [{{"text":"...","label":"...","chunk_ids":["..."]}}]
    }},
    "risk_classification": {{
      "finding": "<narrative>",
      "risk_level": "unacceptable|high|limited|minimal|unknown",
      "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
      "article_references": ["Article 6", "Annex III"],
      "claims": [{{"text":"...","label":"...","chunk_ids":["..."]}}]
    }},
    "prohibited_practices_check": {{
      "finding": "<narrative>",
      "prohibited": true|false|null,
      "triggered_articles": ["Article 5(1)(a)", ...],
      "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
      "claims": [{{"text":"...","label":"...","chunk_ids":["..."]}}]
    }},
    "transparency_gpai_obligations": {{
      "finding": "<narrative>",
      "obligations": ["<obligation 1>", ...],
      "applies_to_gpai": true|false,
      "labelling_required": true|false,
      "notification_required": true|false,
      "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
      "claims": [{{"text":"...","label":"...","chunk_ids":["..."]}}]
    }},
    "roles": {{
      "finding": "<narrative>",
      "is_provider": true|false,
      "is_deployer": true|false,
      "is_both": true|false,
      "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
      "claims": [{{"text":"...","label":"...","chunk_ids":["..."]}}]
    }},
    "governance_observations": {{
      "finding": "<narrative>",
      "obligations": ["<obligation 1>", ...],
      "documentation_required": true|false,
      "oversight_required": true|false,
      "monitoring_required": true|false,
      "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
      "claims": [{{"text":"...","label":"...","chunk_ids":["..."]}}]
    }},
    "missing_information": {{
      "gaps": ["<gap 1>", ...],
      "uncertain_claims": ["<claim text>", ...],
      "unresolved_dimensions": ["<dimension_id>", ...]
    }},
    "citations_by_source": {{
      "legislation": [{{"chunk_id":"...","article_id":"...","excerpt":"..."}}],
      "official_guidance": [{{"chunk_id":"...","article_id":"...","excerpt":"..."}}],
      "uploaded_doc": [{{"chunk_id":"...","excerpt":"..."}}]
    }}
  }},
  "follow_up": {{
    "questions": ["<question for the user>", ...],
    "missing_evidence": ["<what additional info would help>", ...]
  }},
  "confidence": {{
    "definition_check": "HIGH|MEDIUM|LOW|INSUFFICIENT",
    "risk_classification": "HIGH|MEDIUM|LOW|INSUFFICIENT",
    "prohibited_practices": "HIGH|MEDIUM|LOW|INSUFFICIENT",
    "transparency": "HIGH|MEDIUM|LOW|INSUFFICIENT",
    "roles": "HIGH|MEDIUM|LOW|INSUFFICIENT",
    "governance": "HIGH|MEDIUM|LOW|INSUFFICIENT",
    "overall": "HIGH|MEDIUM|LOW|INSUFFICIENT"
  }}
}}
"""
