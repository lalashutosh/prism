"""
prompts/validation_prompts.py
──────────────────────────────
Raw prompt templates for the validation agent.

Pure string constants — no logic, no imports.
"""

# The system prompt frames the model as an *independent* reviewer to reduce
# anchoring bias: the model should not simply restate the original claim
# but must genuinely re-evaluate it against newly retrieved evidence.
VALIDATION_SYSTEM_PROMPT = """\
You are an independent EU AI Act legal reviewer.  Your task is to
re-examine a specific claim from an earlier analysis pass and determine
whether additional evidence confirms, overturns, or leaves it unresolved.

CRITICAL RULES
──────────────
1. Only cite chunk_ids listed in the RETRIEVED CHUNKS section.
2. Your verdict must be one of exactly: "CONFIRMED", "OVERTURNED", "UNRESOLVED".
3. CONFIRMED  – new evidence clearly supports the original claim.
4. OVERTURNED – new evidence clearly contradicts the original claim.
5. UNRESOLVED – evidence is still ambiguous or absent; neither confirms nor overturns.
6. Be conservative: prefer UNRESOLVED over a weak CONFIRMED or OVERTURNED.
7. Output ONLY valid JSON.
"""

CLAIM_VALIDATION_TEMPLATE = """\
ORIGINAL CLAIM UNDER REVIEW
─────────────────────────────
Dimension  : {dimension_id}
Claim ID   : {claim_id}
Claim text : {claim_text}
Weakness   : {weakness_reason}
Original confidence : {original_confidence}
Original label      : {original_label}

RETRIEVED CHUNKS FOR INDEPENDENT REVIEW
─────────────────────────────────────────
{chunks_text}

USE CASE FACTS (for context)
──────────────────────────────
{facts_text}

Respond with ONLY this JSON structure:
{{
  "claim_id": "{claim_id}",
  "status": "CONFIRMED|OVERTURNED|UNRESOLVED",
  "finding": "<explanation of your verdict>",
  "supporting_chunk_ids": ["<chunk_id>", ...],
  "new_confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
  "new_label": "RETRIEVED|FACT|ASSUMPTION|UNCERTAIN"
}}
"""

# WEAKNESS_RETRIEVAL_TERMS maps each weakness reason code to additional search
# terms that help the retrieval layer surface chunks more likely to resolve it.
# These terms supplement the dimension-level DIMENSION_KEYWORDS from the
# analysis prompts when building a validation-specific retrieval query.
WEAKNESS_RETRIEVAL_TERMS: dict[str, list[str]] = {
    "LOW_CONFIDENCE": ["evidence", "clarification", "definition"],
    "ASSUMPTION":     ["statutory", "article", "regulation", "requirement"],
    "UNSUPPORTED":    ["provision", "obligation", "compliance"],
}
