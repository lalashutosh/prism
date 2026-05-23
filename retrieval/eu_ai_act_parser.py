"""
Extract atomic EU AI Act legal nodes from the official EUR-Lex PDF with Gemini.

The script reads the local PDF with pdfplumber, batches text along legal
structure boundaries, sends each batch to Gemini 3 Flash with a Pydantic
response schema, and writes a single JSON array of atomic legal nodes.

Execution:

    $env:GOOGLE_API_KEY = "your-key"
    python retrieval/eu_ai_act_parser.py ^
        --pdf "docs/Regulation - EU - 2024_1689 - EN - EUR-Lex.pdf" ^
        --output retrieval/eu_ai_act_nodes.json

Dependencies:
    pip install pdfplumber google-genai pydantic
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from google import genai
from google.genai import types
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


TARGET_PDF_BASENAME = "Regulation - EU - 2024_1689 - EN - EUR-Lex.pdf"
DOCUMENT_SOURCE = "Regulation (EU) 2024/1689"
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_MAX_BATCH_CHARS = 6_000
DEFAULT_OVERLAP_CHARS = 1_800
DEFAULT_MAX_OUTPUT_TOKENS = 16_384
MAX_ADAPTIVE_SPLIT_DEPTH = 8

DimensionTag = Literal[
    "definition_check",
    "risk_classification",
    "prohibited_practices",
    "transparency",
    "roles",
    "governance",
]
NodeType = Literal["Core_Article", "Core_Recital", "Core_Annex"]
LegalWeight = Literal["Binding", "Binding_List", "Interpretive"]

DIMENSION_TAGS: tuple[str, ...] = (
    "definition_check",
    "risk_classification",
    "prohibited_practices",
    "transparency",
    "roles",
    "governance",
)

NOISE_PATTERNS = (
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2},\s+\d{1,2}:\d{2}\s+[AP]M\s+Regulation - EU - 2024/1689 - EN - EUR-Lex$"),
    re.compile(r"^https://eur-lex\.europa\.eu/eli/reg/2024/1689/oj\?.*\s+\d+/\d+$"),
    re.compile(r"^\d+/\d+$"),
)
PAGE_NUMBER_RE = re.compile(r"^(?:\d+\s*)+$")
RECITAL_RE = re.compile(r"^\((?P<number>\d{1,3})\)\s*")
CHAPTER_RE = re.compile(r"^CHAPTER\s+(?P<number>[IVXLCDM]+)\b", re.IGNORECASE)
SECTION_RE = re.compile(r"^SECTION\s+(?P<number>\d+[A-Z]?)\b", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^Article\s+(?P<number>\d+[a-z]?)$", re.IGNORECASE)
ANNEX_RE = re.compile(r"^ANNEX\s+(?P<number>[IVXLCDM]+)$")
POINT_BOUNDARY_RE = re.compile(
    r"^(?:\(\d{1,3}\)|\d+(?:\.\d+)*\.|\([a-z]\)|[a-z]\))\s+",
    re.IGNORECASE,
)


class AtomicLegalNode(BaseModel):
    """Strict JSON node shape expected from Gemini."""

    # This is the local post-LLM validator. Gemini gets a compatible SDK schema
    # later, then this stricter Pydantic model rejects extra or malformed fields.
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(
        ...,
        description="Uppercase/snake_case ID such as EU_AI_ACT_ART_5_1_A.",
        pattern=r"^EU_AI_ACT_[A-Z0-9_]+$",
    )
    node_type: NodeType
    hierarchy: list[str] = Field(default_factory=list)
    document_source: Literal["Regulation (EU) 2024/1689"]
    atomic_content: str
    legal_weight: LegalWeight
    exceptions: list[str] = Field(default_factory=list)
    applicable_dimensions: list[DimensionTag] = Field(default_factory=list)
    internal_references: list[str] = Field(default_factory=list)
    external_legislation: list[str] = Field(default_factory=list)

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, value: str) -> str:
        if value != value.upper() or "__" in value or value.endswith("_"):
            raise ValueError("node_id must be strict uppercase snake case")
        return value

    @field_validator(
        "hierarchy",
        "exceptions",
        "internal_references",
        "external_legislation",
        mode="before",
    )
    @classmethod
    def empty_list_for_none(cls, value: object) -> object:
        return [] if value is None else value

    @field_validator("atomic_content")
    @classmethod
    def validate_atomic_content(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("atomic_content must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def validate_legal_weight(self) -> "AtomicLegalNode":
        if self.node_type == "Core_Recital" and self.legal_weight != "Interpretive":
            raise ValueError("Recitals must have Interpretive legal_weight")
        if self.node_type in {"Core_Article", "Core_Annex"}:
            if self.legal_weight not in {"Binding", "Binding_List"}:
                raise ValueError("Articles and annexes must be binding")
        return self


class ExtractionResult(BaseModel):
    """Top-level response object used as google-genai response_schema."""

    # The LLM is instructed to return one object with a single "nodes" array.
    # Keeping a top-level object is easier for Gemini than asking for a bare list.
    model_config = ConfigDict(extra="forbid")

    nodes: list[AtomicLegalNode] = Field(default_factory=list)


@dataclass(frozen=True)
class PageLine:
    page_number: int
    text: str


@dataclass
class Segment:
    kind: Literal["recital", "article", "annex", "other"]
    title: str
    hierarchy: list[str]
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()

    def clone_with_lines(self, title: str, lines: list[str]) -> "Segment":
        return Segment(
            kind=self.kind,
            title=title,
            hierarchy=list(self.hierarchy),
            lines=lines,
        )


@dataclass(frozen=True)
class TextBatch:
    batch_index: int
    segments: list[Segment]
    context_before: str
    context_after: str

    @property
    def structural_context(self) -> list[str]:
        context: list[str] = []
        for segment in self.segments:
            for item in segment.hierarchy:
                if item not in context:
                    context.append(item)
        return context

    @property
    def text(self) -> str:
        parts: list[str] = []
        for segment in self.segments:
            hierarchy = " > ".join(segment.hierarchy) or segment.title
            parts.append(
                "\n".join(
                    [
                        f"[BEGIN {segment.kind.upper()} SEGMENT: {segment.title}]",
                        f"[HIERARCHY: {hierarchy}]",
                        segment.text,
                        f"[END {segment.kind.upper()} SEGMENT]",
                    ]
                )
            )
        return "\n\n".join(parts).strip()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    try:
        pdf_path = Path(args.pdf).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve()

        validate_pdf_path(pdf_path)
        page_lines = extract_pdf_lines(pdf_path)
        segments = split_structural_segments(page_lines)
        if not segments:
            raise RuntimeError("No legal segments were extracted from the PDF.")

        split_segments = split_oversized_segments(
            segments=segments,
            max_chars=args.max_batch_chars,
        )
        batches = build_text_batches(
            split_segments,
            max_chars=args.max_batch_chars,
            overlap_chars=args.overlap_chars,
        )

        logging.info(
            "Prepared %s segments into %s Gemini batch(es)",
            len(split_segments),
            len(batches),
        )

        if "GOOGLE_API_KEY" not in os.environ:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set in the process environment. "
                "Set it before running; this script intentionally does not read .env."
            )

        # genai.Client() reads GOOGLE_API_KEY from the environment by design.
        client = genai.Client()
        # This is the full LLM extraction phase: one Gemini call per batch.
        nodes = run_extraction_loop(
            client=client,
            batches=batches,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            retries=args.retries,
            continue_on_error=args.continue_on_error,
        )
        if not nodes:
            raise RuntimeError("Gemini returned no atomic legal nodes.")

        # Sliding-window context can cause overlapping node proposals, so collapse
        # duplicate node_ids before the master JSON file is written.
        deduped_nodes = deduplicate_nodes(nodes)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            # model_dump(mode="json") converts Pydantic/enums/literals into plain
            # JSON-serializable Python objects before json.dumps formats the file.
            json.dumps(
                [node.model_dump(mode="json") for node in deduped_nodes],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logging.info("Wrote %s nodes to %s", len(deduped_nodes), output_path)
        return 0
    except Exception as exc:  # noqa: BLE001 - command line should fail cleanly.
        logging.error("%s", exc)
        return 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract atomic JSON nodes from the EU AI Act PDF with Gemini.",
        epilog=(
            "Set GOOGLE_API_KEY in the process environment, then run this "
            "script to execute the full Gemini extraction pipeline."
        ),
    )
    parser.add_argument(
        "--pdf",
        default=str(Path("docs") / TARGET_PDF_BASENAME),
        help="Path to the official EU AI Act EUR-Lex PDF.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("eu_ai_act_nodes.json")),
        help="Destination JSON file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model name.",
    )
    parser.add_argument(
        "--max-batch-chars",
        type=int,
        default=DEFAULT_MAX_BATCH_CHARS,
        help="Maximum current-batch character budget before calling Gemini.",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=DEFAULT_OVERLAP_CHARS,
        help="Characters of neighboring context supplied outside current batch.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum Gemini output tokens per batch.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per Gemini batch after an API or validation failure.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log failed batches and keep going instead of aborting.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )
    return parser


def validate_pdf_path(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"PDF path is not a file: {pdf_path}")
    if pdf_path.name.casefold() != TARGET_PDF_BASENAME.casefold():
        raise ValueError(
            f"Expected {TARGET_PDF_BASENAME!r}; got {pdf_path.name!r}."
        )


def extract_pdf_lines(pdf_path: Path) -> list[PageLine]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is not installed.") from exc

    page_lines: list[PageLine] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text(layout=False, x_tolerance=2, y_tolerance=3)
            if not raw_text:
                logging.warning("Page %s produced no text", page_number)
                continue
            for raw_line in raw_text.splitlines():
                line = clean_pdf_line(raw_line)
                if line:
                    page_lines.append(PageLine(page_number=page_number, text=line))
    return page_lines


def clean_pdf_line(raw_line: str) -> str:
    line = raw_line.replace("\u00a0", " ").replace("\u00ad", "")
    line = line.replace("`", "")
    line = re.sub(r"\s+", " ", line).strip()
    if not line:
        return ""
    if line in {"EN", "*"}:
        return ""
    if PAGE_NUMBER_RE.fullmatch(line):
        return ""
    if any(pattern.match(line) for pattern in NOISE_PATTERNS):
        return ""
    return line


def split_structural_segments(page_lines: Iterable[PageLine]) -> list[Segment]:
    segments: list[Segment] = []
    current: Segment | None = None
    state: Literal["front", "recitals", "body", "annex"] = "front"
    chapter: str | None = None
    section: str | None = None
    pending_heading: tuple[Literal["chapter", "section", "annex"], str] | None = None
    pending_article_title = False

    def flush_current() -> None:
        nonlocal current
        if current is not None and current.text:
            segments.append(current)
        current = None

    def current_hierarchy(article_title: str | None = None) -> list[str]:
        hierarchy = [item for item in (chapter, section, article_title) if item]
        return hierarchy or ([article_title] if article_title else [])

    for page_line in page_lines:
        line = page_line.text

        if line == "Whereas:":
            flush_current()
            state = "recitals"
            continue

        if line == "HAVE ADOPTED THIS REGULATION:":
            flush_current()
            state = "body"
            chapter = None
            section = None
            pending_heading = None
            pending_article_title = False
            continue

        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            flush_current()
            state = "body"
            section = None
            chapter = line
            pending_heading = ("chapter", line)
            pending_article_title = False
            continue

        section_match = SECTION_RE.match(line)
        if section_match and state == "body":
            flush_current()
            section = line
            pending_heading = ("section", line)
            pending_article_title = False
            continue

        annex_match = ANNEX_RE.match(line)
        if annex_match:
            flush_current()
            state = "annex"
            chapter = line
            section = None
            pending_heading = ("annex", line)
            current = Segment(kind="annex", title=line, hierarchy=[line], lines=[line])
            pending_article_title = False
            continue

        article_match = ARTICLE_RE.match(line)
        if article_match and state == "body":
            flush_current()
            article_number = article_match.group("number")
            title = f"Article {article_number}"
            current = Segment(
                kind="article",
                title=title,
                hierarchy=current_hierarchy(title),
                lines=[line],
            )
            pending_article_title = True
            pending_heading = None
            continue

        if pending_heading is not None:
            heading_kind, heading_number = pending_heading
            full_heading = f"{heading_number} - {line}"
            if heading_kind == "chapter":
                chapter = full_heading
            elif heading_kind == "section":
                section = full_heading
            elif heading_kind == "annex":
                chapter = full_heading
                if current is not None and current.kind == "annex":
                    current.title = full_heading
                    current.hierarchy = [full_heading]
            pending_heading = None
            if current is None:
                continue

        if state == "recitals":
            recital_match = RECITAL_RE.match(line)
            if recital_match:
                flush_current()
                recital_number = recital_match.group("number")
                title = f"Recital {recital_number}"
                current = Segment(
                    kind="recital",
                    title=title,
                    hierarchy=["Recitals", title],
                    lines=[line],
                )
            elif current is not None:
                current.lines.append(line)
            continue

        if current is None:
            continue

        if pending_article_title and current.kind == "article":
            current.title = f"{current.title} - {line}"
            current.hierarchy = current_hierarchy(current.title)
            pending_article_title = False

        current.lines.append(line)

    flush_current()
    return segments


def split_oversized_segments(
    segments: Iterable[Segment],
    max_chars: int,
) -> list[Segment]:
    split_segments: list[Segment] = []
    target_chars = max(max_chars - 1_500, max_chars // 2)
    for segment in segments:
        if segment.kind == "recital":
            split_segments.append(segment)
            continue
        if len(segment.text) <= max_chars:
            split_segments.append(segment)
            continue
        split_segments.extend(split_single_segment(segment, target_chars))
    return split_segments


def split_single_segment(segment: Segment, target_chars: int) -> list[Segment]:
    header_lines = segment.lines[:2] if segment.kind in {"article", "annex"} else []
    parts: list[Segment] = []
    current_lines: list[str] = list(header_lines)

    def flush_part() -> None:
        nonlocal current_lines
        body = [line for line in current_lines if line]
        if len(body) > len(header_lines):
            title = f"{segment.title} (part {len(parts) + 1})"
            parts.append(segment.clone_with_lines(title, body))
        current_lines = list(header_lines)

    for line in segment.lines[len(header_lines) :]:
        projected = len("\n".join(current_lines + [line]))
        is_boundary = bool(POINT_BOUNDARY_RE.match(line))
        if current_lines and projected > target_chars and is_boundary:
            flush_part()
        elif projected > target_chars and len(current_lines) > len(header_lines):
            flush_part()
        current_lines.append(line)

    flush_part()
    if not parts:
        return [segment]
    return parts


def build_text_batches(
    segments: list[Segment],
    max_chars: int,
    overlap_chars: int,
) -> list[TextBatch]:
    batches: list[TextBatch] = []
    current: list[Segment] = []

    def current_size() -> int:
        return sum(rendered_segment_size(segment) for segment in current)

    def flush_current() -> None:
        nonlocal current
        if current:
            index = len(batches)
            batches.append(
                TextBatch(
                    batch_index=index,
                    segments=list(current),
                    context_before=make_neighbor_context(
                        segments[: segment_start_index(segments, current[0])],
                        overlap_chars,
                        from_end=True,
                    ),
                    context_after=make_neighbor_context(
                        segments[segment_end_index(segments, current[-1]) + 1 :],
                        overlap_chars,
                        from_end=False,
                    ),
                )
            )
            current = []

    for segment in segments:
        projected_size = current_size() + rendered_segment_size(segment) + 2
        if current and projected_size > max_chars:
            flush_current()
        current.append(segment)
    flush_current()
    return batches


def rendered_segment_size(segment: Segment) -> int:
    hierarchy = " > ".join(segment.hierarchy) or segment.title
    return (
        len(segment.text)
        + len(segment.kind) * 2
        + len(segment.title)
        + len(hierarchy)
        + 64
    )


def segment_start_index(segments: list[Segment], target: Segment) -> int:
    for index, segment in enumerate(segments):
        if segment is target:
            return index
    return 0


def segment_end_index(segments: list[Segment], target: Segment) -> int:
    for index, segment in enumerate(segments):
        if segment is target:
            return index
    return len(segments) - 1


def make_neighbor_context(
    neighbor_segments: list[Segment],
    max_chars: int,
    from_end: bool,
) -> str:
    if max_chars <= 0 or not neighbor_segments:
        return ""
    text = "\n\n".join(segment.text for segment in neighbor_segments)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:] if from_end else text[:max_chars]


def split_batch_after_failure(batch: TextBatch) -> list[TextBatch]:
    """Split a failed batch into smaller real batches for another LLM pass."""

    if len(batch.segments) > 1:
        midpoint = max(1, len(batch.segments) // 2)
        left = batch.segments[:midpoint]
        right = batch.segments[midpoint:]
        return [
            make_child_batch(batch=batch, segments=left, before=[], after=right),
            make_child_batch(batch=batch, segments=right, before=left, after=[]),
        ]

    if not batch.segments:
        return []

    segment = batch.segments[0]
    if segment.kind == "recital":
        return []

    target_chars = max(1_500, len(segment.text) // 2)
    split_segments = split_single_segment(segment, target_chars)
    if len(split_segments) <= 1:
        split_segments = force_split_article_or_annex(segment)
    if len(split_segments) <= 1:
        return []

    child_batches: list[TextBatch] = []
    for index, split_segment in enumerate(split_segments):
        child_batches.append(
            make_child_batch(
                batch=batch,
                segments=[split_segment],
                before=split_segments[:index],
                after=split_segments[index + 1 :],
            )
        )
    return child_batches


def make_child_batch(
    batch: TextBatch,
    segments: list[Segment],
    before: list[Segment],
    after: list[Segment],
) -> TextBatch:
    """Create a smaller batch while preserving local neighbor context."""

    context_limit = max(
        DEFAULT_OVERLAP_CHARS,
        len(batch.context_before),
        len(batch.context_after),
    )
    before_text = "\n\n".join(segment.text for segment in before)
    after_text = "\n\n".join(segment.text for segment in after)
    context_before = combine_context(
        batch.context_before,
        before_text,
        max_chars=context_limit,
        from_end=True,
    )
    context_after = combine_context(
        after_text,
        batch.context_after,
        max_chars=context_limit,
        from_end=False,
    )
    return TextBatch(
        batch_index=batch.batch_index,
        segments=segments,
        context_before=context_before,
        context_after=context_after,
    )


def combine_context(
    left: str,
    right: str,
    max_chars: int,
    from_end: bool,
) -> str:
    text = "\n\n".join(part for part in (left.strip(), right.strip()) if part)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:] if from_end else text[:max_chars]


def force_split_article_or_annex(segment: Segment) -> list[Segment]:
    """Last-resort split for a single long article or annex segment."""

    if segment.kind not in {"article", "annex"} or len(segment.lines) < 6:
        return [segment]

    header_count = 2 if len(segment.lines) > 2 else 1
    header_lines = segment.lines[:header_count]
    body_lines = segment.lines[header_count:]
    if len(body_lines) < 2:
        return [segment]

    midpoint = len(body_lines) // 2
    return [
        segment.clone_with_lines(
            f"{segment.title} (part 1)",
            header_lines + body_lines[:midpoint],
        ),
        segment.clone_with_lines(
            f"{segment.title} (part 2)",
            header_lines + body_lines[midpoint:],
        ),
    ]


def run_extraction_loop(
    client: genai.Client,
    batches: Iterable[TextBatch],
    model: str,
    max_output_tokens: int,
    retries: int,
    continue_on_error: bool,
) -> list[AtomicLegalNode]:
    """Call Gemini for every text batch and collect validated JSON nodes."""

    all_nodes: list[AtomicLegalNode] = []
    for batch in batches:
        logging.info(
            "Extracting batch %s (%s segment(s), %s chars)",
            batch.batch_index + 1,
            len(batch.segments),
            len(batch.text),
        )
        try:
            batch_nodes = extract_batch_adaptively(
                client=client,
                batch=batch,
                model=model,
                max_output_tokens=max_output_tokens,
                retries=retries,
            )
            # batch_nodes have already passed local Pydantic validation.
            logging.info(
                "Batch %s returned %s node(s)",
                batch.batch_index + 1,
                len(batch_nodes),
            )
            all_nodes.extend(batch_nodes)
        except Exception as exc:  # noqa: BLE001 - configurable batch behavior.
            if not continue_on_error:
                raise
            logging.error("Batch %s failed: %s", batch.batch_index + 1, exc)
    return all_nodes


def extract_batch_adaptively(
    client: genai.Client,
    batch: TextBatch,
    model: str,
    max_output_tokens: int,
    retries: int,
    split_depth: int = 0,
) -> list[AtomicLegalNode]:
    """Extract one batch, splitting it if Gemini returns truncated JSON."""

    try:
        result = extract_batch_with_retries(
            client=client,
            batch=batch,
            model=model,
            max_output_tokens=max_output_tokens,
            retries=retries,
        )
        return result.nodes
    except Exception as exc:  # noqa: BLE001 - adaptive split boundary.
        if not is_truncated_json_error(exc):
            raise
        sub_batches = split_batch_after_failure(batch)
        if split_depth >= MAX_ADAPTIVE_SPLIT_DEPTH or not sub_batches:
            raise

        logging.warning(
            "Batch %s failed after retries and will be split into %s smaller "
            "real extraction call(s): %s",
            batch.batch_index + 1,
            len(sub_batches),
            exc,
        )

        nodes: list[AtomicLegalNode] = []
        for sub_batch in sub_batches:
            nodes.extend(
                extract_batch_adaptively(
                    client=client,
                    batch=sub_batch,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    retries=retries,
                    split_depth=split_depth + 1,
                )
            )
        return nodes


def extract_batch_with_retries(
    client: genai.Client,
    batch: TextBatch,
    model: str,
    max_output_tokens: int,
    retries: int,
) -> ExtractionResult:
    """Retry one batch when Gemini or JSON validation fails transiently."""

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            # A successful call returns a fully validated ExtractionResult.
            return call_gemini_for_batch(
                client=client,
                batch=batch,
                model=model,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - retry boundary.
            last_error = exc
            if attempt >= retries:
                break
            if is_truncated_json_error(exc):
                logging.warning(
                    "Batch %s returned truncated JSON; skipping same-size "
                    "retry so adaptive splitting can continue.",
                    batch.batch_index + 1,
                )
                break
            sleep_seconds = min(2**attempt, 8)
            logging.warning(
                "Batch %s attempt %s failed: %s. Retrying in %ss.",
                batch.batch_index + 1,
                attempt + 1,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(
        f"Gemini extraction failed for batch {batch.batch_index + 1}: {last_error}"
    )


def is_truncated_json_error(exc: Exception) -> bool:
    """Return True when Gemini output ended before valid JSON completed."""

    message = str(exc).lower()
    return (
        "eof while parsing" in message
        or "unterminated string" in message
        or "finish_reason=max_tokens" in message
        or "finish_reason=maximum_tokens" in message
    )


def call_gemini_for_batch(
    client: genai.Client,
    batch: TextBatch,
    model: str,
    max_output_tokens: int,
) -> ExtractionResult:
    """Send one prompt to Gemini and parse the structured JSON response."""

    # The prompt contains legal extraction rules plus the current batch text.
    prompt = build_extraction_prompt(batch)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=max_output_tokens,
            # Force Gemini into JSON mode; no prose should be returned.
            response_mime_type="application/json",
            # Use an API-compatible schema, then enforce the stricter Pydantic
            # model after the response is received.
            response_schema=build_gemini_response_schema(),
        ),
    )

    # Newer google-genai versions may hydrate response.parsed automatically.
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, ExtractionResult):
        return parsed
    if isinstance(parsed, dict):
        # Validate SDK-parsed dicts against the strict local Pydantic model.
        return ExtractionResult.model_validate(parsed)
    if parsed is not None:
        # Some SDK versions return a model-like parsed object; normalize it.
        return ExtractionResult.model_validate(parsed)

    # Fallback for SDKs/models that only expose raw JSON text.
    text = getattr(response, "text", "") or ""
    if not text:
        raise RuntimeError("Gemini response had no parsed object or text payload.")
    # model_validate_json is the last gate before nodes enter the pipeline.
    try:
        return ExtractionResult.model_validate_json(text)
    except ValidationError as exc:
        finish_reason = get_gemini_finish_reason(response)
        raise RuntimeError(
            "Gemini returned invalid JSON "
            f"(finish_reason={finish_reason}, chars={len(text)}): {exc}; "
            f"response_tail={text[-500:]!r}"
        ) from exc


def get_gemini_finish_reason(response: types.GenerateContentResponse) -> str:
    """Return the first candidate finish reason for LLM JSON diagnostics."""

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return "unknown"
    finish_reason = getattr(candidates[0], "finish_reason", None)
    if finish_reason is None:
        return "unknown"
    return str(getattr(finish_reason, "name", finish_reason))


def build_gemini_response_schema() -> types.Schema:
    """Return a Gemini-compatible schema without unsupported JSON Schema keys.

    Pydantic's strict model schema includes ``additionalProperties: false``.
    The Gemini API rejects that field for controlled generation, so the
    request uses this explicit API schema and Pydantic still validates the
    returned object locally.
    """

    # Reusable list-of-string schema for optional metadata arrays.
    string_array = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.STRING),
    )
    # This is the node shape Gemini sees while generating JSON. It mirrors the
    # Pydantic model but avoids JSON Schema keywords the API rejects.
    node_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            # Stable retrieval ID produced by the model from legal structure.
            "node_id": types.Schema(
                type=types.Type.STRING,
                pattern=r"^EU_AI_ACT_[A-Z0-9_]+$",
            ),
            # Node type controls legal_weight validation after generation.
            "node_type": types.Schema(
                type=types.Type.STRING,
                enum=["Core_Article", "Core_Recital", "Core_Annex"],
            ),
            # Parent headings and point labels preserve legal context.
            "hierarchy": string_array,
            # Hard-coded source value prevents mixed-document outputs.
            "document_source": types.Schema(
                type=types.Type.STRING,
                enum=[DOCUMENT_SOURCE],
            ),
            # Verbatim legal text lives here; the prompt forbids summaries.
            "atomic_content": types.Schema(type=types.Type.STRING),
            # Binding_List is reserved for standalone split list obligations.
            "legal_weight": types.Schema(
                type=types.Type.STRING,
                enum=["Binding", "Binding_List", "Interpretive"],
            ),
            # Explicit carve-outs are copied separately for downstream filters.
            "exceptions": string_array,
            # Dimension tags make the extracted law searchable by agent task.
            "applicable_dimensions": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.STRING,
                    enum=list(DIMENSION_TAGS),
                ),
            ),
            # Internal references point to other EU AI Act nodes/sections.
            "internal_references": string_array,
            # External legal frameworks are separated from internal citations.
            "external_legislation": string_array,
        },
        # Gemini must emit every field, even when the correct value is [].
        required=[
            "node_id",
            "node_type",
            "hierarchy",
            "document_source",
            "atomic_content",
            "legal_weight",
            "exceptions",
            "applicable_dimensions",
            "internal_references",
            "external_legislation",
        ],
    )
    # Top-level object wrapper around the nodes array.
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "nodes": types.Schema(
                type=types.Type.ARRAY,
                items=node_schema,
            ),
        },
        required=["nodes"],
    )


def build_extraction_prompt(batch: TextBatch) -> str:
    """Build the exact legal/JSON instructions sent to Gemini."""

    # The prompt repeats schema rules in natural language because response_schema
    # constrains shape, while the prompt controls legal boundary interpretation.
    # The structural context is serialized as JSON for stable list syntax.
    return f"""
You are extracting perfectly bounded atomic JSON nodes from the official EU AI Act.

Return only JSON matching the provided schema:
{{"nodes": [AtomicLegalNode, ...]}}

Global constants and allowed values:
- document_source must be exactly "{DOCUMENT_SOURCE}".
- node_type must be one of: Core_Article, Core_Recital, Core_Annex.
- legal_weight must be one of: Binding, Binding_List, Interpretive.
- applicable_dimensions tags must come only from: {list(DIMENSION_TAGS)}.

CRITICAL LEGAL ATOMIC BOUNDARY LOGIC:
- Recitals: Each individual Recital, for example Recital 14, is a single indivisible node.
- Article 3 Definitions: Each individual numbered definition is its own node.
- Articles: Split by numbered paragraph. If a paragraph has a list of independent items
  that operate as alternatives, the OR rule applies: split those list items into separate
  nodes and prepend the paragraph's introductory text to each node's atomic_content.
  If list items are cumulative conditions, the AND rule applies: keep the paragraph and
  the list unified in one node.
- Annexes: Split down to the lowest-level lettered or numbered point, for example
  Annex III, Point 1(a).

Extraction rules:
- Extract nodes only from CURRENT BATCH TEXT. CONTEXT BEFORE and CONTEXT AFTER are
  supplied only to understand boundaries; do not create nodes from context-only text.
- atomic_content must be exact, verbatim legal text copied from the current batch. Do
  not summarise, paraphrase, modernise punctuation, or invent missing text.
- Do not create nodes for source headers, browser footers, page numbers, or the bracketed
  segment markers inserted by this script.
- node_id must be strict uppercase snake case beginning with EU_AI_ACT_.
  Examples: EU_AI_ACT_RECITAL_14, EU_AI_ACT_ART_5_1_A,
  EU_AI_ACT_ART_3_1, EU_AI_ACT_ANNEX_III_1_A.
- hierarchy must track parent headers such as Recitals, Chapter, Section, Article, Annex,
  and point labels where applicable.
- legal_weight is Interpretive for recitals, Binding for binding paragraphs, and
  Binding_List for standalone list items that are split as independent binding nodes.
- exceptions must contain explicit exemptions, carve-outs, derogations, or "shall not
  apply" clauses copied as separate strings.
- internal_references must contain structural references within this Act, normalized to
  node-like IDs where possible, for example EU_AI_ACT_ART_5 or EU_AI_ACT_ANNEX_III.
- external_legislation must list adjacent frameworks mentioned, for example
  Regulation (EU) 2016/679 (GDPR), Directive (EU) 2016/680, or Regulation (EU) 2019/1020.

Batch number: {batch.batch_index + 1}
Current structural context:
{json.dumps(batch.structural_context, ensure_ascii=False, indent=2)}

CONTEXT BEFORE (do not extract nodes from this block):
<<<CONTEXT_BEFORE
{batch.context_before}
CONTEXT_BEFORE>>>

CURRENT BATCH TEXT (extract only from this block):
<<<CURRENT_BATCH
{batch.text}
CURRENT_BATCH>>>

CONTEXT AFTER (do not extract nodes from this block):
<<<CONTEXT_AFTER
{batch.context_after}
CONTEXT_AFTER>>>
""".strip()


def deduplicate_nodes(nodes: Iterable[AtomicLegalNode]) -> list[AtomicLegalNode]:
    """Remove duplicate LLM nodes caused by neighboring batch context."""

    deduped: list[AtomicLegalNode] = []
    seen: dict[str, AtomicLegalNode] = {}
    for node in nodes:
        existing = seen.get(node.node_id)
        if existing is None:
            # First occurrence wins; later duplicates are usually overlap echoes.
            seen[node.node_id] = node
            deduped.append(node)
            continue
        if existing.atomic_content != node.atomic_content:
            # Conflicting duplicates are logged for human review instead of
            # merging possibly incompatible legal text.
            logging.warning(
                "Duplicate node_id with conflicting content skipped: %s",
                node.node_id,
            )
    return deduped


if __name__ == "__main__":
    sys.exit(main())
