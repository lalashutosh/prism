"""
Parse the EU AI Act prohibited-practices guidelines into atomic legal nodes.

This script is intentionally layout-aware and section-boundary driven. It does
not use token chunking, character chunking, or recursive text splitters.

Default run from the repository root:

    python retrieval/parse_eu_ai_guidelines.py ^
        "docs/EU AI Act Prohibited AI Practices Guidelines.PDF" ^
        --output retrieval/eu_ai_act_prohibited_practices_atomic_nodes.json

Dependencies:
    Preferred: unstructured[pdf]
    Fallback: pdfplumber, pdfminer.six
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal


ElementCategory = Literal[
    "Header_1",
    "Header_2",
    "Header_3",
    "List_Item",
    "Standard_Text",
]

TARGET_PDF_BASENAME = "EU AI Act Prohibited AI Practices Guidelines.PDF"
DOCUMENT_SOURCE = "EU_AI_Act_Guidelines_Prohibited_Practices.pdf"
NODE_LABEL = "Guideline_Rule"

# The target PDF has a title page and four contents pages before the body.
DEFAULT_SKIP_FRONT_MATTER_PAGES = 5

HEADER_TO_LEVEL = {
    "Header_1": 1,
    "Header_2": 2,
    "Header_3": 3,
}

NUMBERED_HEADING_RE = re.compile(
    r"^(?P<number>\d+(?:\.\d+)*)(?:\.)\s+(?P<title>\S.*)$"
)
LEGAL_PARAGRAPH_RE = re.compile(r"^\(\d+\)\s+")
LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*•]\s+|\((?:[ivxlcdm]+|[a-z])\)\s+)", re.IGNORECASE
)
PAGE_NUMBER_RE = re.compile(r"^(?:\d+|[IVXLCDM]+)$", re.IGNORECASE)


@dataclass
class LegalElement:
    """A layout-classified document element in original reading order."""

    category: ElementCategory
    text: str
    page_number: int | None = None
    source_parser: str = ""


@dataclass
class PdfLine:
    """A physical PDF line with enough layout data for heading detection."""

    text: str
    page_number: int
    x0: float
    top: float
    bottom: float
    avg_size: float
    bold_ratio: float
    category: ElementCategory
    is_heading_continuation: bool = False


@dataclass
class ASTNode:
    """Hierarchical legal AST node.

    Header_2 and Header_3 nodes are the atomic-node bucket boundaries. Text and
    list elements are attached to the deepest open bucket, so exceptions and
    lists stay with their parent legal section.
    """

    title: str
    level: int
    elements: list[LegalElement] = field(default_factory=list)
    children: list["ASTNode"] = field(default_factory=list)


@dataclass
class TextBlockState:
    """Mutable state used while stitching physical PDF lines into blocks."""

    category: ElementCategory
    text: str
    page_number: int
    last_line: PdfLine


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    try:
        pdf_path = Path(args.pdf_path).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve()

        validate_target_pdf(pdf_path)
        elements = extract_layout_elements(
            pdf_path=pdf_path,
            parser=args.parser,
            skip_front_matter_pages=args.skip_front_matter_pages,
        )
        if not elements:
            raise RuntimeError("No parseable layout elements were extracted.")

        ast = build_ast(elements)
        atomic_nodes = compile_atomic_nodes(ast)
        if not atomic_nodes:
            raise RuntimeError(
                "No atomic legal nodes were produced. Check heading detection."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(atomic_nodes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logging.info("Wrote %s atomic legal nodes to %s", len(atomic_nodes), output_path)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should report cleanly.
        logging.error("%s", exc)
        return 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse the EU AI Act Prohibited Practices Guidelines PDF into a "
            "single JSON array of atomic legal nodes."
        ),
        epilog=(
            "Example: python retrieval/parse_eu_ai_guidelines.py "
            "\"docs/EU AI Act Prohibited AI Practices Guidelines.PDF\" "
            "--output retrieval/eu_ai_act_prohibited_practices_atomic_nodes.json"
        ),
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default=str(Path("docs") / TARGET_PDF_BASENAME),
        help="Path to the target PDF. Other PDF basenames are rejected.",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).with_name(
                "eu_ai_act_prohibited_practices_atomic_nodes.json"
            )
        ),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--parser",
        choices=("auto", "unstructured", "pdfplumber"),
        default="auto",
        help="Layout parser to use. 'auto' tries unstructured, then pdfplumber.",
    )
    parser.add_argument(
        "--skip-front-matter-pages",
        type=int,
        default=DEFAULT_SKIP_FRONT_MATTER_PAGES,
        help=(
            "Number of physical PDF pages to skip before parsing. The target "
            "document's legal body starts after five front-matter pages."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def validate_target_pdf(pdf_path: Path) -> None:
    """Ensure the script is used only for the requested source document."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"PDF path is not a file: {pdf_path}")
    if pdf_path.name.casefold() != TARGET_PDF_BASENAME.casefold():
        raise ValueError(
            "This parser is locked to "
            f"{TARGET_PDF_BASENAME!r}; got {pdf_path.name!r}."
        )


def extract_layout_elements(
    pdf_path: Path,
    parser: Literal["auto", "unstructured", "pdfplumber"],
    skip_front_matter_pages: int,
) -> list[LegalElement]:
    """Extract and classify legal elements with a layout-aware parser."""

    if parser in {"auto", "unstructured"}:
        try:
            return extract_with_unstructured(pdf_path, skip_front_matter_pages)
        except ImportError as exc:
            if parser == "unstructured":
                raise RuntimeError(
                    "unstructured is not installed. Install unstructured[pdf] "
                    "or run with --parser pdfplumber."
                ) from exc
            logging.info("unstructured unavailable; falling back to pdfplumber")
        except Exception as exc:  # noqa: BLE001 - fallback path is intentional.
            if parser == "unstructured":
                raise RuntimeError(f"unstructured parsing failed: {exc}") from exc
            logging.warning(
                "unstructured parsing failed (%s); falling back to pdfplumber",
                exc,
            )

    return extract_with_pdfplumber(pdf_path, skip_front_matter_pages)


def extract_with_unstructured(
    pdf_path: Path,
    skip_front_matter_pages: int,
) -> list[LegalElement]:
    """Preferred extractor using unstructured.io's PDF partitioner."""

    from unstructured.partition.pdf import partition_pdf

    raw_elements = partition_pdf(
        filename=str(pdf_path),
        infer_table_structure=False,
        strategy="hi_res",
    )

    elements: list[LegalElement] = []
    for raw in raw_elements:
        text = clean_text(getattr(raw, "text", ""))
        if not text:
            continue

        metadata = getattr(raw, "metadata", None)
        page_number = getattr(metadata, "page_number", None)
        if page_number and page_number <= skip_front_matter_pages:
            continue

        category = map_unstructured_category(
            category_name=getattr(raw, "category", ""),
            text=text,
        )
        if category is None:
            continue
        elements.append(
            LegalElement(
                category=category,
                text=text,
                page_number=page_number,
                source_parser="unstructured",
            )
        )

    return discard_until_first_header_1(elements)


def map_unstructured_category(
    category_name: str,
    text: str,
) -> ElementCategory | None:
    """Map unstructured element categories onto the required legal categories."""

    if category_name in {"Header", "Footer", "PageBreak"}:
        return None
    if is_noise_line(text):
        return None
    if category_name == "ListItem" or is_list_item(text):
        return "List_Item"
    if category_name == "Title":
        heading_category = classify_numbered_heading(text, require_numbering=True)
        return heading_category or "Standard_Text"
    return "Standard_Text"


def extract_with_pdfplumber(
    pdf_path: Path,
    skip_front_matter_pages: int,
) -> list[LegalElement]:
    """Fallback extractor using pdfplumber line geometry and font metadata."""

    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is not installed. Install pdfplumber/pdfminer.six "
            "or install unstructured[pdf]."
        ) from exc

    lines: list[PdfLine] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            if page_index <= skip_front_matter_pages:
                continue

            extracted = page.extract_text_lines(
                layout=True,
                strip=True,
                return_chars=True,
            )
            for line in extracted or []:
                parsed = parse_pdf_line(line, page_index)
                if parsed is not None:
                    lines.append(parsed)

    mark_heading_continuations(lines)
    return stitch_pdf_lines(lines)


def parse_pdf_line(raw_line: dict, page_number: int) -> PdfLine | None:
    """Convert one pdfplumber line dict into a classified PdfLine."""

    text = clean_text(raw_line.get("text", ""))
    if not text or is_noise_line(text):
        return None

    chars = [
        char
        for char in raw_line.get("chars", [])
        if clean_text(char.get("text", ""))
    ]
    if not chars:
        return None

    sizes = [float(char.get("size") or 0) for char in chars]
    avg_size = sum(sizes) / len(sizes)
    bold_count = sum(
        1
        for char in chars
        if "bold" in str(char.get("fontname", "")).casefold()
    )
    bold_ratio = bold_count / len(chars)

    # Footnotes in the target PDF are materially smaller than body text.
    if avg_size < 9.5:
        return None

    category = classify_pdf_text(text=text, bold_ratio=bold_ratio, avg_size=avg_size)
    return PdfLine(
        text=text,
        page_number=page_number,
        x0=float(raw_line.get("x0") or 0),
        top=float(raw_line.get("top") or 0),
        bottom=float(raw_line.get("bottom") or 0),
        avg_size=avg_size,
        bold_ratio=bold_ratio,
        category=category,
    )


def classify_pdf_text(
    text: str,
    bold_ratio: float,
    avg_size: float,
) -> ElementCategory:
    """Classify a physical PDF line into the requested element categories."""

    heading_category = classify_numbered_heading(
        text,
        require_numbering=True,
        require_bold=True,
        bold_ratio=bold_ratio,
        avg_size=avg_size,
    )
    if heading_category:
        return heading_category
    if is_list_item(text):
        return "List_Item"
    return "Standard_Text"


def classify_numbered_heading(
    text: str,
    require_numbering: bool = True,
    require_bold: bool = False,
    bold_ratio: float = 0,
    avg_size: float = 0,
) -> ElementCategory | None:
    """Classify numbered headings such as '2.1.' or '10.4.3.'."""

    if require_bold and not is_bold_enough(bold_ratio=bold_ratio, avg_size=avg_size):
        return None

    match = NUMBERED_HEADING_RE.match(text)
    if require_numbering and not match:
        return None
    if not match:
        return None

    number = match.group("number")
    depth = number.count(".") + 1
    if depth <= 1:
        return "Header_1"
    if depth == 2:
        return "Header_2"
    return "Header_3"


def mark_heading_continuations(lines: list[PdfLine]) -> None:
    """Mark wrapped heading lines so they merge with the prior header."""

    previous_header: PdfLine | None = None
    for line in lines:
        if line.category in HEADER_TO_LEVEL:
            previous_header = line
            continue

        if previous_header is None:
            continue
        if line.page_number != previous_header.page_number:
            previous_header = None
            continue
        if not is_bold_enough(line.bold_ratio, line.avg_size):
            previous_header = None
            continue
        if is_list_item(line.text):
            previous_header = None
            continue
        if starts_new_paragraph(line.text) and not previous_header.text.endswith("-"):
            previous_header = None
            continue

        vertical_gap = line.top - previous_header.bottom
        if vertical_gap <= max(6.0, previous_header.avg_size * 0.8):
            line.category = previous_header.category
            line.is_heading_continuation = True
            previous_header = line
        else:
            previous_header = None


def stitch_pdf_lines(lines: list[PdfLine]) -> list[LegalElement]:
    """Merge physical lines into logical elements without token chunking."""

    elements: list[LegalElement] = []
    block: TextBlockState | None = None

    def flush_block() -> None:
        nonlocal block
        if block is None:
            return
        elements.append(
            LegalElement(
                category=block.category,
                text=block.text,
                page_number=block.page_number,
                source_parser="pdfplumber",
            )
        )
        block = None

    for line in lines:
        if line.category in HEADER_TO_LEVEL:
            flush_block()
            if (
                line.is_heading_continuation
                and elements
                and elements[-1].category == line.category
            ):
                elements[-1].text = join_wrapped_text(elements[-1].text, line.text)
                continue
            elements.append(
                LegalElement(
                    category=line.category,
                    text=line.text,
                    page_number=line.page_number,
                    source_parser="pdfplumber",
                )
            )
            continue

        if line.category not in {"List_Item", "Standard_Text"}:
            logging.warning("Skipping unknown element category: %s", line.category)
            continue

        if block is None or should_start_new_text_block(block, line):
            flush_block()
            block = TextBlockState(
                category=line.category,
                text=line.text,
                page_number=line.page_number,
                last_line=line,
            )
            continue

        block.text = join_wrapped_text(block.text, line.text)
        block.last_line = line

    flush_block()
    return discard_until_first_header_1(elements)


def should_start_new_text_block(block: TextBlockState, line: PdfLine) -> bool:
    """Decide whether a PDF line starts a new paragraph/list block."""

    if line.category == "List_Item":
        return True
    if block.category == "List_Item" and starts_new_paragraph(line.text):
        return True
    if line.page_number != block.last_line.page_number:
        return True
    if starts_new_paragraph(line.text):
        return True

    vertical_gap = line.top - block.last_line.bottom
    return vertical_gap > max(8.0, line.avg_size * 0.95)


def build_ast(elements: Iterable[LegalElement]) -> ASTNode:
    """Build a legal AST using Header_2/Header_3 as node-bucket boundaries."""

    root = ASTNode(title="Document", level=0)
    current_h1: ASTNode | None = None
    current_h2: ASTNode | None = None
    current_h3: ASTNode | None = None

    for element in elements:
        text = clean_text(element.text)
        if not text:
            logging.warning(
                "Skipping malformed empty %s on page %s",
                element.category,
                element.page_number,
            )
            continue

        if element.category == "Header_1":
            current_h1 = ASTNode(title=text, level=1)
            root.children.append(current_h1)
            current_h2 = None
            current_h3 = None
            continue

        if element.category == "Header_2":
            if current_h1 is None:
                logging.warning("Header_2 without Header_1 on page %s", element.page_number)
                current_h1 = ASTNode(title="Missing Header_1", level=1)
                root.children.append(current_h1)
            current_h2 = ASTNode(title=text, level=2)
            current_h1.children.append(current_h2)
            current_h3 = None
            continue

        if element.category == "Header_3":
            if current_h1 is None:
                logging.warning("Header_3 without Header_1 on page %s", element.page_number)
                current_h1 = ASTNode(title="Missing Header_1", level=1)
                root.children.append(current_h1)
            if current_h2 is None:
                logging.warning("Header_3 without Header_2 on page %s", element.page_number)
                current_h2 = ASTNode(title="Missing Header_2", level=2)
                current_h1.children.append(current_h2)
            current_h3 = ASTNode(title=text, level=3)
            current_h2.children.append(current_h3)
            continue

        if element.category not in {"List_Item", "Standard_Text"}:
            logging.warning(
                "Skipping unsupported category %s on page %s",
                element.category,
                element.page_number,
            )
            continue

        bucket = current_h3 or current_h2 or current_h1
        if bucket is None:
            logging.warning(
                "Text before any header skipped on page %s: %.80r",
                element.page_number,
                text,
            )
            continue
        bucket.elements.append(
            LegalElement(
                category=element.category,
                text=text,
                page_number=element.page_number,
                source_parser=element.source_parser,
            )
        )

    return root


def compile_atomic_nodes(ast: ASTNode) -> list[dict[str, object]]:
    """Compile Header_2 and Header_3 AST buckets into the required JSON schema."""

    atomic_nodes: list[dict[str, object]] = []
    used_node_ids: set[str] = set()

    def visit(node: ASTNode, parent_hierarchy: list[str]) -> None:
        hierarchy = parent_hierarchy + ([node.title] if node.level > 0 else [])
        if node.level in {2, 3}:
            atomic_content = combine_bucket_content(node.elements)
            if atomic_content:
                atomic_nodes.append(
                    {
                        "node_id": make_node_id(
                            hierarchy=hierarchy,
                            level=node.level,
                            used_node_ids=used_node_ids,
                        ),
                        "node_label": NODE_LABEL,
                        "hierarchy": hierarchy,
                        "atomic_content": atomic_content,
                        "document_source": DOCUMENT_SOURCE,
                    }
                )
            else:
                logging.warning("Skipping empty bucket: %s", " > ".join(hierarchy))

        for child in node.children:
            visit(child, hierarchy)

    visit(ast, [])
    return atomic_nodes


def combine_bucket_content(elements: Iterable[LegalElement]) -> str:
    """Combine bucket paragraphs and lists without breaking legal exceptions."""

    blocks = [clean_text(element.text) for element in elements if clean_text(element.text)]
    return "\n\n".join(blocks)


def make_node_id(
    hierarchy: list[str],
    level: int,
    used_node_ids: set[str],
) -> str:
    """Create a stable unique node_id from the full heading hierarchy."""

    raw = " / ".join(hierarchy)
    slug = slugify(raw)[:90] or "untitled"
    digest = hashlib.blake2s(raw.encode("utf-8"), digest_size=4).hexdigest()
    base = f"GL_H{level}_{slug}_{digest}"
    candidate = base
    counter = 2
    while candidate in used_node_ids:
        candidate = f"{base}_{counter}"
        counter += 1
    used_node_ids.add(candidate)
    return candidate


def slugify(text: str) -> str:
    """Return an ASCII-ish identifier slug from a heading hierarchy."""

    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def discard_until_first_header_1(elements: list[LegalElement]) -> list[LegalElement]:
    """Drop any remaining title/contents material before the first Header_1."""

    for index, element in enumerate(elements):
        if element.category == "Header_1":
            return elements[index:]
    logging.warning("No Header_1 found; keeping all extracted elements")
    return elements


def clean_text(value: object) -> str:
    """Normalize whitespace and common non-breaking characters."""

    if not isinstance(value, str):
        return ""
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def join_wrapped_text(left: str, right: str) -> str:
    """Join two visual lines into one logical block."""

    left = clean_text(left)
    right = clean_text(right)
    if not left:
        return right
    if not right:
        return left
    if left.endswith("-"):
        if right[:1].islower():
            return f"{left[:-1]}{right}"
        return f"{left}{right}"
    return f"{left} {right}"


def is_bold_enough(bold_ratio: float, avg_size: float) -> bool:
    """Return True for heading-like bold lines in the target PDF."""

    return bold_ratio >= 0.65 and avg_size >= 10.5


def starts_new_paragraph(text: str) -> bool:
    """Detect numbered legal paragraphs such as '(58)'."""

    return bool(LEGAL_PARAGRAPH_RE.match(text))


def is_list_item(text: str) -> bool:
    """Detect actual list bullets/enumerators, excluding legal paragraphs."""

    return bool(LIST_ITEM_RE.match(text)) and not starts_new_paragraph(text)


def is_noise_line(text: str) -> bool:
    """Filter page numbers and known front/back matter noise."""

    cleaned = clean_text(text)
    if not cleaned:
        return True
    if cleaned in {"EN"}:
        return True
    return bool(PAGE_NUMBER_RE.match(cleaned))


if __name__ == "__main__":
    sys.exit(main())
