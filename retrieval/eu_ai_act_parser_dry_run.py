"""
Inspect EU AI Act PDF segmentation without calling Gemini.

This file is intentionally separate from eu_ai_act_parser.py. The main parser
executes the real Gemini extraction pipeline only; this helper is for checking
how the PDF will be batched before running the expensive extraction.

Example:

    python retrieval/eu_ai_act_parser_dry_run.py --limit-batches 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from eu_ai_act_parser import (
    DEFAULT_MAX_BATCH_CHARS,
    DEFAULT_OVERLAP_CHARS,
    TARGET_PDF_BASENAME,
    Segment,
    TextBatch,
    build_text_batches,
    extract_pdf_lines,
    split_oversized_segments,
    split_structural_segments,
    validate_pdf_path,
)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    try:
        pdf_path = Path(args.pdf).expanduser().resolve()
        validate_pdf_path(pdf_path)

        page_lines = extract_pdf_lines(pdf_path)
        segments = split_structural_segments(page_lines)
        split_segments = split_oversized_segments(
            segments=segments,
            max_chars=args.max_batch_chars,
        )
        batches = build_text_batches(
            split_segments,
            max_chars=args.max_batch_chars,
            overlap_chars=args.overlap_chars,
        )
        if args.limit_batches:
            batches = batches[: args.limit_batches]

        print_segmentation_summary(segments=split_segments, batches=batches)
        return 0
    except Exception as exc:  # noqa: BLE001 - command line should fail cleanly.
        logging.error("%s", exc)
        return 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect EU AI Act parser segmentation without Gemini calls.",
    )
    parser.add_argument(
        "--pdf",
        default=str(Path("docs") / TARGET_PDF_BASENAME),
        help="Path to the official EU AI Act EUR-Lex PDF.",
    )
    parser.add_argument(
        "--max-batch-chars",
        type=int,
        default=DEFAULT_MAX_BATCH_CHARS,
        help="Maximum current-batch character budget used by the real parser.",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=DEFAULT_OVERLAP_CHARS,
        help="Neighboring context characters used by the real parser.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="Show only the first N batches in the diagnostic summary.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )
    return parser


def print_segmentation_summary(
    segments: list[Segment],
    batches: list[TextBatch],
) -> None:
    by_kind: dict[str, int] = {}
    for segment in segments:
        by_kind[segment.kind] = by_kind.get(segment.kind, 0) + 1

    print(json.dumps(
        {
            "segments": len(segments),
            "segments_by_kind": by_kind,
            "batches": len(batches),
            "first_batches": [
                {
                    "batch_index": batch.batch_index + 1,
                    "segments": len(batch.segments),
                    "chars": len(batch.text),
                    "titles": [segment.title for segment in batch.segments[:8]],
                }
                for batch in batches[:5]
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    sys.exit(main())
