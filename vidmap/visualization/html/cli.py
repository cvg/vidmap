"""Command line interface for the browser reconstruction viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .exporter import write_viewer_html


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vidmap.visualization.html",
        allow_abbrev=False,
        description="Write the VidMap browser reconstruction viewer.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory to package into an embedded viewer.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        help="Optional image root for embedded keyframe previews (requires --run-dir).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path (defaults to <run>/vidmap-viewer-embedded.html with --run-dir).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.images is not None and args.run_dir is None:
        parser.error("--images requires --run-dir")
    if args.run_dir is None and args.output is None:
        parser.error("--output is required unless --run-dir is provided")
    if args.run_dir is not None:
        from .embedded import write_embedded_viewer_html

        output = args.run_dir / "vidmap-viewer-embedded.html" if args.output is None else args.output
        written = write_embedded_viewer_html(args.run_dir, output, images_dir=args.images)
        print(written.resolve())
        return 0
    written = write_viewer_html(args.output)
    print(written.resolve())
    return 0
