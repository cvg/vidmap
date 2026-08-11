"""Export COLMAP reconstructions as interactive Three.js HTML."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .interactive_html import InteractiveHtmlExporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vidmap.visualization.html",
        allow_abbrev=False,
        description="Export a COLMAP reconstruction as interactive Three.js HTML.",
    )
    parser.add_argument(
        "--rec",
        type=Path,
        required=True,
        help="COLMAP reconstruction directory.",
    )
    parser.add_argument(
        "--ground-truth-rec",
        type=Path,
        help="Optional COLMAP ground-truth reconstruction directory to align and display.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="Optional COLMAP database used to display valid loop-closure pairs.",
    )
    parser.add_argument(
        "--include-cameras",
        action="store_true",
        help="Embed estimated and ground-truth camera frusta (disabled by default).",
    )
    parser.add_argument(
        "--point-covariance-percentile",
        type=float,
        metavar="PERCENT",
        help="Keep the requested percentage of 3D points with lowest trace covariance.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path (default: <rec>/3d.html).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exporter = InteractiveHtmlExporter.from_paths(
        args.rec,
        ground_truth=args.ground_truth_rec,
        database=args.database,
        include_cameras=args.include_cameras,
        point_covariance_percentile=args.point_covariance_percentile,
    )
    output = args.rec / "3d.html" if args.output is None else args.output
    exporter.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
