from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .api import DEFAULT_RESOLUTION, Playback, Resolution
from .scene import DEPTH_LIFT_RADIUS_UI_POINTS

FIXED_VIEWS = ("topdown", "side", "isometric")
FLYTHROUGH_VIEWS = ("follow", *FIXED_VIEWS)
COMMANDS = ("playback", "final", "flythrough")
logger = logging.getLogger(__name__)


def parse_command(command: str, argv: list[str] | None = None) -> argparse.Namespace:
    if command not in COMMANDS:
        raise ValueError(f"Unknown Rerun command: {command}")
    parser = argparse.ArgumentParser(
        prog=f"python -m vidmap.visualization.rerun.{command}",
        description={
            "playback": "Record the GP and optional BA solver timeline.",
            "final": "Record the final solver state as a one-second hold.",
            "flythrough": "Record a reconstruction flythrough as RRD.",
        }[command],
    )
    _configure_command(parser, command)
    args = parser.parse_args(argv)
    args.command = command
    _validate_args(parser, args)
    return args


def _configure_command(parser: argparse.ArgumentParser, command: str) -> None:
    if command in {"playback", "final"}:
        _source_arguments(
            parser,
            views=FIXED_VIEWS,
            gp_only=True,
            default_view="topdown",
        )
        return
    if command != "flythrough":
        raise ValueError(f"Unknown Rerun command: {command}")

    _source_arguments(
        parser,
        views=FLYTHROUGH_VIEWS,
        gp_only=False,
        default_view="follow",
    )
    reconstruction = parser.add_argument_group("reconstruction and images")
    reconstruction.add_argument(
        "--reconstruction",
        default="final",
        type=_reconstruction,
        metavar="{final,ba,gp}|PATH",
        help="Reconstruction to visualize (default: final).",
    )
    reconstruction.add_argument(
        "--images",
        type=Path,
        metavar="DIR",
        help="Override automatic RGB image discovery.",
    )
    reconstruction.add_argument(
        "--ground-truth",
        type=Path,
        metavar="MODEL",
        help="Override automatic ground-truth reconstruction discovery.",
    )
    reconstruction.add_argument(
        "--sparse-point-mode",
        choices=("progressive", "static"),
        default="static",
        help=(
            "Sparse reconstruction point display: progressively reveal color batches, "
            "or log one static neutral-color entity. Default: static."
        ),
    )
    reconstruction.add_argument(
        "--sparse-point-covariance-percentile",
        type=_positive_float,
        default=90.0,
        metavar="PERCENT",
        help="Keep the requested percentage of sparse points with the lowest trace covariance (default: 90).",
    )

    depth = parser.add_argument_group("depth lift")
    depth.add_argument(
        "--depth-lift",
        action="store_true",
        help="Replace the current sparse highlight with the RGB-colored full depth map.",
    )
    depth.add_argument(
        "--depth-lift-keep",
        action="store_true",
        help="Lift each full depth map and keep earlier clouds visible cumulatively.",
    )
    depth.add_argument(
        "--depth-lift-refit-scale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Refit each raw-depth multiplier as median reconstructed camera depth / raw depth (default: enabled).",
    )
    depth.add_argument(
        "--depth-maps",
        type=Path,
        metavar="H5",
        help="Full depth-map H5 override; normally inferred from the run.",
    )
    depth.add_argument(
        "--depth-lift-stride",
        type=_positive_int,
        default=1,
        metavar="N",
        help="Use every Nth prediction-grid depth sample (default: 1).",
    )
    depth.add_argument(
        "--depth-lift-max",
        type=_positive_float,
        default=20.0,
        metavar="DEPTH",
        help="Maximum reconstruction-aligned lifted depth (default: 20).",
    )
    depth.add_argument(
        "--depth-lift-point-radius",
        type=_positive_float,
        default=DEPTH_LIFT_RADIUS_UI_POINTS,
        metavar="UI_POINTS",
        help=f"Lifted-depth point radius in UI units (default: {DEPTH_LIFT_RADIUS_UI_POINTS:g}).",
    )

    advanced = parser.add_argument_group("advanced flythrough controls")
    advanced.add_argument(
        "--duration-fraction",
        type=_positive_float,
        default=1.0,
        metavar="FRACTION",
        help=(
            "Emit only the first fraction of flythrough frames while retaining "
            "the complete reconstruction geometry and paths (default: 1)."
        ),
    )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    output_suffix = args.output.suffix.lower()
    if output_suffix != ".rrd":
        parser.error(f"{args.command} --output must end in .rrd, got {args.output}")


def _source_arguments(
    parser: argparse.ArgumentParser,
    *,
    views: tuple[str, ...],
    gp_only: bool,
    default_view: str,
) -> None:
    source = parser.add_argument_group("input")
    source.add_argument(
        "source",
        metavar="RUN",
        type=Path,
        help="Run directory or compact playback_trace.",
    )
    output = parser.add_argument_group("output")
    output.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="RRD",
        help="Output .rrd recording.",
    )
    view = parser.add_argument_group("view")
    view.add_argument(
        "--view",
        choices=views,
        default=default_view,
        help=f"Camera view. Default: {default_view}.",
    )
    view.add_argument(
        "--theme",
        choices=("dark", "light", "neon"),
        default="dark",
        help="Semantic color theme. Default: dark.",
    )
    if gp_only:
        view.add_argument("--gp-only", action="store_true", help="Record GP without BA.")
    view.add_argument(
        "--no-gt",
        action="store_true",
        help="Do not align to or draw ground truth; use for reconstruction-only sequences.",
    )
    _output_arguments(output)


def _output_arguments(parser) -> None:
    parser.add_argument(
        "--resolution",
        type=_resolution,
        default=DEFAULT_RESOLUTION,
        metavar="WIDTH,HEIGHT",
        help="Spatial view size (default: 1280,720).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")


def main_command(command: str, argv: list[str] | None = None) -> int:
    return _main(lambda: parse_command(command, argv), f"rerun.{command}")


def _main(parse, error_prefix: str) -> int:
    try:
        from vidmap.utils.logging import configure_logging

        configure_logging(1)
        args = parse()
        if args.command == "flythrough":
            playback = Playback(
                args.source,
                mode=args.command,
                view=args.view,
                theme=args.theme,
                reconstruction=args.reconstruction,
                align_to_ground_truth=not args.no_gt,
                ground_truth=args.ground_truth,
                image_dir=args.images,
                sparse_point_mode=args.sparse_point_mode,
                sparse_point_covariance_percentile=args.sparse_point_covariance_percentile,
                depth_lift=args.depth_lift,
                depth_lift_keep=args.depth_lift_keep,
                depth_lift_refit_scale=args.depth_lift_refit_scale,
                depth_maps=args.depth_maps,
                depth_lift_stride=args.depth_lift_stride,
                depth_lift_max=args.depth_lift_max,
                depth_lift_point_radius=args.depth_lift_point_radius,
                duration_fraction=args.duration_fraction,
            )
            logger.info("Creating flythrough RRD: source=%s output=%s", args.source, args.output)
        else:
            logger.info("Creating %s RRD: source=%s output=%s", args.command, args.source, args.output)
            playback = Playback(
                args.source,
                gp_only=args.gp_only if args.command == "playback" else False,
                mode=args.command,
                view=args.view,
                theme=args.theme,
                align_to_ground_truth=not args.no_gt,
            )
        playback.save_rrd(args.output, resolution=args.resolution, overwrite=args.overwrite)
        logger.info("RRD written: %s", args.output)
        return 0
    except (
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        TypeError,
        ValueError,
        ModuleNotFoundError,
    ) as exc:
        raise SystemExit(f"{error_prefix}: error: {exc}") from None


def _resolution(value: str) -> Resolution:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected WIDTH,HEIGHT, got {value!r}")
    try:
        resolution = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected WIDTH,HEIGHT, got {value!r}") from exc
    if resolution[0] <= 0 or resolution[1] <= 0:
        raise argparse.ArgumentTypeError(f"expected positive WIDTH,HEIGHT, got {value!r}")
    return resolution


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return result


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}") from exc
    if result <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}")
    return result


def _reconstruction(value: str) -> str | Path:
    return value if value in {"final", "ba", "gp"} else Path(value)
