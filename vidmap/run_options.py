"""CLI-only execution and presentation controls."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass


@dataclass(frozen=True)
class RunOptions:
    """Controls for one process invocation, never reconstruction config."""

    verbosity: int = 1
    terminate_on_error: bool = False
    save_3d_html: bool = False
    save_playback_trace: bool = False
    playback_trace_stride: int = 3
    playback_trace_point_cap: int | None = None
    html_point_covariance_percentile: float | None = None
    profile: bool = False

    def __post_init__(self) -> None:
        if self.verbosity < 0:
            raise ValueError("verbosity must be nonnegative")
        if (
            not isinstance(self.playback_trace_stride, int)
            or isinstance(self.playback_trace_stride, bool)
            or self.playback_trace_stride < 1
        ):
            raise ValueError("playback trace stride must be a positive integer")
        if self.playback_trace_point_cap is not None and (
            not isinstance(self.playback_trace_point_cap, int)
            or isinstance(self.playback_trace_point_cap, bool)
            or self.playback_trace_point_cap < 1
        ):
            raise ValueError("playback trace point cap must be a positive integer")
        if self.playback_trace_point_cap is not None and self.playback_trace_point_cap > 200000:
            raise ValueError("playback trace point cap cannot exceed 200000")
        if self.html_point_covariance_percentile is not None:
            if not self.save_3d_html:
                raise ValueError("HTML point covariance percentile requires save_3d_html")
            if not 0 < self.html_point_covariance_percentile <= 100:
                raise ValueError("HTML point covariance percentile must be in (0, 100]")

    @classmethod
    def from_namespace(cls, args: Namespace) -> RunOptions:
        """Build options from the shared CLI destination names."""
        return cls(
            verbosity=args.verbose,
            terminate_on_error=args.terminate,
            save_3d_html=args.save_3d_html,
            save_playback_trace=args.save_playback_trace,
            playback_trace_stride=args.playback_trace_stride,
            playback_trace_point_cap=args.playback_trace_point_cap,
            html_point_covariance_percentile=args.html_point_covariance_percentile,
            profile=args.profile,
        )


def add_run_arguments(parser, *, mapping: bool, profiling: bool = False) -> None:
    """Add the common process-only arguments to an argparse parser."""
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
        metavar="LEVEL",
        help="Runtime output level: 0 warnings/errors, 1 lifecycle/stages, 2 debug.",
    )
    parser.add_argument("-t", "--terminate", action="store_true", help="Raise on the first failed case.")
    parser.set_defaults(
        save_3d_html=False,
        save_playback_trace=False,
        playback_trace_stride=3,
        playback_trace_point_cap=None,
        html_point_covariance_percentile=None,
        profile=False,
    )
    if mapping:
        parser.add_argument(
            "--save-3d-html",
            action="store_true",
            help="Write an interactive 3D reconstruction.",
        )
        parser.add_argument(
            "--html-point-covariance-percentile",
            type=float,
            default=None,
            metavar="PERCENT",
            help="Keep the requested percentage of HTML points with lowest trace covariance.",
        )
        parser.add_argument(
            "--save-playback-trace",
            action="store_true",
            help="Capture compact GP/BA playback data.",
        )
        parser.add_argument(
            "--playback-trace-stride",
            type=int,
            default=3,
            metavar="N",
            help="Keep every Nth solver iteration in playback traces (default: 3; use 1 for full fidelity).",
        )
        parser.add_argument(
            "--playback-trace-point-cap",
            type=int,
            default=None,
            metavar="N",
            help="Deterministically retain at most N playback points.",
        )
    if profiling:
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Profile exactly one selected case and write stage timing and sampled RSS results.",
        )
