"""CLI-only execution and presentation controls."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass


@dataclass(frozen=True)
class RunOptions:
    """Controls for one process invocation, never reconstruction config."""

    verbosity: int = 1
    terminate_on_error: bool = False
    save_playback_trace: bool = False
    playback_trace_stride: int = 3
    playback_trace_point_cap: int | None = None
    profile: bool = False
    device: str = "auto"
    split_cuts: bool = False
    min_covisibility_points: int = 20
    min_model_size: int = 5

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
        if (
            not isinstance(self.min_covisibility_points, int)
            or isinstance(self.min_covisibility_points, bool)
            or self.min_covisibility_points < 1
        ):
            raise ValueError("min_covisibility_points must be a positive integer")
        if (
            not isinstance(self.min_model_size, int)
            or isinstance(self.min_model_size, bool)
            or self.min_model_size < 1
        ):
            raise ValueError("min_model_size must be a positive integer")

    @classmethod
    def from_namespace(cls, args: Namespace) -> RunOptions:
        """Build options from the shared CLI destination names."""
        return cls(
            verbosity=args.verbose,
            terminate_on_error=args.terminate,
            save_playback_trace=args.save_playback_trace,
            playback_trace_stride=args.playback_trace_stride,
            playback_trace_point_cap=args.playback_trace_point_cap,
            profile=args.profile,
            device=getattr(args, "device", "auto"),
            split_cuts=getattr(args, "split_cuts", False),
            min_covisibility_points=getattr(args, "min_covisibility_points", 20),
            min_model_size=getattr(args, "min_model_size", 5),
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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda", "mps"],
        help="Compute device for neural networks (default: auto; choices: auto, cpu, gpu, cuda, mps).",
    )
    parser.set_defaults(
        save_playback_trace=False,
        playback_trace_stride=3,
        playback_trace_point_cap=None,
        profile=False,
        device="auto",
        split_cuts=False,
        min_covisibility_points=20,
        min_model_size=5,
    )
    if mapping:
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
        parser.add_argument(
            "--split-cuts",
            action="store_true",
            help="Decompose reconstruction into multiple sub-models if video cuts or disconnected covisibility components exist.",
        )
        parser.add_argument(
            "--min-covisibility-points",
            type=int,
            default=20,
            metavar="N",
            help="Minimum shared 3D points between frames to keep a covisibility edge when detecting cuts (default: 20).",
        )
        parser.add_argument(
            "--min-model-size",
            type=int,
            default=5,
            metavar="N",
            help="Minimum number of registered images required to keep a sub-reconstruction (default: 5).",
        )
    if profiling:
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Profile exactly one selected case and write stage timing and sampled RSS results.",
        )
