"""Persist the standard benchmark trajectory comparison plot."""

from pathlib import Path

from evo.tools import plot
from matplotlib.figure import Figure


def write_trajectory_artifacts(output_dir, estimated_trajectory, ground_truth_trajectory) -> None:
    """Write an XY comparison PDF for aligned estimate and ground truth."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(8, 6))
    axis = plot.prepare_axis(figure, plot.PlotMode.xy)
    plot.traj(axis, plot.PlotMode.xy, ground_truth_trajectory, color="black", label="gt")
    plot.traj(axis, plot.PlotMode.xy, estimated_trajectory, color="tab:blue", label="estimate")
    axis.legend()
    figure.savefig(output_dir / "traj.pdf", bbox_inches="tight")
