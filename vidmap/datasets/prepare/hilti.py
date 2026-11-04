"""Download and prepare the Hilti 360-degree benchmark."""

from vidmap.datasets.prepare.helpers.ros_benchmarks import main as prepare_dataset
from vidmap.datasets.prepare.helpers.ros_benchmarks import run_cli


def main(scenes=None, *, mode="panorama", download=True, delete_files=False, workers=1):
    prepare_dataset("hilti", scenes, mode=mode, download=download, delete_files=delete_files, workers=workers)


if __name__ == "__main__":
    run_cli("hilti")
