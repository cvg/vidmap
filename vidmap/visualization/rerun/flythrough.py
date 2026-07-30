"""Record or render a VidMap reconstruction flythrough."""

from vidmap.visualization.playback.cli import main_command


def main() -> int:
    return main_command("flythrough")


if __name__ == "__main__":
    raise SystemExit(main())
