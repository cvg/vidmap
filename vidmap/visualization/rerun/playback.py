"""Record VidMap solver optimization playback."""

from vidmap.visualization.playback.cli import main_command


def main() -> int:
    return main_command("playback")


if __name__ == "__main__":
    raise SystemExit(main())
