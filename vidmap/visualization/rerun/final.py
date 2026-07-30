"""Record the final VidMap solver state."""

from vidmap.visualization.playback.cli import main_command


def main() -> int:
    return main_command("final")


if __name__ == "__main__":
    raise SystemExit(main())
