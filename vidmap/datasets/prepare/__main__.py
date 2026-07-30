"""Dispatch dataset-specific download and preparation commands."""

from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Sequence

from vidmap.datasets.names import get_dataset_definition, supported_preparers


def _module_name(dataset: str) -> str:
    module = get_dataset_definition(dataset).preparer_module
    if module is None:
        raise ValueError(f"Dataset {dataset!r} does not provide automatic preparation")
    return module


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and prepare a supported benchmark dataset.",
        add_help=add_help,
        allow_abbrev=False,
    )
    parser.add_argument("-d", "--dataset", required=True, choices=supported_preparers())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    selects_dataset = any(token in {"-d", "--dataset"} or token.startswith("--dataset=") for token in tokens)
    if not tokens or (not selects_dataset and any(token in {"-h", "--help"} for token in tokens)):
        build_parser().print_help()
        return 0

    args, forwarded = build_parser(add_help=False).parse_known_args(tokens)
    module = _module_name(args.dataset)
    previous_argv = sys.argv
    try:
        sys.argv = [module, *forwarded]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = previous_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
