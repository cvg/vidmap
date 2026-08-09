"""Canonical mapping-only module entrypoint."""

import logging
from argparse import ArgumentParser
from pathlib import Path

from vidmap.configuration.names import DEFAULT_FRONTEND_CONFIG, DEFAULT_MAPPING_CONFIG

logger = logging.getLogger("vidmap.cli.map")


def _split_stage_overrides(override_tokens):
    """Route frontend-qualified overrides and unqualified mapping overrides."""

    frontend = []
    mapping = []
    for token in override_tokens:
        if token.startswith("frontend."):
            frontend.append(token.removeprefix("frontend."))
        elif token.startswith("mapping."):
            mapping.append(token.removeprefix("mapping."))
        else:
            mapping.append(token)
    return tuple(frontend), tuple(mapping)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        allow_abbrev=False,
        description="Map one explicit finalized mapper-input boundary.",
    )
    parser.add_argument("--mapper-inputs", dest="mapper_inputs", required=True)
    parser.add_argument("--frontend-conf", default=DEFAULT_FRONTEND_CONFIG)
    parser.add_argument("-c", "--mapping-conf", default=DEFAULT_MAPPING_CONFIG)
    parser.add_argument(
        "--output",
        required=True,
        help="Run directory that receives resolved configs and reconstruction outputs.",
    )
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("--name", type=str)
    from vidmap.run_options import add_run_arguments

    add_run_arguments(parser, mapping=True)
    return parser


def _run_local(mapping_conf, frontend_conf, args, run_options) -> int:
    from vidmap.reconstruction import extract_point_colors, propagate_local_input_provenance, run_mapping

    output_dir = Path(args.output).expanduser()
    reconstruction = run_mapping(
        mapping_conf,
        frontend_conf=frontend_conf,
        mapper_inputs=args.mapper_inputs,
        run_options=run_options,
        scene_parser=None,
        overwrite_outputs=args.overwrite,
        output_dir=output_dir,
    )
    extract_point_colors(reconstruction, args.mapper_inputs)
    reconstruction_dir = output_dir / "rec"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(reconstruction_dir)
    propagate_local_input_provenance(args.mapper_inputs, output_dir, overwrite=args.overwrite)
    logger.info("Reconstruction written to %s", reconstruction_dir)
    return 0


def main(argv=None):
    from vidmap.configuration.build import (
        build_frontend_config,
        build_mapping_config,
        parse_config_args,
        reject_routing_overrides,
    )
    from vidmap.configuration.names import FRONTEND_CONFIG_DIR, MAPPING_CONFIG_DIR, resolve_config_path

    parser = build_parser()
    args, overrides = parse_config_args(parser, argv)
    try:
        reject_routing_overrides(overrides, {"name": "--name"})
    except ValueError as error:
        parser.error(str(error))

    from vidmap.run_options import RunOptions
    from vidmap.utils.logging import configure_logging

    run_options = RunOptions.from_namespace(args)
    configure_logging(run_options.verbosity)
    frontend_overrides, mapping_overrides = _split_stage_overrides(overrides)
    frontend_conf = build_frontend_config(
        resolve_config_path(args.frontend_conf, FRONTEND_CONFIG_DIR),
        source_name=args.frontend_conf,
        override_tokens=frontend_overrides,
    )
    mapping_conf = build_mapping_config(
        resolve_config_path(args.mapping_conf, MAPPING_CONFIG_DIR),
        source_name=args.mapping_conf,
        name=args.name,
        override_tokens=mapping_overrides,
    )
    from vidmap.mapper.runtime import load_mapping_runtime

    load_mapping_runtime()
    return _run_local(mapping_conf, frontend_conf, args, run_options)


if __name__ == "__main__":
    raise SystemExit(main())
