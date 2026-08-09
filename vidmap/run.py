"""Local-media frontend-to-mapping module entrypoint."""

import logging
from argparse import ArgumentParser
from pathlib import Path

from vidmap.configuration.names import DEFAULT_FRONTEND_CONFIG, DEFAULT_MAPPING_CONFIG

logger = logging.getLogger("vidmap.cli.run")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--input_data",
        required=True,
        help="Ordered image directory or MP4 to reconstruct.",
    )
    parser.add_argument("--frontend-conf", default=DEFAULT_FRONTEND_CONFIG)
    parser.add_argument("--mapping-conf", default=DEFAULT_MAPPING_CONFIG)
    parser.add_argument(
        "--output",
        required=True,
        help="Run directory for resolved configs, frontend artifacts, decoded frames, and reconstruction outputs.",
    )
    parser.add_argument("--imnames", nargs="*", type=str)
    parser.add_argument("--intrinsics", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--force-frontend", action="store_true")
    parser.add_argument(
        "--cache-depth-maps",
        action="store_true",
        help="Retain full depth maps for depth-lift flythroughs.",
    )
    parser.add_argument("-o", "--overwrite", action="store_true")
    from vidmap.run_options import add_run_arguments

    add_run_arguments(parser, mapping=True)
    return parser


def main(argv=None):
    from vidmap.configuration.build import parse_config_args, split_stage_overrides

    parser = build_parser()
    args, override_tokens = parse_config_args(parser, argv)
    try:
        frontend_overrides, mapping_overrides = split_stage_overrides(override_tokens)
    except ValueError as error:
        parser.error(str(error).replace("Pipeline config", "End-to-end config"))

    from vidmap.configuration.build import build_frontend_config, build_mapping_config
    from vidmap.configuration.names import FRONTEND_CONFIG_DIR, MAPPING_CONFIG_DIR, resolve_config_path

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
    from vidmap.run_options import RunOptions
    from vidmap.utils.logging import configure_logging

    run_options = RunOptions.from_namespace(args)
    configure_logging(run_options.verbosity)

    from vidmap.mapper.runtime import load_mapping_runtime

    load_mapping_runtime()

    # Import torch only after the native mapping runtime: libtorch_cpu.so exports its own
    # statically linked BLAS/LAPACK (dgemm_, dpotrf_, ...), and if it is loaded first those
    # symbols win global resolution for SuiteSparse/Ceres, which makes CHOLMOD report
    # "matrix not positive definite" and bundle adjustment fail.
    import torch

    torch.set_float32_matmul_precision("high")

    from vidmap.frontend.runner import run_local_frontend

    frontend = run_local_frontend(
        frontend_conf,
        args.input_data,
        workspace=args.output,
        imnames=args.imnames,
        intrinsics_path=args.intrinsics,
        force_frontend=args.force_frontend,
        cache_depth_maps=args.cache_depth_maps,
    )
    logger.info("Mapper inputs ready: tag=%s path=%s", frontend.tag, frontend.path)

    from vidmap.reconstruction import propagate_local_input_provenance, reconstruct

    output_dir = Path(args.output).expanduser()
    reconstruction = reconstruct(
        mapping_conf,
        frontend_conf,
        args.input_data,
        workspace=args.output,
        mapper_inputs_dir=frontend.mapper_inputs,
        run_options=run_options,
        imnames=args.imnames,
        intrinsics_path=args.intrinsics,
        overwrite_outputs=args.overwrite,
        output_dir=output_dir,
    )
    reconstruction_dir = output_dir / "rec"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(reconstruction_dir)
    propagate_local_input_provenance(frontend.mapper_inputs, output_dir, overwrite=args.overwrite)
    logger.info("Reconstruction written to %s", reconstruction_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
