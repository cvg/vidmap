"""Canonical frontend-only module for concrete local media."""

import argparse
import logging
import sys

from vidmap.configuration.names import DEFAULT_FRONTEND_CONFIG

logger = logging.getLogger("vidmap.cli.frontend")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("input", metavar="INPUT", help="Ordered image directory or MP4 to process.")
    parser.add_argument("-c", "--frontend-conf", default=DEFAULT_FRONTEND_CONFIG)
    parser.add_argument(
        "--output",
        required=True,
        help="Project directory for decoded frames, frontend artifacts, and mapper inputs.",
    )
    parser.add_argument("--imnames", nargs="*", type=str)
    parser.add_argument("--intrinsics", type=str)
    parser.add_argument(
        "--mapper-inputs",
        help="Finalized mapper inputs to use when backfilling full depth maps into --output.",
    )
    parser.add_argument("--force-frontend", action="store_true")
    parser.add_argument(
        "--cache-depth-maps",
        action="store_true",
        help=(
            "Retain full prediction-grid depth maps for depth-lift visualization; "
            "on an existing frontend, infer only the missing full maps."
        ),
    )
    from vidmap.run_options import add_run_arguments

    add_run_arguments(parser, mapping=False, profiling=True)
    return parser


def config_from_args(args, overrides):
    from vidmap.configuration.build import build_frontend_config_from_args

    return build_frontend_config_from_args(args, overrides)


def main(argv=None):
    try:
        from vidmap.configuration.build import parse_config_args

        parser = build_parser()
        args, overrides = parse_config_args(parser, argv)
        from vidmap.run_options import RunOptions
        from vidmap.utils.logging import configure_logging

        run_options = RunOptions.from_namespace(args)
        configure_logging(run_options.verbosity)
        conf = config_from_args(args, overrides)
        if args.mapper_inputs and not args.cache_depth_maps:
            parser.error("--mapper-inputs requires --cache-depth-maps")
        if args.mapper_inputs and args.force_frontend:
            parser.error("--mapper-inputs cannot be combined with --force-frontend")

        from vidmap.mapper.runtime import load_pycolmap_runtime

        load_pycolmap_runtime()

        import time

        from vidmap.frontend.runner import run_local_frontend
        from vidmap.utils.profiling import profiling_output_dir, profiling_session

        with profiling_session(run_options.profile) as profile:
            started = time.time()
            result = run_local_frontend(
                conf,
                args.input,
                workspace=args.output,
                imnames=args.imnames,
                intrinsics_path=args.intrinsics,
                force_frontend=args.force_frontend,
                cache_depth_maps=args.cache_depth_maps,
                mapper_inputs_path=args.mapper_inputs,
            )
            profile.record_timing("wall_total", time.time() - started)
        logger.setLevel(logging.INFO)
        logger.info("Mapper inputs ready: tag=%s path=%s", result.tag, result.path)
        if profile.enabled:
            profile_args = argparse.Namespace(
                frontend_conf=args.frontend_conf,
                dataset="local",
                scene=["local"],
                testset_id=["input"],
                mode="all",
            )
            profile_dir = profiling_output_dir(conf, profile_args, operation="frontend")
            profile.write(profile_dir / "important.txt")
            logger.info("Profiling written to %s", profile_dir / "important.txt")
        return 0
    except KeyboardInterrupt:
        logger.warning("Received keyboard interrupt; exiting")
        sys.exit(1)
    except Exception as error:
        logger.exception("Frontend failed: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
