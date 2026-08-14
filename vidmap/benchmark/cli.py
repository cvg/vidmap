"""Prepared-dataset benchmark command implementation."""

import argparse
import logging
import os
import pathlib
import sys

from vidmap.configuration.names import DEFAULT_FRONTEND_CONFIG, DEFAULT_MAPPING_CONFIG

logger = logging.getLogger("vidmap.cli.run_for_benchmark")


def _load_runtime(*, mapping: bool):
    from vidmap.mapper.runtime import load_mapping_runtime, load_pycolmap_runtime

    if mapping:
        load_mapping_runtime()
    else:
        load_pycolmap_runtime()


def build_parser():
    from vidmap.datasets.names import SUPPORTED_DATASETS

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "-d",
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="lamar",
        help="Dataset to use (test harness name).",
    )
    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="Produce finalized boundaries without mapping.",
    )
    parser.add_argument("--force-frontend", action="store_true")
    parser.add_argument(
        "--run-only",
        action="store_true",
        help="Skip the final aggregate report; per-trajectory evaluation still runs.",
    )
    parser.add_argument("--deterministic_frontend", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--pre_geom_db_stop", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--pre_geom_repro_dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "-c",
        "--mapping-conf",
        type=str,
        default=None,
        help="Mapping config relative to vidmap/configs/mapping/.",
    )
    parser.add_argument(
        "--frontend-conf",
        default=None,
        help="Frontend config relative to vidmap/configs/frontend/.",
    )
    parser.add_argument(
        "--testset_id",
        nargs="+",
        type=str,
        default=argparse.SUPPRESS,
        help="Testset id to run",
    )
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("-s", "--scene", nargs="+", type=str, default=argparse.SUPPRESS)
    parser.add_argument("-m", "--mode", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "--workspace_outputs",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Write experiment outputs under this checkout instead of the dataset experiment directory.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=argparse.SUPPRESS,
        help="Override the benchmark experiment root. Output layout remains reconstruction/<mode>/<scene>/<testset>/<conf>.",
    )
    parser.add_argument(
        "--frontend_cache_root",
        type=str,
        default=argparse.SUPPRESS,
        help="Override frontend cache root.",
    )
    parser.add_argument(
        "--mapper-inputs",
        dest="mapper_inputs_dir",
        type=str,
        help="Run mapping directly from a prepared current-schema mapper-input directory.",
    )
    from vidmap.run_options import add_run_arguments

    add_run_arguments(parser, mapping=True, profiling=True)
    from vidmap.configuration.evaluation import add_evaluation_arguments

    add_evaluation_arguments(parser, suppress_defaults=True)
    parser.add_argument("--name", type=str, default=argparse.SUPPRESS, help="Experiment/output name.")
    return parser


def validate_stage_arguments(parser, args) -> None:
    """Reject switches owned by the other benchmark stage."""
    if args.frontend_only:
        mapping_only = {
            "--mapper-inputs": args.mapper_inputs_dir is not None,
            "--mapping-conf": args.mapping_conf is not None,
            "--run-only": args.run_only,
            "--overwrite": args.overwrite,
            "--name": hasattr(args, "name"),
            "--save-playback-trace": args.save_playback_trace,
            "--playback-trace-point-cap": args.playback_trace_point_cap is not None,
        }
        from vidmap.configuration.evaluation import EVALUATION_POLICY_FIELDS

        mapping_only.update(
            {f"--{field.replace('_', '-')}": True for field in EVALUATION_POLICY_FIELDS if hasattr(args, field)}
        )
        rejected = [option for option, present in mapping_only.items() if present]
        if rejected:
            parser.error(f"--frontend-only rejects mapping arguments: {', '.join(rejected)}")
        if args.frontend_conf is None:
            args.frontend_conf = DEFAULT_FRONTEND_CONFIG
        return

    frontend_only = {
        "--force-frontend": args.force_frontend,
        "--deterministic_frontend": hasattr(args, "deterministic_frontend"),
        "--pre_geom_db_stop": hasattr(args, "pre_geom_db_stop"),
        "--pre_geom_repro_dir": hasattr(args, "pre_geom_repro_dir"),
    }
    rejected = [option for option, present in frontend_only.items() if present]
    if args.frontend_conf is None:
        rejected.append("missing --frontend-conf")
    if rejected:
        parser.error(f"mapping rejects frontend arguments: {', '.join(rejected)}")
    if args.mapping_conf is None:
        args.mapping_conf = DEFAULT_MAPPING_CONFIG


def config_from_args(args, overrides):
    if hasattr(args, "name"):
        output_name = str(args.name)
        if (
            not output_name
            or pathlib.Path(output_name).name != output_name
            or "\\" in output_name
            or output_name in {".", ".."}
        ):
            raise ValueError(f"Experiment name must be one safe output path component, got {args.name!r}")
    if not hasattr(args, "mode"):
        from vidmap.datasets.registry import get_dataset_spec

        args.mode = get_dataset_spec(args.dataset).default_mode
    if args.frontend_only:
        if args.frontend_conf is None:
            args.frontend_conf = DEFAULT_FRONTEND_CONFIG
        from vidmap.configuration.build import build_frontend_config_from_args
        from vidmap.repro.frontend_bootstrap import deterministic_config_patch

        projected_cli_values = deterministic_config_patch() if getattr(args, "deterministic_frontend", False) else None
        runtime = {
            key: getattr(args, key)
            for key in (
                "deterministic_frontend",
                "pre_geom_db_stop",
                "pre_geom_repro_dir",
            )
            if hasattr(args, key)
        }
        return build_frontend_config_from_args(
            args,
            overrides,
            projected_cli_values={**(projected_cli_values or {}), **runtime},
        )

    from vidmap.configuration.build import build_mapping_config_from_args
    from vidmap.configuration.evaluation import evaluation_config_policy

    if args.mapping_conf is None:
        args.mapping_conf = DEFAULT_MAPPING_CONFIG
    return build_mapping_config_from_args(
        args,
        overrides,
        config_policy=evaluation_config_policy(args),
        use_cli_name=False,
    )


def split_config_overrides(parser, args, override_tokens):
    """Use stage prefixes for combined runs while preserving frontend-only composition."""
    if args.frontend_only:
        return tuple(override_tokens), ()
    from vidmap.configuration.build import split_stage_overrides

    try:
        return split_stage_overrides(override_tokens)
    except ValueError as error:
        parser.error(str(error).replace("Pipeline config", "Benchmark config"))


def frontend_config_for_mapping(args, mapping_conf, override_tokens=()):
    """Build the frontend stage selected by one end-to-end benchmark command."""
    import dataclasses

    from vidmap.configuration.build import build_frontend_config
    from vidmap.configuration.names import FRONTEND_CONFIG_DIR, resolve_config_path

    config_name = args.frontend_conf
    projected_cli_values = dataclasses.asdict(mapping_conf.selection)
    projected_cli_values["frontend_cache_root"] = mapping_conf.run.frontend_cache_root
    frontend_conf = build_frontend_config(
        resolve_config_path(config_name, FRONTEND_CONFIG_DIR),
        source_name=config_name,
        projected_cli_values=projected_cli_values,
        override_tokens=override_tokens,
    )
    return frontend_conf


def run_selected_targets(args, conf, frontend_conf, selection, run_options):
    """Run optional frontend followed by mapping for one finite selection."""
    if frontend_conf is not None and args.mapper_inputs_dir is None:
        from vidmap.frontend.runner import FrontendRunner

        FrontendRunner(
            frontend_conf,
            selection,
            run_options,
            output_root=getattr(args, "output_root", None),
        ).run_with_results(force_frontend=False)

    from vidmap.benchmarking import BenchmarkRunner

    runner_kwargs = {"output_name": args.name} if hasattr(args, "name") else {}
    return BenchmarkRunner(conf, frontend_conf, selection, run_options, **runner_kwargs).run(
        mapper_inputs_dir=args.mapper_inputs_dir,
        overwrite_results=args.overwrite,
    )


def render_final_evaluation(args, conf, frontend_conf):
    """Render the aggregate saved-result report unless explicitly disabled."""
    if args.run_only:
        return None

    from vidmap.benchmark.report import build_evaluation_parser, render_reconstruction_report
    from vidmap.configuration.names import benchmark_config_pair_slug

    output_name = args.name if hasattr(args, "name") else benchmark_config_pair_slug(frontend_conf.name, conf.name)
    evaluation_argv = [
        "--dataset",
        args.dataset,
        "--name",
        output_name,
        "--mode",
        conf.selection.mode,
        "-v",
        str(args.verbose),
    ]
    if conf.selection.scene is not None:
        evaluation_argv.extend(("--scene", *conf.selection.scene))
    if conf.selection.testset_id is not None:
        evaluation_argv.extend(("--testset_id", *conf.selection.testset_id))
    if conf.run.workspace_outputs:
        evaluation_argv.append("--workspace_outputs")
    if conf.run.output_root is not None:
        evaluation_argv.extend(("--output_root", conf.run.output_root))
    if args.terminate:
        evaluation_argv.append("--terminate")
    evaluation_args = build_evaluation_parser().parse_args(evaluation_argv)
    return render_reconstruction_report(evaluation_args, evaluation_options=conf.evaluation)


def main(argv=None):
    from vidmap.configuration.build import parse_config_args

    parser = build_parser()
    args, overrides = parse_config_args(parser, argv)
    validate_stage_arguments(parser, args)
    frontend_overrides, mapping_overrides = split_config_overrides(parser, args, overrides)
    from vidmap.run_options import RunOptions

    run_options = RunOptions.from_namespace(args)
    if (
        args.frontend_only
        and getattr(args, "deterministic_frontend", False)
        and os.environ.get("VIDMAP_DETERMINISTIC_BOOTSTRAPPED") != "1"
    ):
        from vidmap.repro.frontend_bootstrap import deterministic_env

        env = dict(os.environ)
        env.update(deterministic_env())
        env["VIDMAP_DETERMINISTIC_BOOTSTRAPPED"] = "1"
        process_argv = sys.argv[1:] if argv is None else argv
        os.execvpe(
            sys.executable,
            [sys.executable, "-m", "vidmap.run_for_benchmark", *process_argv],
            env,
        )

    from vidmap.utils.logging import configure_logging

    configure_logging(run_options.verbosity)
    conf = config_from_args(args, frontend_overrides if args.frontend_only else mapping_overrides)
    frontend_conf = None
    if not args.frontend_only:
        try:
            frontend_conf = frontend_config_for_mapping(args, conf, frontend_overrides)
        except ValueError as error:
            parser.error(str(error))
    _load_runtime(mapping=not args.frontend_only)

    import pycolmap

    from vidmap.datasets.selection import PreparedTargetSelection, validate_profile_selection
    from vidmap.utils.profiling import profiling_output_dir

    logger.info("pycolmap %s from %s", pycolmap.__version__, pycolmap.__file__)

    if args.frontend_only and getattr(args, "deterministic_frontend", False):
        from vidmap.repro.frontend import apply_deterministic_runtime

        apply_deterministic_runtime()

    selection = PreparedTargetSelection(conf.selection, args.dataset)
    try:
        validate_profile_selection(selection, enabled=run_options.profile)
    except ValueError as error:
        parser.error(str(error))
    selection.dataset_spec.prepare(conf.selection.scene)

    import time

    from vidmap.utils.profiling import profiling_session

    with profiling_session(run_options.profile) as profile:
        started = time.time()
        if args.frontend_only:
            from vidmap.frontend.runner import FrontendRunner

            result = FrontendRunner(
                conf,
                selection,
                run_options,
                output_root=getattr(args, "output_root", None),
            ).run_with_results(force_frontend=args.force_frontend)
            failure_count = result.failure_count
            for target in result.targets:
                logger.info("Mapper inputs ready: tag=%s path=%s", target.tag, target.path)
        else:
            failure_count = run_selected_targets(args, conf, frontend_conf, selection, run_options)
            render_final_evaluation(args, conf, frontend_conf)
        profile.record_timing("wall_total", time.time() - started)
    if profile.enabled:
        prof_dir = profiling_output_dir(conf, args, operation="frontend" if args.frontend_only else "benchmark")
        profile.write(prof_dir / "important.txt")
        logger.info("Profiling written to %s", prof_dir / "important.txt")
    if failure_count:
        logger.error("%d target(s) failed", failure_count)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
