import argparse
import dataclasses

import numpy as np

from vidmap.benchmark.trajectory import wate_auc_result_key, windowed_ate_auc_fields
from vidmap.configuration.build import (
    ROUTING_CLI_OPTIONS,
    build_mapping_config,
    parse_config_args,
    project_config_args,
    reject_routing_overrides,
)
from vidmap.configuration.names import DEFAULT_MAPPING_CONFIG
from vidmap.datasets.names import SUPPORTED_DATASETS

WATE_AUC_FULL = "full"


def build_evaluation_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="lamar",
        help="Prepared dataset containing the saved reconstruction results.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Experiment/output name used by the saved benchmark run.",
    )
    parser.add_argument("--testset_id", nargs="+", type=str, help="Testset id to run")
    parser.add_argument("-s", "--scene", nargs="+", type=str)
    parser.add_argument("-m", "--mode", nargs="+", type=str)
    from vidmap.run_options import add_run_arguments

    add_run_arguments(parser, mapping=False)
    parser.add_argument("--workspace_outputs", action="store_true")
    parser.add_argument("--output_root", type=str)
    from vidmap.configuration.evaluation import add_evaluation_arguments

    add_evaluation_arguments(parser)
    parser.add_argument(
        "--only-wate-auc",
        action="store_true",
        help="Only print WATE-AUC tables/CSV; suppress WATE median/mean/count and WRRE output.",
    )
    return parser


def _parse_wate_auc_key(key):
    prefix = "wate_auc_"
    if not key.startswith(prefix) or "@" not in key:
        return None
    window_s, threshold_s = key[len(prefix) :].split("@", 1)
    is_percent = threshold_s.endswith("pct")
    is_meter = threshold_s.endswith("m")
    if not is_percent and not is_meter:
        return None
    threshold_s = threshold_s[: -len("pct")] if is_percent else threshold_s[:-1]
    try:
        window_size = WATE_AUC_FULL if window_s == WATE_AUC_FULL else int(window_s)
        threshold = float(threshold_s) / 100.0 if is_percent else float(threshold_s)
        return window_size, threshold
    except ValueError:
        return None


def _format_threshold(threshold):
    return f"{threshold:g}"


def _format_wate_auc_col(window_size, threshold):
    if window_size == WATE_AUC_FULL:
        return f"full@{_format_threshold(threshold * 100.0)}%"
    return f"{window_size}m@{_format_threshold(threshold)}m"


def _requested_wate_auc_pairs(window_sizes, args):
    if not args.wate_auc_thresholds and not args.wate_auc_percents:
        return None
    pairs = []
    for W in window_sizes:
        pairs.extend((W, th) for th in (args.wate_auc_thresholds or []))
        pairs.extend((W, W * percent / 100.0) for percent in (args.wate_auc_percents or []))
    pairs.extend((WATE_AUC_FULL, percent / 100.0) for percent in (args.wate_auc_percents or []))
    return pairs


def _wate_auc_pair_sort_key(pair):
    window_size, threshold = pair
    return (float("inf") if window_size == WATE_AUC_FULL else window_size, threshold)


def _filter_wate_auc_pairs(pairs, window_sizes, args):
    requested = _requested_wate_auc_pairs(window_sizes, args)
    if requested is None:
        return pairs
    return [
        pair
        for pair in pairs
        if any(pair[0] == req[0] and np.isclose(pair[1], req[1], rtol=0.0, atol=1e-9) for req in requested)
    ]


def _parse_difficulty(scene_name):
    for diff in ("easy", "medium", "hard"):
        if f"_{diff}" in scene_name:
            return diff
    return None


def _group_scenes_by_difficulty(scenes):
    from collections import OrderedDict

    groups = OrderedDict()
    for s in scenes:
        diff = _parse_difficulty(s)
        key = diff if diff else "other"
        groups.setdefault(key, []).append(s)
    return groups


def _print_group_avg(label, scene_list, coll):
    vals_in_group = [coll[s][0] for s in scene_list if s in coll]
    if not vals_in_group:
        return
    mvals = np.mean(vals_in_group, axis=0)
    formatted = [f"{v:^6.1f}" for v in mvals]
    result = " | ".join(formatted)
    comp = np.sum([coll[s][1]["completed"] for s in scene_list if s in coll])
    fail = np.sum([coll[s][1]["failed"] for s in scene_list if s in coll])
    print(f"  --> {label}: {result}    recs: {comp}/{fail}")


class ReconstructionEvaluationReport:
    """Collected reconstruction metrics with deterministic text and CSV rendering."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        modes: list[str],
        scenes: list[str],
        prints: dict,
        colls: dict,
        wate_colls: dict,
    ):
        self.args = args
        self.modes = modes
        self.scenes = scenes
        self.prints = prints
        self.colls = colls
        self.wate_colls = wate_colls

    @classmethod
    def collect(
        cls,
        *,
        args: argparse.Namespace,
        override_tokens: tuple[str, ...],
        evaluation_options=None,
    ) -> "ReconstructionEvaluationReport":
        from vidmap.datasets.registry import get_dataset_spec

        dataset = get_dataset_spec(args.dataset)
        if getattr(args, "mode", None) is None:
            modes = [dataset.default_mode]
        else:
            modes = args.mode

        from vidmap.run_options import RunOptions

        run_options = RunOptions.from_namespace(args)
        from vidmap.benchmark.result_evaluation import BenchmarkResultEvaluator
        from vidmap.configuration.evaluation import evaluation_config_policy

        prints = {}
        colls = {}
        wate_colls = {}
        wate_auc_full_thresholds = sorted({percent / 100.0 for percent in (args.wate_auc_percents or [])})
        for mode in modes:
            args.mode = mode
            conf = build_mapping_config(
                None,
                source_name=DEFAULT_MAPPING_CONFIG,
                projected_cli_values=project_config_args(
                    args,
                    ("scene", "mode", "testset_id", "workspace_outputs", "output_root"),
                ),
                override_tokens=override_tokens,
                name=args.name,
                config_policy=evaluation_config_policy(args),
            )
            if evaluation_options is not None:
                conf = dataclasses.replace(conf, evaluation=evaluation_options)
                wate_auc_full_thresholds = evaluation_options.windowed_ate_auc_full_thresholds
            dataset.prepare()
            result_evaluator = BenchmarkResultEvaluator(
                conf,
                dataset,
                output_name=args.name,
                run_options=run_options,
            )
            out, counts, _, out_wate = result_evaluator.evaluate_saved()
            prints[mode] = {}
            colls[mode] = {}
            for scene, _AUCs in out.items():
                if not _AUCs:
                    print(f"Scene {scene} has no results in {mode} for {args.name}")
                    continue
                auc_arr = np.array(list(_AUCs.values()))
                if auc_arr.ndim < 2:
                    print(f"Scene {scene} has malformed AUCs (shape={auc_arr.shape}) in {mode} for {args.name}")
                    continue
                values = [float(el) for el in list(np.mean(auc_arr, axis=1)[0] * 100)][:-1]
                formatted = [f"{v:^6.1f}" for v in values]
                result = " | ".join(formatted)
                prints[mode][
                    scene
                ] = f"     {scene}: {result}    recs: {counts[scene]['completed']}/{counts[scene]['completed'] + counts[scene]['failed']}"
                colls[mode][scene] = (
                    [float(el) for el in list(np.mean(list(_AUCs.values()), axis=1)[0] * 100)][:-1],
                    counts[scene],
                )
            # Windowed ATE — separate median, mean, and metadata
            wate_colls[mode] = {}
            for scene, wate_data in out_wate.items():
                for wkey, testsets in wate_data.items():
                    if not testsets:
                        continue
                    all_vals = sum(testsets.values(), [])
                    if all_vals:
                        if wkey.startswith("wate_auc_errors_"):
                            window_s = wkey.removeprefix("wate_auc_errors_")
                            if window_s == WATE_AUC_FULL:
                                W = WATE_AUC_FULL
                                thresholds = wate_auc_full_thresholds
                            else:
                                W = int(window_s)
                                thresholds = conf.evaluation.windowed_ate_auc_thresholds
                            for auc_key, auc in windowed_ate_auc_fields(W, all_vals, thresholds).items():
                                wate_colls[mode].setdefault(scene, {})[auc_key] = auc
                        elif wkey.endswith("_count"):
                            # Sum counts, also track number of contributing trajectories
                            wate_colls[mode].setdefault(scene, {})[wkey] = int(np.sum(all_vals))
                            wate_colls[mode][scene][wkey + "_trajs"] = len(all_vals)
                        else:
                            wate_colls[mode].setdefault(scene, {})[wkey] = float(np.mean(all_vals))
        if getattr(args, "scene", None) is None:
            scenes = result_evaluator.selection.dataset_layout.scenes
        else:
            scenes = args.scene
        return cls(
            args=args,
            modes=modes,
            scenes=scenes,
            prints=prints,
            colls=colls,
            wate_colls=wate_colls,
        )

    def render(self) -> None:
        args = self.args
        modes = self.modes
        scenes = self.scenes
        prints = self.prints
        colls = self.colls
        wate_colls = self.wate_colls
        scene_groups = _group_scenes_by_difficulty(scenes)
        use_groups = len(scene_groups) > 1
        from vidmap.datasets.registry import get_dataset_spec

        default_metric = get_dataset_spec(args.dataset).default_evaluation_metric
        show_pose_auc = default_metric == "pose_auc" and not args.only_wate_auc
        show_wate_auc = default_metric == "wate_auc" or args.only_wate_auc
        wate_alignment_label = "Sim(3)-window scale-aligned"

        for mode in modes:
            print(mode, args.name)
            coll = colls[mode]
            if show_pose_auc:
                print("  AUC:")
                if use_groups:
                    for diff, group_scenes in scene_groups.items():
                        for scene in group_scenes:
                            if scene in prints[mode]:
                                print(prints[mode][scene])
                            else:
                                print(f"Scene {scene} not found in {mode} for {args.name}")
                        _print_group_avg(diff.upper(), group_scenes, coll)
                        print()
                else:
                    for scene in scenes:
                        if scene in prints[mode]:
                            print(prints[mode][scene])
                        else:
                            print(f"Scene {scene} not found in {mode} for {args.name}")
                comp_sum = np.sum([counts["completed"] for values, counts in coll.values()])
                failed_sum = np.sum([counts["failed"] for values, counts in coll.values()])
                print(f"     recs total: {comp_sum}/{comp_sum + failed_sum}")

            # Windowed ATE
            wcoll = wate_colls[mode]
            # Find window sizes present across scenes (e.g. 10, 25, 50, 100)
            all_window_sizes = sorted(
                set(
                    int(k.split("_")[1])
                    for s in scenes
                    if s in wcoll
                    for k in wcoll[s]
                    if k.startswith("wate_") and k[5:].isdigit()
                )
                if show_wate_auc
                else set()
            )
            if all_window_sizes:
                wate_auc_pairs = sorted(
                    {
                        parsed
                        for s in scenes
                        if s in wcoll
                        for k in wcoll[s]
                        for parsed in [_parse_wate_auc_key(k)]
                        if parsed is not None
                    },
                    key=_wate_auc_pair_sort_key,
                )
                wate_auc_pairs = _filter_wate_auc_pairs(wate_auc_pairs, all_window_sizes, args)
                if wate_auc_pairs:
                    auc_cols = ["path"] + [_format_wate_auc_col(W, th) for W, th in wate_auc_pairs]
                    auc_header = "  " + " | ".join(f"{c:>9s}" for c in auc_cols)
                    print()
                    print(f"  WATE-AUC {wate_alignment_label} (%): {auc_header}")
                    for scene in scenes:
                        if scene not in wcoll:
                            continue
                        sd = wcoll[scene]
                        cells = [f"{sd['wate_path_length']:9.1f}" if "wate_path_length" in sd else f"{'—':>9s}"]
                        for W, th in wate_auc_pairs:
                            wk = wate_auc_result_key(W, th)
                            cells.append(f"{sd[wk] * 100:9.1f}" if wk in sd else f"{'—':>9s}")
                        print(f"     {scene}: {' | '.join(cells)}")
                    avg_path = np.mean(
                        [wcoll[s]["wate_path_length"] for s in scenes if s in wcoll and "wate_path_length" in wcoll[s]]
                    )
                    avg_auc = []
                    for W, th in wate_auc_pairs:
                        wk = wate_auc_result_key(W, th)
                        vals = [wcoll[s][wk] for s in scenes if s in wcoll and wk in wcoll[s]]
                        avg_auc.append(f"{np.mean(vals) * 100:9.1f}" if vals else f"{'—':>9s}")
                    print(f"  --> AVG: {avg_path:9.1f} | {' | '.join(avg_auc)}")

        sep = "\t"
        if show_pose_auc:
            # CSV output (tab-separated for Google Sheets paste)
            print("\n<!-- csv -->")
            if len(modes) > 1:
                cols = [s for s in scenes if any(s in colls[m] for m in modes)]
                if use_groups:
                    group_labels = [d.upper() for d in scene_groups]
                    cols = cols + group_labels
                print(sep.join(["Mode"] + cols))
                for mode in modes:
                    cells = []
                    for scene in scenes:
                        if scene in colls[mode]:
                            values, _ = colls[mode][scene]
                            cells.append(" / ".join(f"{v:.1f}" for v in values))
                    coll = colls[mode]
                    if use_groups:
                        for diff, group_scenes in scene_groups.items():
                            gvals = [coll[s][0] for s in group_scenes if s in coll]
                            if gvals:
                                gm = np.mean(gvals, axis=0)
                                cells.append(" / ".join(f"{v:.1f}" for v in gm))
                    print(sep.join([mode] + cells))
            else:
                mode = modes[0]
                coll = colls[mode]
                print(sep.join(["Scene", "AUC"]))
                if use_groups:
                    for diff, group_scenes in scene_groups.items():
                        for scene in group_scenes:
                            auc_str = ""
                            if scene in coll:
                                values, _ = coll[scene]
                                auc_str = " / ".join(f"{v:.1f}" for v in values)
                            print(sep.join([scene, auc_str]))
                        gvals = [coll[s][0] for s in group_scenes if s in coll]
                        auc_str = " / ".join(f"{v:.1f}" for v in np.mean(gvals, axis=0)) if gvals else ""
                        print(sep.join([diff.upper(), auc_str]))
                else:
                    for scene in scenes:
                        auc_str = ""
                        if scene in coll:
                            values, _ = coll[scene]
                            auc_str = " / ".join(f"{v:.1f}" for v in values)
                        print(sep.join([scene, auc_str]))

        # WATE CSV (separate from AUC/Recall)
        all_window_sizes_csv = sorted(
            set(
                int(k.split("_")[1])
                for m in modes
                for s in scenes
                if s in wate_colls[m]
                for k in wate_colls[m][s]
                if k.startswith("wate_") and k[5:].isdigit()
            )
            if show_wate_auc
            else set()
        )
        if all_window_sizes_csv:
            wate_auc_pairs_csv = sorted(
                {
                    parsed
                    for m in modes
                    for s in scenes
                    if s in wate_colls[m]
                    for k in wate_colls[m][s]
                    for parsed in [_parse_wate_auc_key(k)]
                    if parsed is not None
                },
                key=_wate_auc_pair_sort_key,
            )
            wate_auc_pairs_csv = _filter_wate_auc_pairs(wate_auc_pairs_csv, all_window_sizes_csv, args)
            if wate_auc_pairs_csv:
                print(f"\n<!-- wate auc csv: {wate_alignment_label} -->")
                wate_auc_cols = ["path"] + [_format_wate_auc_col(W, th) for W, th in wate_auc_pairs_csv]
                if len(modes) > 1:
                    print(sep.join(["Mode"] + wate_auc_cols))
                    for mode in modes:
                        wcoll = wate_colls[mode]
                        path_vals = [
                            wcoll[s]["wate_path_length"]
                            for s in scenes
                            if s in wcoll and "wate_path_length" in wcoll[s]
                        ]
                        cells = [f"{np.mean(path_vals):.1f}" if path_vals else ""]
                        for W, th in wate_auc_pairs_csv:
                            wk = wate_auc_result_key(W, th)
                            vals = [wcoll[s][wk] for s in scenes if s in wcoll and wk in wcoll[s]]
                            cells.append(f"{np.mean(vals) * 100:.1f}" if vals else "")
                        print(sep.join([mode] + cells))
                else:
                    mode = modes[0]
                    wcoll = wate_colls[mode]
                    print(sep.join(["Scene"] + wate_auc_cols))
                    for scene in scenes:
                        if scene not in wcoll:
                            continue
                        sd = wcoll[scene]
                        cells = [f"{sd['wate_path_length']:.1f}" if "wate_path_length" in sd else ""]
                        for W, th in wate_auc_pairs_csv:
                            wk = wate_auc_result_key(W, th)
                            cells.append(f"{sd[wk] * 100:.1f}" if wk in sd else "")
                        print(sep.join([scene] + cells))


def collect_reconstruction_report(
    args: argparse.Namespace,
    *,
    override_tokens: tuple[str, ...] = (),
    evaluation_options=None,
) -> ReconstructionEvaluationReport:
    """Collect one evaluation report from a resolved evaluation request."""

    return ReconstructionEvaluationReport.collect(
        args=args,
        override_tokens=override_tokens,
        evaluation_options=evaluation_options,
    )


def render_reconstruction_report(
    args: argparse.Namespace,
    *,
    override_tokens: tuple[str, ...] = (),
    evaluation_options=None,
) -> ReconstructionEvaluationReport:
    """Collect and render one evaluation report, returning the collected data."""

    report = collect_reconstruction_report(
        args,
        override_tokens=override_tokens,
        evaluation_options=evaluation_options,
    )
    report.render()
    return report


def main(argv=None):
    parser = build_evaluation_parser()
    args, override_tokens = parse_config_args(parser, argv)
    from vidmap.utils.logging import configure_logging

    configure_logging(args.verbose)
    try:
        reject_routing_overrides(
            override_tokens,
            {
                **ROUTING_CLI_OPTIONS,
                "name": "--name",
            },
        )
    except ValueError as exc:
        parser.error(str(exc))
    render_reconstruction_report(args, override_tokens=override_tokens)
    return 0
