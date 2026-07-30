"""CLI and config policy for current trajectory evaluation options."""

import argparse
from functools import partial

EVALUATION_POLICY_FIELDS = (
    "wate",
    "wate_auc_thresholds",
    "wate_auc_percents",
    "wate_auc_missing_error",
    "eval_on_full_gt_timeline",
    "no_eval_on_full_gt_timeline",
)


def add_evaluation_arguments(parser, *, suppress_defaults: bool = False) -> None:
    """Add the shared WATE/WATE-AUC policy arguments to ``parser``."""
    value_default = argparse.SUPPRESS if suppress_defaults else None
    flag_default = argparse.SUPPRESS if suppress_defaults else False
    parser.add_argument(
        "--wate",
        nargs="+",
        type=int,
        default=value_default,
        help="Windowed ATE sizes in meters, e.g. --wate 10 25 50 100.",
    )
    parser.add_argument(
        "--wate-auc",
        dest="wate_auc_thresholds",
        nargs="+",
        type=float,
        default=value_default,
        help="Compute WATE-AUC at meter thresholds, e.g. --wate-auc 5 10.",
    )
    parser.add_argument(
        "--wate-auc-percent",
        dest="wate_auc_percents",
        nargs="+",
        type=float,
        default=value_default,
        help="Compute WATE-AUC thresholds as a percent of each WATE window.",
    )
    parser.add_argument(
        "--wate-auc-missing-error",
        type=float,
        default=value_default,
        help=(
            "Assign this meter error to GT-pose frames missing a windowed error. "
            "Implies --eval-on-full-gt-timeline unless explicitly disabled."
        ),
    )
    parser.add_argument(
        "--eval-on-full-gt-timeline",
        action="store_true",
        default=flag_default,
        help="Evaluate on every GT/evaluation frame instead of shrinking the GT set.",
    )
    parser.add_argument(
        "--no-eval-on-full-gt-timeline",
        action="store_true",
        default=flag_default,
        help="Evaluate only directly matched keyframes when exact source timestamps exist.",
    )


def project_evaluation_policy(args) -> dict:
    """Project explicitly requested evaluation behavior from an argparse namespace."""
    explicit = vars(args)
    return {
        field: explicit[field]
        for field in EVALUATION_POLICY_FIELDS
        if field in explicit and explicit[field] is not None and explicit[field] is not False
    }


def apply_evaluation_policy(raw, *, policy: dict) -> None:
    """Apply projected evaluation policy after config composition and before overrides."""
    if "wate" in policy:
        raw["windowed_ate_sizes"] = policy["wate"]
    thresholds = list(policy.get("wate_auc_thresholds", []))
    if "wate_auc_percents" in policy:
        for window in raw.get("windowed_ate_sizes", []):
            thresholds.extend(window * percent / 100.0 for percent in policy["wate_auc_percents"])
        raw["windowed_ate_auc_full_thresholds"] = sorted({percent / 100.0 for percent in policy["wate_auc_percents"]})
    if thresholds:
        raw["windowed_ate_auc_thresholds"] = sorted(set(thresholds))
    if "wate_auc_missing_error" in policy:
        raw["windowed_ate_auc_missing_error"] = policy["wate_auc_missing_error"]
    if policy.get("no_eval_on_full_gt_timeline"):
        raw["eval_on_full_gt_timeline"] = False
    elif policy.get("eval_on_full_gt_timeline") or "wate_auc_missing_error" in policy:
        raw["eval_on_full_gt_timeline"] = True


def evaluation_config_policy(args):
    """Return the config-composition policy for one parsed CLI namespace."""
    return partial(apply_evaluation_policy, policy=project_evaluation_policy(args))
