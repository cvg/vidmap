"""Stage-specific config composition and CLI projection."""

import dataclasses
import os
import re
import sys
from argparse import ArgumentParser, Namespace
from collections.abc import Mapping
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from vidmap.configuration.names import FRONTEND_CONFIG_DIR, MAPPING_CONFIG_DIR, config_name_to_output_slug

TARGET_CLI_KEYS = "testset_id scene mode".split()
FRONTEND_CLI_KEYS = (*TARGET_CLI_KEYS, "frontend_cache_root")
MAPPING_CLI_KEYS = (
    *TARGET_CLI_KEYS,
    "workspace_outputs",
    "output_root",
    "frontend_cache_root",
)
# Kept as the target-selection subset for callers which do not own a stage.
BASE_CLI_KEYS = TARGET_CLI_KEYS
ROUTING_CLI_OPTIONS = {
    "mode": "--mode",
    "scene": "--scene",
    "testset_id": "--testset_id",
    "name": "--name",
    "output_root": "--output_root",
    "workspace_outputs": "--workspace_outputs",
}

_CONFIG_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_OVERRIDE_SHAPED = re.compile(r"(?:\+\+?|~)?[A-Za-z_][A-Za-z0-9_.]*=")
_SUPPRESSED_DESTINATIONS = frozenset(FRONTEND_CLI_KEYS) | frozenset(MAPPING_CLI_KEYS)
_ROUTING_OPTION_STRINGS = {
    "mode": ("-m", "--mode"),
    "scene": ("-s", "--scene"),
    "testset_id": ("--testset_id",),
    "workspace_outputs": ("--workspace_outputs",),
    "output_root": ("--output_root",),
    "frontend_cache_root": ("--frontend_cache_root",),
}
_VARIADIC_OPTIONS = {
    "--imnames": "--imnames",
    "--testset_id": "--testset_id",
    "-s": "--scene",
    "--scene": "--scene",
    "-m": "--mode",
    "--mode": "--mode",
    "--wate": "--wate",
    "--wate-auc": "--wate-auc",
    "--wate_auc": "--wate-auc",
    "--wate-auc-percent": "--wate-auc-percent",
    "--wate_auc_percent": "--wate-auc-percent",
}


def parse_config_args(parser: ArgumentParser, argv=None) -> tuple[Namespace, tuple[str, ...]]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser.allow_abbrev = False
    _reject_overrides_swallowed_by_variadic_args(parser, raw_argv)
    args, tokens = parser.parse_known_args(raw_argv)
    _remove_implicit_routing_defaults(args, raw_argv)
    if tokens.count("--") > 1:
        parser.error("config overrides accept at most one '--' separator")
    tokens = [token for token in tokens if token != "--"]
    unknown_options = [token for token in tokens if token.startswith("-")]
    if unknown_options:
        parser.error(f"unrecognized arguments: {' '.join(unknown_options)}")
    try:
        parse_override_tokens(tokens)
    except ValueError as exc:
        parser.error(str(exc))
    return args, tuple(tokens)


def project_config_args(args: Namespace, keys):
    explicit = vars(args)
    return {key: explicit[key] for key in keys if key in explicit and explicit[key] is not None}


def reject_routing_overrides(override_tokens, options=ROUTING_CLI_OPTIONS):
    rejected = sorted({key for key, _ in parse_override_tokens(override_tokens)} & options.keys())
    if rejected:
        directions = ", ".join(f"{key} (use {options[key]})" for key in rejected)
        raise ValueError(f"Config overrides cannot change entrypoint routing fields: {directions}")


def split_stage_overrides(override_tokens):
    """Route pipeline overrides to frontend or mapping, removing the stage prefix."""
    frontend = []
    mapping = []
    for token, (key, _value) in zip(override_tokens, parse_override_tokens(override_tokens), strict=True):
        stage, separator, field = key.partition(".")
        if separator == "" or stage not in {"frontend", "mapping"}:
            raise ValueError(f"Pipeline config override {token!r} must start with 'frontend.' or 'mapping.'")
        if not field:
            raise ValueError(f"Pipeline config override {token!r} is missing a config field")
        routed = token[len(stage) + 1 :]
        (frontend if stage == "frontend" else mapping).append(routed)
    return tuple(frontend), tuple(mapping)


def build_frontend_config_from_args(args, override_tokens=(), *, projected_cli_values=None):
    from vidmap.configuration.names import resolve_config_path

    config_name = args.frontend_conf
    merged_cli_values = OmegaConf.merge(
        OmegaConf.create(projected_cli_values or {}),
        project_config_args(args, FRONTEND_CLI_KEYS),
    )
    return build_frontend_config(
        resolve_config_path(config_name, FRONTEND_CONFIG_DIR),
        source_name=config_name,
        projected_cli_values=OmegaConf.to_container(merged_cli_values, resolve=False),
        override_tokens=override_tokens,
    )


def build_mapping_config_from_args(args, override_tokens=(), *, config_policy=None, use_cli_name=True):
    from vidmap.configuration.names import resolve_config_path

    config_name = args.mapping_conf
    projected_cli_values = project_config_args(args, MAPPING_CLI_KEYS)
    return build_mapping_config(
        resolve_config_path(config_name, MAPPING_CONFIG_DIR),
        source_name=config_name,
        projected_cli_values=projected_cli_values,
        override_tokens=override_tokens,
        name=vars(args).get("name") if use_cli_name else None,
        config_policy=config_policy,
    )


def parse_override_tokens(tokens):
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from hydra.errors import HydraException

    parser = OverridesParser.create()
    parsed = []
    for token in tokens:
        try:
            override = parser.parse_override(token)
        except HydraException as exc:
            raise ValueError(f"Malformed config override {token!r}: {exc}") from exc
        if override.is_sweep_override():
            raise ValueError(f"Hydra sweeps are not supported: {token!r}")
        if override.is_delete():
            raise ValueError(f"Hydra deletion overrides are not supported: {token!r}")
        if override.is_add() or override.is_force_add():
            raise ValueError(f"Hydra add/force-add overrides are not supported: {token!r}")
        if override.package is not None or _CONFIG_KEY.fullmatch(override.key_or_group) is None:
            raise ValueError(f"Expected a config field dot path in override {token!r}")
        if override.key_or_group == "deterministic_frontend":
            raise ValueError("use --deterministic_frontend so runtime bootstrap precedes imports")
        parsed.append((override.key_or_group, override.value()))
    return parsed


def build_frontend_config(source, *, source_name, projected_cli_values=None, override_tokens=()):
    raw = compose_raw_config(
        source,
        source_name=source_name,
        stage="frontend",
        projected_cli_values=projected_cli_values,
        override_tokens=override_tokens,
    )
    if raw.get("deterministic_frontend") and os.environ.get("VIDMAP_DETERMINISTIC_BOOTSTRAPPED") != "1":
        raise ValueError("deterministic_frontend requires the --deterministic_frontend runtime bootstrap")
    from vidmap.configuration.config import FrontendConfig, FrontendRunSpec
    from vidmap.configuration.defaults import FrontendRunOptions, TargetSelectionOptions
    from vidmap.mapper.options import ReplayCacheOptions

    plain = materialize_config_values(
        raw,
        {
            **dataclasses.asdict(FrontendConfig()),
            **dataclasses.asdict(TargetSelectionOptions()),
            **dataclasses.asdict(FrontendRunOptions()),
            "name": None,
            "colmap_runtime": "stock",
        },
    )
    name = plain.pop("name")
    runtime = plain.pop("colmap_runtime")
    _validate_safe_component(name, "Config name", allow_path=True)
    _validate_safe_component(runtime, "COLMAP runtime")
    selection = _pop_dataclass_values(plain, TargetSelectionOptions)
    plain["replay_cache"] = ReplayCacheOptions(**plain["replay_cache"])
    run = _pop_dataclass_values(plain, FrontendRunOptions)
    return FrontendRunSpec(
        pipeline=FrontendConfig(**plain),
        name=name,
        colmap_runtime=runtime,
        selection=selection,
        run=run,
    )


def build_mapping_config(
    source,
    *,
    source_name,
    projected_cli_values=None,
    override_tokens=(),
    name=None,
    config_policy=None,
):
    raw = compose_raw_config(
        source,
        source_name=source_name,
        stage="mapping",
        projected_cli_values=projected_cli_values,
        override_tokens=override_tokens,
        name=name,
        config_policy=config_policy,
    )
    from vidmap.configuration.config import MappingConfig, MappingRunSpec
    from vidmap.configuration.defaults import EvaluationOptions, MappingRunOptions, TargetSelectionOptions

    plain = materialize_config_values(
        raw,
        {
            **dataclasses.asdict(MappingConfig()),
            **dataclasses.asdict(TargetSelectionOptions()),
            **dataclasses.asdict(MappingRunOptions()),
            **dataclasses.asdict(EvaluationOptions()),
            "name": None,
            "colmap_runtime": "stock",
        },
    )
    name = plain.pop("name")
    runtime = plain.pop("colmap_runtime")
    _validate_safe_component(name, "Config name")
    _validate_safe_component(runtime, "COLMAP runtime")
    selection = _pop_dataclass_values(plain, TargetSelectionOptions)
    run = _pop_dataclass_values(plain, MappingRunOptions)
    evaluation = _pop_dataclass_values(plain, EvaluationOptions)
    return MappingRunSpec(
        pipeline=MappingConfig(**plain),
        name=name,
        colmap_runtime=runtime,
        selection=selection,
        run=run,
        evaluation=evaluation,
    )


def compose_raw_config(
    source,
    *,
    source_name,
    stage,
    projected_cli_values=None,
    override_tokens=(),
    name=None,
    config_policy=None,
):
    stage_root = FRONTEND_CONFIG_DIR if stage == "frontend" else MAPPING_CONFIG_DIR
    source_path = None if source is None else Path(source).resolve()
    root = (
        stage_root if source_path is None or source_path.is_relative_to(stage_root.resolve()) else source_path.parent
    )
    raw = OmegaConf.create({}) if source is None else load_config_source(Path(source), config_root=root)
    unresolved = OmegaConf.to_container(raw, resolve=False)
    if stage == "frontend" and "name" in unresolved:
        raise ValueError("Frontend config YAML cannot set name; identity comes from its root-relative config name")
    parsed_overrides = parse_override_tokens(override_tokens)
    for key, value in parsed_overrides:
        if _is_reproducibility_key(key):
            OmegaConf.update(raw, key, value, merge=True, force_add=True)

    if stage == "mapping":
        if "frontend_tag" in raw:
            raise ValueError("Mapping configs no longer accept frontend_tag; select the frontend config explicitly")
        from vidmap.repro.profile import apply_mapper_reproducibility_policy, apply_reproducibility_profile

        raw = apply_reproducibility_profile(raw)
    elif OmegaConf.select(raw, "reproducibility") is not None:
        raise ValueError("reproducibility profiles are mapping-only; use frontend-owned deterministic controls")

    projected_cli_conf = OmegaConf.create(dict(projected_cli_values or {}))
    raw = OmegaConf.merge(raw, projected_cli_conf)
    if stage == "mapping" and "frontend_tag" in raw:
        raise ValueError("Mapping configs no longer accept frontend_tag; select the frontend config explicitly")
    if config_policy is not None:
        config_policy(raw)
    explicit_keys = set()
    for key, value in parsed_overrides:
        if _is_reproducibility_key(key):
            continue
        explicit_keys.add(key)
        explicit_keys.update(_mapping_leaf_keys(key, value))
        OmegaConf.update(raw, key, value, merge=True, force_add=True)
    if stage == "mapping":
        raw = apply_mapper_reproducibility_policy(raw, projected_cli_conf, explicit_keys)

    if stage == "frontend":
        raw.name = source_name
    else:
        if "name" not in raw:
            raw.name = config_name_to_output_slug(source_name)
        if name is not None:
            raw.name = name
    return raw


def materialize_config_values(raw, defaults):
    raw = OmegaConf.create(OmegaConf.to_container(raw, resolve=False))
    merged = OmegaConf.merge(OmegaConf.create(defaults), raw)
    return OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)


def _pop_dataclass_values(plain, model_type):
    values = {field.name: plain.pop(field.name) for field in dataclasses.fields(model_type)}
    return model_type(**values)


def _validate_safe_component(value, label, *, allow_path=False):
    if allow_path:
        valid = (
            isinstance(value, str)
            and "\\" not in value
            and not value.startswith("/")
            and all(part not in {"", ".", ".."} for part in value.split("/"))
        )
        requirement = "a non-empty safe root-relative config name"
    else:
        valid = isinstance(value, str) and re.fullmatch(r"(?!\.{1,2}$)[^/\\]+", value) is not None
        requirement = "one non-empty safe output path component"
    if not valid:
        raise ValueError(f"{label} must be {requirement}; got {value!r}")


def _mapping_leaf_keys(prefix, value):
    if not isinstance(value, Mapping):
        return set()
    leaves = set()
    for key, child in value.items():
        path = f"{prefix}.{key}"
        nested = _mapping_leaf_keys(path, child)
        leaves.update(nested or {path})
    return leaves


def config_source_name(source, *, stage):
    path = Path(source).resolve()
    root = (FRONTEND_CONFIG_DIR if stage == "frontend" else MAPPING_CONFIG_DIR).resolve()
    if path.is_relative_to(root):
        return path.relative_to(root).as_posix().removesuffix(".yaml").removesuffix(".yml")
    return path.stem


def _is_reproducibility_key(key):
    return key == "reproducibility" or key.startswith("reproducibility.")


def load_config_source(path, override_tokens=(), *, config_root=None):
    path = Path(path).resolve()
    root = path.parent if config_root is None else Path(config_root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Config composition escapes stage root {root}: {path}")
    try:
        conf = OmegaConf.load(path)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise TypeError(f"Expected mapping config in {str(path)!r}") from exc
    if not isinstance(conf, DictConfig):
        raise TypeError(f"Expected mapping config in {str(path)!r}, got {type(conf).__name__}")
    for default in conf.pop("defaults", []):
        if isinstance(default, str):
            base_path = (path.parent / f"{default}.yaml").resolve()
            conf = OmegaConf.merge(load_config_source(base_path, config_root=root), conf)
            continue
        if not isinstance(default, (dict, DictConfig)) or len(default) != 1:
            raise TypeError(f"Expected string or single-item dict in defaults, got {type(default).__name__}")
        key, value = next(iter(default.items()))
        match = re.fullmatch(r"(.+?)@(.+)", key)
        if match is None:
            raise ValueError(f"Unexpected format in defaults: {default}")
        config_dir, target = match.groups()
        base = load_config_source((path.parent / config_dir / f"{value}.yaml").resolve(), config_root=root)
        if target == ".":
            conf = OmegaConf.merge(base, conf)
        else:
            current = OmegaConf.select(conf, target)
            OmegaConf.update(conf, target, OmegaConf.merge(base, current) if current else base)
    for key, value in parse_override_tokens(override_tokens):
        OmegaConf.update(conf, key, value, merge=True, force_add=True)
    return conf


def _matches_option(token, option) -> bool:
    if token == option:
        return True
    if option.startswith("--"):
        return token.startswith(f"{option}=")
    return len(option) == 2 and token.startswith(option) and len(token) > len(option)


def _option_is_present(argv, option_strings) -> bool:
    return any(_matches_option(token, option) for token in argv for option in option_strings)


def _remove_implicit_routing_defaults(args: Namespace, argv) -> None:
    for destination in _SUPPRESSED_DESTINATIONS:
        if destination in vars(args) and not _option_is_present(argv, _ROUTING_OPTION_STRINGS[destination]):
            delattr(args, destination)


def _reject_overrides_swallowed_by_variadic_args(parser: ArgumentParser, argv):
    for index, token in enumerate(argv):
        option = next((option for option in _VARIADIC_OPTIONS if _matches_option(token, option)), None)
        canonical = None if option is None else _VARIADIC_OPTIONS[option]
        if canonical is None:
            continue
        for candidate in argv[index + 1 :]:
            if candidate == "--" or candidate.startswith("-"):
                break
            if _OVERRIDE_SHAPED.match(candidate):
                parser.error(
                    f"config override {candidate!r} would be consumed by variadic option {canonical}; "
                    f"put '--' before config overrides"
                )
