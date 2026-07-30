from omegaconf import DictConfig, OmegaConf

_PROFILE_FIELDS = {"profile", "write_stage"}


def apply_mapper_reproducibility_policy(
    conf: DictConfig,
    projected_cli_conf: DictConfig,
    explicit_keys,
) -> DictConfig:
    canonical_key = "mapper.gp.common.canonical_checkpoint"
    update_errors_key = "mapper.gp.track_filter.update_point3d_errors"
    explicit_keys = set(explicit_keys)
    explicit_keys.update(
        key for key in (canonical_key, update_errors_key) if OmegaConf.select(projected_cli_conf, key) is not None
    )
    if canonical_key in explicit_keys and update_errors_key not in explicit_keys:
        OmegaConf.update(conf, update_errors_key, OmegaConf.select(conf, canonical_key))
    return conf


def apply_reproducibility_profile(conf: DictConfig) -> DictConfig:
    node = conf.pop("reproducibility", None)
    if node is None:
        return conf
    if not isinstance(node, DictConfig):
        raise TypeError(f"Expected reproducibility mapping, got {type(node).__name__}")
    repro = OmegaConf.to_container(node, resolve=True)
    unknown_fields = sorted(set(repro) - _PROFILE_FIELDS)
    if unknown_fields:
        raise ValueError(f"Unsupported reproducibility fields: {', '.join(unknown_fields)}")

    profile = repro.get("profile", "off")
    if profile in {None, "off"}:
        return conf
    if profile != "byte_check":
        raise ValueError(f"Unsupported reproducibility.profile={profile!r}")

    canonical_gp_checkpoint = bool(OmegaConf.select(conf, "mapper.gp.common.canonical_checkpoint", default=False))
    patch = {
        "mapper": {
            "gp": {
                "common": {
                    "roundtrip_before_ba": True,
                    "canonical_checkpoint": canonical_gp_checkpoint,
                    "num_threads": 1,
                    "parameter_ordering_strategy": "deterministic_singleton_groups",
                    "camera_center_strategy": "image",
                },
                "second_pass": {"center_init_mode": "python_frame_centers"},
                "track_filter": {
                    "update_point3d_errors": canonical_gp_checkpoint,
                    "skip_zero_observation_points": True,
                },
            },
            "replay_cache": {
                "mode": "byte_check",
                **({"write_stage": repro["write_stage"]} if repro.get("write_stage") is not None else {}),
            },
            "ba": {"num_threads": 1},
        }
    }
    return OmegaConf.merge(conf, OmegaConf.create(patch))
