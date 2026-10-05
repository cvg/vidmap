from omegaconf import DictConfig, OmegaConf

_PROFILE_FIELDS = {"profile", "write_stage"}


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

    patch = {
        "mapper": {
            "gp": {
                "common": {
                    "num_threads": 1,
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
