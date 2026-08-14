"""Researcher-facing reconstruction workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path

from PIL import Image

from vidmap.datasets.local import IMAGE_SUFFIXES, LocalImageParser
from vidmap.mapper import Mapper
from vidmap.mapper.inputs import MapperInputs
from vidmap.run_options import RunOptions

logger = logging.getLogger(__name__)

_VIDEO_SUFFIXES = frozenset({".mp4"})
_FRAME_PATTERN = "%06d.jpg"
_TIMESTAMP_FRAME_PATTERN = "timestamp-seconds.jpg"
_TIMESTAMP_FILENAME_WIDTH = 20
_VIDEO_MANIFEST_NAME = "video_frames.json"
_VIDEO_MANIFEST_SCHEMA_VERSION = 1
_VIDEO_DECODE_PROFILE = "ffmpeg-jpeg-q2-vsync0-best-effort-timestamps-v1"
LOCAL_INPUT_MANIFEST_NAME = "local_input.json"
_LOCAL_INPUT_SCHEMA_VERSION = 1


def reconstruct(
    mapping_conf,
    frontend_conf,
    input_path: str | Path,
    *,
    workspace: str | Path,
    mapper_inputs_dir: str | Path | MapperInputs,
    run_options: RunOptions | None = None,
    imnames=None,
    intrinsics_path: str | Path | None = None,
    overwrite_outputs: bool = False,
    output_dir: str | Path | None = None,
):
    """Reconstruct an image directory or MP4 using the shared mapping API."""
    mapper_inputs = resolve_mapper_inputs(frontend_conf, mapper_inputs_dir)
    use_geocalib = frontend_conf.pipeline.use_geocalib
    if use_geocalib and intrinsics_path is not None:
        raise ValueError("The selected uncalibrated config cannot be combined with --intrinsics")
    if not use_geocalib and intrinsics_path is None:
        raise ValueError("The selected calibrated config requires --intrinsics")

    workspace = Path(workspace).expanduser()
    image_dir = prepare_reconstruction_images(input_path, workspace)
    scene_parser = LocalImageParser(
        image_dir=image_dir,
        imnames=imnames,
        intrinsics_path=intrinsics_path,
        use_geocalib=use_geocalib,
    )
    output_dir = workspace if output_dir is None else Path(output_dir).expanduser()
    run_options = RunOptions() if run_options is None else run_options
    return run_mapping(
        mapping_conf,
        frontend_conf=frontend_conf,
        mapper_inputs=mapper_inputs,
        run_options=run_options,
        scene_parser=scene_parser,
        output_dir=output_dir,
        scene_name=scene_parser.scene,
        overwrite_outputs=overwrite_outputs,
    )


def resolve_mapper_inputs(frontend_conf, mapper_inputs: str | Path | MapperInputs) -> MapperInputs:
    """Attach the expected frontend identity without hashing mapper payloads."""
    from vidmap.frontend.identity import frontend_config_identity
    from vidmap.mapper.inputs import validate_mapper_inputs_identity

    expected_identity = frontend_config_identity(frontend_conf)
    if isinstance(mapper_inputs, MapperInputs):
        combined_identity = dict(mapper_inputs.expected_identity or {})
        conflicts = {
            key: (combined_identity[key], value)
            for key, value in expected_identity.items()
            if key in combined_identity and combined_identity[key] != value
        }
        if conflicts:
            raise ValueError(f"Mapper-input frontend identity mismatch with the selected config: {conflicts!r}")
        combined_identity.update(expected_identity)
        inputs = replace(mapper_inputs, expected_identity=combined_identity)
    else:
        inputs = MapperInputs.from_directory(
            Path(mapper_inputs),
            expected_identity=expected_identity,
        )
    validate_mapper_inputs_identity(inputs.directory, inputs.expected_identity or {})
    return inputs


def prepare_reconstruction_images(input_path: str | Path, workspace: str | Path) -> Path:
    """Return an image directory, decoding an MP4 into the workspace when needed."""
    source = Path(input_path).expanduser().resolve()
    if source.is_dir():
        return source
    if not source.is_file():
        raise FileNotFoundError(f"Reconstruction input does not exist: {source}")
    if source.suffix.lower() not in _VIDEO_SUFFIXES:
        raise ValueError(f"Reconstruction input must be an image directory or an MP4 file, got {source}")
    return decode_video_frames(source, Path(workspace).expanduser())


def write_local_input_provenance(directory: str | Path, image_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Record the resolved RGB root needed to visualize a local reconstruction."""
    directory = Path(directory).expanduser()
    image_dir = Path(image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Local input image directory is missing: {image_dir}")
    _validate_video_frame_manifest(image_dir)
    path = directory / LOCAL_INPUT_MANIFEST_NAME
    payload = {
        "schema_version": _LOCAL_INPUT_SCHEMA_VERSION,
        "image_dir": str(image_dir),
    }
    if path.is_file() and not overwrite:
        if _read_local_input_provenance(path) != image_dir:
            raise ValueError(f"Local input provenance already identifies different images: {path}")
        return path
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def local_input_image_dir(mapper_inputs: str | Path | MapperInputs) -> Path | None:
    """Return the RGB root recorded in a mapper-inputs boundary, when present."""
    directory = (
        Path(mapper_inputs).expanduser() if isinstance(mapper_inputs, (str, Path)) else Path(mapper_inputs.directory)
    )
    source = directory / LOCAL_INPUT_MANIFEST_NAME
    return _read_local_input_provenance(source) if source.is_file() else None


def extract_point_colors(
    reconstruction,
    mapper_inputs: str | Path | MapperInputs,
    *,
    image_dir: str | Path | None = None,
) -> bool:
    """Sample point colors from the run's RGB images so COLMAP output is not uniformly black.

    Colors are cosmetic, so a missing or unreadable image root degrades to a warning
    rather than discarding a finished reconstruction.
    """
    if image_dir is None:
        try:
            image_dir = local_input_image_dir(mapper_inputs)
        except (FileNotFoundError, ValueError) as error:
            logger.warning("Leaving points uncolored: %s", error)
            return False
    if image_dir is None:
        logger.warning("Leaving points uncolored: no local input provenance beside %s", mapper_inputs)
        return False
    image_dir = Path(image_dir).expanduser()
    try:
        reconstruction.extract_colors_for_all_images(str(image_dir))
    except Exception:
        logger.warning("Leaving points uncolored: color extraction failed for %s", image_dir, exc_info=True)
        return False
    logger.info("Extracted point colors from %s", image_dir)
    return True


def local_run_image_dir(run: str | Path) -> Path | None:
    """Return the RGB root recorded in a local run's mapper inputs."""
    return local_input_image_dir(Path(run).expanduser() / "mapper_inputs")


def _read_local_input_provenance(path: Path) -> Path:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid local input provenance: {path}") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "image_dir"}
        or payload["schema_version"] != _LOCAL_INPUT_SCHEMA_VERSION
        or isinstance(payload["schema_version"], bool)
        or not isinstance(payload["image_dir"], str)
        or not payload["image_dir"]
    ):
        raise ValueError(f"Invalid local input provenance: {path}")
    image_dir = Path(payload["image_dir"]).expanduser()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Recorded local input image directory is missing: {image_dir}")
    return image_dir


def decode_video_frames(video_path: str | Path, workspace: str | Path) -> Path:
    """Decode an MP4 to a persistent, content-addressed image directory."""
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if video_path.suffix.lower() not in _VIDEO_SUFFIXES:
        raise ValueError(f"Expected an MP4 video, got {video_path}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("FFmpeg is required to reconstruct an MP4 video")

    digest = _file_digest(video_path)
    frames_root = Path(workspace).expanduser() / "video_frames"
    frames_dir = frames_root / f"{video_path.stem}-v{_VIDEO_MANIFEST_SCHEMA_VERSION}-{digest[:16]}"
    frames_root.mkdir(parents=True, exist_ok=True)
    if frames_dir.exists():
        _validate_decoded_frames(frames_dir, digest)
        return frames_dir

    staging = Path(tempfile.mkdtemp(prefix=f".{frames_dir.name}.", dir=frames_root))
    try:
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-q:v",
            "2",
            str(staging / _FRAME_PATTERN),
        ]
        subprocess.run(command, check=True)
        sequential_frames = _validated_frame_names(staging)
        frame_times = _probe_video_frame_timestamps(video_path)
        if len(frame_times) != len(sequential_frames):
            raise RuntimeError(
                f"Decoded frame count does not match video timestamps: "
                f"{len(sequential_frames)} images vs {len(frame_times)} timestamps"
            )
        frames = []
        for source, timestamp in zip(sequential_frames, frame_times, strict=True):
            name = f"{timestamp:0{_TIMESTAMP_FILENAME_WIDTH}.9f}.jpg"
            destination = staging / name
            if destination.exists() and destination.name != source:
                raise RuntimeError(f"Video frame timestamps are not unique at nanosecond precision: {name}")
            (staging / source).rename(destination)
            frames.append(name)
        (staging / _VIDEO_MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "schema_version": _VIDEO_MANIFEST_SCHEMA_VERSION,
                    "decode_profile": _VIDEO_DECODE_PROFILE,
                    "source": str(video_path),
                    "sha256": digest,
                    "frame_pattern": _TIMESTAMP_FRAME_PATTERN,
                    "frames": frames,
                    "frame_timestamps_seconds": frame_times,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.rename(frames_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return frames_dir


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_decoded_frames(path: Path, expected_digest: str) -> None:
    manifest_path = path / _VIDEO_MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(f"Decoded video cache has no manifest: {path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Decoded video cache has invalid provenance: {path}") from error
    if (
        not _is_current_video_frame_manifest(manifest)
        or manifest["sha256"] != expected_digest
        or manifest["frames"] != _validated_frame_names(path)
    ):
        raise RuntimeError(f"Decoded video cache has invalid provenance: {path}")
    _validate_frame_timestamps(manifest["frames"], manifest["frame_timestamps_seconds"], path)


def decoded_video_frame_timestamps(image_dir: str | Path) -> dict[str, float] | None:
    """Return persisted MP4 presentation timestamps keyed by decoded image name."""
    image_dir = Path(image_dir).expanduser()
    manifest_path = image_dir / _VIDEO_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    _validate_video_frame_manifest(image_dir)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid decoded-video provenance: {manifest_path}") from error
    if "frame_timestamps_seconds" not in manifest:
        raise ValueError(f"Decoded-video provenance does not use the current timestamped format: {manifest_path}")
    frames = manifest["frames"]
    timestamps = manifest["frame_timestamps_seconds"]
    _validate_frame_timestamps(frames, timestamps, Path(image_dir))
    return dict(zip(frames, (float(value) for value in timestamps), strict=True))


def _validate_video_frame_manifest(image_dir: Path) -> None:
    manifest_path = image_dir / _VIDEO_MANIFEST_NAME
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid decoded-video provenance: {manifest_path}") from error
    if not _is_current_video_frame_manifest(manifest):
        raise ValueError(f"Decoded-video provenance does not use the current format: {manifest_path}")
    try:
        decoded_frames = _validated_frame_names(image_dir)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Decoded-video provenance has invalid frames: {manifest_path}") from error
    if manifest["frames"] != decoded_frames:
        raise ValueError(f"Decoded-video provenance does not match decoded frames: {manifest_path}")
    _validate_frame_timestamps(manifest["frames"], manifest["frame_timestamps_seconds"], image_dir)


def _is_current_video_frame_manifest(manifest: object) -> bool:
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "decode_profile",
        "frame_pattern",
        "frames",
        "frame_timestamps_seconds",
        "sha256",
        "source",
    }:
        return False
    schema_version = manifest["schema_version"]
    digest = manifest["sha256"]
    frames = manifest["frames"]
    return (
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version == _VIDEO_MANIFEST_SCHEMA_VERSION
        and manifest["decode_profile"] == _VIDEO_DECODE_PROFILE
        and manifest["frame_pattern"] == _TIMESTAMP_FRAME_PATTERN
        and isinstance(manifest["source"], str)
        and bool(manifest["source"])
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        and isinstance(frames, list)
        and bool(frames)
        and all(isinstance(name, str) and name for name in frames)
    )


def _probe_video_frame_timestamps(video_path: Path) -> list[float]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FileNotFoundError("FFprobe is required to preserve MP4 frame timestamps")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    try:
        entries = json.loads(result.stdout)["frames"]
        timestamps = [float(entry["best_effort_timestamp_time"]) for entry in entries]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"FFprobe returned invalid frame timestamps for {video_path}") from error
    _validate_frame_timestamps(tuple(str(index) for index in range(len(timestamps))), timestamps, video_path)
    start = timestamps[0]
    return [timestamp - start for timestamp in timestamps]


def _validate_frame_timestamps(frames, timestamps, source: Path) -> None:
    if (
        not isinstance(frames, (list, tuple))
        or not isinstance(timestamps, (list, tuple))
        or len(frames) != len(timestamps)
        or not frames
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in timestamps)
    ):
        raise ValueError(f"Invalid decoded-video frame timestamps: {source}")
    values = [float(value) for value in timestamps]
    if any(not math.isfinite(value) for value in values) or any(
        current <= previous for previous, current in zip(values, values[1:])
    ):
        raise ValueError(f"Invalid decoded-video frame timestamps: {source}")


def _validated_frame_names(path: Path) -> list[str]:
    images = sorted(entry for entry in path.iterdir() if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"FFmpeg did not decode any supported images into {path}")
    for image_path in images:
        with Image.open(image_path) as image:
            image.verify()
    return [image_path.name for image_path in images]


def run_mapping(
    mapping_conf,
    *,
    frontend_conf,
    mapper_inputs: str | Path | MapperInputs,
    output_dir: str | Path,
    scene_name: str | None = None,
    run_options: RunOptions | None = None,
    scene_parser=None,
    overwrite_outputs: bool = False,
):
    """Map one finalized boundary and return its reconstruction."""
    from vidmap.configuration.dump import dump_mapping_configs, validate_mapping_config_provenance

    mapper_inputs = resolve_mapper_inputs(frontend_conf, mapper_inputs)
    if scene_name is None:
        scene_name = str(mapper_inputs.frontend_identity()["scene"])
    run_options = RunOptions() if run_options is None else run_options
    output_dir = Path(output_dir).expanduser()

    def initialize_output() -> None:
        validate_mapping_config_provenance(
            frontend_conf,
            mapping_conf,
            output_dir,
            overwrite=overwrite_outputs,
            context="Reconstruction run",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_mapping_configs(
            frontend_conf,
            mapping_conf,
            output_dir,
            overwrite=overwrite_outputs,
        )

    logger.info("Starting mapping for %s from %s", scene_name, mapper_inputs.directory)
    reconstruction = Mapper(
        conf=mapping_conf.pipeline.mapper,
        mapper_inputs=mapper_inputs,
        sfm_outputs_dir=output_dir,
        persist_intermediate_reconstructions=mapping_conf.run.persist_intermediate_reconstructions,
    ).run(
        save_playback_trace=run_options.save_playback_trace,
        playback_trace_stride=run_options.playback_trace_stride,
        playback_trace_point_cap=run_options.playback_trace_point_cap,
        overwrite_outputs=overwrite_outputs,
        on_inputs_validated=initialize_output,
    )
    extract_point_colors(
        reconstruction,
        mapper_inputs,
        image_dir=None if scene_parser is None else scene_parser.rgb_dir,
    )
    if mapper_inputs.full_depth_maps_path is not None:
        from vidmap.depth_artifacts import write_reference

        write_reference(output_dir, mapper_inputs)
    logger.info(
        "Reconstruction complete with %d/%d registered images",
        reconstruction.num_reg_images(),
        reconstruction.num_images(),
    )
    return reconstruction
