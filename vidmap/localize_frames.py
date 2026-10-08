"""Localize all video frames of a finished VidMap run against its reconstruction.

Usage:
    python -m vidmap.localize_frames --run-dir <run> [--fps 20] [--batch-size 8] [--num-refs 1] [--leave-one-out]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

logger = logging.getLogger("vidmap.localize_frames")


def _frames_dir(run_dir: Path) -> Path:
    root = run_dir / "video_frames"
    subs = [p for p in root.iterdir() if p.is_dir()]
    return subs[0] if len(subs) == 1 else root


def leave_one_out_errors(result, models) -> dict[str, float]:
    """Compare leave-one-out poses of keyframes to their bundle-adjusted poses."""
    import pycolmap

    from vidmap.localization.frames import collect_keyframes

    keyframes = {k.name: k for k in collect_keyframes(models)}
    ba = {name: k.cam_from_world for name, k in keyframes.items()}
    rot, pos, focal = [], [], []
    for i, name in enumerate(result.names):
        if not result.success[i]:
            continue
        est = pycolmap.Rigid3d(pycolmap.Rotation3d(result.rotation_xyzw[i]), result.translation[i])
        ref = ba[name]
        dR = (est.rotation * ref.rotation.inverse()).angle()
        rot.append(np.degrees(dR))
        pos.append(np.linalg.norm(est.inverse().translation - ref.inverse().translation))
        f_ba = keyframes[name].camera.mean_focal_length()
        focal.append(abs(result.focal_length[i] - f_ba) / f_ba)
    rot, pos, focal = np.array(rot), np.array(pos), np.array(focal)
    return {
        "num": int(len(result.names)),
        "success_rate": float(np.mean(result.success)),
        "rot_err_median_deg": float(np.median(rot)) if len(rot) else float("nan"),
        "rot_err_p90_deg": float(np.percentile(rot, 90)) if len(rot) else float("nan"),
        "pos_err_median_m": float(np.median(pos)) if len(pos) else float("nan"),
        "pos_err_p90_m": float(np.percentile(pos, 90)) if len(pos) else float("nan"),
        "focal_rel_err_median": float(np.median(focal)) if len(focal) else float("nan"),
        "focal_rel_err_p90": float(np.percentile(focal, 90)) if len(focal) else float("nan"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rec-dir", type=Path, help="Defaults to <run-dir>/rec")
    parser.add_argument("--frames-dir", type=Path, help="Defaults to <run-dir>/video_frames[/<single subdir>]")
    parser.add_argument("--output", type=Path, help="Defaults to <run-dir>/frame_poses_<fps>fps.npz")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-refs", type=int, default=1)
    parser.add_argument("--max-error-px", type=float, default=8.0)
    parser.add_argument("--min-certainty", type=float, default=0.1)
    parser.add_argument("--pnp-workers", type=int, default=8)
    parser.add_argument("--io-workers", type=int, default=6)
    parser.add_argument("--leave-one-out", action="store_true", help="Validate by re-localizing the keyframes.")
    parser.add_argument("--coarse-only", action="store_true", help="Skip the RoMaV2 refiners (faster, less accurate).")
    parser.add_argument("--fixed-focal", action="store_true", help="Do not refine the focal length per frame.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    from vidmap.mapper.runtime import load_mapping_runtime

    load_mapping_runtime()
    from vidmap.localization.frames import FrameLocalizationOptions, FrameLocalizer, load_reconstructions

    options = FrameLocalizationOptions(
        fps=args.fps,
        batch_size=args.batch_size,
        num_reference_keyframes=args.num_refs,
        max_error_px=args.max_error_px,
        min_certainty=args.min_certainty,
        num_pnp_workers=args.pnp_workers,
        num_io_workers=args.io_workers,
        coarse_only=args.coarse_only,
        refine_focal_length=not args.fixed_focal,
    )
    import time

    import torch

    run_dir = args.run_dir
    models = load_reconstructions(args.rec_dir or run_dir / "rec")
    frames_dir = args.frames_dir or _frames_dir(run_dir)
    t0 = time.perf_counter()
    localizer = FrameLocalizer(options)
    model_load_s = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    result = localizer.localize(models, frames_dir, leave_one_out=args.leave_one_out)
    result.timings["model_load"] = model_load_s
    result.timings["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9

    tag = "loo" if args.leave_one_out else f"{args.fps:g}fps"
    tag += "_coarse" if args.coarse_only else ""
    output = args.output or run_dir / f"frame_poses_{tag}_b{args.batch_size}_r{args.num_refs}.npz"
    result.save(output)
    summary = {
        "options": asdict(options),
        "num_queries": len(result.names),
        "num_matched": int((~result.is_keyframe).sum()),
        "success_rate_matched": (
            float(result.success[~result.is_keyframe].mean()) if (~result.is_keyframe).any() else 1.0
        ),
        "median_inliers": (
            float(np.median(result.num_inliers[result.success & ~result.is_keyframe]))
            if (result.success & ~result.is_keyframe).any()
            else 0.0
        ),
        "timings": result.timings,
    }
    matched = int((~result.is_keyframe).sum())
    summary["matched_frames_per_second"] = matched / result.timings["total"] if result.timings.get("total") else 0.0
    if args.leave_one_out:
        summary["leave_one_out"] = leave_one_out_errors(result, models)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
