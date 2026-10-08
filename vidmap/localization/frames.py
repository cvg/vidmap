"""Localize all video frames against a finished VidMap reconstruction.

Each query frame is matched with RoMaV2 (batched on the GPU) to its temporally nearest keyframe, and to the keyframe
on its other side in time if that match explains few of the reference's 3D points (e.g. across a shot cut). The dense
warp keyframe -> frame is sampled at the keyframe's observations of triangulated 3D points, which yields 2D-3D
correspondences for a robust absolute pose with per-frame focal refinement (LO-RANSAC + refinement, run in a CPU
thread pool that overlaps the GPU matching). No new points are triangulated and no bundle adjustment is run.
"""

from __future__ import annotations

import logging
import time
from bisect import bisect_left
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameLocalizationOptions:
    fps: float = 20.0
    batch_size: int = 8
    resolution: int = 560
    # Candidate references are the keyframes immediately before/after the frame. With 1, the nearest is matched
    # first and the other only if the pose is weak (see fallback_min_covisibility); with 2, both are always matched.
    # Each (frame, reference) pair is solved independently and the pose with the most inliers is kept, which
    # resolves frames next to shot cuts whose nearest keyframe lies on the other side of the cut.
    num_reference_keyframes: int = 1
    min_certainty: float = 0.1
    max_error_px: float = 8.0
    min_inliers: int = 30
    # Also match the other bracketing keyframe if fewer than this fraction of the reference's 3D points are inliers.
    fallback_min_covisibility: float = 0.5
    # Refine the focal length per frame, initialized from the reference keyframe (handles zoom changes).
    refine_focal_length: bool = True
    num_pnp_workers: int = 8
    num_io_workers: int = 6
    keyframe_feature_cache_size: int = 16
    # Skip the RoMaV2 refiners (coarse warp only): faster, less accurate.
    coarse_only: bool = False


@dataclass
class _Keyframe:
    name: str
    timestamp: float
    model_index: int
    camera: pycolmap.Camera
    cam_from_world: pycolmap.Rigid3d
    xy: np.ndarray  # (N, 2) observations of triangulated points, in image pixels
    xyz: np.ndarray  # (N, 3)


@dataclass
class FrameLocalizationResult:
    names: list[str]
    timestamps: np.ndarray
    success: np.ndarray
    is_keyframe: np.ndarray
    model_index: np.ndarray
    rotation_xyzw: np.ndarray  # cam_from_world
    translation: np.ndarray
    focal_length: np.ndarray
    num_correspondences: np.ndarray
    num_inliers: np.ndarray
    reference_names: list[str]
    timings: dict[str, float] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            names=np.array(self.names),
            timestamps=self.timestamps,
            success=self.success,
            is_keyframe=self.is_keyframe,
            model_index=self.model_index,
            rotation_xyzw=self.rotation_xyzw,
            translation=self.translation,
            focal_length=self.focal_length,
            num_correspondences=self.num_correspondences,
            num_inliers=self.num_inliers,
            reference_names=np.array(self.reference_names),
        )
        with open(path.with_suffix(".txt"), "w") as f:
            f.write("# name timestamp model qw qx qy qz tx ty tz focal num_inliers (cam_from_world)\n")
            for i, name in enumerate(self.names):
                if not self.success[i]:
                    continue
                qx, qy, qz, qw = self.rotation_xyzw[i]
                tx, ty, tz = self.translation[i]
                f.write(
                    f"{name} {self.timestamps[i]:.6f} {self.model_index[i]} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tx:.6f} {ty:.6f} {tz:.6f} {self.focal_length[i]:.3f} {self.num_inliers[i]}\n"
                )


def frame_timestamp(name: str) -> float:
    return float(Path(name).stem)


def load_reconstructions(rec_dir: str | Path) -> list[pycolmap.Reconstruction]:
    rec_dir = Path(rec_dir)
    if (rec_dir / "images.bin").exists() or (rec_dir / "images.txt").exists():
        return [pycolmap.Reconstruction(str(rec_dir))]
    subs = sorted(p for p in rec_dir.iterdir() if p.is_dir() and (p / "images.bin").exists())
    if not subs:
        raise FileNotFoundError(f"No reconstruction found in {rec_dir}")
    return [pycolmap.Reconstruction(str(p)) for p in subs]


def collect_keyframes(models: list[pycolmap.Reconstruction]) -> list[_Keyframe]:
    keyframes = []
    for model_index, rec in enumerate(models):
        for image in rec.images.values():
            if not image.has_pose:
                continue
            points3D = rec.points3D
            obs = [
                (p.xy, points3D[p.point3D_id].xyz)
                for p in image.points2D
                if p.has_point3D() and p.point3D_id in points3D
            ]
            if not obs:
                continue
            keyframes.append(
                _Keyframe(
                    name=Path(image.name).name,
                    timestamp=frame_timestamp(image.name),
                    model_index=model_index,
                    camera=rec.cameras[image.camera_id],
                    cam_from_world=image.cam_from_world(),
                    xy=np.array([o[0] for o in obs], dtype=np.float64),
                    xyz=np.array([o[1] for o in obs], dtype=np.float64),
                )
            )
    keyframes.sort(key=lambda k: k.timestamp)
    return keyframes


def subsample_frames(names: list[str], fps: float) -> list[str]:
    """Pick the frame closest to each tick of a regular grid at the requested rate."""
    times = np.array([frame_timestamp(n) for n in names])
    order = np.argsort(times)
    names = [names[i] for i in order]
    times = times[order]
    if fps <= 0:
        return names
    ticks = np.arange(times[0], times[-1] + 1e-9, 1.0 / fps)
    idx = np.clip(np.searchsorted(times, ticks), 1, len(times) - 1)
    idx = np.where(np.abs(times[idx - 1] - ticks) <= np.abs(times[idx] - ticks), idx - 1, idx)
    return [names[i] for i in sorted(set(idx.tolist()))]


def bracketing_keyframes(
    timestamp: float, keyframes: list[_Keyframe], key_times: list[float], exclude: str | None = None
) -> list[int]:
    """Indices of the keyframes immediately before and after a timestamp (any submodel), nearest first."""
    pos = bisect_left(key_times, timestamp)
    lo, hi = pos - 1, pos
    while lo >= 0 and keyframes[lo].name == exclude:
        lo -= 1
    while hi < len(keyframes) and keyframes[hi].name == exclude:
        hi += 1
    candidates = [i for i in (lo, hi) if 0 <= i < len(keyframes)]
    return sorted(candidates, key=lambda i: abs(key_times[i] - timestamp))


class _Timer:
    def __init__(self):
        self.totals: OrderedDict[str, float] = OrderedDict()
        self.counts: dict[str, int] = {}

    def add(self, key: str, seconds: float):
        self.totals[key] = self.totals.get(key, 0.0) + seconds
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.counts[key] == 1:
            self.totals[f"{key}_first_call"] = seconds

    def gpu(self, key: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                torch.cuda.synchronize()
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                torch.cuda.synchronize()
                timer.add(key, time.perf_counter() - self.t0)

        return _Ctx()


def _pnp_job(points2D, points3D, camera, options: FrameLocalizationOptions):
    t0 = time.perf_counter()
    if len(points2D) < max(options.min_inliers, 6):
        return None, time.perf_counter() - t0
    est = pycolmap.AbsolutePoseEstimationOptions()
    est.ransac.max_error = options.max_error_px
    ref = pycolmap.AbsolutePoseRefinementOptions()
    ref.refine_focal_length = options.refine_focal_length
    ref.refine_extra_params = False
    ref.print_summary = False
    result = pycolmap.estimate_and_refine_absolute_pose(points2D, points3D, camera, est, ref)
    return result, time.perf_counter() - t0


class FrameLocalizer:
    def __init__(self, options: FrameLocalizationOptions, device: torch.device | str = "cuda", model=None):
        self.options = options
        self.device = torch.device(device)
        if model is None:
            from vidmap.frontend.models.romav2 import load_romav2_model
            from vidmap.frontend.options.matching import RoMaV2Options

            model = load_romav2_model(RoMaV2Options(compile=True), device=self.device)
        self.model = model

    def _dataset(self, frames_dir: Path, names: list[str]):
        from vidmap.frontend.image_dataset import ImageDatasetOptions
        from vidmap.frontend.video_images import RomaVideoImageDataset

        r = self.options.resolution
        dataset = RomaVideoImageDataset(
            frames_dir, ImageDatasetOptions(resize_to_shape=(r, r), interpolation="cv2_area"), names
        )
        dataset.normalize = False
        return dataset

    @staticmethod
    def _query_camera(kf: _Keyframe) -> pycolmap.Camera:
        """Initial query camera: a copy of the reference keyframe's camera (focal refined by PnP if enabled).

        Interpolating the focal length between bracketing keyframes breaks at shot cuts, where the zoom jumps.
        """
        return pycolmap.Camera(
            model=kf.camera.model, width=kf.camera.width, height=kf.camera.height, params=np.array(kf.camera.params)
        )

    @torch.inference_mode()
    def localize(
        self,
        models: list[pycolmap.Reconstruction],
        frames_dir: str | Path,
        query_names: list[str] | None = None,
        *,
        leave_one_out: bool = False,
    ) -> FrameLocalizationResult:
        """Localize query frames (default: all frames subsampled to options.fps).

        With leave_one_out=True, the queries are the keyframes themselves, each localized without using itself as
        reference; this validates the accuracy against the bundle-adjusted poses.
        """
        opts = self.options
        frames_dir = Path(frames_dir)
        timer = _Timer()
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        keyframes = collect_keyframes(models)
        key_times = [k.timestamp for k in keyframes]
        key_by_name = {k.name: i for i, k in enumerate(keyframes)}
        if leave_one_out:
            query_names = [k.name for k in keyframes]
        elif query_names is None:
            all_names = sorted(p.name for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
            query_names = subsample_frames(all_names, opts.fps)
        timer.add("setup", time.perf_counter() - t0)

        n = len(query_names)
        res = FrameLocalizationResult(
            names=list(query_names),
            timestamps=np.array([frame_timestamp(q) for q in query_names]),
            success=np.zeros(n, bool),
            is_keyframe=np.zeros(n, bool),
            model_index=np.full(n, -1, int),
            rotation_xyzw=np.zeros((n, 4)),
            translation=np.zeros((n, 3)),
            focal_length=np.zeros(n),
            num_correspondences=np.zeros(n, int),
            num_inliers=np.zeros(n, int),
            reference_names=[""] * n,
        )

        # Keyframes queried outside leave-one-out mode copy their BA pose; other frames get candidate references.
        candidates: dict[int, list[int]] = {}
        for qi, name in enumerate(query_names):
            if not leave_one_out and name in key_by_name:
                kf = keyframes[key_by_name[name]]
                res.success[qi] = res.is_keyframe[qi] = True
                res.model_index[qi] = kf.model_index
                res.rotation_xyzw[qi] = kf.cam_from_world.rotation.quat
                res.translation[qi] = kf.cam_from_world.translation
                res.focal_length[qi] = kf.camera.mean_focal_length()
                res.reference_names[qi] = name
                continue
            exclude = name if leave_one_out else None
            refs = bracketing_keyframes(res.timestamps[qi], keyframes, key_times, exclude)
            if refs:
                candidates[qi] = refs

        # Best (keyframe, PnP result, #correspondences, camera) per query.
        best: dict[int, tuple[int, dict | None, int, pycolmap.Camera]] = {}
        context = (keyframes, frames_dir, query_names, res.timestamps, timer)
        num_first = 1 if opts.num_reference_keyframes <= 1 else 2
        self._match_and_solve({qi: refs[:num_first] for qi, refs in candidates.items()}, best, *context)
        num_retried = 0
        if num_first == 1 and opts.fallback_min_covisibility > 0:

            def covisibility(qi):
                ki, result, _, _ = best[qi]
                return 0.0 if result is None else result["num_inliers"] / len(keyframes[ki].xy)

            retry = {
                qi: refs[1:]
                for qi, refs in candidates.items()
                if len(refs) > 1 and covisibility(qi) < opts.fallback_min_covisibility
            }
            num_retried = len(retry)
            if retry:
                self._match_and_solve(retry, best, *context)

        for qi, (ki, result, num_corr, camera) in best.items():
            kf = keyframes[ki]
            res.model_index[qi] = kf.model_index
            res.reference_names[qi] = kf.name
            res.focal_length[qi] = camera.mean_focal_length()
            res.num_correspondences[qi] = num_corr
            if result is None or result["num_inliers"] < opts.min_inliers:
                continue
            pose = result["cam_from_world"]
            res.success[qi] = True
            res.rotation_xyzw[qi] = pose.rotation.quat
            res.translation[qi] = pose.translation
            res.num_inliers[qi] = int(result["num_inliers"])
        res.timings = dict(timer.totals, total=time.perf_counter() - t_start, num_retried=num_retried)
        return res

    def _match_and_solve(self, refs_by_query, best, keyframes, frames_dir, query_names, timestamps, timer) -> None:
        """Match each query to its references (batched on the GPU) and solve one absolute pose per pair (in a CPU
        thread pool overlapping the GPU work); keep in `best` the pose with the most inliers per query."""
        opts = self.options
        refs_by_query = {qi: refs for qi, refs in refs_by_query.items() if refs}
        if not refs_by_query:
            return
        # Load each query frame once; keyframe images are loaded lazily for their features.
        query_list = sorted(refs_by_query, key=lambda q: timestamps[q])
        dataset = self._dataset(frames_dir, [query_names[q] for q in query_list])
        kf_dataset = self._dataset(frames_dir, [k.name for k in keyframes])
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opts.batch_size,
            num_workers=opts.num_io_workers,
            pin_memory=self.device.type == "cuda",
            prefetch_factor=4 if opts.num_io_workers > 0 else None,
        )

        # Keyframes in order of first use, so that their features can be extracted in look-ahead batches.
        kf_order = list(OrderedDict.fromkeys(ki for qi in query_list for ki in refs_by_query[qi]))
        kf_position = {ki: i for i, ki in enumerate(kf_order)}
        kf_cache: OrderedDict[int, tuple[torch.Tensor, ...]] = OrderedDict()
        cache_size = max(opts.keyframe_feature_cache_size, 2 * opts.batch_size)
        io_pool = ThreadPoolExecutor(max_workers=max(opts.num_io_workers, 1))

        def keyframe_features(ki: int):
            if ki in kf_cache:
                kf_cache.move_to_end(ki)
                return kf_cache[ki]
            start = kf_position[ki]
            todo = [k for k in kf_order[start:] if k not in kf_cache][: opts.batch_size]
            t0 = time.perf_counter()
            images = torch.stack([d["image"] for d in io_pool.map(kf_dataset.__getitem__, todo)])
            timer.add("io_keyframes", time.perf_counter() - t0)
            with timer.gpu("features_keyframes"):
                if len(todo) < opts.batch_size:
                    images = torch.cat((images, images.new_zeros((opts.batch_size - len(todo), *images.shape[1:]))))
                feats = self.model._extract_features(images, coarse_only=opts.coarse_only)
            for j, k in enumerate(todo):
                # Clone: compiled graphs may reuse their output buffers across calls.
                kf_cache[k] = tuple(f[j : j + 1].clone() for f in feats)
            while len(kf_cache) > cache_size:
                kf_cache.popitem(last=False)
            return kf_cache[ki]

        # Padded keypoints per keyframe on the GPU, in normalized coordinates of the matching grid.
        kp_cache: dict[int, tuple[torch.Tensor, int]] = {}

        def keyframe_grid(ki: int):
            if ki not in kp_cache:
                kf = keyframes[ki]
                w, h = kf.camera.width, kf.camera.height
                g = np.stack((2 * kf.xy[:, 0] / w - 1, 2 * kf.xy[:, 1] / h - 1), -1)
                kp_cache[ki] = (torch.from_numpy(g).float().to(self.device), len(g))
            return kp_cache[ki]

        executor = ThreadPoolExecutor(max_workers=opts.num_pnp_workers)
        futures: list[tuple[int, int, int, pycolmap.Camera, Future]] = []

        def flush(features_b, rows):
            """Match a chunk of (query, keyframe) pairs whose query features are rows of features_b."""
            B = opts.batch_size
            for start in range(0, len(rows), B):
                chunk = rows[start : start + B]
                count = len(chunk)
                fa = [keyframe_features(ki) for _, ki, _ in chunk]
                with timer.gpu("matching"):
                    idx = torch.tensor([row for _, _, row in chunk] + [chunk[-1][2]] * (B - count), device=self.device)
                    features_a = tuple(
                        torch.cat([f[j] for f in fa] + [fa[-1][j]] * (B - count)).contiguous()
                        for j in range(len(fa[0]))
                    )
                    feats_b = tuple(f.index_select(0, idx).contiguous() for f in features_b)
                    raw = self.model._feature_matcher(features_a, feats_b, refine=not opts.coarse_only)
                    w, h = keyframes[chunk[0][1]].camera.width, keyframes[chunk[0][1]].camera.height
                    out = self.model._pixel_match(raw, output_size=(w, h), return_covariance=False)
                with timer.gpu("sampling"):
                    grids = [keyframe_grid(ki) for _, ki, _ in chunk]
                    nmax = max(g[1] for g in grids)
                    grid = torch.zeros((count, 1, nmax, 2), device=self.device)
                    for j, (g, ng) in enumerate(grids):
                        grid[j, 0, :ng] = g
                    warp = out.matches[:count].permute(0, 3, 1, 2).float()
                    cert = out.certainty[:count, None].float()
                    xy_b = F.grid_sample(warp, grid, mode="bilinear", align_corners=False)[:, :, 0].permute(0, 2, 1)
                    c_b = F.grid_sample(cert, grid, mode="bilinear", align_corners=False)[:, 0, 0]
                    xy_b, c_b = xy_b.cpu().numpy(), c_b.cpu().numpy()
                for j, (qi, ki, _) in enumerate(chunk):
                    kf = keyframes[ki]
                    ng = grids[j][1]
                    pts, c = xy_b[j, :ng].astype(np.float64), c_b[j, :ng]
                    keep = (
                        (c > opts.min_certainty)
                        & (pts[:, 0] >= 0)
                        & (pts[:, 1] >= 0)
                        & (pts[:, 0] < kf.camera.width)
                        & (pts[:, 1] < kf.camera.height)
                    )
                    camera = self._query_camera(kf)
                    p2, p3 = pts[keep], kf.xyz[keep]
                    futures.append((qi, ki, len(p2), camera, executor.submit(_pnp_job, p2, p3, camera, opts)))

        offset = 0
        t_io = time.perf_counter()
        for batch in loader:
            timer.add("io_wait_queries", time.perf_counter() - t_io)
            images = batch["image"]
            count = images.shape[0]
            batch_queries = query_list[offset : offset + count]
            offset += count
            with timer.gpu("features_queries"):
                if count < opts.batch_size:
                    images = torch.cat((images, images.new_zeros((opts.batch_size - count, *images.shape[1:]))))
                features_b = tuple(
                    f.clone() for f in self.model._extract_features(images, coarse_only=opts.coarse_only)
                )
            rows = [(qi, ki, row) for row, qi in enumerate(batch_queries) for ki in refs_by_query[qi]]
            flush(features_b, rows)
            t_io = time.perf_counter()

        t0 = time.perf_counter()
        pnp_cpu = 0.0
        for qi, ki, num_corr, camera, fut in futures:
            result, cpu = fut.result()
            pnp_cpu += cpu
            score = -1 if result is None else result["num_inliers"]
            prev = best.get(qi)
            if prev is None or score > (-1 if prev[1] is None else prev[1]["num_inliers"]):
                best[qi] = (ki, result, num_corr, camera)
        executor.shutdown()
        io_pool.shutdown()
        timer.add("pnp_drain_wall", time.perf_counter() - t0)
        timer.add("pnp_cpu_sum", pnp_cpu)
