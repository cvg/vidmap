"""Packed Three.js scene export for large sparse reconstructions."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

THREE_JS_URL = "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"
ORBIT_CONTROLS_URL = "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"
WIDE_LINE_URLS = (
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/lines/LineSegmentsGeometry.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/lines/LineGeometry.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/lines/LineMaterial.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/lines/LineSegments2.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/lines/Line2.js",
)
LOOP_CLOSURE_MIN_SHARED_POINTS = 50


@dataclass(frozen=True)
class SceneGeometry:
    """Packed reconstruction geometry used by the viewer."""

    points: np.ndarray
    estimated_frusta: np.ndarray
    estimated_path: np.ndarray
    ground_truth_frusta: np.ndarray
    ground_truth_path: np.ndarray
    point_colors: np.ndarray | None = None
    loop_closure_edges: np.ndarray | None = None
    loop_closure_shared_points: np.ndarray | None = None


def _float32_xyz(values) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype="<f4").reshape(-1, 3))


def _all_points(model, point_ids=None) -> np.ndarray:
    if point_ids is None:
        return _float32_xyz([point.xyz for _, point in model.points3D.items()])
    return _float32_xyz([model.point3D(int(point_id)).xyz for point_id in point_ids])


def sample_point_colors(model, images_dir: str | Path) -> np.ndarray:
    """Run COLMAP's native all-image point-color extractor."""
    images_dir = Path(images_dir)
    for image_id in sorted(model.images):
        image = model.images[image_id]
        if not image.has_pose:
            continue
        image_path = images_dir / image.name
        if not image_path.is_file():
            raise FileNotFoundError(f"Registered image does not exist: {image_path}")
        with Image.open(image_path) as source:
            actual_size = source.size
        camera = model.cameras[image.camera_id]
        expected_size = (int(camera.width), int(camera.height))
        if actual_size != expected_size:
            raise ValueError(
                f"Image dimensions do not match reconstruction camera for {image.name}: "
                f"expected {expected_size[0]}x{expected_size[1]}, got {actual_size[0]}x{actual_size[1]}"
            )

    model.extract_colors_for_all_images(str(images_dir))
    return np.ascontiguousarray(np.asarray([point.color for point in model.points3D.values()], dtype=np.uint8))


def _frustum_segments(image, camera, *, size: float) -> np.ndarray:
    world_t_camera = image.cam_from_world().inverse()
    rotation = world_t_camera.rotation.matrix()
    center = np.asarray(world_t_camera.translation)
    calibration = camera.calibration_matrix()
    width, height = calibration[0, 2] * 2, calibration[1, 2] * 2
    corners = np.asarray([[0, 0], [width, 0], [width, height], [0, height]])

    image_extent = max(size * width / 1024.0, size * height / 1024.0)
    world_extent = max(width, height) / (calibration[0, 0] + calibration[1, 1]) / 0.5
    scale = 0.5 * image_extent / world_extent
    homogeneous = np.concatenate((corners, np.ones((4, 1))), axis=1)
    corners = homogeneous @ np.linalg.inv(calibration).T
    corners = (corners / 2 * scale) @ rotation.T + center

    segments = []
    for corner in corners:
        segments.extend((center, corner))
    for index in range(4):
        segments.extend((corners[index], corners[(index + 1) % 4]))
    return _float32_xyz(segments)


def _camera_geometry(model, *, include_frusta: bool, size: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    frusta = []
    centers = []
    for image_id in sorted(model.images):
        image = model.images[image_id]
        if not image.has_pose:
            continue
        if include_frusta:
            frusta.append(_frustum_segments(image, model.cameras[image.camera_id], size=size))
        centers.append(image.projection_center())
    packed_frusta = _float32_xyz(np.concatenate(frusta, axis=0) if frusta else [])
    return packed_frusta, _float32_xyz(centers)


def build_scene_geometry(
    model,
    ground_truth=None,
    *,
    point_ids=None,
    point_colors=None,
    loop_closure_edges=None,
    loop_closure_shared_points=None,
    include_cameras: bool = False,
) -> SceneGeometry:
    """Pack model and optional ground truth into a fixed number of buffers."""
    estimated_frusta, estimated_path = _camera_geometry(model, include_frusta=include_cameras)
    if ground_truth is None or ground_truth.num_reg_images() == 0:
        ground_truth_frusta = _float32_xyz([])
        ground_truth_path = _float32_xyz([])
    else:
        ground_truth_frusta, ground_truth_path = _camera_geometry(ground_truth, include_frusta=include_cameras)
    return SceneGeometry(
        points=_all_points(model, point_ids),
        estimated_frusta=estimated_frusta,
        estimated_path=estimated_path,
        ground_truth_frusta=ground_truth_frusta,
        ground_truth_path=ground_truth_path,
        point_colors=point_colors,
        loop_closure_edges=None if loop_closure_edges is None else _float32_xyz(loop_closure_edges),
        loop_closure_shared_points=loop_closure_shared_points,
    )


def _encode_array(
    values: np.ndarray | None,
    *,
    dtype: str | type[np.generic] = "<f4",
    columns: int = 3,
    count_key: str = "vertices",
) -> dict[str, str | int] | None:
    if values is None:
        return None
    values = np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(-1, columns))
    return {
        "base64": base64.b64encode(values.tobytes()).decode("ascii"),
        count_key: len(values),
    }


def scene_payload(scene: SceneGeometry) -> dict[str, dict[str, str | int] | None]:
    """Serialize geometry as packed little-endian float32 buffers."""
    if scene.point_colors is not None and len(scene.point_colors) != len(scene.points):
        raise ValueError(
            f"Point colors must match point positions: {len(scene.point_colors)} colors for {len(scene.points)} points"
        )
    if (scene.loop_closure_edges is None) != (scene.loop_closure_shared_points is None):
        raise ValueError("Loop-closure edge positions and shared-point counts must be provided together")
    if scene.loop_closure_edges is not None:
        if len(scene.loop_closure_edges) % 2:
            raise ValueError("Loop-closure edge positions must contain endpoint pairs")
        if len(scene.loop_closure_shared_points) * 2 != len(scene.loop_closure_edges):
            raise ValueError(
                "Loop-closure shared-point counts must match edge count: "
                f"{len(scene.loop_closure_shared_points)} counts for {len(scene.loop_closure_edges) // 2} edges"
            )
    return {
        "points": _encode_array(scene.points),
        "pointColors": _encode_array(scene.point_colors, dtype=np.uint8),
        "estimatedFrusta": _encode_array(scene.estimated_frusta),
        "estimatedPath": _encode_array(scene.estimated_path),
        "groundTruthFrusta": _encode_array(scene.ground_truth_frusta),
        "groundTruthPath": _encode_array(scene.ground_truth_path),
        "loopClosureEdges": _encode_array(scene.loop_closure_edges),
        "loopClosureSharedPoints": _encode_array(scene.loop_closure_shared_points, columns=1, count_key="values"),
    }


def _camera_controls(frusta: np.ndarray, *, name: str, color: str) -> str:
    if not len(frusta):
        return ""
    return f"""
    <div class="control-row">
      <label class="toggle"><input id="{name}-frusta-toggle" type="checkbox" checked> {name} cameras</label>
      <label class="setting">size <input id="{name}-frusta-size" type="number" min="0.01" max="20" step="0.05" value="0.3"></label>
      <label class="setting">line <input id="{name}-frusta-width" type="number" min="0.5" max="20" step="0.5" value="1"></label>
      <label class="setting">color <input id="{name}-frusta-color" type="color" value="{color}"></label>
    </div>"""


def render_scene_html(scene: SceneGeometry) -> str:
    """Return a scene document with inline geometry and a pinned renderer URL."""
    payload = json.dumps(scene_payload(scene), separators=(",", ":"))
    point_color_mode = ""
    if scene.point_colors is not None:
        point_color_mode = """
      <label class="setting">mode
        <select id="points-color-mode">
          <option value="solid">solid</option>
          <option value="rgb" selected>image RGB</option>
        </select>
      </label>"""
    background_color = "#ffffff" if scene.point_colors is not None else "#000000"
    vertex_colors = "true" if scene.point_colors is not None else "false"
    loop_closure_controls = ""
    if scene.loop_closure_edges is not None and len(scene.loop_closure_edges):
        loop_closure_controls = f"""
    <div class="control-row">
      <label class="toggle"><input id="loop-closures-toggle" type="checkbox" checked> loop closures</label>
      <label class="setting">line <input id="loop-closures-width" type="number" min="0.5" max="20" step="0.5" value="3"></label>
      <label class="setting"><span class="lc-rejected">red</span>/<span class="lc-accepted">green</span> min shared <input id="loop-closures-min-shared" type="number" min="0" step="1" value="{LOOP_CLOSURE_MIN_SHARED_POINTS}"></label>
    </div>"""
    estimated_camera_controls = _camera_controls(scene.estimated_frusta, name="estimated", color="#d62728")
    ground_truth_camera_controls = _camera_controls(scene.ground_truth_frusta, name="ground-truth", color="#2ca02c")
    return (
        _HTML_TEMPLATE.replace("__THREE_JS_URL__", THREE_JS_URL)
        .replace("__ORBIT_CONTROLS_URL__", ORBIT_CONTROLS_URL)
        .replace(
            "__WIDE_LINE_SCRIPTS__",
            "\n".join(f'  <script src="{url}"></script>' for url in WIDE_LINE_URLS),
        )
        .replace("__POINT_COLOR_MODE__", point_color_mode)
        .replace("__DEFAULT_BACKGROUND_COLOR__", background_color)
        .replace("__DEFAULT_VERTEX_COLORS__", vertex_colors)
        .replace("__ESTIMATED_CAMERA_CONTROLS__", estimated_camera_controls)
        .replace("__GROUND_TRUTH_CAMERA_CONTROLS__", ground_truth_camera_controls)
        .replace("__LOOP_CLOSURE_CONTROLS__", loop_closure_controls)
        .replace("__LOOP_CLOSURE_MIN_SHARED_POINTS__", str(LOOP_CLOSURE_MIN_SHARED_POINTS))
        .replace("__VIDMAP_SCENE_PAYLOAD__", payload)
    )


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>VidMap reconstruction</title>
  <style>
    html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: __DEFAULT_BACKGROUND_COLOR__; }
    #background-layer {
      position: fixed; z-index: 0; inset: 0; background-position: center;
      background-repeat: no-repeat; background-size: contain; pointer-events: none;
    }
    canvas { position: fixed; z-index: 1; inset: 0; display: block; }
    #toolbar {
      position: fixed; z-index: 10; top: 12px; left: 12px; padding: 9px 12px;
      min-width: 470px;
      border-radius: 6px; background: rgba(255,255,255,.9); color: #222;
      font: 13px/1.5 system-ui, sans-serif; box-shadow: 0 1px 5px rgba(0,0,0,.2);
    }
    #toolbar label { display: block; white-space: nowrap; cursor: pointer; }
    .control-row { display: flex; align-items: center; gap: 10px; min-height: 27px; }
    .toggle { width: 165px; }
    .setting { color: #555; }
    .setting input {
      width: 58px; margin-left: 3px; padding: 2px 4px; border: 1px solid #bbb;
      border-radius: 3px; background: white; color: #222;
    }
    .setting input[type="color"] {
      width: 30px; height: 24px; padding: 1px; vertical-align: middle; cursor: pointer;
    }
    .setting select {
      margin-left: 3px; padding: 2px 4px; border: 1px solid #bbb;
      border-radius: 3px; background: white; color: #222;
    }
    .lc-rejected { color: rgb(240,70,70); }
    .lc-accepted { color: rgb(0,210,130); }
    button, .file-button {
      padding: 3px 7px; border: 1px solid #aaa; border-radius: 3px;
      background: white; color: #222; font: inherit; cursor: pointer;
    }
    button:hover, .file-button:hover { background: #eee; }
    #help { margin-top: 5px; color: #666; }
    #error { display: none; color: #a00; max-width: 480px; }
  </style>
  <script src="__THREE_JS_URL__"></script>
  <script src="__ORBIT_CONTROLS_URL__"></script>
__WIDE_LINE_SCRIPTS__
</head>
<body>
  <div id="background-layer"></div>
  <div id="toolbar">
    <div class="control-row">
      <label class="toggle"><input id="points-toggle" type="checkbox" checked> points</label>
      <label class="setting">size <input id="points-size" type="number" min="0.01" max="20" step="0.01" value="0.7"></label>
      __POINT_COLOR_MODE__
      <label class="setting">color <input id="points-color" type="color" value="#ffffff"></label>
    </div>
__ESTIMATED_CAMERA_CONTROLS__
__GROUND_TRUTH_CAMERA_CONTROLS__
    <div class="control-row">
      <label class="toggle"><input id="paths-toggle" type="checkbox" checked> trajectories</label>
      <label class="setting">line <input id="paths-width" type="number" min="0.5" max="20" step="0.5" value="6"></label>
      <label class="setting">estimated <input id="estimated-path-color" type="color" value="#0064ff"></label>
      <label class="setting">GT <input id="ground-truth-path-color" type="color" value="#ffdc00"></label>
    </div>
__LOOP_CLOSURE_CONTROLS__
    <div class="control-row">
      <span class="toggle">view</span>
      <label class="setting">projection
        <select id="projection">
          <option value="perspective">perspective</option>
          <option value="orthographic">orthographic</option>
        </select>
      </label>
    </div>
    <div class="control-row">
      <span class="toggle">background</span>
      <label class="setting">color <input id="background-color" type="color" value="__DEFAULT_BACKGROUND_COLOR__"></label>
      <label class="file-button" for="background-file">choose image</label>
      <input id="background-file" type="file" accept="image/*" hidden>
      <label class="setting">fit
        <select id="background-fit">
          <option value="contain">contain</option>
          <option value="cover">cover</option>
          <option value="stretch">stretch</option>
        </select>
      </label>
      <button id="background-clear" type="button">clear</button>
    </div>
    <div id="help">drag: orbit · middle-drag: roll view · Shift+middle-drag: orbit gravity axis · wheel: zoom · right-drag: pan</div>
    <div id="error"></div>
  </div>
  <script>
  (() => {
    "use strict";
    const payload = __VIDMAP_SCENE_PAYLOAD__;
    const error = document.getElementById("error");
    if (typeof THREE === "undefined") {
      error.style.display = "block";
      error.textContent = "Three.js could not be loaded. This viewer requires network access to its pinned renderer.";
      return;
    }

    function decodeFloat32(encoded) {
      const binary = atob(encoded);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; ++index) bytes[index] = binary.charCodeAt(index);
      return new Float32Array(bytes.buffer);
    }

    function decodedPositions(buffer) {
      return decodeFloat32(buffer.base64);
    }

    function decodedColors(buffer) {
      if (buffer === null) return null;
      const binary = atob(buffer.base64);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; ++index) bytes[index] = binary.charCodeAt(index);
      return bytes;
    }

    function pointGeometry(positions) {
      const result = new THREE.BufferGeometry();
      result.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      return result;
    }

    const renderer = new THREE.WebGLRenderer({
      antialias: true, alpha: true, powerPreference: "high-performance"
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    document.body.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const perspectiveCamera = new THREE.PerspectiveCamera(
      45, window.innerWidth / window.innerHeight, 0.001, 1e9
    );
    const orthographicCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.001, 1e9);
    let camera = perspectiveCamera;
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.mouseButtons.LEFT = null;
    controls.mouseButtons.MIDDLE = null;

    const viewAxis = new THREE.Vector3();
    const viewRight = new THREE.Vector3();
    const gravityAxis = new THREE.Vector3(0, 0, 1);
    const orbitYaw = new THREE.Quaternion();
    const orbitPitch = new THREE.Quaternion();
    let orbitPointerId = null;
    let previousOrbitX = 0;
    let previousOrbitY = 0;
    let rollPointerId = null;
    let previousRollX = 0;
    let middleRotationMode = "roll";
    function orbitInCurrentView(deltaX, deltaY) {
      const offset = camera.position.clone().sub(controls.target);
      camera.getWorldDirection(viewAxis).normalize();
      viewRight.crossVectors(viewAxis, camera.up).normalize();
      orbitYaw.setFromAxisAngle(camera.up.clone().normalize(), -deltaX * 0.005);
      offset.applyQuaternion(orbitYaw);
      viewRight.applyQuaternion(orbitYaw);
      orbitPitch.setFromAxisAngle(viewRight, -deltaY * 0.005);
      offset.applyQuaternion(orbitPitch);
      camera.up.applyQuaternion(orbitYaw).applyQuaternion(orbitPitch).normalize();
      camera.position.copy(controls.target).add(offset);
      camera.lookAt(controls.target);
      controls.update();
    }
    function rollAroundViewAxis(deltaX) {
      camera.getWorldDirection(viewAxis).normalize();
      camera.up.applyAxisAngle(viewAxis, -deltaX * 0.005).normalize();
      camera.lookAt(controls.target);
      controls.update();
    }
    function orbitAroundGravityAxis(deltaX) {
      const offset = camera.position.clone().sub(controls.target);
      orbitYaw.setFromAxisAngle(gravityAxis, -deltaX * 0.005);
      offset.applyQuaternion(orbitYaw);
      camera.up.applyQuaternion(orbitYaw).normalize();
      camera.position.copy(controls.target).add(offset);
      camera.lookAt(controls.target);
      controls.update();
    }
    renderer.domElement.addEventListener("pointerdown", event => {
      if (event.button === 0) {
        event.preventDefault();
        orbitPointerId = event.pointerId;
        previousOrbitX = event.clientX;
        previousOrbitY = event.clientY;
        renderer.domElement.setPointerCapture(event.pointerId);
        return;
      }
      if (event.button !== 1) return;
      event.preventDefault();
      rollPointerId = event.pointerId;
      previousRollX = event.clientX;
      middleRotationMode = event.shiftKey ? "orbit-gravity" : "roll";
      renderer.domElement.setPointerCapture(event.pointerId);
    });
    renderer.domElement.addEventListener("pointermove", event => {
      if (event.pointerId === orbitPointerId) {
        event.preventDefault();
        orbitInCurrentView(
          event.clientX - previousOrbitX,
          event.clientY - previousOrbitY
        );
        previousOrbitX = event.clientX;
        previousOrbitY = event.clientY;
      } else if (event.pointerId === rollPointerId) {
        event.preventDefault();
        const deltaX = event.clientX - previousRollX;
        if (middleRotationMode === "orbit-gravity") orbitAroundGravityAxis(deltaX);
        else rollAroundViewAxis(deltaX);
        previousRollX = event.clientX;
      }
    });
    function finishViewRotation(event) {
      if (event.pointerId !== orbitPointerId && event.pointerId !== rollPointerId) return;
      if (renderer.domElement.hasPointerCapture(event.pointerId)) {
        renderer.domElement.releasePointerCapture(event.pointerId);
      }
      if (event.pointerId === orbitPointerId) orbitPointerId = null;
      if (event.pointerId === rollPointerId) rollPointerId = null;
    }
    renderer.domElement.addEventListener("pointerup", finishViewRotation);
    renderer.domElement.addEventListener("pointercancel", finishViewRotation);
    renderer.domElement.addEventListener("auxclick", event => {
      if (event.button === 1) event.preventDefault();
    });
    const pointPositions = decodedPositions(payload.points);
    const pointColors = decodedColors(payload.pointColors);
    const estimatedFrustaPositions = decodedPositions(payload.estimatedFrusta);
    const estimatedPathPositions = decodedPositions(payload.estimatedPath);
    const groundTruthFrustaPositions = decodedPositions(payload.groundTruthFrusta);
    const groundTruthPathPositions = decodedPositions(payload.groundTruthPath);
    const loopClosureEdgePositions = payload.loopClosureEdges === null
      ? new Float32Array()
      : decodedPositions(payload.loopClosureEdges);
    const loopClosureSharedPoints = payload.loopClosureSharedPoints === null
      ? new Float32Array()
      : decodeFloat32(payload.loopClosureSharedPoints.base64);

    const pointsGeometry = pointGeometry(pointPositions);
    if (pointColors !== null) {
      pointsGeometry.setAttribute(
        "color",
        new THREE.Uint8BufferAttribute(pointColors, 3, true)
      );
    }
    const points = new THREE.Points(
      pointsGeometry,
      new THREE.PointsMaterial({
        color: 0xffffff, size: 0.7, sizeAttenuation: false, vertexColors: __DEFAULT_VERTEX_COLORS__
      })
    );
    points.name = "points";
    scene.add(points);

    function wideLineMaterial(color, opacity, width, depthTest) {
      const material = new THREE.LineMaterial({
        color: color, transparent: opacity < 1, opacity: opacity,
        linewidth: width, worldUnits: false, depthTest: depthTest
      });
      material.resolution.set(window.innerWidth, window.innerHeight);
      return material;
    }

    function wideLineSegments(positions, color, opacity, width) {
      const geometry = new THREE.LineSegmentsGeometry();
      geometry.setPositions(positions);
      return new THREE.LineSegments2(
        geometry,
        wideLineMaterial(color, opacity, width, true)
      );
    }
    function loopClosureVertexColors(sharedPoints, minimumSharedPoints) {
      const rejected = [240, 70, 70];
      const accepted = [0, 210, 130];
      const colors = new Float32Array(sharedPoints.length * 6);
      for (let edge = 0; edge < sharedPoints.length; ++edge) {
        const color = sharedPoints[edge] >= minimumSharedPoints ? accepted : rejected;
        for (let endpoint = 0; endpoint < 2; ++endpoint) {
          for (let channel = 0; channel < 3; ++channel) {
            colors[edge * 6 + endpoint * 3 + channel] = color[channel] / 255;
          }
        }
      }
      return colors;
    }
    function coloredWideLineSegments(positions, sharedPoints, width, minimumSharedPoints) {
      const geometry = new THREE.LineSegmentsGeometry();
      geometry.setPositions(positions);
      geometry.setColors(loopClosureVertexColors(sharedPoints, minimumSharedPoints));
      const material = wideLineMaterial(0xffffff, 0.8, width, true);
      material.vertexColors = true;
      material.needsUpdate = true;
      return new THREE.LineSegments2(geometry, material);
    }
    const estimatedFrusta = wideLineSegments(estimatedFrustaPositions, 0xd62728, 0.45, 1);
    const groundTruthFrusta = wideLineSegments(groundTruthFrustaPositions, 0x2ca02c, 0.45, 1);
    scene.add(estimatedFrusta);
    scene.add(groundTruthFrusta);
    const loopClosures = coloredWideLineSegments(
      loopClosureEdgePositions, loopClosureSharedPoints, 3, __LOOP_CLOSURE_MIN_SHARED_POINTS__
    );
    scene.add(loopClosures);

    function path(positions, color) {
      const geometry = new THREE.LineGeometry();
      const hasSegments = positions.length >= 6;
      if (hasSegments) geometry.setPositions(positions);
      const material = wideLineMaterial(color, 1, 6, false);
      material.transparent = true;
      material.depthWrite = false;
      const result = new THREE.Line2(
        geometry,
        material
      );
      result.visible = hasSegments;
      result.renderOrder = 100;
      return result;
    }
    const estimatedPath = path(estimatedPathPositions, 0x0064ff);
    const groundTruthPath = path(groundTruthPathPositions, 0xffdc00);
    scene.add(estimatedPath);
    scene.add(groundTruthPath);

    const bounds = new THREE.Box3();
    const fitPositions = estimatedPathPositions.length > 0
      ? [estimatedPathPositions, groundTruthPathPositions]
      : [pointPositions];
    const fitPoint = new THREE.Vector3();
    for (const positions of fitPositions) {
      for (let index = 0; index < positions.length; index += 3) {
        fitPoint.set(positions[index], positions[index + 1], positions[index + 2]);
        bounds.expandByPoint(fitPoint);
      }
    }
    const center = new THREE.Vector3();
    const extent = new THREE.Vector3();
    if (bounds.isEmpty()) {
      bounds.setFromCenterAndSize(new THREE.Vector3(), new THREE.Vector3(2, 2, 2));
    }
    bounds.getCenter(center);
    bounds.getSize(extent);
    const radius = Math.max(extent.length() * 0.5, 1);
    controls.target.copy(center);
    const initialPosition = center.clone().add(
      new THREE.Vector3(1, -1, 0.8).normalize().multiplyScalar(radius * 2.2)
    );
    for (const viewCamera of [perspectiveCamera, orthographicCamera]) {
      viewCamera.near = Math.max(radius / 10000, 0.001);
      viewCamera.far = radius * 100;
      viewCamera.position.copy(initialPosition);
    }

    function updateProjectionDimensions() {
      const aspect = window.innerWidth / window.innerHeight;
      perspectiveCamera.aspect = aspect;
      perspectiveCamera.updateProjectionMatrix();
      const halfHeight = radius * 1.25;
      orthographicCamera.left = -halfHeight * aspect;
      orthographicCamera.right = halfHeight * aspect;
      orthographicCamera.top = halfHeight;
      orthographicCamera.bottom = -halfHeight;
      orthographicCamera.updateProjectionMatrix();
    }
    updateProjectionDimensions();
    controls.update();

    document.getElementById("points-toggle").onchange = event => points.visible = event.target.checked;
    const estimatedFrustaToggle = document.getElementById("estimated-frusta-toggle");
    if (estimatedFrustaToggle !== null) {
      estimatedFrustaToggle.onchange = event => estimatedFrusta.visible = event.target.checked;
    }
    const groundTruthFrustaToggle = document.getElementById("ground-truth-frusta-toggle");
    if (groundTruthFrustaToggle !== null) {
      groundTruthFrustaToggle.onchange = event => groundTruthFrusta.visible = event.target.checked;
    }
    document.getElementById("paths-toggle").onchange = event => {
      estimatedPath.visible = event.target.checked && estimatedPathPositions.length >= 6;
      groundTruthPath.visible = event.target.checked && groundTruthPathPositions.length >= 6;
    };
    const loopClosuresToggle = document.getElementById("loop-closures-toggle");
    if (loopClosuresToggle !== null) {
      loopClosuresToggle.onchange = event => loopClosures.visible = event.target.checked;
      bindNumber("loop-closures-width", value => loopClosures.material.linewidth = value);
      const minimumSharedPoints = document.getElementById("loop-closures-min-shared");
      function updateLoopClosureColors() {
        const minimum = Number(minimumSharedPoints.value);
        if (!Number.isFinite(minimum) || minimum < 0) return;
        const replacement = new THREE.LineSegmentsGeometry();
        replacement.setPositions(loopClosureEdgePositions);
        replacement.setColors(loopClosureVertexColors(loopClosureSharedPoints, minimum));
        loopClosures.geometry.dispose();
        loopClosures.geometry = replacement;
      }
      minimumSharedPoints.oninput = updateLoopClosureColors;
    }

    function bindNumber(id, update) {
      const input = document.getElementById(id);
      input.oninput = () => {
        const value = Number(input.value);
        if (Number.isFinite(value) && value > 0) update(value);
      };
    }

    function scaledFrusta(base, centers, size) {
      const result = new Float32Array(base.length);
      const scale = size / 0.3;
      const verticesPerCamera = 16;
      for (let cameraIndex = 0; cameraIndex < centers.length / 3; ++cameraIndex) {
        const centerOffset = cameraIndex * 3;
        const centerX = centers[centerOffset];
        const centerY = centers[centerOffset + 1];
        const centerZ = centers[centerOffset + 2];
        const begin = cameraIndex * verticesPerCamera * 3;
        const end = begin + verticesPerCamera * 3;
        for (let index = begin; index < end; index += 3) {
          result[index] = centerX + (base[index] - centerX) * scale;
          result[index + 1] = centerY + (base[index + 1] - centerY) * scale;
          result[index + 2] = centerZ + (base[index + 2] - centerZ) * scale;
        }
      }
      return result;
    }

    function updateFrustaSize(object, base, centers, size) {
      const replacement = new THREE.LineSegmentsGeometry();
      replacement.setPositions(scaledFrusta(base, centers, size));
      object.geometry.dispose();
      object.geometry = replacement;
    }

    function updatePointSize(value) {
      const minimumSize = 1 / renderer.getPixelRatio();
      const coverage = Math.min(value / minimumSize, 1);
      points.material.size = Math.max(value, minimumSize);
      points.material.opacity = coverage * coverage;
      const transparent = coverage < 1;
      if (points.material.transparent !== transparent) {
        points.material.transparent = transparent;
        points.material.needsUpdate = true;
      }
    }
    bindNumber("points-size", updatePointSize);
    updatePointSize(Number(document.getElementById("points-size").value));
    const pointsColorInput = document.getElementById("points-color");
    pointsColorInput.oninput = event => points.material.color.set(event.target.value);
    const pointsColorMode = document.getElementById("points-color-mode");
    pointsColorInput.disabled = pointColors !== null;
    if (pointsColorMode !== null) {
      pointsColorMode.onchange = event => {
        const useRgb = event.target.value === "rgb";
        points.material.vertexColors = useRgb;
        points.material.color.set(useRgb ? 0xffffff : pointsColorInput.value);
        points.material.needsUpdate = true;
        pointsColorInput.disabled = useRgb;
      };
    }
    if (estimatedFrustaToggle !== null) {
      bindNumber(
        "estimated-frusta-size",
        value => updateFrustaSize(estimatedFrusta, estimatedFrustaPositions, estimatedPathPositions, value)
      );
      bindNumber("estimated-frusta-width", value => estimatedFrusta.material.linewidth = value);
      document.getElementById("estimated-frusta-color").oninput =
        event => estimatedFrusta.material.color.set(event.target.value);
    }
    if (groundTruthFrustaToggle !== null) {
      bindNumber(
        "ground-truth-frusta-size",
        value => updateFrustaSize(groundTruthFrusta, groundTruthFrustaPositions, groundTruthPathPositions, value)
      );
      bindNumber("ground-truth-frusta-width", value => groundTruthFrusta.material.linewidth = value);
      document.getElementById("ground-truth-frusta-color").oninput =
        event => groundTruthFrusta.material.color.set(event.target.value);
    }
    bindNumber("paths-width", value => {
      estimatedPath.material.linewidth = value;
      groundTruthPath.material.linewidth = value;
    });
    document.getElementById("estimated-path-color").oninput =
      event => estimatedPath.material.color.set(event.target.value);
    document.getElementById("ground-truth-path-color").oninput =
      event => groundTruthPath.material.color.set(event.target.value);

    document.getElementById("projection").onchange = event => {
      const nextCamera = event.target.value === "orthographic"
        ? orthographicCamera
        : perspectiveCamera;
      nextCamera.position.copy(camera.position);
      nextCamera.quaternion.copy(camera.quaternion);
      nextCamera.up.copy(camera.up);
      camera = nextCamera;
      controls.object = camera;
      controls.update();
    };

    const backgroundLayer = document.getElementById("background-layer");
    const backgroundFile = document.getElementById("background-file");
    document.getElementById("background-color").oninput =
      event => document.body.style.backgroundColor = event.target.value;
    let backgroundUrl = null;
    function clearBackground() {
      if (backgroundUrl !== null) URL.revokeObjectURL(backgroundUrl);
      backgroundUrl = null;
      backgroundLayer.style.backgroundImage = "none";
      backgroundFile.value = "";
    }
    backgroundFile.onchange = () => {
      const file = backgroundFile.files[0];
      if (file === undefined) return;
      if (backgroundUrl !== null) URL.revokeObjectURL(backgroundUrl);
      backgroundUrl = URL.createObjectURL(file);
      backgroundLayer.style.backgroundImage = `url("${backgroundUrl}")`;
    };
    document.getElementById("background-fit").onchange = event => {
      backgroundLayer.style.backgroundSize = event.target.value === "stretch"
        ? "100% 100%"
        : event.target.value;
    };
    document.getElementById("background-clear").onclick = clearBackground;
    window.addEventListener("beforeunload", () => {
      if (backgroundUrl !== null) URL.revokeObjectURL(backgroundUrl);
    });

    function resize() {
      updateProjectionDimensions();
      renderer.setSize(window.innerWidth, window.innerHeight);
      for (const object of [
        estimatedFrusta, groundTruthFrusta, loopClosures, estimatedPath, groundTruthPath
      ]) {
        object.material.resolution.set(window.innerWidth, window.innerHeight);
      }
    }
    window.addEventListener("resize", resize);

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  })();
  </script>
</body>
</html>
"""
