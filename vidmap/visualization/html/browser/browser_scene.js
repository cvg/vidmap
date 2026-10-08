  (() => {
    "use strict";
    const embeddedRun = __VIDMAP_EMBEDDED_RUN__;
    const error = document.getElementById("error");
    function reportError(message) {
      error.style.display = "block";
      error.textContent = message;
    }
    if (typeof THREE === "undefined") {
      error.style.display = "block";
      error.textContent = "Three.js could not be loaded. This viewer requires network access to its pinned renderer.";
      return;
    }

    const VIDMAP_BROWSER_WORKER_SOURCE = __BROWSER_WORKER_SOURCE__;
__RECONSTRUCTION_SCRIPT__

    function pointGeometry(positions) {
      const result = new THREE.BufferGeometry();
      result.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      return result;
    }

    function resetPointMetadata(geometry, pointCount) {
      geometry.setAttribute(
        "vidmapFirstSeen", new THREE.BufferAttribute(new Float32Array(pointCount).fill(-1), 1)
      );
      geometry.setAttribute(
        "vidmapTrackLength", new THREE.BufferAttribute(new Float32Array(pointCount).fill(1), 1)
      );
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
    let sceneRenderDirty = true;
    function invalidateSceneRender() {
      sceneRenderDirty = true;
    }
    controls.addEventListener("change", invalidateSceneRender);
    controls.mouseButtons.MIDDLE = null;

    const viewAxis = new THREE.Vector3();
    const viewRight = new THREE.Vector3();
    const gravityAxis = new THREE.Vector3(0, 0, 1);
    const orbitYaw = new THREE.Quaternion();
    const orbitPitch = new THREE.Quaternion();
    let orbitPointerId = null;
    let orbitMoved = false;
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
        orbitMoved = false;
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
        if (Math.abs(event.clientX - previousOrbitX) + Math.abs(event.clientY - previousOrbitY) > 2) {
          orbitMoved = true;
        }
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
    let pointPositions = new Float32Array();
    let pointColors = new Uint8Array();
    const keyframeTimeline = {names: [], pointOffsets: [0], imageIds: [], timestampsSeconds: null};
    let imageDirectory = null;
    let estimatedFrustaPositions = new Float32Array();
    let estimatedPathPositions = new Float32Array();
    // Optional dense path from all-frame localization; the slider still steps through keyframes.
    let estimatedFramesPathPositions = new Float32Array();
    let estimatedFramesKeyframeCounts = [];
    let gtPathPositions = new Float32Array();
    let gtKeyframePathCounts = [];
    let gtKeyframeFrustaCounts = [];
    let gtKeyframeCenters = new Float32Array();
    let gtFrustaPositions = new Float32Array();
    let gtCovEllipsesPositions = new Float32Array();
    let gtKeyframeConfidences = [];
    let gtKeyframeNames = [];
    let gtKeyframeInliers = [];
    let gtKeyframePosStdMeters = [];
    let gtKeyframeReprojRmsPx = [];
    let gtMetadataByName = new Map();
    let denseGtPathPositions = new Float32Array();
    let denseGtKeyframePathCounts = [];
    let denseGtKeyframeFrustaCounts = [];
    let denseGtKeyframeCenters = new Float32Array();
    let denseGtFrustaPositions = new Float32Array();
    let loopClosureSharedPoints = new Float32Array();
    let loopClosureKeyframeIndices = new Uint32Array();

    const pointsGeometry = pointGeometry(pointPositions);
    resetPointMetadata(pointsGeometry, pointPositions.length / 3);
    pointsGeometry.setAttribute(
      "color",
      new THREE.Uint8BufferAttribute(pointColors, 3, true)
    );
    const keyframeLimit = {value: -1};
    const minimumTrackLength = {value: 1};
    let covarianceIndexActive = false;
    let covarianceDrawCount = pointPositions.length / 3;
    const pointMaterial = new THREE.PointsMaterial({
      color: 0xffffff, size: 3, sizeAttenuation: false, vertexColors: true
    });
    pointMaterial.onBeforeCompile = shader => {
      shader.uniforms.vidmapKeyframeLimit = keyframeLimit;
      shader.uniforms.vidmapMinimumTrackLength = minimumTrackLength;
      shader.vertexShader = `
        attribute float vidmapFirstSeen;
        attribute float vidmapTrackLength;
        uniform float vidmapKeyframeLimit;
        uniform float vidmapMinimumTrackLength;
      ` + shader.vertexShader.replace(
        "#include <project_vertex>",
        `#include <project_vertex>
        if ((vidmapKeyframeLimit >= 0.0 && vidmapFirstSeen > vidmapKeyframeLimit) ||
            vidmapTrackLength < vidmapMinimumTrackLength) {
          gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        }`
      );
    };
    pointMaterial.customProgramCacheKey = () => "vidmap-first-seen-v1";
    const points = new THREE.Points(pointsGeometry, pointMaterial);
    points.name = "points";
    scene.add(points);
    const loopClosureHighlightGeometry = pointGeometry(new Float32Array());
    const loopClosureHighlights = new THREE.Points(
      loopClosureHighlightGeometry,
      new THREE.PointsMaterial({
        color: 0xff1493, size: 3.5, sizeAttenuation: false,
        depthTest: false, depthWrite: false
      })
    );
    loopClosureHighlights.renderOrder = 110;
    loopClosureHighlights.visible = false;
    scene.add(loopClosureHighlights);
    let trackedKeypointPositions = null;
    let trackedKeypointPointIds = null;
    let trackedKeypointOffsets = null;
    let loopClosureSelectionHandler = null;
    let activeLoopClosurePair = null;

    function setLoopClosurePointHighlights(positions) {
      loopClosureHighlightGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      loopClosureHighlightGeometry.computeBoundingSphere();
      loopClosureHighlights.visible = positions.length > 0;
      invalidateSceneRender();
    }

    function installLoopClosureSelectionHandler(handler) {
      loopClosureSelectionHandler = handler;
    }

    function replacePointGeometry(positions, colors) {
      pointPositions = positions;
      pointColors = colors;
      pointsGeometry.setAttribute("position", new THREE.BufferAttribute(pointPositions, 3));
      resetPointMetadata(pointsGeometry, pointPositions.length / 3);
      pointsGeometry.setIndex(null);
      keyframeLimit.value = -1;
      covarianceIndexActive = false;
      covarianceDrawCount = pointPositions.length / 3;
      pointsGeometry.setDrawRange(0, pointPositions.length / 3);
      if (pointColors === null) pointsGeometry.deleteAttribute("color");
      else pointsGeometry.setAttribute("color", new THREE.Uint8BufferAttribute(pointColors, 3, true));
      const colorMode = document.getElementById("points-color-mode");
      const useRgb = pointColors !== null && colorMode.value === "rgb";
      points.material.vertexColors = useRgb;
      points.material.color.set(useRgb ? 0xffffff : document.getElementById("points-color").value);
      points.material.needsUpdate = true;
      pointsGeometry.computeBoundingSphere();
      invalidateSceneRender();
    }

    function installCovarianceRanks(ranks, firstSeen) {
      const pointCount = pointPositions.length / 3;
      if (ranks.length !== pointCount || firstSeen.length !== pointCount) {
        throw new Error(`Covariance ranks must match point count: ${ranks.length} for ${pointPositions.length / 3}`);
      }
      let eligibleCount = 0;
      for (const rank of ranks) {
        if (rank >= 0) eligibleCount = Math.max(eligibleCount, rank + 1);
      }
      const indices = new Uint32Array(pointCount);
      indices.fill(0xffffffff);
      let invalidIndex = eligibleCount;
      for (let pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
        const rank = ranks[pointIndex];
        if (rank < 0) indices[invalidIndex++] = pointIndex;
        else if (rank >= eligibleCount || indices[rank] !== 0xffffffff) {
          throw new Error("Covariance ranks must be unique and contiguous");
        } else indices[rank] = pointIndex;
      }
      for (let rank = 0; rank < eligibleCount; ++rank) {
        if (indices[rank] === 0xffffffff) throw new Error("Covariance ranks must be unique and contiguous");
      }
      pointsGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
      pointsGeometry.setAttribute(
        "vidmapFirstSeen",
        new THREE.BufferAttribute(Float32Array.from(firstSeen), 1)
      );
      covarianceIndexActive = true;
      covarianceDrawCount = pointCount;
      pointsGeometry.setDrawRange(0, covarianceDrawCount);
      invalidateSceneRender();
    }

    function installTrackLengths(trackLengths) {
      if (trackLengths.length !== pointPositions.length / 3) {
        throw new Error(
          `Track lengths must match point count: ${trackLengths.length} for ${pointPositions.length / 3}`
        );
      }
      pointsGeometry.setAttribute(
        "vidmapTrackLength",
        new THREE.BufferAttribute(Float32Array.from(trackLengths), 1)
      );
      invalidateSceneRender();
    }

    function setMinimumTrackLength(value) {
      minimumTrackLength.value = value;
      invalidateSceneRender();
    }

    function imageSource(directory, name) {
      if (typeof directory === "object") {
        const preview = directory[name];
        if (
          preview === null || typeof preview !== "object" || typeof preview.url !== "string" ||
          !Number.isSafeInteger(preview.width) || preview.width < 1 ||
          !Number.isSafeInteger(preview.height) || preview.height < 1
        ) {
          throw new Error(`Embedded preview is invalid or missing: ${name}`);
        }
        return preview;
      }
      const normalizedName = name.replaceAll("\\", "/");
      const parts = normalizedName.split("/");
      if (normalizedName.startsWith("/") || parts.some(part => part === "..")) {
        throw new Error(`Reconstruction image name is not a safe relative path: ${name}`);
      }
      const url = new URL("file:///");
      url.pathname = `${directory.replace(/\/+$/, "")}/${normalizedName}`;
      return {url: url.href, width: null, height: null};
    }

    function loadImage(image, directory, name) {
      const source = imageSource(directory, name);
      image.dataset.sourceWidth = source.width === null ? "" : String(source.width);
      image.dataset.sourceHeight = source.height === null ? "" : String(source.height);
      image.src = source.url;
    }

    function imageDimensions(image) {
      const width = Number(image.dataset.sourceWidth) || image.naturalWidth;
      const height = Number(image.dataset.sourceHeight) || image.naturalHeight;
      return {width: width, height: height};
    }

    function drawTrackedKeypoints(index) {
      const canvas = document.getElementById("tracked-keypoints-overlay");
      const preview = document.getElementById("keyframe-preview");
      const width = preview.clientWidth;
      const height = preview.clientHeight;
      const pixelRatio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(width * pixelRatio));
      canvas.height = Math.max(1, Math.round(height * pixelRatio));
      const drawing = canvas.getContext("2d");
      drawing.clearRect(0, 0, canvas.width, canvas.height);
      const toggle = document.getElementById("tracked-keypoints-toggle");
      if (
        trackedKeypointPositions === null || trackedKeypointOffsets === null ||
        !toggle.checked || !preview.complete ||
        !(preview.naturalWidth > 0) || !(preview.naturalHeight > 0)
      ) return;
      const begin = trackedKeypointOffsets[index];
      const end = trackedKeypointOffsets[index + 1];
      const sourceDimensions = imageDimensions(preview);
      const scaleX = canvas.width / sourceDimensions.width;
      const scaleY = canvas.height / sourceDimensions.height;
      const radius = Math.max(1.7 * pixelRatio, 1);
      drawing.fillStyle = "rgba(0, 255, 170, 0.9)";
      drawing.strokeStyle = "rgba(0, 0, 0, 0.85)";
      drawing.lineWidth = Math.max(pixelRatio, 1);
      for (let point = begin; point < end; ++point) {
        if (trackedKeypointPointIds[point] === 0xffffffffffffffffn) continue;
        const x = trackedKeypointPositions[point * 2] * scaleX;
        const y = trackedKeypointPositions[point * 2 + 1] * scaleY;
        drawing.beginPath();
        drawing.arc(x, y, radius, 0, 2 * Math.PI);
        drawing.fill();
        drawing.stroke();
      }
    }

    function installTrackedKeypoints(positions, pointIds, offsets) {
      if (offsets.length !== keyframeTimeline.names.length + 1) {
        throw new Error("Tracked-keypoint offsets must match the keyframe timeline");
      }
      if (positions.length !== offsets[offsets.length - 1] * 2 || pointIds.length !== offsets[offsets.length - 1]) {
        throw new Error("Tracked-keypoint positions do not match their offsets");
      }
      trackedKeypointPositions = positions;
      trackedKeypointPointIds = pointIds;
      trackedKeypointOffsets = offsets;
      drawTrackedKeypoints(Number(document.getElementById("keyframe-slider").value));
    }

    function installKeyframeTimeline(timeline) {
      if (
        timeline === null || !Array.isArray(timeline.names) || !Array.isArray(timeline.imageIds) ||
        timeline.names.length === 0 || timeline.names.length !== timeline.imageIds.length
      ) {
        throw new Error("Loaded reconstruction has an invalid keyframe timeline");
      }
      pauseKeyframePlayback();
      keyframeTimeline.names = timeline.names;
      keyframeTimeline.imageIds = timeline.imageIds;
      keyframeTimeline.timestampsSeconds = timeline.timestampsSeconds;
      keyframeTimeline.pointOffsets = new Array(timeline.names.length + 1).fill(0);
      const slider = document.getElementById("keyframe-slider");
      slider.max = String(timeline.names.length - 1);
      slider.value = slider.max;
      slider.disabled = false;
      const play = document.getElementById("keyframe-play");
      const status = document.getElementById("keyframe-playback-status");
      play.disabled = timeline.timestampsSeconds === null;
      status.textContent = timeline.timestampsSeconds === null ? "timestamps unavailable" : "ready";
    }

    function drawLoopClosureKeypoints(image, canvas, keypoints, matchClasses) {
      if (!image.complete || !(image.naturalWidth > 0) || !(image.naturalHeight > 0)) return;
      const pixelRatio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(image.clientWidth * pixelRatio));
      canvas.height = Math.max(1, Math.round(image.clientHeight * pixelRatio));
      const drawing = canvas.getContext("2d");
      drawing.clearRect(0, 0, canvas.width, canvas.height);
      const sourceDimensions = imageDimensions(image);
      const scaleX = canvas.width / sourceDimensions.width;
      const scaleY = canvas.height / sourceDimensions.height;
      const radius = Math.max(2 * pixelRatio, 1);
      drawing.strokeStyle = "rgba(0, 0, 0, 0.9)";
      drawing.lineWidth = Math.max(pixelRatio, 1);
      const colors = ["rgba(0, 140, 255, 0.95)", "rgba(255, 0, 0, 1)", "rgba(0, 255, 170, 0.92)"];
      for (let matchClass = 0; matchClass < colors.length; ++matchClass) {
        drawing.fillStyle = colors[matchClass];
        for (let point = 0; point < keypoints.length; point += 2) {
          if (matchClasses[point / 2] !== matchClass) continue;
          drawing.beginPath();
          drawing.arc(keypoints[point] * scaleX, keypoints[point + 1] * scaleY, radius, 0, 2 * Math.PI);
          drawing.fill();
          drawing.stroke();
        }
      }
    }

    function redrawLoopClosurePair(selection) {
      drawLoopClosureKeypoints(
        document.getElementById("loop-closure-left-image"),
        document.getElementById("loop-closure-left-keypoints"),
        selection.leftKeypoints,
        selection.matchClasses
      );
      drawLoopClosureKeypoints(
        document.getElementById("loop-closure-right-image"),
        document.getElementById("loop-closure-right-keypoints"),
        selection.rightKeypoints,
        selection.matchClasses
      );
    }

    function showLoopClosurePair(selection) {
      if (imageDirectory === null) throw new Error("Load the reconstruction image source before opening a pair");
      const panel = document.getElementById("loop-closure-pair-panel");
      activeLoopClosurePair = selection;
      const leftImage = document.getElementById("loop-closure-left-image");
      const rightImage = document.getElementById("loop-closure-right-image");
      document.getElementById("loop-closure-left-name").textContent = selection.leftName;
      document.getElementById("loop-closure-right-name").textContent = selection.rightName;
      document.getElementById("loop-closure-pair-status").textContent =
        `${selection.totalMatchCount.toLocaleString()} matches · ` +
        `${selection.matchCount.toLocaleString()} LC · ` +
        `${selection.sharedLoopClosureCount.toLocaleString()} LC sharing a 3D point · ` +
        `${selection.sharedPointCount.toLocaleString()} highlighted 3D points`;
      leftImage.onload = () => requestAnimationFrame(() => drawLoopClosureKeypoints(
        leftImage,
        document.getElementById("loop-closure-left-keypoints"),
        selection.leftKeypoints,
        selection.matchClasses
      ));
      rightImage.onload = () => requestAnimationFrame(() => drawLoopClosureKeypoints(
        rightImage,
        document.getElementById("loop-closure-right-keypoints"),
        selection.rightKeypoints,
        selection.matchClasses
      ));
      panel.hidden = false;
      loadImage(leftImage, imageDirectory, selection.leftName);
      loadImage(rightImage, imageDirectory, selection.rightName);
      setLoopClosurePointHighlights(selection.sharedPointPositions);
    }

    function closeLoopClosurePair() {
      const panel = document.getElementById("loop-closure-pair-panel");
      panel.hidden = true;
      activeLoopClosurePair = null;
      setLoopClosurePointHighlights(new Float32Array());
    }

    function installImageDirectory(directory) {
      imageDirectory = directory;
      const slider = document.getElementById("keyframe-slider");
      showKeyframe(Number(slider.value));
    }

    function resetRunVisualState() {
      imageDirectory = null;
      closeLoopClosurePair();
      trackedKeypointPositions = null;
      trackedKeypointPointIds = null;
      trackedKeypointOffsets = null;
      const empty = new Float32Array();
      replacePointGeometry(empty, null);
      replaceEstimatedGeometry(empty, empty);
      replaceGtGeometry(null, null);
      replaceLoopClosureGeometry(new Uint32Array(), new Float32Array());
      error.style.display = "none";
      error.textContent = "";
      pauseKeyframePlayback();
      keyframeTimeline.names = [];
      keyframeTimeline.imageIds = [];
      keyframeTimeline.timestampsSeconds = null;
      keyframeTimeline.pointOffsets = [0];
      const slider = document.getElementById("keyframe-slider");
      slider.max = "0";
      slider.value = "0";
      slider.disabled = true;
      document.getElementById("keyframe-play").disabled = true;
      document.getElementById("keyframe-playback-status").textContent = "timestamps unavailable";
      document.getElementById("keyframe-label").textContent = "";
      const preview = document.getElementById("keyframe-preview");
      preview.removeAttribute("src");
      preview.removeAttribute("data-source-width");
      preview.removeAttribute("data-source-height");
      drawTrackedKeypoints(0);
    }

    function setCovarianceRankLimit(limit) {
      if (!covarianceIndexActive) return;
      covarianceDrawCount = limit < 0 ? pointPositions.length / 3 : limit;
      pointsGeometry.setDrawRange(0, covarianceDrawCount);
      invalidateSceneRender();
    }

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
    function visibleLoopClosureData(maximumKeyframe) {
      const positions = [];
      const sharedPoints = [];
      const edgeIndices = [];
      for (let edge = 0; edge < loopClosureSharedPoints.length; ++edge) {
        const left = loopClosureKeyframeIndices[edge * 2];
        const right = loopClosureKeyframeIndices[edge * 2 + 1];
        if (left > maximumKeyframe || right > maximumKeyframe) continue;
        if ((left + 1) * 3 > estimatedPathPositions.length || (right + 1) * 3 > estimatedPathPositions.length) {
          continue;
        }
        positions.push(
          estimatedPathPositions[left * 3],
          estimatedPathPositions[left * 3 + 1],
          estimatedPathPositions[left * 3 + 2],
          estimatedPathPositions[right * 3],
          estimatedPathPositions[right * 3 + 1],
          estimatedPathPositions[right * 3 + 2]
        );
        sharedPoints.push(loopClosureSharedPoints[edge]);
        edgeIndices.push(edge);
      }
      return {
        positions: Float32Array.from(positions),
        sharedPoints: Float32Array.from(sharedPoints),
        edgeIndices: Uint32Array.from(edgeIndices)
      };
    }
    const estimatedFrusta = wideLineSegments(estimatedFrustaPositions, 0xd62728, 0.45, 1);
    scene.add(estimatedFrusta);
    const gtFrusta = wideLineSegments(gtFrustaPositions, 0x00c853, 0.85, 2);
    gtFrusta.visible = false;
    gtFrusta.renderOrder = 95;
    scene.add(gtFrusta);
    const gtCovEllipses = wideLineSegments(gtCovEllipsesPositions, 0x00897b, 0.78, 1.5);
    gtCovEllipses.visible = false;
    gtCovEllipses.renderOrder = 93;
    scene.add(gtCovEllipses);
    const denseGtFrusta = wideLineSegments(denseGtFrustaPositions, 0xeab308, 0.75, 1.5);
    denseGtFrusta.visible = false;
    denseGtFrusta.renderOrder = 90;
    scene.add(denseGtFrusta);
    const initialLoopClosures = visibleLoopClosureData(keyframeTimeline.names.length - 1);
    const loopClosures = coloredWideLineSegments(
      initialLoopClosures.positions,
      initialLoopClosures.sharedPoints,
      3,
      __LOOP_CLOSURE_MIN_SHARED_POINTS__
    );
    loopClosures.visible = false;
    let visibleLoopClosurePositions = initialLoopClosures.positions;
    let visibleLoopClosureIndices = initialLoopClosures.edgeIndices;
    scene.add(loopClosures);
    function updateLoopClosureGeometry(maximumKeyframe) {
      const visible = visibleLoopClosureData(maximumKeyframe);
      visibleLoopClosurePositions = visible.positions;
      visibleLoopClosureIndices = visible.edgeIndices;
      const minimumInput = document.getElementById("loop-closures-min-shared");
      const minimum = Number(minimumInput.value);
      if (!Number.isFinite(minimum) || minimum < 0) return;
      const replacement = new THREE.LineSegmentsGeometry();
      replacement.setPositions(visible.positions);
      replacement.setColors(loopClosureVertexColors(visible.sharedPoints, minimum));
      loopClosures.geometry.dispose();
      loopClosures.geometry = replacement;
      const toggle = document.getElementById("loop-closures-toggle");
      loopClosures.visible = visible.sharedPoints.length > 0 && toggle.checked;
      invalidateSceneRender();
    }
    function replaceLoopClosureGeometry(keyframeIndices, sharedPoints) {
      if (keyframeIndices.length !== sharedPoints.length * 2) {
        throw new Error("Shared-track pair indices and counts must match");
      }
      loopClosureKeyframeIndices = keyframeIndices;
      loopClosureSharedPoints = Float32Array.from(sharedPoints);
      const slider = document.getElementById("keyframe-slider");
      updateLoopClosureGeometry(Number(slider.value));
    }
    function pickedLoopClosure(clientX, clientY) {
      if (!loopClosures.visible || visibleLoopClosurePositions.length === 0) return null;
      const bounds = renderer.domElement.getBoundingClientRect();
      if (
        clientX < bounds.left || clientX >= bounds.right ||
        clientY < bounds.top || clientY >= bounds.bottom
      ) return null;
      const point = new THREE.Vector2(clientX - bounds.left, clientY - bounds.top);
      const start = new THREE.Vector3();
      const end = new THREE.Vector3();
      let selected = null;
      let bestDistance = 10;
      for (let edge = 0; edge < visibleLoopClosureIndices.length; ++edge) {
        start.fromArray(visibleLoopClosurePositions, edge * 6).project(camera);
        end.fromArray(visibleLoopClosurePositions, edge * 6 + 3).project(camera);
        if ((start.z < -1 && end.z < -1) || (start.z > 1 && end.z > 1)) continue;
        start.set((start.x + 1) * bounds.width / 2, (1 - start.y) * bounds.height / 2, 0);
        end.set((end.x + 1) * bounds.width / 2, (1 - end.y) * bounds.height / 2, 0);
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const lengthSquared = dx * dx + dy * dy;
        const position = lengthSquared === 0 ? 0 : Math.max(
          0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared)
        );
        const distance = Math.hypot(point.x - start.x - position * dx, point.y - start.y - position * dy);
        if (distance < bestDistance) {
          bestDistance = distance;
          selected = visibleLoopClosureIndices[edge];
        }
      }
      return selected;
    }
    renderer.domElement.addEventListener("click", async event => {
      if (orbitMoved) {
        orbitMoved = false;
        return;
      }
      const edge = pickedLoopClosure(event.clientX, event.clientY);
      if (edge === null) return;
      const left = loopClosureKeyframeIndices[edge * 2];
      const right = loopClosureKeyframeIndices[edge * 2 + 1];
      try {
        await loopClosureSelectionHandler(left, right);
      } catch (selectionError) {
        error.style.display = "block";
        error.textContent = selectionError instanceof Error ? selectionError.message : String(selectionError);
      }
    });

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
    function updatePathPrefix(object, positions, pointCount, extraVisible = true) {
      const replacement = new THREE.LineGeometry();
      const hasSegments = pointCount >= 2;
      if (hasSegments) replacement.setPositions(positions.subarray(0, pointCount * 3));
      object.geometry.dispose();
      object.geometry = replacement;
      object.visible = hasSegments && document.getElementById("paths-toggle").checked && extraVisible;
      invalidateSceneRender();
    }
    const estimatedPath = path(estimatedPathPositions, 0x0064ff);
    scene.add(estimatedPath);
    const gtPath = path(gtPathPositions, 0x00c853);
    scene.add(gtPath);
    const denseGtPath = path(denseGtPathPositions, 0xeab308);
    denseGtPath.visible = false;
    denseGtPath.renderOrder = 98;
    scene.add(denseGtPath);

    function priorConfidenceRgb(conf) {
      const t = Math.max(0, Math.min(1, (Number(conf) - 0.95) / 0.05));
      if (t < 0.5) {
        const u = t * 2;
        return [(235 - 15 * u) / 255, (45 + 115 * u) / 255, 0];
      }
      const u = (t - 0.5) * 2;
      return [(220 * (1 - u)) / 255, (160 + 40 * u) / 255, (85 * u) / 255];
    }
    function perCameraVertexColors(confidences, cameraCount, verticesPerCamera) {
      const colors = new Float32Array(cameraCount * verticesPerCamera * 3);
      for (let c = 0; c < cameraCount; ++c) {
        const conf = c < confidences.length ? confidences[c] : 1.0;
        const rgb = priorConfidenceRgb(conf);
        const base = c * verticesPerCamera * 3;
        for (let v = 0; v < verticesPerCamera; ++v) {
          colors[base + v * 3] = rgb[0];
          colors[base + v * 3 + 1] = rgb[1];
          colors[base + v * 3 + 2] = rgb[2];
        }
      }
      return colors;
    }
    function updateCameraSegmentsPrefix(
      object,
      base,
      centers,
      pointCount,
      scaleFactor,
      verticesPerCamera,
      confidences = null,
      solidColorHex = null
    ) {
      const scaled = scaledCameraSegments(base, centers, scaleFactor, pointCount, verticesPerCamera);
      const replacement = new THREE.LineSegmentsGeometry();
      if (scaled.length > 0) {
        replacement.setPositions(scaled);
        if (confidences !== null && confidences.length > 0) {
          const camCount = scaled.length / (verticesPerCamera * 3);
          replacement.setColors(perCameraVertexColors(confidences, camCount, verticesPerCamera));
          object.material.vertexColors = true;
          object.material.color.set(0xffffff);
          object.material.needsUpdate = true;
        } else if (solidColorHex !== null) {
          object.material.vertexColors = false;
          object.material.color.set(solidColorHex);
          object.material.needsUpdate = true;
        }
      }
      object.geometry.dispose();
      object.geometry = replacement;
      invalidateSceneRender();
    }
    function updateFrustaPrefix(object, base, centers, pointCount, size, confidences = null, solidColorHex = null) {
      updateCameraSegmentsPrefix(object, base, centers, pointCount, size / 0.3, 16, confidences, solidColorHex);
    }
    function updateCovEllipsesPrefix(
      object,
      base,
      centers,
      pointCount,
      sigmaScale,
      confidences = null,
      solidColorHex = null
    ) {
      updateCameraSegmentsPrefix(object, base, centers, pointCount, sigmaScale, 144, confidences, solidColorHex);
    }
    function replaceEstimatedGeometry(centers, frusta, framesTrajectory = null) {
      estimatedPathPositions = centers;
      estimatedFrustaPositions = frusta;
      const hasFrames = (
        framesTrajectory !== null && typeof framesTrajectory === "object" &&
        Array.isArray(framesTrajectory.centers) && framesTrajectory.centers.length >= 6
      );
      estimatedFramesPathPositions = hasFrames ? Float32Array.from(framesTrajectory.centers) : new Float32Array();
      estimatedFramesKeyframeCounts = (
        hasFrames && Array.isArray(framesTrajectory.keyframePathCounts) ? framesTrajectory.keyframePathCounts : []
      );
      updateLoopClosureGeometry(Number(document.getElementById("keyframe-slider").value));
      fitScene();
    }
    function replaceGtGeometry(gtTrajectory, denseGtTrajectory = null) {
      const hasGt = (
        gtTrajectory !== null && typeof gtTrajectory === "object" &&
        Array.isArray(gtTrajectory.centers) && gtTrajectory.centers.length >= 6
      );
      gtPathPositions = hasGt ? Float32Array.from(gtTrajectory.centers) : new Float32Array();
      gtKeyframePathCounts = (
        hasGt && Array.isArray(gtTrajectory.keyframePathCounts) ? gtTrajectory.keyframePathCounts : []
      );
      gtKeyframeFrustaCounts = (
        hasGt && Array.isArray(gtTrajectory.keyframeFrustaCounts)
          ? gtTrajectory.keyframeFrustaCounts
          : gtKeyframePathCounts
      );
      gtKeyframeCenters = (
        hasGt && Array.isArray(gtTrajectory.keyframeCenters)
          ? Float32Array.from(gtTrajectory.keyframeCenters)
          : new Float32Array()
      );
      gtFrustaPositions = (
        hasGt && Array.isArray(gtTrajectory.keyframeFrusta)
          ? Float32Array.from(gtTrajectory.keyframeFrusta)
          : new Float32Array()
      );
      gtCovEllipsesPositions = (
        hasGt && Array.isArray(gtTrajectory.keyframeCovEllipses)
          ? Float32Array.from(gtTrajectory.keyframeCovEllipses)
          : new Float32Array()
      );
      gtKeyframeConfidences = (
        hasGt && Array.isArray(gtTrajectory.keyframeConfidences)
          ? gtTrajectory.keyframeConfidences
          : []
      );
      gtKeyframeNames = (
        hasGt && Array.isArray(gtTrajectory.keyframeNames)
          ? gtTrajectory.keyframeNames
          : []
      );
      gtKeyframeInliers = (
        hasGt && Array.isArray(gtTrajectory.keyframeInliers)
          ? gtTrajectory.keyframeInliers
          : []
      );
      gtKeyframePosStdMeters = (
        hasGt && Array.isArray(gtTrajectory.keyframePosStdMeters)
          ? gtTrajectory.keyframePosStdMeters
          : []
      );
      gtKeyframeReprojRmsPx = (
        hasGt && Array.isArray(gtTrajectory.keyframeReprojRmsPx)
          ? gtTrajectory.keyframeReprojRmsPx
          : []
      );
      gtMetadataByName = new Map();
      for (let i = 0; i < gtKeyframeNames.length; ++i) {
        gtMetadataByName.set(gtKeyframeNames[i], {
          index: i,
          name: gtKeyframeNames[i],
          confidence: i < gtKeyframeConfidences.length ? gtKeyframeConfidences[i] : null,
          inliers: i < gtKeyframeInliers.length ? gtKeyframeInliers[i] : null,
          posStdMeters: i < gtKeyframePosStdMeters.length ? gtKeyframePosStdMeters[i] : null,
          reprojRmsPx: i < gtKeyframeReprojRmsPx.length ? gtKeyframeReprojRmsPx[i] : null
        });
      }
      const hasDenseGt = (
        denseGtTrajectory !== null && typeof denseGtTrajectory === "object" &&
        Array.isArray(denseGtTrajectory.centers) && denseGtTrajectory.centers.length >= 6
      );
      denseGtPathPositions = hasDenseGt ? Float32Array.from(denseGtTrajectory.centers) : new Float32Array();
      denseGtKeyframePathCounts = (
        hasDenseGt && Array.isArray(denseGtTrajectory.keyframePathCounts)
          ? denseGtTrajectory.keyframePathCounts
          : []
      );
      denseGtKeyframeFrustaCounts = (
        hasDenseGt && Array.isArray(denseGtTrajectory.keyframeFrustaCounts)
          ? denseGtTrajectory.keyframeFrustaCounts
          : denseGtKeyframePathCounts
      );
      denseGtKeyframeCenters = (
        hasDenseGt && Array.isArray(denseGtTrajectory.keyframeCenters)
          ? Float32Array.from(denseGtTrajectory.keyframeCenters)
          : new Float32Array()
      );
      denseGtFrustaPositions = (
        hasDenseGt && Array.isArray(denseGtTrajectory.keyframeFrusta)
          ? Float32Array.from(denseGtTrajectory.keyframeFrusta)
          : new Float32Array()
      );
      const hasConf = hasGt && gtKeyframeConfidences.length > 0;
      const hasCov = hasGt && gtCovEllipsesPositions.length > 0;
      const gtFrustaLabel = document.getElementById("gt-frusta-label");
      const gtPathLabel = document.getElementById("gt-path-label");
      if (gtFrustaLabel !== null) gtFrustaLabel.textContent = (hasDenseGt || hasConf) ? "prior cameras" : "GT cameras";
      if (gtPathLabel !== null) gtPathLabel.textContent = (hasDenseGt || hasConf) ? "prior" : "GT";
      document.getElementById("gt-path-controls").hidden = !hasGt;
      document.getElementById("gt-frusta-row").hidden = !(hasGt && gtFrustaPositions.length > 0);
      const gtFrustaModeSetting = document.getElementById("gt-frusta-mode-setting");
      if (gtFrustaModeSetting !== null) gtFrustaModeSetting.hidden = !hasConf;
      const gtCovRow = document.getElementById("gt-cov-row");
      if (gtCovRow !== null) gtCovRow.hidden = !hasCov;
      document.getElementById("dense-gt-path-controls").hidden = !hasDenseGt;
      document.getElementById("dense-gt-frusta-row").hidden = !(hasDenseGt && denseGtFrustaPositions.length > 0);
      const gtFrustaColorInput = document.getElementById("gt-frusta-color");
      const gtFrustaColorMode = document.getElementById("gt-frusta-color-mode");
      if (gtFrustaColorInput !== null && gtFrustaColorMode !== null) {
        gtFrustaColorInput.disabled = hasConf && gtFrustaColorMode.value === "confidence";
      }
      const gtCovColorInput = document.getElementById("gt-cov-color");
      const gtCovColorMode = document.getElementById("gt-cov-color-mode");
      if (gtCovColorInput !== null && gtCovColorMode !== null) {
        gtCovColorInput.disabled = hasConf && gtCovColorMode.value === "confidence";
      }
      if (!hasGt) {
        gtPath.visible = false;
        gtFrusta.visible = false;
        gtCovEllipses.visible = false;
      }
      if (!hasDenseGt) {
        denseGtPath.visible = false;
        denseGtFrusta.visible = false;
      }
      fitScene();
    }
    function showKeyframe(index) {
      if (keyframeTimeline.names.length === 0) return;
      const bounded = Math.max(0, Math.min(index, keyframeTimeline.names.length - 1));
      document.getElementById("keyframe-slider").value = String(bounded);
      const pointCount = keyframeTimeline.pointOffsets[bounded + 1];
      if (covarianceIndexActive) {
        keyframeLimit.value = bounded;
        pointsGeometry.setDrawRange(0, covarianceDrawCount);
      } else {
        pointsGeometry.setDrawRange(0, pointCount);
      }
      const timestampText = keyframeTimeline.timestampsSeconds === null
        ? ""
        : ` · ${keyframeTimeline.timestampsSeconds[bounded].toFixed(3)} s`;
      const kfName = keyframeTimeline.names[bounded];
      let priorStatusText = "";
      if (gtMetadataByName.has(kfName)) {
        const meta = gtMetadataByName.get(kfName);
        if (meta.confidence !== null) {
          priorStatusText += ` · prior conf=${Number(meta.confidence).toFixed(4)}`;
        }
        if (Array.isArray(meta.posStdMeters) && meta.posStdMeters.length >= 1) {
          priorStatusText += ` · σ_pos=${Number(meta.posStdMeters[0]).toFixed(2)}m`;
        }
      }
      document.getElementById("keyframe-label").textContent =
        `${bounded + 1}/${keyframeTimeline.names.length}${timestampText} · ${kfName} · ${pointCount.toLocaleString()} cumulative points${priorStatusText}`;
      const preview = document.getElementById("keyframe-preview");
      if (imageDirectory !== null) {
        loadImage(preview, imageDirectory, kfName);
      }
      preview.dataset.keyframeIndex = String(bounded);
      drawTrackedKeypoints(bounded);
      if (estimatedFramesPathPositions.length >= 6) {
        const framesCount = bounded < estimatedFramesKeyframeCounts.length
          ? estimatedFramesKeyframeCounts[bounded]
          : Math.floor(estimatedFramesPathPositions.length / 3);
        updatePathPrefix(estimatedPath, estimatedFramesPathPositions, framesCount);
      } else {
        updatePathPrefix(estimatedPath, estimatedPathPositions, bounded + 1);
      }
      const estimatedSize = Number(document.getElementById("estimated-frusta-size").value);
      updateFrustaPrefix(
        estimatedFrusta, estimatedFrustaPositions, estimatedPathPositions, bounded + 1, estimatedSize
      );
      const gtPointCount = bounded < gtKeyframePathCounts.length
        ? gtKeyframePathCounts[bounded]
        : Math.floor(gtPathPositions.length / 3);
      if (gtPathPositions.length >= 6) {
        updatePathPrefix(
          gtPath,
          gtPathPositions,
          gtPointCount,
          document.getElementById("gt-path-toggle").checked
        );
      } else {
        gtPath.visible = false;
      }
      const gtFrustaCount = bounded < gtKeyframeFrustaCounts.length
        ? gtKeyframeFrustaCounts[bounded]
        : Math.floor(gtKeyframeCenters.length / 3);
      if (gtFrustaPositions.length > 0) {
        const gtSize = Number(document.getElementById("gt-frusta-size").value);
        const gtColorMode = document.getElementById("gt-frusta-color-mode");
        const useGtConf = gtKeyframeConfidences.length > 0 && gtColorMode !== null && gtColorMode.value === "confidence";
        updateFrustaPrefix(
          gtFrusta,
          gtFrustaPositions,
          gtKeyframeCenters,
          gtFrustaCount,
          gtSize,
          useGtConf ? gtKeyframeConfidences : null,
          document.getElementById("gt-frusta-color").value
        );
        gtFrusta.visible = gtFrustaCount > 0 && document.getElementById("gt-frusta-toggle").checked;
      } else {
        gtFrusta.visible = false;
      }
      if (gtCovEllipsesPositions.length > 0) {
        const covScale = Number(document.getElementById("gt-cov-scale").value);
        const covColorMode = document.getElementById("gt-cov-color-mode");
        const useCovConf = gtKeyframeConfidences.length > 0 && covColorMode !== null && covColorMode.value === "confidence";
        updateCovEllipsesPrefix(
          gtCovEllipses,
          gtCovEllipsesPositions,
          gtKeyframeCenters,
          gtFrustaCount,
          covScale,
          useCovConf ? gtKeyframeConfidences : null,
          document.getElementById("gt-cov-color").value
        );
        const covToggle = document.getElementById("gt-cov-toggle");
        gtCovEllipses.visible = gtFrustaCount > 0 && covToggle !== null && covToggle.checked;
      } else {
        gtCovEllipses.visible = false;
      }
      const denseGtPointCount = bounded < denseGtKeyframePathCounts.length
        ? denseGtKeyframePathCounts[bounded]
        : Math.floor(denseGtPathPositions.length / 3);
      if (denseGtPathPositions.length >= 6) {
        updatePathPrefix(
          denseGtPath,
          denseGtPathPositions,
          denseGtPointCount,
          document.getElementById("dense-gt-path-toggle").checked
        );
      } else {
        denseGtPath.visible = false;
      }
      if (denseGtFrustaPositions.length > 0) {
        const denseGtSize = Number(document.getElementById("dense-gt-frusta-size").value);
        const denseGtFrustaCount = bounded < denseGtKeyframeFrustaCounts.length
          ? denseGtKeyframeFrustaCounts[bounded]
          : Math.floor(denseGtKeyframeCenters.length / 3);
        updateFrustaPrefix(
          denseGtFrusta, denseGtFrustaPositions, denseGtKeyframeCenters, denseGtFrustaCount, denseGtSize
        );
        denseGtFrusta.visible = denseGtFrustaCount > 0 && document.getElementById("dense-gt-frusta-toggle").checked;
      } else {
        denseGtFrusta.visible = false;
      }
      updateLoopClosureGeometry(bounded);
    }
    const keyframePreview = document.getElementById("keyframe-preview");
    keyframePreview.onload = () => drawTrackedKeypoints(Number(keyframePreview.dataset.keyframeIndex));
    const keyframePreviewPanel = document.getElementById("keyframe-preview-panel");
    keyframePreviewPanel.ontoggle = () => {
      if (keyframePreviewPanel.open) {
        requestAnimationFrame(() => drawTrackedKeypoints(Number(keyframeSlider.value)));
      }
    };
    const loopClosurePairClose = document.getElementById("loop-closure-pair-close");
    loopClosurePairClose.onclick = closeLoopClosurePair;
    if (typeof ResizeObserver !== "undefined") {
      const imagePanelResizeObserver = new ResizeObserver(() => {
        requestAnimationFrame(() => {
          drawTrackedKeypoints(Number(keyframeSlider.value));
          if (activeLoopClosurePair !== null) redrawLoopClosurePair(activeLoopClosurePair);
        });
      });
      imagePanelResizeObserver.observe(keyframePreviewPanel);
      const loopClosurePairPanel = document.getElementById("loop-closure-pair-panel");
      imagePanelResizeObserver.observe(loopClosurePairPanel);
    }

    const bounds = new THREE.Box3();
    const fitPoint = new THREE.Vector3();
    const center = new THREE.Vector3();
    const extent = new THREE.Vector3();
    let radius = 1;

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
    function fitScene() {
      bounds.makeEmpty();
      const fitPositions = estimatedPathPositions.length > 0 ? [estimatedPathPositions] : [pointPositions];
      if (gtPathPositions.length > 0) fitPositions.push(gtPathPositions);
      else if (denseGtPathPositions.length > 0) fitPositions.push(denseGtPathPositions);
      for (const positions of fitPositions) {
        for (let index = 0; index < positions.length; index += 3) {
          fitPoint.set(positions[index], positions[index + 1], positions[index + 2]);
          bounds.expandByPoint(fitPoint);
        }
      }
      if (bounds.isEmpty()) {
        bounds.setFromCenterAndSize(new THREE.Vector3(), new THREE.Vector3(2, 2, 2));
      }
      bounds.getCenter(center);
      bounds.getSize(extent);
      radius = Math.max(extent.length() * 0.5, 1);
      controls.target.copy(center);
      const initialPosition = center.clone().add(
        new THREE.Vector3(1, -1, 0.8).normalize().multiplyScalar(radius * 2.2)
      );
      for (const viewCamera of [perspectiveCamera, orthographicCamera]) {
        viewCamera.near = Math.max(radius / 10000, 0.001);
        viewCamera.far = radius * 100;
        viewCamera.position.copy(initialPosition);
      }
      updateProjectionDimensions();
      controls.update();
      invalidateSceneRender();
    }
    fitScene();

    document.getElementById("points-toggle").onchange = event => points.visible = event.target.checked;
    const keyframeSlider = document.getElementById("keyframe-slider");
    const keyframePlay = document.getElementById("keyframe-play");
    const keyframeSpeed = document.getElementById("keyframe-speed");
    const keyframePlaybackStatus = document.getElementById("keyframe-playback-status");
    let keyframePlaybackRequest = null;
    function timelineIndexAtElapsed(timestamps, startIndex, elapsedSeconds, speed) {
      const target = timestamps[startIndex] + elapsedSeconds * speed;
      let low = startIndex;
      let high = timestamps.length;
      while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (timestamps[middle] <= target) low = middle + 1;
        else high = middle;
      }
      return Math.max(startIndex, low - 1);
    }
    function pauseKeyframePlayback() {
      if (keyframePlaybackRequest !== null) cancelAnimationFrame(keyframePlaybackRequest);
      keyframePlaybackRequest = null;
      keyframePlay.textContent = "play";
      if (keyframeTimeline.timestampsSeconds !== null) {
        keyframePlaybackStatus.textContent = "paused";
      }
    }
    function startKeyframePlayback() {
      if (keyframeTimeline.timestampsSeconds === null) return;
      let startIndex = Number(keyframeSlider.value);
      if (startIndex >= keyframeTimeline.names.length - 1) {
        startIndex = 0;
        showKeyframe(startIndex);
      }
      const startedAt = performance.now();
      const speed = Number(keyframeSpeed.value);
      let displayedIndex = startIndex;
      keyframePlay.textContent = "pause";
      keyframePlaybackStatus.textContent = `${speed}× real time`;
      function advance(now) {
        const index = timelineIndexAtElapsed(
          keyframeTimeline.timestampsSeconds,
          startIndex,
          (now - startedAt) / 1000,
          speed
        );
        if (index !== displayedIndex) {
          showKeyframe(index);
          displayedIndex = index;
        }
        if (index >= keyframeTimeline.names.length - 1) {
          pauseKeyframePlayback();
          keyframePlaybackStatus.textContent = "finished";
          return;
        }
        keyframePlaybackRequest = requestAnimationFrame(advance);
      }
      keyframePlaybackRequest = requestAnimationFrame(advance);
    }
    keyframeSlider.oninput = event => {
      pauseKeyframePlayback();
      showKeyframe(Number(event.target.value));
    };
    keyframePlay.onclick = () => {
      if (keyframePlaybackRequest === null) startKeyframePlayback();
      else pauseKeyframePlayback();
    };
    keyframeSpeed.onchange = () => {
      if (keyframePlaybackRequest !== null) {
        pauseKeyframePlayback();
        startKeyframePlayback();
      }
    };
    const trackedKeypointsToggle = document.getElementById("tracked-keypoints-toggle");
    trackedKeypointsToggle.onchange = () => drawTrackedKeypoints(Number(keyframeSlider.value));
    const estimatedFrustaToggle = document.getElementById("estimated-frusta-toggle");
    estimatedFrustaToggle.onchange = event => estimatedFrusta.visible = event.target.checked;
    const gtFrustaToggle = document.getElementById("gt-frusta-toggle");
    gtFrustaToggle.onchange = () => showKeyframe(Number(keyframeSlider.value));
    const gtCovToggle = document.getElementById("gt-cov-toggle");
    if (gtCovToggle !== null) {
      gtCovToggle.onchange = () => showKeyframe(Number(keyframeSlider.value));
    }
    const denseGtFrustaToggle = document.getElementById("dense-gt-frusta-toggle");
    denseGtFrustaToggle.onchange = () => showKeyframe(Number(keyframeSlider.value));
    document.getElementById("paths-toggle").onchange = () => showKeyframe(Number(keyframeSlider.value));
    document.getElementById("gt-path-toggle").onchange = () => showKeyframe(Number(keyframeSlider.value));
    document.getElementById("dense-gt-path-toggle").onchange = () => showKeyframe(Number(keyframeSlider.value));
    const loopClosuresToggle = document.getElementById("loop-closures-toggle");
    loopClosuresToggle.onchange = () => updateLoopClosureGeometry(Number(keyframeSlider.value));
    bindNumber("loop-closures-width", value => loopClosures.material.linewidth = value);
    const minimumSharedPoints = document.getElementById("loop-closures-min-shared");
    function updateLoopClosureColors() {
      updateLoopClosureGeometry(Number(keyframeSlider.value));
    }
    minimumSharedPoints.oninput = updateLoopClosureColors;

    function bindNumber(id, update) {
      const input = document.getElementById(id);
      if (input === null) return;
      input.oninput = () => {
        const value = Number(input.value);
        if (Number.isFinite(value) && value > 0) update(value);
      };
    }

    function scaledCameraSegments(
      base,
      centers,
      scale,
      requestedCount = centers.length / 3,
      verticesPerCamera = 16
    ) {
      const cameraCount = Math.max(0, Math.floor(Math.min(
        requestedCount, centers.length / 3, base.length / (verticesPerCamera * 3)
      )));
      const result = new Float32Array(cameraCount * verticesPerCamera * 3);
      for (let cameraIndex = 0; cameraIndex < cameraCount; ++cameraIndex) {
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

    function scaledFrusta(base, centers, size, requestedCount = centers.length / 3) {
      return scaledCameraSegments(base, centers, size / 0.3, requestedCount, 16);
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
    pointsColorInput.disabled = true;
    pointsColorMode.onchange = event => {
      const useRgb = event.target.value === "rgb";
      points.material.vertexColors = useRgb;
      points.material.color.set(useRgb ? 0xffffff : pointsColorInput.value);
      points.material.needsUpdate = true;
      pointsColorInput.disabled = useRgb;
    };
    bindNumber(
      "estimated-frusta-size",
      () => showKeyframe(Number(keyframeSlider.value))
    );
    bindNumber("estimated-frusta-width", value => estimatedFrusta.material.linewidth = value);
    document.getElementById("estimated-frusta-color").oninput =
      event => estimatedFrusta.material.color.set(event.target.value);
    bindNumber(
      "gt-frusta-size",
      () => showKeyframe(Number(keyframeSlider.value))
    );
    bindNumber("gt-frusta-width", value => gtFrusta.material.linewidth = value);
    const gtFrustaColorInput = document.getElementById("gt-frusta-color");
    const gtFrustaColorMode = document.getElementById("gt-frusta-color-mode");
    gtFrustaColorInput.oninput = () => showKeyframe(Number(keyframeSlider.value));
    if (gtFrustaColorMode !== null) {
      gtFrustaColorMode.onchange = event => {
        gtFrustaColorInput.disabled = gtKeyframeConfidences.length > 0 && event.target.value === "confidence";
        showKeyframe(Number(keyframeSlider.value));
      };
    }
    bindNumber(
      "gt-cov-scale",
      () => showKeyframe(Number(keyframeSlider.value))
    );
    bindNumber("gt-cov-width", value => gtCovEllipses.material.linewidth = value);
    const gtCovColorInput = document.getElementById("gt-cov-color");
    const gtCovColorMode = document.getElementById("gt-cov-color-mode");
    if (gtCovColorInput !== null) {
      gtCovColorInput.oninput = () => showKeyframe(Number(keyframeSlider.value));
    }
    if (gtCovColorMode !== null) {
      gtCovColorMode.onchange = event => {
        if (gtCovColorInput !== null) {
          gtCovColorInput.disabled = gtKeyframeConfidences.length > 0 && event.target.value === "confidence";
        }
        showKeyframe(Number(keyframeSlider.value));
      };
    }
    bindNumber(
      "dense-gt-frusta-size",
      () => showKeyframe(Number(keyframeSlider.value))
    );
    bindNumber("dense-gt-frusta-width", value => denseGtFrusta.material.linewidth = value);
    document.getElementById("dense-gt-frusta-color").oninput =
      event => denseGtFrusta.material.color.set(event.target.value);
    bindNumber("paths-width", value => {
      estimatedPath.material.linewidth = value;
      gtPath.material.linewidth = value;
      denseGtPath.material.linewidth = value;
    });
    document.getElementById("estimated-path-color").oninput =
      event => estimatedPath.material.color.set(event.target.value);
    document.getElementById("gt-path-color").oninput =
      event => gtPath.material.color.set(event.target.value);
    document.getElementById("dense-gt-path-color").oninput =
      event => denseGtPath.material.color.set(event.target.value);

    const priorHoverTooltip = document.getElementById("prior-hover-tooltip");
    const hoverProjVec = new THREE.Vector3();
    function updatePriorHoverTooltip(clientX, clientY) {
      if (priorHoverTooltip === null) return;
      if (
        orbitPointerId !== null ||
        rollPointerId !== null ||
        (!gtFrusta.visible && !gtCovEllipses.visible) ||
        gtKeyframeCenters.length === 0
      ) {
        priorHoverTooltip.hidden = true;
        return;
      }
      const rect = renderer.domElement.getBoundingClientRect();
      if (clientX < rect.left || clientX >= rect.right || clientY < rect.top || clientY >= rect.bottom) {
        priorHoverTooltip.hidden = true;
        return;
      }
      const px = clientX - rect.left;
      const py = clientY - rect.top;
      const bounded = Math.max(0, Math.min(Number(keyframeSlider.value), keyframeTimeline.names.length - 1));
      const visibleCount = bounded < gtKeyframeFrustaCounts.length
        ? gtKeyframeFrustaCounts[bounded]
        : Math.floor(gtKeyframeCenters.length / 3);
      let bestIdx = -1;
      let bestDist = 18;
      for (let i = 0; i < visibleCount; ++i) {
        hoverProjVec.fromArray(gtKeyframeCenters, i * 3).project(camera);
        if (hoverProjVec.z < -1 || hoverProjVec.z > 1) continue;
        const sx = (hoverProjVec.x + 1) * rect.width * 0.5;
        const sy = (1 - hoverProjVec.y) * rect.height * 0.5;
        const d = Math.hypot(px - sx, py - sy);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }
      if (bestIdx < 0) {
        priorHoverTooltip.hidden = true;
        return;
      }
      const name = bestIdx < gtKeyframeNames.length ? gtKeyframeNames[bestIdx] : `prior #${bestIdx + 1}`;
      const lines = [`prior anchor: ${name}`];
      if (bestIdx < gtKeyframeConfidences.length) {
        lines.push(`Confidence: ${Number(gtKeyframeConfidences[bestIdx]).toFixed(4)}`);
      }
      if (bestIdx < gtKeyframePosStdMeters.length && Array.isArray(gtKeyframePosStdMeters[bestIdx])) {
        const s = gtKeyframePosStdMeters[bestIdx];
        lines.push(
          `Position std (1σ): ${Number(s[0]).toFixed(3)} m  (axes: ${Number(s[1]).toFixed(2)}, ${Number(s[2]).toFixed(2)}, ${Number(s[3]).toFixed(2)} m)`
        );
      }
      if (bestIdx < gtKeyframeInliers.length) {
        const inl = gtKeyframeInliers[bestIdx];
        const rms = bestIdx < gtKeyframeReprojRmsPx.length ? Number(gtKeyframeReprojRmsPx[bestIdx]).toFixed(2) : null;
        lines.push(rms !== null ? `2D-3D inliers: ${inl}  (reproj RMS: ${rms} px)` : `2D-3D inliers: ${inl}`);
      }
      priorHoverTooltip.textContent = lines.join("\n");
      priorHoverTooltip.style.left = `${Math.min(window.innerWidth - 280, clientX + 14)}px`;
      priorHoverTooltip.style.top = `${Math.min(window.innerHeight - 90, clientY + 14)}px`;
      priorHoverTooltip.hidden = false;
    }
    renderer.domElement.addEventListener("pointermove", event => {
      updatePriorHoverTooltip(event.clientX, event.clientY);
    });
    renderer.domElement.addEventListener("pointerleave", () => {
      if (priorHoverTooltip !== null) priorHoverTooltip.hidden = true;
    });

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
      invalidateSceneRender();
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
    document.addEventListener("input", invalidateSceneRender);
    document.addEventListener("change", invalidateSceneRender);
    window.addEventListener("beforeunload", () => {
      if (backgroundUrl !== null) URL.revokeObjectURL(backgroundUrl);
    });

    initializeReconstructionViewer(embeddedRun);

    function resize() {
      updateProjectionDimensions();
      renderer.setSize(window.innerWidth, window.innerHeight);
      for (const object of [estimatedFrusta, gtFrusta, gtCovEllipses, denseGtFrusta, loopClosures, estimatedPath, gtPath, denseGtPath]) {
        object.material.resolution.set(window.innerWidth, window.innerHeight);
      }
      drawTrackedKeypoints(Number(keyframeSlider.value));
      if (activeLoopClosurePair !== null) redrawLoopClosurePair(activeLoopClosurePair);
      invalidateSceneRender();
    }
    window.addEventListener("resize", resize);

    function animate() {
      requestAnimationFrame(animate);
      const controlsChanged = controls.update();
      if (!sceneRenderDirty && !controlsChanged) return;
      sceneRenderDirty = false;
      renderer.render(scene, camera);
    }
    animate();
  })();
