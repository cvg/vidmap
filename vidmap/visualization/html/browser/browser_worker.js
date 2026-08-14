(function (root) {
  "use strict";

  const CAMERA_PARAMETER_COUNTS = [3, 4, 4, 5, 8, 8, 12, 5, 4, 5, 12, 16, 4, 5, 3, 4, 6];
  const SINGLE_FOCAL_MODELS = new Set([0, 2, 3, 8, 9, 12, 14]);
  const CRC32_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let index = 0; index < 256; ++index) {
      let value = index;
      for (let bit = 0; bit < 8; ++bit) value = (value >>> 1) ^ ((value & 1) ? 0xedb88320 : 0);
      table[index] = value >>> 0;
    }
    return table;
  })();

  function checkedCount(value, label) {
    const count = Number(value);
    if (!Number.isSafeInteger(count) || count < 0) throw new Error(`${label} is outside JavaScript's safe range`);
    return count;
  }

  function requireBytes(view, offset, count, label) {
    if (offset < 0 || count < 0 || offset + count > view.byteLength) {
      throw new Error(`Truncated ${label} at byte ${offset}`);
    }
  }

  function crc32(buffer) {
    const bytes = new Uint8Array(buffer);
    let value = 0xffffffff;
    for (let index = 0; index < bytes.length; ++index) {
      value = CRC32_TABLE[(value ^ bytes[index]) & 0xff] ^ (value >>> 8);
    }
    return (value ^ 0xffffffff) >>> 0;
  }

  function fingerprints(buffers) {
    const result = {};
    for (const name of ["cameras.bin", "images.bin", "points3D.bin"]) {
      const buffer = buffers[name];
      result[name] = {size: buffer.byteLength, crc32: crc32(buffer)};
    }
    return result;
  }

  function parseCameras(buffer) {
    const view = new DataView(buffer);
    requireBytes(view, 0, 8, "cameras.bin header");
    const count = checkedCount(view.getBigUint64(0, true), "camera count");
    const cameras = new Map();
    let offset = 8;
    for (let index = 0; index < count; ++index) {
      requireBytes(view, offset, 24, "camera record");
      const cameraId = view.getUint32(offset, true);
      const modelId = view.getInt32(offset + 4, true);
      if (modelId === 17) throw new Error("EQUIRECTANGULAR cameras are not supported by the 3D viewer");
      const parameterCount = CAMERA_PARAMETER_COUNTS[modelId];
      if (parameterCount === undefined) throw new Error(`Unsupported COLMAP camera model id ${modelId}`);
      const width = checkedCount(view.getBigUint64(offset + 8, true), "camera width");
      const height = checkedCount(view.getBigUint64(offset + 16, true), "camera height");
      offset += 24;
      requireBytes(view, offset, parameterCount * 8, "camera parameters");
      const fx = view.getFloat64(offset, true);
      const fy = SINGLE_FOCAL_MODELS.has(modelId) ? fx : view.getFloat64(offset + 8, true);
      const principalOffset = SINGLE_FOCAL_MODELS.has(modelId) ? 8 : 16;
      const cx = view.getFloat64(offset + principalOffset, true);
      const cy = view.getFloat64(offset + principalOffset + 8, true);
      cameras.set(cameraId, {fx: fx, fy: fy, cx: cx, cy: cy, width: width, height: height});
      offset += parameterCount * 8;
    }
    if (offset !== view.byteLength) throw new Error(`Unexpected trailing cameras.bin bytes: ${view.byteLength - offset}`);
    return cameras;
  }

  function quaternionRotation(w, x, y, z) {
    const norm = Math.hypot(w, x, y, z);
    if (!(norm > 0)) throw new Error("Invalid zero-norm image quaternion");
    w /= norm;
    x /= norm;
    y /= norm;
    z /= norm;
    return new Float64Array([
      1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
      2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
    ]);
  }

  function parseImages(buffer) {
    const view = new DataView(buffer);
    requireBytes(view, 0, 8, "images.bin header");
    const count = checkedCount(view.getBigUint64(0, true), "image count");
    const images = new Map();
    let offset = 8;
    for (let index = 0; index < count; ++index) {
      requireBytes(view, offset, 64, "image pose");
      const imageId = view.getUint32(offset, true);
      const rotation = quaternionRotation(
        view.getFloat64(offset + 4, true),
        view.getFloat64(offset + 12, true),
        view.getFloat64(offset + 20, true),
        view.getFloat64(offset + 28, true)
      );
      const translation = new Float64Array([
        view.getFloat64(offset + 36, true),
        view.getFloat64(offset + 44, true),
        view.getFloat64(offset + 52, true)
      ]);
      const cameraId = view.getUint32(offset + 60, true);
      offset += 64;
      const nameStart = offset;
      while (true) {
        requireBytes(view, offset, 1, "image name");
        if (view.getUint8(offset++) === 0) break;
      }
      const name = new TextDecoder().decode(new Uint8Array(buffer, nameStart, offset - nameStart - 1));
      requireBytes(view, offset, 8, "image point count");
      const pointCount = checkedCount(view.getBigUint64(offset, true), "image point count");
      offset += 8;
      const pointBytes = pointCount * 24;
      requireBytes(view, offset, pointBytes, "image points2D");
      offset += pointBytes;
      images.set(imageId, {
        cameraId: cameraId,
        rotation: rotation,
        translation: translation,
        name: name,
        pointDataOffset: offset - pointBytes,
        pointCount: pointCount
      });
    }
    if (offset !== view.byteLength) throw new Error(`Unexpected trailing images.bin bytes: ${view.byteLength - offset}`);
    return images;
  }

  function numericTimestampToken(name) {
    const basename = name.replaceAll("\\", "/").split("/").pop();
    const stem = basename.includes(".") ? basename.slice(0, basename.lastIndexOf(".")) : basename;
    const match = stem.match(/^(\d+(?:\.\d+)?)(?:[-_]|$)/);
    return match === null ? null : match[1];
  }

  function inferredTimestampScale(tokens, values) {
    if (tokens.some(token => token.includes("."))) return 1;
    const conventionalScales = new Map([[10, 1], [13, 1e3], [16, 1e6], [19, 1e9]]);
    if (values.length === 1) return conventionalScales.get(tokens[0].length) || null;
    const deltas = values.slice(1).map((value, index) => value - values[index]).sort((a, b) => a - b);
    const median = deltas[Math.floor(deltas.length / 2)];
    const sameWidth = tokens.every(token => token.length === tokens[0].length);
    const conventional = sameWidth ? conventionalScales.get(tokens[0].length) : undefined;
    if (conventional !== undefined) {
      const interval = median / conventional;
      if (interval >= 0.001 && interval <= 10) return conventional;
    }
    const candidates = [1, 1e3, 1e6, 1e9].filter(scale => {
      const interval = median / scale;
      return interval >= 0.001 && interval <= 10;
    });
    return candidates.length === 1 ? candidates[0] : null;
  }

  function imageTimeline(images) {
    const entries = Array.from(images.entries()).map(([imageId, image]) => ({
      imageId: imageId,
      name: image.name,
      token: numericTimestampToken(image.name)
    }));
    const hasNumericNames = entries.every(entry => entry.token !== null);
    if (hasNumericNames) {
      entries.sort((left, right) => {
        const difference = Number(left.token) - Number(right.token);
        return difference !== 0 ? difference : left.name.localeCompare(right.name);
      });
    } else {
      entries.sort((left, right) => left.imageId - right.imageId);
    }
    let timestampsSeconds = null;
    if (hasNumericNames && entries.length > 0) {
      const tokens = entries.map(entry => entry.token);
      const values = tokens.map(Number);
      const unique = values.every(
        (value, index) => Number.isFinite(value) && (index === 0 || value > values[index - 1])
      );
      const scale = unique ? inferredTimestampScale(tokens, values) : null;
      if (scale !== null) timestampsSeconds = values.map(value => (value - values[0]) / scale);
    }
    if (timestampsSeconds === null) entries.sort((left, right) => left.imageId - right.imageId);
    return {
      imageIds: entries.map(entry => entry.imageId),
      names: entries.map(entry => entry.name),
      timestampsSeconds: timestampsSeconds
    };
  }

  function cameraCenter(image) {
    const r = image.rotation;
    const t = image.translation;
    return [
      -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
      -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
      -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2])
    ];
  }

  function cameraGeometry(camerasBuffer, imagesBuffer, images, imageIds) {
    const cameras = parseCameras(camerasBuffer);
    const centers = new Float32Array(imageIds.length * 3);
    const frusta = new Float32Array(imageIds.length * 16 * 3);
    const observationOffsets = new Uint32Array(imageIds.length + 1);
    let observationCount = 0;
    for (let index = 0; index < imageIds.length; ++index) {
      const image = images.get(imageIds[index]);
      if (image === undefined) throw new Error(`images.bin is missing registered image ${imageIds[index]}`);
      observationCount += image.pointCount;
      observationOffsets[index + 1] = observationCount;
      const camera = cameras.get(image.cameraId);
      if (camera === undefined) throw new Error(`cameras.bin is missing camera ${image.cameraId}`);
      const center = cameraCenter(image);
      centers.set(center, index * 3);
      const imageExtent = Math.max(0.3 * camera.width / 1024, 0.3 * camera.height / 1024);
      const worldExtent = 2 * Math.max(camera.width, camera.height) / (camera.fx + camera.fy);
      const scale = 0.5 * imageExtent / worldExtent;
      const pixelCorners = [[0, 0], [camera.width, 0], [camera.width, camera.height], [0, camera.height]];
      const corners = pixelCorners.map(([u, v]) => {
        const ray = [(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, 1];
        const r = image.rotation;
        const world = [
          center[0] + 0.5 * scale * (r[0] * ray[0] + r[3] * ray[1] + r[6] * ray[2]),
          center[1] + 0.5 * scale * (r[1] * ray[0] + r[4] * ray[1] + r[7] * ray[2]),
          center[2] + 0.5 * scale * (r[2] * ray[0] + r[5] * ray[1] + r[8] * ray[2])
        ];
        return Float32Array.from(world);
      });
      let vertex = index * 16;
      for (const corner of corners) {
        frusta.set(centers.subarray(index * 3, index * 3 + 3), vertex++ * 3);
        frusta.set(corner, vertex++ * 3);
      }
      for (let corner = 0; corner < 4; ++corner) {
        frusta.set(corners[corner], vertex++ * 3);
        frusta.set(corners[(corner + 1) % 4], vertex++ * 3);
      }
    }
    const observationPositions = new Float32Array(observationCount * 2);
    const observationPointIds = new BigUint64Array(observationCount);
    const imageView = new DataView(imagesBuffer);
    let observationOffset = 0;
    for (const imageId of imageIds) {
      const image = images.get(imageId);
      for (let point = 0; point < image.pointCount; ++point) {
        const source = image.pointDataOffset + point * 24;
        const destination = observationOffset + point;
        observationPositions[destination * 2] = imageView.getFloat64(source, true);
        observationPositions[destination * 2 + 1] = imageView.getFloat64(source + 8, true);
        observationPointIds[destination] = imageView.getBigUint64(source + 16, true);
      }
      observationOffset += image.pointCount;
    }
    return {
      centers: centers,
      frusta: frusta,
      observationPositions: observationPositions,
      observationPointIds: observationPointIds,
      observationOffsets: observationOffsets
    };
  }

  function parsePointsForDisplay(buffer, keyframeImageIds) {
    const view = new DataView(buffer);
    requireBytes(view, 0, 8, "points3D.bin header");
    const count = checkedCount(view.getBigUint64(0, true), "3D point count");
    if (count > 0xffffffff) throw new Error("Browser viewer supports at most 2^32-1 points");
    const pointIds = new BigUint64Array(count);
    const firstSeen = new Int32Array(count);
    const trackLengths = new Uint32Array(count);
    firstSeen.fill(-1);
    const imageOrder = new Map();
    keyframeImageIds.forEach((imageId, index) => imageOrder.set(Number(imageId), index));
    let offset = 8;
    for (let index = 0; index < count; ++index) {
      requireBytes(view, offset, 51, "3D point record");
      pointIds[index] = view.getBigUint64(offset, true);
      const trackLength = checkedCount(view.getBigUint64(offset + 43, true), "point track length");
      trackLengths[index] = trackLength;
      offset += 51;
      requireBytes(view, offset, trackLength * 8, "point track");
      let earliest = keyframeImageIds.length;
      for (let observation = 0; observation < trackLength; ++observation) {
        const rank = imageOrder.get(view.getUint32(offset + observation * 8, true));
        if (rank === undefined) continue;
        if (rank < earliest) earliest = rank;
      }
      if (earliest < keyframeImageIds.length) firstSeen[index] = earliest;
      offset += trackLength * 8;
    }
    if (offset !== view.byteLength) throw new Error(`Unexpected trailing points3D.bin bytes: ${view.byteLength - offset}`);

    const orderStorage = new Uint32Array(count);
    let orderCount = 0;
    for (let index = 0; index < count; ++index) {
      if (firstSeen[index] >= 0) orderStorage[orderCount++] = index;
    }
    const order = orderStorage.subarray(0, orderCount);
    order.sort((left, right) => {
      const byFrame = firstSeen[left] - firstSeen[right];
      if (byFrame !== 0) return byFrame;
      if (pointIds[left] < pointIds[right]) return -1;
      if (pointIds[left] > pointIds[right]) return 1;
      return left - right;
    });

    const orderedPositions = new Float32Array(order.length * 3);
    const orderedColors = new Uint8Array(order.length * 3);
    const orderedPointIds = new BigUint64Array(order.length);
    const sourceIndices = new Uint32Array(order.length);
    const orderedFirstSeen = new Int32Array(order.length);
    const orderedTrackLengths = new Uint32Array(order.length);
    const pointOffsets = new Uint32Array(keyframeImageIds.length + 1);
    const destinationBySource = new Uint32Array(count);
    destinationBySource.fill(0xffffffff);
    for (let destination = 0; destination < order.length; ++destination) {
      const source = order[destination];
      destinationBySource[source] = destination;
      orderedPointIds[destination] = pointIds[source];
      sourceIndices[destination] = source;
      orderedFirstSeen[destination] = firstSeen[source];
      orderedTrackLengths[destination] = trackLengths[source];
      pointOffsets[firstSeen[source] + 1] += 1;
    }
    for (let index = 1; index < pointOffsets.length; ++index) pointOffsets[index] += pointOffsets[index - 1];
    offset = 8;
    for (let source = 0; source < count; ++source) {
      const destination = destinationBySource[source];
      const trackLength = trackLengths[source];
      if (destination !== 0xffffffff) {
        orderedPositions[destination * 3] = view.getFloat64(offset + 8, true);
        orderedPositions[destination * 3 + 1] = view.getFloat64(offset + 16, true);
        orderedPositions[destination * 3 + 2] = view.getFloat64(offset + 24, true);
        orderedColors[destination * 3] = view.getUint8(offset + 32);
        orderedColors[destination * 3 + 1] = view.getUint8(offset + 33);
        orderedColors[destination * 3 + 2] = view.getUint8(offset + 34);
      }
      offset += 51 + trackLength * 8;
    }
    return {
      positions: orderedPositions,
      colors: orderedColors,
      pointIds: orderedPointIds,
      sourceIndices: sourceIndices,
      firstSeen: orderedFirstSeen,
      trackLengths: orderedTrackLengths,
      pointOffsets: pointOffsets,
      sourcePointCount: count
    };
  }

  function inverseTrace(h00, h01, h02, h11, h12, h22) {
    const offDiagonalSquare = h01 * h01 + h02 * h02 + h12 * h12;
    let smallest;
    let largest;
    if (offDiagonalSquare === 0) {
      smallest = Math.min(h00, h11, h22);
      largest = Math.max(h00, h11, h22);
    } else {
      const mean = (h00 + h11 + h22) / 3;
      const scale = Math.sqrt(
        ((h00 - mean) ** 2 + (h11 - mean) ** 2 + (h22 - mean) ** 2 + 2 * offDiagonalSquare) / 6
      );
      const b00 = (h00 - mean) / scale;
      const b01 = h01 / scale;
      const b02 = h02 / scale;
      const b11 = (h11 - mean) / scale;
      const b12 = h12 / scale;
      const b22 = (h22 - mean) / scale;
      const halfDeterminant = (
        b00 * (b11 * b22 - b12 * b12) -
        b01 * (b01 * b22 - b02 * b12) +
        b02 * (b01 * b12 - b02 * b11)
      ) / 2;
      const angle = Math.acos(Math.max(-1, Math.min(1, halfDeterminant))) / 3;
      largest = mean + 2 * scale * Math.cos(angle);
      smallest = mean + 2 * scale * Math.cos(angle + 2 * Math.PI / 3);
    }
    if (!(smallest > largest * Number.EPSILON * 3)) return Infinity;
    const c00 = h11 * h22 - h12 * h12;
    const c11 = h00 * h22 - h02 * h02;
    const c22 = h00 * h11 - h01 * h01;
    const determinant = h00 * c00 - h01 * (h01 * h22 - h02 * h12) + h02 * (h01 * h12 - h02 * h11);
    return (c00 + c11 + c22) / determinant;
  }

  function computeCovariances(camerasBuffer, imagesBuffer, pointsBuffer, progress) {
    const cameras = parseCameras(camerasBuffer);
    const images = parseImages(imagesBuffer);
    const view = new DataView(pointsBuffer);
    requireBytes(view, 0, 8, "points3D.bin header");
    const count = checkedCount(view.getBigUint64(0, true), "3D point count");
    const scores = new Float64Array(count);
    let offset = 8;
    for (let index = 0; index < count; ++index) {
      requireBytes(view, offset, 51, "3D point record");
      const px = view.getFloat64(offset + 8, true);
      const py = view.getFloat64(offset + 16, true);
      const pz = view.getFloat64(offset + 24, true);
      const trackLength = checkedCount(view.getBigUint64(offset + 43, true), "point track length");
      offset += 51;
      requireBytes(view, offset, trackLength * 8, "point track");
      if (trackLength < 2) {
        scores[index] = NaN;
        offset += trackLength * 8;
        continue;
      }
      let h00 = 0;
      let h01 = 0;
      let h02 = 0;
      let h11 = 0;
      let h12 = 0;
      let h22 = 0;
      for (let observation = 0; observation < trackLength; ++observation) {
        const image = images.get(view.getUint32(offset + observation * 8, true));
        if (image === undefined) continue;
        const camera = cameras.get(image.cameraId);
        if (camera === undefined) throw new Error(`Image references missing camera ${image.cameraId}`);
        const r = image.rotation;
        const t = image.translation;
        const x = r[0] * px + r[1] * py + r[2] * pz + t[0];
        const y = r[3] * px + r[4] * py + r[5] * pz + t[1];
        const z = r[6] * px + r[7] * py + r[8] * pz + t[2];
        if (!(z > 0)) continue;
        const ax = camera.fx / z;
        const az = -camera.fx * x / (z * z);
        const by = camera.fy / z;
        const bz = -camera.fy * y / (z * z);
        const gx0 = ax * r[0] + az * r[6];
        const gx1 = ax * r[1] + az * r[7];
        const gx2 = ax * r[2] + az * r[8];
        const gy0 = by * r[3] + bz * r[6];
        const gy1 = by * r[4] + bz * r[7];
        const gy2 = by * r[5] + bz * r[8];
        h00 += gx0 * gx0 + gy0 * gy0;
        h01 += gx0 * gx1 + gy0 * gy1;
        h02 += gx0 * gx2 + gy0 * gy2;
        h11 += gx1 * gx1 + gy1 * gy1;
        h12 += gx1 * gx2 + gy1 * gy2;
        h22 += gx2 * gx2 + gy2 * gy2;
      }
      scores[index] = inverseTrace(h00, h01, h02, h11, h12, h22);
      offset += trackLength * 8;
      if (progress !== undefined && (index % 10000 === 0 || index + 1 === count)) progress(index + 1, count);
    }
    if (offset !== view.byteLength) throw new Error(`Unexpected trailing points3D.bin bytes: ${view.byteLength - offset}`);
    return scores;
  }

  function covarianceRanks(scores, sourceIndices, pointIds) {
    if (sourceIndices.length !== pointIds.length) throw new Error("Point order arrays have inconsistent lengths");
    const eligible = new Uint32Array(sourceIndices.length);
    let eligibleCount = 0;
    for (let index = 0; index < sourceIndices.length; ++index) {
      if (!Number.isNaN(scores[sourceIndices[index]])) eligible[eligibleCount++] = index;
    }
    const ordered = eligible.subarray(0, eligibleCount);
    ordered.sort((left, right) => {
      const leftScore = scores[sourceIndices[left]];
      const rightScore = scores[sourceIndices[right]];
      if (leftScore < rightScore) return -1;
      if (leftScore > rightScore) return 1;
      if (pointIds[left] < pointIds[right]) return -1;
      if (pointIds[left] > pointIds[right]) return 1;
      return left - right;
    });
    const ranks = new Int32Array(sourceIndices.length);
    ranks.fill(-1);
    for (let rank = 0; rank < eligibleCount; ++rank) ranks[ordered[rank]] = rank;
    return {ranks: ranks, eligibleCount: eligibleCount};
  }

  const api = {
    crc32: crc32,
    fingerprints: fingerprints,
    parseCameras: parseCameras,
    parseImages: parseImages,
    imageTimeline: imageTimeline,
    cameraGeometry: cameraGeometry,
    parsePointsForDisplay: parsePointsForDisplay,
    computeCovariances: computeCovariances,
    covarianceRanks: covarianceRanks
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;

  if (root !== undefined && typeof root.postMessage === "function" && typeof document === "undefined") {
    root.onmessage = event => {
      const message = event.data;
      try {
        if (message.command === "load") {
          const buffers = message.buffers;
          const images = parseImages(buffers["images.bin"]);
          const timeline = imageTimeline(images);
          const keyframeImageIds = timeline.imageIds;
          const points = parsePointsForDisplay(
            buffers["points3D.bin"],
            keyframeImageIds
          );
          const cameras = cameraGeometry(
            buffers["cameras.bin"],
            buffers["images.bin"],
            images,
            keyframeImageIds
          );
          const sourceFingerprints = fingerprints(buffers);
          root.postMessage(
            {type: "loaded", points: points, cameras: cameras, fingerprints: sourceFingerprints, timeline: timeline},
            [
              points.positions.buffer,
              points.colors.buffer,
              points.pointIds.buffer,
              points.sourceIndices.buffer,
              points.firstSeen.buffer,
              points.trackLengths.buffer,
              points.pointOffsets.buffer,
              cameras.centers.buffer,
              cameras.frusta.buffer,
              cameras.observationPositions.buffer,
              cameras.observationPointIds.buffer,
              cameras.observationOffsets.buffer
            ]
          );
        } else if (message.command === "covariance") {
          const buffers = message.buffers;
          const scores = computeCovariances(
            buffers["cameras.bin"],
            buffers["images.bin"],
            buffers["points3D.bin"],
            (done, total) => root.postMessage({type: "progress", done: done, total: total})
          );
          root.postMessage({type: "covariance", scores: scores}, [scores.buffer]);
        } else if (message.command === "rank") {
          const ranked = covarianceRanks(message.scores, message.sourceIndices, message.pointIds);
          root.postMessage(
            {type: "ranked", ...ranked, scores: message.scores},
            [ranked.ranks.buffer, message.scores.buffer]
          );
        } else {
          throw new Error(`Unknown worker command: ${message.command}`);
        }
      } catch (error) {
        root.postMessage({type: "error", message: error instanceof Error ? error.message : String(error)});
      }
    };
  }
})(typeof self === "undefined" ? undefined : self);
