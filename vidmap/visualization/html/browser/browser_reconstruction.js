    const COVARIANCE_CACHE_MAGIC = "VIDCOV2\0";
    const COVARIANCE_CACHE_HEADER_BYTES = 64;
    const COLMAP_FILENAMES = ["cameras.bin", "images.bin", "points3D.bin"];
    const LOOP_CLOSURE_MASKS = "lc_masks.json";
    const MAXIMUM_TRACK_LENGTH_FILTER = 20;
    const CACHE_DIRECTORY = "visualization_cache";
    const CACHE_FILENAME = "point_covariance_rank_v2.bin";
    const LOCAL_INPUT_MANIFEST = "local_input.json";
    const DATABASE_PATH = "mapper_inputs/database_complete.db";
    const LOOP_CLOSURE_MASKS_PATH = `mapper_inputs/${LOOP_CLOSURE_MASKS}`;

    function errorMessage(error) {
      return error instanceof Error ? error.message : String(error);
    }

    function workerRequest(command, payload, transferables = [], onProgress = null) {
      return new Promise((resolve, reject) => {
        const sourceUrl = URL.createObjectURL(
          new Blob([VIDMAP_BROWSER_WORKER_SOURCE], {type: "text/javascript"})
        );
        const worker = new Worker(sourceUrl);
        URL.revokeObjectURL(sourceUrl);
        worker.onmessage = event => {
          if (event.data.type === "progress") {
            if (onProgress !== null) onProgress(event.data.done, event.data.total);
            return;
          }
          worker.terminate();
          if (event.data.type === "error") reject(new Error(event.data.message));
          else resolve(event.data);
        };
        worker.onerror = event => {
          worker.terminate();
          reject(new Error(event.message || "Covariance worker failed"));
        };
        worker.postMessage({command: command, ...payload}, transferables);
      });
    }

    function matchingFingerprints(actual, expected) {
      if (actual === null || expected === null || typeof actual !== "object" || typeof expected !== "object") {
        return false;
      }
      return COLMAP_FILENAMES.every(name =>
        actual[name] !== undefined && expected[name] !== undefined &&
        actual[name].size === expected[name].size && actual[name].crc32 === expected[name].crc32
      );
    }

    async function fileFromHandle(root, path) {
      const parts = path.split("/");
      let directory = root;
      for (const part of parts.slice(0, -1)) directory = await directory.getDirectoryHandle(part);
      return (await directory.getFileHandle(parts.at(-1))).getFile();
    }

    async function optionalFile(getFile, path) {
      try {
        return await getFile(path);
      } catch (error) {
        if (error !== null && typeof error === "object" && error.name === "NotFoundError") return null;
        throw error;
      }
    }

    async function pickDirectory() {
      if (typeof window.showDirectoryPicker !== "function") {
        throw new Error("Local folder loading requires a Chromium-based browser");
      }
      return window.showDirectoryPicker({mode: "read"});
    }

    async function selectionFrom(getFile, options = {}) {
      const entries = await Promise.all(
        COLMAP_FILENAMES.map(async name => [name, await getFile(`rec/${name}`)])
      );
      const hasDatabase = Object.prototype.hasOwnProperty.call(options, "database");
      const hasMasks = Object.prototype.hasOwnProperty.call(options, "loopClosureMasks");
      const database = hasDatabase ? options.database : await optionalFile(getFile, DATABASE_PATH);
      const loopClosureMasks = hasMasks
        ? options.loopClosureMasks
        : await optionalFile(getFile, LOOP_CLOSURE_MASKS_PATH);
      if ((database === null) !== (loopClosureMasks === null)) {
        throw new Error(
          `Loop-closure inspection requires both ${DATABASE_PATH} and ${LOOP_CLOSURE_MASKS_PATH}`
        );
      }
      return {
        files: Object.fromEntries(entries),
        reconstructionHandle: options.reconstructionHandle ?? null,
        imagePreviews: options.imagePreviews ?? null,
        localInput: options.localInput ?? null,
        covarianceCache: options.covarianceCache ?? null,
        database: database,
        loopClosureMasks: loopClosureMasks
      };
    }

    async function directorySelection(runHandle) {
      const reconstructionHandle = await runHandle.getDirectoryHandle("rec");
      const getFile = path => fileFromHandle(runHandle, path);
      return selectionFrom(getFile, {
        reconstructionHandle,
        localInput: await optionalFile(getFile, `mapper_inputs/${LOCAL_INPUT_MANIFEST}`)
      });
    }

    async function separateDirectorySelection(reconstructionHandle, mapperInputsHandle) {
      const getFile = path => fileFromHandle(
        path.startsWith("rec/") ? reconstructionHandle : mapperInputsHandle,
        path.split("/")[1]
      );
      return selectionFrom(
        getFile,
        {
          reconstructionHandle,
          localInput: await optionalFile(
            path => fileFromHandle(mapperInputsHandle, path),
            LOCAL_INPUT_MANIFEST
          )
        }
      );
    }

    async function decodedEmbeddedFile(entry, path) {
      if (
        entry === null || typeof entry !== "object" || entry.encoding !== "gzip-base64" ||
        !Number.isSafeInteger(entry.size) || entry.size < 0 || typeof entry.base64 !== "string"
      ) {
        throw new Error(`Embedded run contains an invalid file entry: ${path}`);
      }
      if (typeof DecompressionStream !== "function") {
        throw new Error("This browser does not support embedded gzip decompression");
      }
      const encoded = atob(entry.base64);
      const compressed = new Uint8Array(encoded.length);
      for (let index = 0; index < encoded.length; ++index) compressed[index] = encoded.charCodeAt(index);
      const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
      const data = await new Response(stream).arrayBuffer();
      if (data.byteLength !== entry.size) throw new Error(`Embedded file has the wrong size: ${path}`);
      return new File([data], path.split("/").pop());
    }

    async function embeddedSelection(payload) {
      if (
        payload === null || typeof payload !== "object" ||
        payload.files === null || typeof payload.files !== "object"
      ) {
        throw new Error("Embedded run has an invalid schema");
      }
      const covariancePath = "rec/visualization_cache/point_covariance_rank_v2.bin";
      const covarianceCache = payload.files[covariancePath] === undefined
        ? null
        : await decodedEmbeddedFile(payload.files[covariancePath], covariancePath);
      const databaseEntry = payload.files[DATABASE_PATH];
      const masksEntry = payload.files[LOOP_CLOSURE_MASKS_PATH];
      if ((databaseEntry === undefined) !== (masksEntry === undefined)) {
        throw new Error("Embedded run contains incomplete loop-closure data");
      }
      return selectionFrom(path => decodedEmbeddedFile(payload.files[path], path), {
        imagePreviews: payload.imagePreviews,
        covarianceCache,
        database: databaseEntry === undefined ? null : await decodedEmbeddedFile(databaseEntry, DATABASE_PATH),
        loopClosureMasks: masksEntry === undefined
          ? null
          : await decodedEmbeddedFile(masksEntry, LOOP_CLOSURE_MASKS_PATH)
      });
    }

    async function sourceBuffers(files) {
      const entries = await Promise.all(
        COLMAP_FILENAMES.map(async name => [name, await files[name].arrayBuffer()])
      );
      return Object.fromEntries(entries);
    }

    function parseLoopClosureMasks(text) {
      const payload = JSON.parse(text);
      if (
        payload === null || typeof payload !== "object" || Array.isArray(payload) ||
        Object.keys(payload).sort().join(",") !== "pairs,schemaVersion" ||
        payload.schemaVersion !== 1 || !Array.isArray(payload.pairs)
      ) {
        throw new Error("Loop-closure masks have an invalid schema");
      }
      const pairs = new Map();
      for (const pair of payload.pairs) {
        if (
          pair === null || typeof pair !== "object" || Array.isArray(pair) ||
          Object.keys(pair).sort().join(",") !==
            "first,loopClosureMatchIndices,matchCount,second" ||
          typeof pair.first !== "string" || pair.first.length === 0 ||
          typeof pair.second !== "string" || pair.second.length === 0 || pair.first === pair.second ||
          !Number.isSafeInteger(pair.matchCount) || pair.matchCount < 0 ||
          !Array.isArray(pair.loopClosureMatchIndices) || pair.loopClosureMatchIndices.some(
            (row, index) => !Number.isSafeInteger(row) || row < 0 || row >= pair.matchCount ||
              (index > 0 && row <= pair.loopClosureMatchIndices[index - 1])
          )
        ) {
          throw new Error("Loop-closure masks contain an invalid image pair");
        }
        const ordered = pair.first < pair.second ? [pair.first, pair.second] : [pair.second, pair.first];
        const key = `${ordered[0]}\0${ordered[1]}`;
        if (pairs.has(key)) throw new Error("Loop-closure masks contain a duplicate image pair");
        pairs.set(key, {
          matchCount: pair.matchCount,
          indices: Uint32Array.from(pair.loopClosureMatchIndices)
        });
      }
      return pairs;
    }

    async function openDatabase(databaseFile) {
      if (typeof initSqlJs !== "function") throw new Error("SQLite browser runtime failed to load");
      const SQL = await initSqlJs({locateFile: () => VIDMAP_SQLITE_WASM_URL});
      return new SQL.Database(new Uint8Array(await databaseFile.arrayBuffer()));
    }

    function databaseEdgeKeyframes(database, keyframeNames, loopClosurePairs) {
      const result = database.exec(`
          SELECT first.name, second.name
          FROM two_view_geometries AS geometry
          JOIN images AS first
            ON first.image_id = CAST(geometry.pair_id / 2147483647 AS INTEGER)
          JOIN images AS second
            ON second.image_id = geometry.pair_id % 2147483647
          WHERE geometry.rows > 0 AND geometry.config IN (2, 3, 4, 5, 6, 9)
          ORDER BY geometry.pair_id
        `);
      if (result.length === 0) return new Uint32Array();
      const keyframeByName = new Map(keyframeNames.map((name, index) => [name, index]));
      const pairs = [];
      for (const [leftName, rightName] of result[0].values) {
        const orderedNames = leftName < rightName ? [leftName, rightName] : [rightName, leftName];
        const loopClosure = loopClosurePairs.get(`${orderedNames[0]}\0${orderedNames[1]}`);
        if (loopClosure === undefined || loopClosure.indices.length === 0) continue;
        const left = keyframeByName.get(leftName);
        const right = keyframeByName.get(rightName);
        if (left === undefined || right === undefined || left === right) continue;
        pairs.push(Math.min(left, right), Math.max(left, right));
      }
      return Uint32Array.from(pairs);
    }

    function databaseMatchRows(database, firstImageId, secondImageId) {
      const statement = database.prepare(`
        SELECT rows, cols, data FROM matches
        WHERE pair_id = MIN(?1, ?2) * 2147483647 + MAX(?1, ?2)
      `);
      try {
        statement.bind([firstImageId, secondImageId]);
        if (!statement.step()) throw new Error("Loop-closure pair has no database matches");
        const row = statement.getAsObject();
        if (row.cols !== 2 || !(row.data instanceof Uint8Array) || row.data.byteLength !== row.rows * 8) {
          throw new Error("Loop-closure database matches have an invalid shape");
        }
        return {count: row.rows, bytes: row.data};
      } finally {
        statement.free();
      }
    }

    function databaseEdgeSharedCounts(databasePairs, observationPointIds, observationOffsets) {
      const invalidPointId = 0xffffffffffffffffn;
      function observedPointIds(keyframe) {
        const ids = new Set();
        for (
          let observation = observationOffsets[keyframe];
          observation < observationOffsets[keyframe + 1];
          ++observation
        ) {
          const pointId = observationPointIds[observation];
          if (pointId !== invalidPointId) ids.add(pointId);
        }
        return ids;
      }
      const result = new Uint32Array(databasePairs.length / 2);
      for (let index = 0; index < result.length; ++index) {
        const leftIds = observedPointIds(databasePairs[index * 2]);
        let shared = 0;
        const right = databasePairs[index * 2 + 1];
        for (let observation = observationOffsets[right]; observation < observationOffsets[right + 1]; ++observation) {
          if (leftIds.delete(observationPointIds[observation])) shared += 1;
        }
        result[index] = shared;
      }
      return result;
    }

    function cacheFingerprint(view, offset) {
      return {
        size: Number(view.getBigUint64(offset, true)),
        crc32: view.getUint32(offset + 8, true)
      };
    }

    function parseCovarianceCache(buffer, expectedFingerprints, expectedPointCount) {
      if (buffer.byteLength < COVARIANCE_CACHE_HEADER_BYTES) throw new Error("Covariance cache is truncated");
      const view = new DataView(buffer);
      const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 8));
      if (magic !== COVARIANCE_CACHE_MAGIC) throw new Error("Covariance cache has an unknown format");
      if (view.getUint32(8, true) !== 2 || view.getUint32(12, true) !== COVARIANCE_CACHE_HEADER_BYTES) {
        throw new Error("Covariance cache version is unsupported");
      }
      const pointCount = Number(view.getBigUint64(16, true));
      if (pointCount !== expectedPointCount) throw new Error("Covariance cache point count is stale");
      const cachedFingerprints = {
        "cameras.bin": cacheFingerprint(view, 24),
        "images.bin": cacheFingerprint(view, 36),
        "points3D.bin": cacheFingerprint(view, 48)
      };
      if (!matchingFingerprints(cachedFingerprints, expectedFingerprints)) {
        throw new Error("Covariance cache was computed from different reconstruction bytes");
      }
      const expectedBytes = COVARIANCE_CACHE_HEADER_BYTES + pointCount * 12;
      if (buffer.byteLength !== expectedBytes) throw new Error("Covariance cache score payload has the wrong size");
      return {
        scores: new Float64Array(buffer, COVARIANCE_CACHE_HEADER_BYTES, pointCount),
        sourceRanks: new Int32Array(buffer, COVARIANCE_CACHE_HEADER_BYTES + pointCount * 8, pointCount)
      };
    }

    function writeCacheFingerprint(view, offset, fingerprint) {
      view.setBigUint64(offset, BigInt(fingerprint.size), true);
      view.setUint32(offset + 8, fingerprint.crc32, true);
    }

    function covarianceCacheBuffer(scores, sourceRanks, sourceFingerprints) {
      if (scores.length !== sourceRanks.length) throw new Error("Covariance scores and ranks must match");
      const buffer = new ArrayBuffer(COVARIANCE_CACHE_HEADER_BYTES + scores.length * 12);
      const bytes = new Uint8Array(buffer);
      for (let index = 0; index < 8; ++index) bytes[index] = COVARIANCE_CACHE_MAGIC.charCodeAt(index);
      const view = new DataView(buffer);
      view.setUint32(8, 2, true);
      view.setUint32(12, COVARIANCE_CACHE_HEADER_BYTES, true);
      view.setBigUint64(16, BigInt(scores.length), true);
      writeCacheFingerprint(view, 24, sourceFingerprints["cameras.bin"]);
      writeCacheFingerprint(view, 36, sourceFingerprints["images.bin"]);
      writeCacheFingerprint(view, 48, sourceFingerprints["points3D.bin"]);
      new Float64Array(buffer, COVARIANCE_CACHE_HEADER_BYTES, scores.length).set(scores);
      new Int32Array(buffer, COVARIANCE_CACHE_HEADER_BYTES + scores.length * 8).set(sourceRanks);
      return buffer;
    }

    async function readOptionalCacheFile(directoryHandle, name) {
      try {
        const cacheDirectory = await directoryHandle.getDirectoryHandle(CACHE_DIRECTORY);
        const file = await (await cacheDirectory.getFileHandle(name)).getFile();
        return await file.arrayBuffer();
      } catch (error) {
        if (error instanceof DOMException && error.name === "NotFoundError") return null;
        throw error;
      }
    }

    async function writeHandleCache(directoryHandle, buffer) {
      const cacheDirectory = await directoryHandle.getDirectoryHandle(CACHE_DIRECTORY, {create: true});
      const cacheHandle = await cacheDirectory.getFileHandle(CACHE_FILENAME, {create: true});
      const writer = await cacheHandle.createWritable();
      try {
        await writer.write(buffer);
        await writer.close();
      } catch (error) {
        await writer.abort();
        throw error;
      }
    }

    function parseLocalInput(text) {
      const manifest = JSON.parse(text);
      if (
        manifest === null || typeof manifest !== "object" || Array.isArray(manifest) ||
        Object.keys(manifest).sort().join(",") !== "image_dir,schema_version" ||
        manifest.schema_version !== 1
      ) {
        throw new Error("Local input manifest has an invalid schema");
      }
      if (typeof manifest.image_dir !== "string" || !manifest.image_dir.startsWith("/")) {
        throw new Error("Local input manifest must contain an absolute image directory");
      }
      return manifest.image_dir;
    }

    function sharedTrackObservations(
      observationPositions,
      observationPointIds,
      observationOffsets,
      leftKeyframe,
      rightKeyframe
    ) {
      const invalidPointId = 0xffffffffffffffffn;
      const leftByPointId = new Map();
      for (
        let observation = observationOffsets[leftKeyframe];
        observation < observationOffsets[leftKeyframe + 1];
        ++observation
      ) {
        const pointId = observationPointIds[observation];
        if (pointId !== invalidPointId && !leftByPointId.has(pointId)) {
          leftByPointId.set(pointId, observation);
        }
      }
      const leftKeypoints = [];
      const rightKeypoints = [];
      const pointIds = [];
      for (
        let rightObservation = observationOffsets[rightKeyframe];
        rightObservation < observationOffsets[rightKeyframe + 1];
        ++rightObservation
      ) {
        const pointId = observationPointIds[rightObservation];
        const leftObservation = leftByPointId.get(pointId);
        if (pointId === invalidPointId || leftObservation === undefined) continue;
        leftKeypoints.push(
          observationPositions[leftObservation * 2],
          observationPositions[leftObservation * 2 + 1]
        );
        rightKeypoints.push(
          observationPositions[rightObservation * 2],
          observationPositions[rightObservation * 2 + 1]
        );
        pointIds.push(pointId);
      }
      return {
        leftKeypoints: Float32Array.from(leftKeypoints),
        rightKeypoints: Float32Array.from(rightKeypoints),
        pointIds: pointIds
      };
    }

    function loopClosureMatchObservations(
      database,
      matchRows,
      expectedMatchCount,
      observationPositions,
      observationPointIds,
      observationOffsets,
      leftKeyframe,
      rightKeyframe,
      leftImageId,
      rightImageId
    ) {
      const matches = databaseMatchRows(database, leftImageId, rightImageId);
      if (matches.count !== expectedMatchCount) {
        throw new Error("Loop-closure match count does not match the database");
      }
      const view = new DataView(matches.bytes.buffer, matches.bytes.byteOffset, matches.bytes.byteLength);
      const leftIsFirst = leftImageId < rightImageId;
      const leftCount = observationOffsets[leftKeyframe + 1] - observationOffsets[leftKeyframe];
      const rightCount = observationOffsets[rightKeyframe + 1] - observationOffsets[rightKeyframe];
      const leftKeypoints = new Float32Array(matches.count * 2);
      const rightKeypoints = new Float32Array(matches.count * 2);
      const matchClasses = new Uint8Array(matches.count);
      const invalidPointId = 0xffffffffffffffffn;
      let loopClosureMatchIndex = 0;
      let sharedLoopClosureCount = 0;
      for (let row = 0; row < matches.count; ++row) {
        const isLoopClosure =
          loopClosureMatchIndex < matchRows.length && matchRows[loopClosureMatchIndex] === row;
        if (isLoopClosure) loopClosureMatchIndex += 1;
        const first = view.getUint32(row * 8, true);
        const second = view.getUint32(row * 8 + 4, true);
        const leftPoint = leftIsFirst ? first : second;
        const rightPoint = leftIsFirst ? second : first;
        if (leftPoint >= leftCount || rightPoint >= rightCount) {
          throw new Error("Loop-closure match references a missing reconstruction keypoint");
        }
        const leftObservation = observationOffsets[leftKeyframe] + leftPoint;
        const rightObservation = observationOffsets[rightKeyframe] + rightPoint;
        leftKeypoints[row * 2] = observationPositions[leftObservation * 2];
        leftKeypoints[row * 2 + 1] = observationPositions[leftObservation * 2 + 1];
        rightKeypoints[row * 2] = observationPositions[rightObservation * 2];
        rightKeypoints[row * 2 + 1] = observationPositions[rightObservation * 2 + 1];
        const leftPointId = observationPointIds[leftObservation];
        const rightPointId = observationPointIds[rightObservation];
        const sharesPoint = leftPointId !== invalidPointId && leftPointId === rightPointId;
        matchClasses[row] = isLoopClosure ? (sharesPoint ? 2 : 1) : 0;
        if (isLoopClosure && sharesPoint) sharedLoopClosureCount += 1;
      }
      if (loopClosureMatchIndex !== matchRows.length) {
        throw new Error("Loop-closure match index exceeds database matches");
      }
      return {
        leftKeypoints: leftKeypoints,
        rightKeypoints: rightKeypoints,
        matchClasses: matchClasses,
        totalMatchCount: matches.count,
        sharedLoopClosureCount: sharedLoopClosureCount
      };
    }

    function initializeReconstructionViewer(embeddedRun) {
      const chooseButton = document.getElementById("run-folder-button");
      const reconstructionButton = document.getElementById("reconstruction-folder-button");
      const reconstructionLabel = document.getElementById("reconstruction-folder-label");
      const mapperInputsButton = document.getElementById("mapper-inputs-folder-button");
      const mapperInputsLabel = document.getElementById("mapper-inputs-folder-label");
      const loadSeparateButton = document.getElementById("separate-load-button");
      const status = document.getElementById("load-status");
      const computeButton = document.getElementById("covariance-compute");
      const percentile = document.getElementById("covariance-percentile");
      const percentileLabel = document.getElementById("covariance-percentile-label");
      const minimumTrackLength = document.getElementById("minimum-track-length");
      const minimumTrackLengthLabel = document.getElementById("minimum-track-length-label");
      let state = null;
      let separateReconstruction = null;
      let separateMapperInputs = null;

      function setStatus(message) {
        status.textContent = message;
      }

      function reportPickerError(error) {
        if (!(error instanceof DOMException && error.name === "AbortError")) reportError(errorMessage(error));
      }

      function applyPercentile(value) {
        if (state === null || state.ranks === null) return;
        if (value === 100) {
          setCovarianceRankLimit(-1);
          percentileLabel.textContent = "all points";
        } else {
          const keep = Math.floor(state.eligibleCount * value / 100);
          setCovarianceRankLimit(keep);
          percentileLabel.textContent = `${value}% · ${keep.toLocaleString()} points`;
        }
      }

      function sourceRanksFromDisplay(displayRanks) {
        const sourceRanks = new Int32Array(state.sourcePointCount);
        sourceRanks.fill(-1);
        for (let index = 0; index < displayRanks.length; ++index) {
          sourceRanks[state.sourceIndices[index]] = displayRanks[index];
        }
        return sourceRanks;
      }

      function displayRanksFromSource(sourceRanks) {
        if (sourceRanks.length !== state.sourcePointCount) throw new Error("Cached rank count is stale");
        const sourceRankToDisplay = new Int32Array(state.sourcePointCount);
        sourceRankToDisplay.fill(-1);
        for (let displayIndex = 0; displayIndex < state.sourceIndices.length; ++displayIndex) {
          const sourceRank = sourceRanks[state.sourceIndices[displayIndex]];
          if (sourceRank < 0) continue;
          if (sourceRank >= sourceRankToDisplay.length || sourceRankToDisplay[sourceRank] >= 0) {
            throw new Error("Cached covariance ranks are invalid");
          }
          sourceRankToDisplay[sourceRank] = displayIndex;
        }
        const displayRanks = new Int32Array(state.sourceIndices.length);
        displayRanks.fill(-1);
        let eligibleCount = 0;
        for (const displayIndex of sourceRankToDisplay) {
          if (displayIndex >= 0) displayRanks[displayIndex] = eligibleCount++;
        }
        return {ranks: displayRanks, eligibleCount: eligibleCount};
      }

      function installRanks(ranked, source) {
        state.ranks = ranked.ranks;
        state.eligibleCount = ranked.eligibleCount;
        installCovarianceRanks(state.ranks, state.firstSeen);
        percentile.disabled = false;
        percentile.value = "100";
        applyPercentile(100);
        showKeyframe(Number(keyframeSlider.value));
        setStatus(`${source}; ${ranked.eligibleCount.toLocaleString()} covariance-ranked points`);
        computeButton.textContent = "recompute covariance";
      }

      async function rankScores(scores) {
        setStatus("covariance computed; ranking points…");
        return workerRequest(
          "rank",
          {
            scores: scores,
            sourceIndices: state.sourceIndices,
            pointIds: state.pointIds
          },
          [scores.buffer]
        );
      }

      function loopClosureData(leftKeyframe, rightKeyframe) {
        const shared = sharedTrackObservations(
          state.observationPositions,
          state.observationPointIds,
          state.observationOffsets,
          leftKeyframe,
          rightKeyframe
        );
        const sharedPointIds = new Set(shared.pointIds);
        const sharedPointPositions = [];
        for (let pointIndex = 0; pointIndex < state.pointIds.length; ++pointIndex) {
          if (!sharedPointIds.has(state.pointIds[pointIndex])) continue;
          sharedPointPositions.push(
            state.positions[pointIndex * 3],
            state.positions[pointIndex * 3 + 1],
            state.positions[pointIndex * 3 + 2]
          );
        }
        return {
          sharedPointCount: sharedPointPositions.length / 3,
          sharedPointPositions: Float32Array.from(sharedPointPositions)
        };
      }

      async function selectLoopClosure(leftKeyframe, rightKeyframe) {
        const data = loopClosureData(leftKeyframe, rightKeyframe);
        const leftName = keyframeTimeline.names[leftKeyframe];
        const rightName = keyframeTimeline.names[rightKeyframe];
        const orderedNames = leftName < rightName ? [leftName, rightName] : [rightName, leftName];
        const loopClosure = state.loopClosurePairs.get(`${orderedNames[0]}\0${orderedNames[1]}`);
        if (loopClosure === undefined) throw new Error("Selected edge is missing its loop-closure mask");
        const matches = loopClosureMatchObservations(
          state.database,
          loopClosure.indices,
          loopClosure.matchCount,
          state.observationPositions,
          state.observationPointIds,
          state.observationOffsets,
          leftKeyframe,
          rightKeyframe,
          keyframeTimeline.imageIds[leftKeyframe],
          keyframeTimeline.imageIds[rightKeyframe]
        );
        showLoopClosurePair({
          leftName: leftName,
          rightName: rightName,
          leftKeypoints: matches.leftKeypoints,
          rightKeypoints: matches.rightKeypoints,
          matchClasses: matches.matchClasses,
          totalMatchCount: matches.totalMatchCount,
          matchCount: loopClosure.indices.length,
          sharedLoopClosureCount: matches.sharedLoopClosureCount,
          sharedPointCount: data.sharedPointCount,
          sharedPointPositions: data.sharedPointPositions
        });
      }

      installLoopClosureSelectionHandler(selectLoopClosure);

      async function loadSelected(selection) {
        if (state !== null && state.database !== null) state.database.close();
        state = null;
        resetRunVisualState();
        computeButton.textContent = "compute covariance";
        percentile.disabled = true;
        percentile.value = "100";
        percentileLabel.textContent = "all points";
        minimumTrackLength.disabled = true;
        minimumTrackLength.max = "1";
        minimumTrackLength.value = "1";
        minimumTrackLengthLabel.textContent = "1 observation";
        setStatus("reading COLMAP model…");
        try {
          const files = selection.files;
          const buffers = await sourceBuffers(files);
          const transferables = COLMAP_FILENAMES.map(name => buffers[name]);
          const loaded = await workerRequest(
            "load",
            {buffers: buffers},
            transferables
          );
          installKeyframeTimeline(loaded.timeline);
          state = {
            files: files,
            directoryHandle: selection.reconstructionHandle,
            sourceFingerprints: loaded.fingerprints,
            sourcePointCount: loaded.points.sourcePointCount,
            positions: loaded.points.positions,
            pointIds: loaded.points.pointIds,
            sourceIndices: loaded.points.sourceIndices,
            firstSeen: loaded.points.firstSeen,
            observationPositions: loaded.cameras.observationPositions,
            observationPointIds: loaded.cameras.observationPointIds,
            observationOffsets: loaded.cameras.observationOffsets,
            database: null,
            loopClosurePairs: null,
            ranks: null,
            eligibleCount: 0
          };
          replacePointGeometry(state.positions, loaded.points.colors);
          installTrackLengths(loaded.points.trackLengths);
          let maximumTrackLength = 1;
          for (const length of loaded.points.trackLengths) maximumTrackLength = Math.max(maximumTrackLength, length);
          minimumTrackLength.max = String(Math.min(MAXIMUM_TRACK_LENGTH_FILTER, maximumTrackLength));
          minimumTrackLength.value = "1";
          minimumTrackLength.disabled = false;
          minimumTrackLengthLabel.textContent = "1 observation";
          setMinimumTrackLength(1);
          keyframeTimeline.pointOffsets = Array.from(loaded.points.pointOffsets);
          replaceEstimatedGeometry(loaded.cameras.centers, loaded.cameras.frusta);
          let loopClosureStatus;
          if (selection.database !== null) {
            setStatus("model loaded; validating loop-closure masks…");
            const loopClosurePairs = parseLoopClosureMasks(await selection.loopClosureMasks.text());
            setStatus("loop-closure masks validated; reading verified database edges…");
            state.database = await openDatabase(selection.database);
            state.loopClosurePairs = loopClosurePairs;
            const databasePairs = databaseEdgeKeyframes(
              state.database,
              keyframeTimeline.names,
              loopClosurePairs
            );
            const databaseEdgeCount = databasePairs.length / 2;
            loopClosureStatus = `${databaseEdgeCount.toLocaleString()} loop-closure edges loaded`;
            replaceLoopClosureGeometry(
              databasePairs,
              databaseEdgeSharedCounts(
                databasePairs,
                state.observationPointIds,
                state.observationOffsets
              )
            );
          } else {
            loopClosureStatus = "loop closures unavailable: mapper inputs do not contain loop-closure data";
          }
          installTrackedKeypoints(
            state.observationPositions,
            state.observationPointIds,
            state.observationOffsets
          );
          showKeyframe(Number(keyframeSlider.value));
          if (
            selection.imagePreviews !== null &&
            Object.keys(selection.imagePreviews).length > 0
          ) {
            installImageDirectory(selection.imagePreviews);
          } else if (selection.localInput !== null) {
            installImageDirectory(parseLocalInput(await selection.localInput.text()));
          }
          setStatus(`${state.positions.length / 3} points loaded; checking covariance cache…`);
          let cacheBuffer = null;
          if (selection.covarianceCache !== null) {
            cacheBuffer = await selection.covarianceCache.arrayBuffer();
          } else if (selection.reconstructionHandle !== null) {
            cacheBuffer = await readOptionalCacheFile(selection.reconstructionHandle, CACHE_FILENAME);
          }
          if (cacheBuffer === null) {
            setStatus(
              `${(state.positions.length / 3).toLocaleString()} points and ` +
              `${loopClosureStatus}; covariance not computed`
            );
          } else {
            try {
              const cached = parseCovarianceCache(
                cacheBuffer,
                state.sourceFingerprints,
                state.sourcePointCount
              );
              installRanks(displayRanksFromSource(cached.sourceRanks), "covariance and rank cache loaded");
              setStatus(`${status.textContent}; ${loopClosureStatus}`);
            } catch (error) {
              setStatus(`points loaded; stale covariance cache ignored: ${error.message}; ${loopClosureStatus}`);
            }
          }
        } catch (error) {
          if (state !== null && state.database !== null) state.database.close();
          state = null;
          resetRunVisualState();
          reportError(errorMessage(error));
          setStatus("model load failed");
        } finally {
          setBusy(false);
        }
      }

      chooseButton.onclick = async () => {
        setBusy(true);
        try {
          const runHandle = await pickDirectory();
          await loadSelected(await directorySelection(runHandle));
        } catch (error) {
          reportPickerError(error);
          setBusy(false);
        }
      };

      function updateSeparateControls() {
        reconstructionLabel.textContent = separateReconstruction === null
          ? "not selected"
          : separateReconstruction.name;
        mapperInputsLabel.textContent = separateMapperInputs === null
          ? "not selected"
          : separateMapperInputs.name;
        loadSeparateButton.disabled = separateReconstruction === null || separateMapperInputs === null;
      }

      function setBusy(busy) {
        chooseButton.disabled = busy;
        reconstructionButton.disabled = busy;
        mapperInputsButton.disabled = busy;
        loadSeparateButton.disabled = busy || separateReconstruction === null || separateMapperInputs === null;
        computeButton.disabled = busy || state === null;
      }

      function installSeparatePicker(button, assign) {
        button.onclick = async () => {
          try {
            const handle = await pickDirectory();
            assign(handle);
            updateSeparateControls();
          } catch (error) {
            reportPickerError(error);
          }
        };
      }
      installSeparatePicker(reconstructionButton, value => separateReconstruction = value);
      installSeparatePicker(mapperInputsButton, value => separateMapperInputs = value);

      loadSeparateButton.onclick = async () => {
        setBusy(true);
        try {
          await loadSelected(await separateDirectorySelection(separateReconstruction, separateMapperInputs));
        } catch (error) {
          reportError(errorMessage(error));
          setBusy(false);
        }
      };

      computeButton.onclick = async () => {
        setBusy(true);
        let writable = false;
        try {
          if (state.directoryHandle !== null) {
            writable =
              (await state.directoryHandle.requestPermission({mode: "readwrite"}).catch(() => "denied")) === "granted";
          }
          percentile.disabled = true;
          setStatus("reading model for covariance computation…");
          const buffers = await sourceBuffers(state.files);
          const transferables = COLMAP_FILENAMES.map(name => buffers[name]);
          const result = await workerRequest(
            "covariance",
            {buffers: buffers},
            transferables,
            (done, total) => setStatus(`computing covariance: ${done.toLocaleString()}/${total.toLocaleString()}`)
          );
          const ranked = await rankScores(result.scores);
          const sourceRanks = sourceRanksFromDisplay(ranked.ranks);
          if (writable) {
            setStatus("publishing covariance cache…");
            await writeHandleCache(
              state.directoryHandle,
              covarianceCacheBuffer(ranked.scores, sourceRanks, state.sourceFingerprints)
            );
            installRanks(ranked, "covariance computed and cached");
          } else {
            installRanks(ranked, "covariance computed for this session; cache permission not granted");
          }
        } catch (error) {
          reportError(errorMessage(error));
          setStatus("covariance computation failed");
          percentile.disabled = state.ranks === null;
        } finally {
          setBusy(false);
        }
      };

      percentile.oninput = () => applyPercentile(Number(percentile.value));
      minimumTrackLength.oninput = () => {
        const value = Number(minimumTrackLength.value);
        setMinimumTrackLength(value);
        minimumTrackLengthLabel.textContent = `${value} observation${value === 1 ? "" : "s"}`;
      };
      if (embeddedRun !== null) {
        setBusy(true);
        setStatus("reading embedded run…");
        embeddedSelection(embeddedRun)
          .then(loadSelected)
          .catch(error => {
            reportError(errorMessage(error));
            setStatus("embedded model load failed");
            setBusy(false);
          });
      }
    }

    if (typeof module !== "undefined" && module.exports) {
      module.exports = {
        matchingFingerprints: matchingFingerprints,
        parseCovarianceCache: parseCovarianceCache,
        covarianceCacheBuffer: covarianceCacheBuffer,
        parseLocalInput: parseLocalInput,
        selectionFrom: selectionFrom,
        databaseEdgeSharedCounts: databaseEdgeSharedCounts,
        sharedTrackObservations: sharedTrackObservations,
        loopClosureMatchObservations: loopClosureMatchObservations,
        parseLoopClosureMasks: parseLoopClosureMasks
      };
    }
