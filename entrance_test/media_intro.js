/* The introduction overlays the existing reconstruction viewer. */
(() => {
  const hero = document.querySelector('.entrance-test');
  const card = hero.querySelector('.hero-visual');
  const layer = document.createElement('div');
  layer.className = 'entrance-film';
  layer.hidden = true;
  layer.setAttribute('aria-hidden', 'true');
  const makeVideo = (name, poster, parent = layer) => {
    const video = document.createElement('video');
    video.className = `entrance-film-${name}`;
    video.muted = true;
    video.playsInline = true;
    video.preload = 'none';
    video.disablePictureInPicture = true;
    video.tabIndex = -1;
    video.poster = poster;
    parent.appendChild(video);
    return video;
  };
  const optimization = makeVideo('optimization',
    'entrance_test/web_media/images/complete_official_final_tuners/optimization_gp_smooth/HGE_optimization_blueprint_initial_1920x1080.webp');
  // Keep the accepted GP slowdown; everything after GP plays twice as fast.
  const gpPlaybackRate = 1 / 1.2;
  const baPlaybackRate = 2 / 1.2;
  optimization.defaultPlaybackRate = gpPlaybackRate;
  optimization.playbackRate = optimization.defaultPlaybackRate;
  const montage = document.createElement('div');
  montage.className = 'entrance-film-montage';
  layer.appendChild(montage);
  const tiles = Array.from({ length: 4 }, (_, i) =>
    makeVideo(`tile-${i + 1}`, `entrance_test/hge_tile_${i + 1}.jpg`, montage));
  const montagePlaybackRate = 4.5;
  // Shorten the whole on-screen sequence, including the 1.8-second entrance,
  // rather than just the settled hold. Video speed does not control this clock.
  const montageTotalMs = (1800 + 1400 + 420) * 0.9;
  const montageFadeMs = 420 * 0.9;
  tiles.forEach(video => {
    video.defaultPlaybackRate = montagePlaybackRate;
    video.playbackRate = montagePlaybackRate;
  });
  card.appendChild(layer);
  const progress = document.createElement('div');
  progress.className = 'entrance-optimization-progress';
  progress.hidden = true;
  progress.setAttribute('role', 'progressbar');
  progress.setAttribute('aria-valuemin', '0');
  progress.setAttribute('aria-valuemax', '100');
  const progressLabel = document.createElement('span');
  progress.appendChild(progressLabel);
  const scrubber = document.createElement('input');
  scrubber.type = 'range';
  scrubber.min = '0'; scrubber.max = '1000'; scrubber.step = '1'; scrubber.value = '0';
  scrubber.className = 'entrance-optimization-scrubber';
  scrubber.setAttribute('aria-label', 'Optimization playback');
  scrubber.disabled = true;
  scrubber.hidden = true;
  progress.appendChild(scrubber);
  card.appendChild(progress);
  let progressFrame = null;
  let controller = null;
  let complete = false;
  let prepared = false;
  let entered = false;
  let scrubbingEnabled = false;
  let desiredSeek = null;
  let seekFrame = null;

  function updateProgress(mediaTime = optimization.currentTime) {
    const gpDuration = 2, baDuration = 3;
    const duration = gpDuration + baDuration;
    const t = Math.min(duration, Math.max(0, mediaTime));
    const gpSource = 79 / 30;
    const gpExponent = 2.5680938167353062;
    const baStartSpeed = .8377777777777777;
    const baAcceleration = 2 * (14 - gpSource - baStartSpeed * baDuration) / baDuration ** 2;
    const sourceSeconds = t < gpDuration ? 1/30 + (78/79) * .04 * Math.expm1(gpExponent * t) / gpExponent
      : gpSource + baStartSpeed * (t - gpDuration) + baAcceleration * (t - gpDuration) ** 2 / 2;
    const rate = mediaTime < gpDuration ? gpPlaybackRate : baPlaybackRate;
    if (optimization.playbackRate !== rate) optimization.playbackRate = rate;
    // Account for both rates so the fill continues to represent viewing time.
    const viewingDuration = gpDuration / gpPlaybackRate + baDuration / baPlaybackRate;
    const viewingTime = Math.min(t, gpDuration) / gpPlaybackRate
      + Math.max(0, t - gpDuration) / baPlaybackRate;
    const percent = 100 * viewingTime / viewingDuration;
    const label = t < gpDuration ? 'Global Positioning' : 'Bundle Adjustment + Filtering';
    progressLabel.textContent = label;
    if (!scrubbingEnabled) {
      progress.setAttribute('aria-label', label);
      progress.setAttribute('aria-valuetext', label);
      progress.setAttribute('aria-valuenow', String(Math.round(percent)));
    }
    scrubber.value = String(Math.round(percent * 10));
    scrubber.setAttribute('aria-valuetext', `${label} · ${Math.round(percent)}%`);
    progress.style.setProperty('--progress', `${percent}%`);
    progress.dataset.sourceState = String(Math.min(419, Math.floor(sourceSeconds * 30)));
  }

  function finishSeek() {
    if (!scrubbingEnabled || desiredSeek === null) return;
    if (optimization.seeking) return;
    if (Math.abs(optimization.currentTime - desiredSeek) > .025) {
      optimization.currentTime = desiredSeek;
      return;
    }
    const atEnd = Number(scrubber.value) === Number(scrubber.max);
    if (atEnd) {
      // Wait for the final frame to decode before dissolving in the baselines.
      layer.style.opacity = '0';
      setPhase('complete');
    }
    desiredSeek = null;
  }

  function seekToScrubber() {
    if (!scrubbingEnabled) return;
    optimization.pause();
    const fraction = Number(scrubber.value) / Number(scrubber.max);
    const gpSeconds = 2 / gpPlaybackRate;
    const viewingTime = fraction * (gpSeconds + 3 / baPlaybackRate);
    const mediaTime = fraction >= 1 ? optimization.duration - 1 / 60
      : viewingTime < gpSeconds ? viewingTime * gpPlaybackRate
      : 2 + (viewingTime - gpSeconds) * baPlaybackRate;
    desiredSeek = mediaTime;
    montage.hidden = true;
    optimization.style.opacity = '1';
    layer.hidden = false;
    layer.style.opacity = '1';
    setPhase('scrubbing');
    updateProgress(mediaTime);
    if (seekFrame === null) seekFrame = requestAnimationFrame(() => {
      seekFrame = null;
      finishSeek();
    });
  }

  scrubber.addEventListener('input', seekToScrubber);
  optimization.addEventListener('seeked', finishSeek);
  progress.addEventListener('wheel', event => {
    if (!scrubbingEnabled || scrubber.disabled) return;
    event.preventDefault();
    const delta = (event.deltaY || event.deltaX) * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 300 : 1);
    scrubber.value = String(Math.max(0, Math.min(1000, Number(scrubber.value) + delta)));
    seekToScrubber();
  }, { passive: false });

  document.querySelectorAll('[data-hero-mode]').forEach(button => {
    button.addEventListener('click', () => {
      if (!scrubbingEnabled) return;
      desiredSeek = null;
      if (seekFrame !== null) cancelAnimationFrame(seekFrame);
      seekFrame = null;
      layer.hidden = true;
      updateProgress(optimization.duration);
      setPhase('complete');
      progress.hidden = button.dataset.heroMode !== 'topdown';
    });
  });

  function tickProgress() {
    updateProgress();
    progressFrame = requestAnimationFrame(tickProgress);
  }

  function warmFootage() {
    tiles.forEach((video, i) => {
      if (video.getAttribute('src')) return;
      video.src = ({"entrance_test/hge_tile_1.mp4": "entrance_test/web_media/hge_tile_1.mp4", "entrance_test/hge_tile_2.mp4": "entrance_test/web_media/hge_tile_2.mp4", "entrance_test/hge_tile_3.mp4": "entrance_test/web_media/hge_tile_3.mp4", "entrance_test/hge_tile_4.mp4": "entrance_test/web_media/hge_tile_4.mp4"})[`entrance_test/hge_tile_${i + 1}.mp4`] || `entrance_test/hge_tile_${i + 1}.mp4?v=16s-4p5x`;
      video.preload = 'auto';
      video.load();
    });
  }

  function waitForEntrance(signal) {
    return new Promise((resolve, reject) => {
      function cleanup() {
        window.removeEventListener('vidmap-entrance-settled', done);
        signal.removeEventListener('abort', aborted);
      }
      function done() { cleanup(); resolve(); }
      function aborted() { cleanup(); reject(new DOMException('Cancelled', 'AbortError')); }
      window.addEventListener('vidmap-entrance-settled', done, { once: true });
      signal.addEventListener('abort', aborted, { once: true });
      if (signal.aborted) aborted();
      else if (entered) done();
    });
  }

  function setPhase(phase) {
    hero.dataset.intro = phase;
    window.dispatchEvent(new Event('vidmap-intro-change'));
  }

  function waitUntil(video, condition, signal, timeout = 15000) {
    return new Promise((resolve, reject) => {
      const events = ['loadeddata', 'canplay', 'timeupdate', 'ended', 'seeked'];
      let timer;
      function cleanup() {
        events.forEach(name => video.removeEventListener(name, check));
        video.removeEventListener('error', failed);
        signal.removeEventListener('abort', aborted);
        clearTimeout(timer);
      }
      function check() { if (condition()) { cleanup(); resolve(); } }
      function failed() { cleanup(); reject(new Error('Intro video could not load.')); }
      function aborted() { cleanup(); reject(new DOMException('Cancelled', 'AbortError')); }
      events.forEach(name => video.addEventListener(name, check));
      video.addEventListener('error', failed);
      signal.addEventListener('abort', aborted, { once: true });
      timer = setTimeout(() => { cleanup(); reject(new Error('Intro video timed out.')); }, timeout);
      if (signal.aborted) aborted();
      else if (video.error) failed();
      else check();
    });
  }

  async function fade(element, signal, duration, from = 1, to = 0) {
    signal.throwIfAborted();
    const animation = element.animate([{ opacity: from }, { opacity: to }],
      { duration, easing: 'ease-in-out', fill: 'forwards' });
    const cancel = () => animation.cancel();
    signal.addEventListener('abort', cancel, { once: true });
    try { await animation.finished; }
    finally { signal.removeEventListener('abort', cancel); }
  }

  function waitUntilWallTime(deadline, signal) {
    return new Promise((resolve, reject) => {
      const cleanup = () => { clearTimeout(timer); signal.removeEventListener('abort', aborted); };
      const aborted = () => { cleanup(); reject(new DOMException('Cancelled', 'AbortError')); };
      const timer = setTimeout(() => { cleanup(); resolve(); }, Math.max(0, deadline - performance.now()));
      signal.addEventListener('abort', aborted, { once: true });
      if (signal.aborted) aborted();
    });
  }

  function stop() {
    controller?.abort();
    controller = null;
    [layer, montage, optimization].forEach(element => {
      element.getAnimations().forEach(animation => animation.cancel());
    });
    [...tiles, optimization].forEach(video => video.pause());
    if (progressFrame !== null) cancelAnimationFrame(progressFrame);
    progressFrame = null;
    progress.hidden = true;
    scrubbingEnabled = false;
    desiredSeek = null;
    if (seekFrame !== null) cancelAnimationFrame(seekFrame);
    seekFrame = null;
    scrubber.disabled = true;
    scrubber.hidden = true;
    progress.classList.remove('is-scrubbable');
    progress.removeAttribute('title');
    progress.setAttribute('role', 'progressbar');
    progress.setAttribute('aria-valuemin', '0');
    progress.setAttribute('aria-valuemax', '100');
    montage.hidden = false;
    optimization.style.removeProperty('opacity');
    layer.style.removeProperty('opacity');
  }

  function finish(enableScrubbing = false) {
    stop();
    complete = true;
    layer.hidden = true;
    setPhase('complete');
    if (enableScrubbing && optimization.readyState >= 2) {
      scrubbingEnabled = true;
      scrubber.disabled = false;
      scrubber.hidden = false;
      progress.hidden = false;
      progress.classList.add('is-scrubbable');
      progress.setAttribute('role', 'group');
      progress.setAttribute('aria-label', 'Optimization playback');
      ['aria-valuemin', 'aria-valuemax', 'aria-valuenow', 'aria-valuetext'].forEach(name => progress.removeAttribute(name));
      progress.title = 'Drag or scroll to revisit GP and BA. Move to the end for the baseline comparison.';
      updateProgress(optimization.duration);
    }
  }

  function prepare() {
    stop();
    complete = false;
    entered = false;
    layer.hidden = false;
    warmFootage();
    if (!prepared) {
      optimization.src = 'entrance_test/web_media/hge_optimization_intro.mp4?v=direct-gp-sampling';
      optimization.preload = 'auto';
      optimization.load();
      prepared = true;
    }
    [...tiles, optimization].forEach(video => { if (video.readyState >= 1) video.currentTime = 0; });
    optimization.playbackRate = gpPlaybackRate;
    setPhase('waiting');
  }

  async function start() {
    if (controller || complete) return;
    controller = new AbortController();
    const signal = controller.signal;
    try {
      setPhase('loading');
      // Footage starts with the entrance, without waiting for optimization media.
      await Promise.all(tiles.map(video =>
        waitUntil(video, () => video.readyState >= 2 && !video.seeking, signal)));
      signal.throwIfAborted();
      await Promise.all(tiles.map(video => video.play()));
      signal.throwIfAborted();
      const montageStarted = performance.now();
      setPhase('montage');
      await waitForEntrance(signal);
      await waitUntilWallTime(montageStarted + montageTotalMs - montageFadeMs, signal);
      await waitUntil(optimization, () => optimization.readyState >= 2 && !optimization.seeking, signal);
      signal.throwIfAborted();
      // Start GP with the dissolve, rather than holding its first frame until
      // the footage has gone. The accepted speed curve remains unchanged.
      await optimization.play();
      signal.throwIfAborted();
      setPhase('optimization-reveal');
      progress.hidden = false;
      progress.classList.remove('complete');
      tickProgress();
      await Promise.all([fade(montage, signal, montageFadeMs), fade(optimization, signal, montageFadeMs, 0, 1)]);
      tiles.forEach(video => video.pause());
      signal.throwIfAborted();
      setPhase('optimization');
      await waitUntil(optimization, () => optimization.ended, signal);
      if (progressFrame !== null) cancelAnimationFrame(progressFrame);
      progressFrame = null;
      updateProgress();
      setPhase('baselines');
      // This is the same aligned final filtered map. Dissolving the cover
      // reveals its baseline trajectories and leaves the original viewer live.
      await fade(layer, signal, 1000);
      signal.throwIfAborted();
      finish(true);
    } catch (error) {
      if (signal.aborted) return;
      console.warn('Entrance intro skipped:', error);
      hero.dataset.introError = error.message;
      finish();
    }
  }

  function reset() {
    stop();
    complete = false;
    entered = false;
    layer.hidden = true;
    [...tiles, optimization].forEach(video => { if (video.readyState >= 1) video.currentTime = 0; });
    updateProgress(0);
    delete hero.dataset.introError;
    // Reset the native camera, keyframe prefix and selected tab as well as the
    // intro. Otherwise its next handoff reveals the previous interactive mode.
    window.dispatchEvent(new Event('vidmap-entrance-reset'));
    setPhase('idle');
  }

  window.vidmapEntranceIntro = {
    prepare, start, reset, finish,
    entered() { entered = true; window.dispatchEvent(new Event('vidmap-entrance-settled')); },
    get complete() { return complete; },
    get active() { return ['waiting', 'loading', 'montage', 'optimization-reveal', 'optimization', 'scrubbing'].includes(hero.dataset.intro); },
  };
  setPhase('idle');
  // Only the small RGB tiles warm up on the title screen; WebGL and the larger
  // optimization clip still load on demand. No automatic media for reduced motion.
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches) warmFootage();
})();
