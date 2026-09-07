(() => {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let visible = false, fraction = 0, userPaused = false, started = false;
  let introSeconds = 0, userDivider = false, last = performance.now();
  let previousPointer = null;
  const root = document.documentElement;
  const ending = document.getElementById('city-ending');
  const legend = document.getElementById('city-legend');
  const controls = document.querySelector('.controls');
  let phase = 'flythrough', phaseSeconds = 0, wipeFrom = 0, endingPlayPending = false;
  function setPhase(next) { phase = next; phaseSeconds = 0; root.dataset.cityPhase = next; }
  function notifyMode(mode) { parent.postMessage({ type: 'vidmap-city-mode-changed', mode }, '*'); }
  function playEnding() {
    if (endingPlayPending || !ending.paused || ending.ended || !visible || document.hidden) return;
    endingPlayPending = true;
    ending.play().catch(() => {
      if (phase === 'ending') endingFailure();
    }).finally(() => {
      endingPlayPending = false;
      if (phase !== 'ending' || !visible || document.hidden) ending.pause();
    });
  }
  function endingFailure() {
    returnToFlythrough(false);
    status.textContent = 'The baseline ending could not load. The flythrough is still available.';
    document.querySelector('.source').removeAttribute('data-ready');
  }
  function beginBaselines() {
    if (phase !== 'flythrough' || !ready) return;
    generation++; pause(); userPaused = false; userDivider = true;
    resumeAfterScrub = false; scrubbing = false; dragging = false;
    divider.inert = true; controls.inert = true;
    // A direct tab selection also starts at the exact flythrough endpoint.
    if (reconstruction.currentTime < duration - .05) {
      reconstruction.currentTime = google.currentTime = duration - .001;
    }
    ending.currentTime = 0;
    wipeFrom = split;
    setPhase('seek-end');
    if (ending.error) endingFailure();
  }
  function returnToFlythrough(autoplay = true) {
    generation++; pause(); ending.pause(); setPhase('flythrough');
    root.classList.remove('baseline-mode', 'ending-visible', 'ortho-transition', 'baselines-visible', 'comparison-revealed');
    legend.setAttribute('aria-hidden', 'true');
    legend.querySelectorAll('[data-reveal]').forEach(item => item.classList.remove('revealed'));
    controls.inert = false; timeline.disabled = !ready;
    reconstruction.currentTime = google.currentTime = 0;
    setSplit(100); introSeconds = 0; userDivider = false; divider.inert = true;
    stage.setAttribute('aria-label', 'City-scale comparison. Click or press Space to pause or resume.');
    userPaused = !autoplay;
    notifyMode('flythrough');
    if (autoplay) maybeStart();
  }
  ending.addEventListener('error', () => { if (phase !== 'flythrough') endingFailure(); });
  ending.addEventListener('ended', () => {
    if (phase !== 'ending') return;
    root.classList.remove('ortho-transition');
    root.classList.add('baselines-visible');
    legend.setAttribute('aria-hidden', 'false');
    legend.querySelectorAll('[data-reveal]').forEach(item => item.classList.add('revealed'));
    setPhase('complete');
  });
  reconstruction.addEventListener('ended', beginBaselines);
  setSplit(100);
  divider.inert = true;
  reconstruction.playbackRate = google.playbackRate = 1;

  function maybeStart() {
    if (phase === 'ending') { playEnding(); return; }
    if (phase !== 'flythrough') return;
    if (!visible || document.hidden || userPaused || !ready || playing || reconstruction.ended) return;
    if (reduced.matches && !started) return;
    started = true;
    document.documentElement.classList.add('started');
    start();
  }
  function setVisibility(data) {
    fraction = Math.max(0, Math.min(1, Number(data.fraction) || 0));
    visible = data.visible && fraction >= .03;
    if (!visible || document.hidden) { resumeAfterScrub = false; pause(); ending.pause(); }
    else maybeStart();
  }
  window.addEventListener('message', event => {
    if (event.source !== parent) return;
    if (event.data?.type === 'vidmap-lin-visibility') setVisibility(event.data);
    if (event.data?.type === 'vidmap-pause-media') setVisibility({ visible: false, fraction: 0 });
    if (event.data?.type === 'vidmap-city-mode') {
      if (event.data.mode === 'baselines') beginBaselines();
      if (event.data.mode === 'flythrough' && phase !== 'flythrough') returnToFlythrough();
    }
  });
  new MutationObserver(maybeStart).observe(document.querySelector('.source'), { attributes: true });
  parent.postMessage({ type: 'vidmap-lin-ready' }, '*');

  function togglePlayback() {
    if (!ready || phase !== 'flythrough') return;
    if (playing) { userPaused = true; resumeAfterScrub = false; pause(); }
    else {
      userPaused = false; started = true;
      document.documentElement.classList.add('started');
      start(true);
    }
  }
  stage.addEventListener('pointerdown', event => {
    if (!event.target.closest('#divider')) previousPointer = { x: event.clientX, y: event.clientY };
  });
  stage.addEventListener('click', event => {
    if (event.target.closest('#divider')) return;
    if (previousPointer && Math.hypot(event.clientX - previousPointer.x, event.clientY - previousPointer.y) > 8) return;
    togglePlayback();
  });
  stage.addEventListener('keydown', event => {
    if (event.target !== stage || ![' ', 'Enter'].includes(event.key)) return;
    event.preventDefault(); togglePlayback();
  });
  function takeDivider() {
    if (phase !== 'flythrough') return;
    userDivider = true;
    document.documentElement.classList.add('comparison-revealed');
    divider.inert = false;
  }
  divider.addEventListener('pointerdown', takeDivider);
  divider.addEventListener('keydown', takeDivider);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { resumeAfterScrub = false; pause(); ending.pause(); }
    else maybeStart();
  });
  window.addEventListener('pagehide', () => ending.pause());

  function tick(now) {
    const dt = Math.min(.1, Math.max(0, (now - last) / 1000));
    last = now;
    if (visible && !document.hidden && phase !== 'flythrough') {
      phaseSeconds += dt;
      if (phase === 'seek-end') {
        if (phaseSeconds > 15) endingFailure();
        else if (!reconstruction.seeking && !google.seeking && !ending.seeking && ending.readyState >= 2) setPhase('wipe');
      } else if (phase === 'wipe') {
        const p = reduced.matches ? 1 : Math.min(1, phaseSeconds / .8);
        setSplit(wipeFrom * (.5 + .5 * Math.cos(Math.PI * p)));
        if (p === 1) {
          setSplit(0); setPhase('handoff'); notifyMode('baselines');
          root.classList.add('baseline-mode', 'ortho-transition', 'ending-visible');
          stage.setAttribute('aria-label', 'Baseline trajectories');
        }
      } else if (phase === 'handoff' && (reduced.matches || phaseSeconds >= .3)) {
        setPhase('ending');
        if (reduced.matches) {
          ending.currentTime = ending.duration - .001;
          root.classList.remove('ortho-transition');
          root.classList.add('baselines-visible');
          legend.setAttribute('aria-hidden', 'false');
          legend.querySelectorAll('[data-reveal]').forEach(item => item.classList.add('revealed'));
          setPhase('complete');
        } else playEnding();
      } else if (phase === 'ending') {
        if (ending.currentTime >= 2.5) {
          root.classList.remove('ortho-transition');
          root.classList.add('baselines-visible');
          legend.setAttribute('aria-hidden', 'false');
        }
        legend.querySelectorAll('[data-reveal]').forEach(item => {
          item.classList.toggle('revealed', ending.currentTime >= Number(item.dataset.reveal));
        });
      }
    }
    if (phase === 'flythrough' && visible && (fraction >= .5 || introSeconds >= 2) && playing && !buffering && !scrubbing && !reconstruction.paused) {
      introSeconds += dt;
      if (!userDivider && introSeconds >= 2) {
        document.documentElement.classList.add('comparison-revealed');
        divider.inert = false;
        const progress = reduced.matches ? 1 : Math.min(1, (introSeconds - 2) / 1.2);
        setSplit(100 - 50 * (.5 - .5 * Math.cos(Math.PI * progress)));
        if (progress === 1) userDivider = true;
      }
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();
