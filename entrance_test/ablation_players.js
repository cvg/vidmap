/* Final-first, video-only optimization scrubbers. No trace or WebGL loads here. */
const ablationManifests = JSON.parse(document.getElementById('ablation-playback-manifests').textContent);
document.querySelectorAll('[data-ablation-player]').forEach(card => {
  const frame = card.querySelector('.ablation-frame');
  const video = frame.querySelector('video');
  const slider = frame.querySelector('input');
  const bubble = frame.querySelector('.ablation-scrubber');
  const label = frame.querySelector('.ablation-label');
  let revision = 0, manifest = null, loaded = false, loading = false;
  let targetFrame = 0, painting = false, near = false;
  let selected = card.querySelector('.tab.active');

  function updateLabel() {
    const last = Number(slider.max);
    const value = Number(slider.value);
    bubble.style.setProperty('--progress', `${100 * value / Math.max(1,last)}%`);
    let text = 'Bundle Adjustment + Filtering';
    if (manifest) {
      const time = value / manifest.fps;
      const state = [...manifest.stages].reverse().find(s => s.time <= time);
      text = state?.stage.startsWith('gp') ? 'Global Positioning' : 'Bundle Adjustment + Filtering';
    }
    label.textContent = text;
    slider.setAttribute('aria-valuetext', `${text}, ${Math.round(value / Math.max(1,last) * 100)} percent`);
  }
  function seekLatest() {
    if (!loaded || video.seeking || painting) return;
    const target = Math.min(targetFrame / manifest.fps, video.duration - .001);
    if (Math.abs(video.currentTime-target) > .0001) video.currentTime = target;
    else present();
  }
  function present() {
    if (!loaded || painting || video.seeking) return;
    const token = revision;
    const displayed = targetFrame;
    if (Math.abs(video.currentTime - Math.min(displayed / manifest.fps,video.duration-.001)) > .002) {
      seekLatest(); return;
    }
    painting = true;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (token !== revision) return;
      painting = false;
      if (displayed !== targetFrame) { seekLatest(); return; }
      // Keep the exact existing high-resolution still at the end. Earlier
      // positions show only the matching video, never a second cloud layer.
      frame.classList.toggle('show-video', displayed !== Number(slider.max));
      slider.disabled = false;
    }));
  }
  async function load() {
    if (loading || loaded) return;
    const token = revision;
    loading = true;
    try {
      const data = ablationManifests[selected.dataset.ablationKey];
      if (!data) throw Error('Missing playback metadata');
      if (token !== revision) return;
      manifest = data; targetFrame = data.frames - 1;
      slider.max = slider.value = String(targetFrame);
      updateLabel();
      video.preload = 'auto';
      video.src = selected.dataset.ablationVideo;
      video.load();
    } catch (error) {
      if (token !== revision) return;
      loading = false; label.textContent = 'Playback unavailable';
      console.error('Ablation playback unavailable', error);
    }
  }
  video.addEventListener('loadedmetadata', () => {
    if (!manifest) return;
    if (Math.abs(video.duration-manifest.frames/manifest.fps) > .05) {
      label.textContent = 'Playback unavailable'; return;
    }
    loaded = true; loading = false; seekLatest();
  });
  video.addEventListener('seeked', seekLatest);
  video.addEventListener('error', () => {
    loaded = loading = false;
    slider.disabled = true; frame.classList.remove('show-video');
    label.textContent = 'Playback unavailable';
  });
  slider.addEventListener('input', () => {
    targetFrame = Number(slider.value); updateLabel(); seekLatest();
  });
  card.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
    if (selected === tab) return;
    revision++; selected = tab; manifest = null; loaded = loading = painting = false;
    video.pause(); video.removeAttribute('src'); video.load();
    frame.classList.remove('show-video'); slider.disabled = true;
    slider.max = slider.value = '1000'; updateLabel();
    if (near) load();
  }));
  new IntersectionObserver(entries => {
    near = entries[0].isIntersecting;
    if (near) load();
  }, {rootMargin:'350px'}).observe(card);
  updateLabel();
});
