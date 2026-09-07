/* Small priority queue for complete scrub-video buffers; autoplay uses native streaming. */
(() => {
  const records = new Map();
  const limit = 96 * 1024 * 1024;
  let bytes = 0, active = 0, introWarm = false, cityVisible = false, galleryActive = false;
  const hero = document.querySelector('.entrance-test');
  let resolveWarm;
  const warm = new Promise(resolve => { resolveWarm = resolve; });
  const autoplayBusy = () => !introWarm || cityVisible || galleryActive ||
    ['waiting', 'loading', 'montage', 'optimization-reveal', 'optimization', 'baselines'].includes(hero.dataset.intro);
  function pump() {
    if (active >= 2 || document.hidden) return;
    const record = [...records.values()].filter(r => r.state === 'queued' &&
      (r.priority < 10 || !autoplayBusy())).sort((a, b) => a.priority - b.priority)[0];
    if (!record || (active && record.priority >= 10)) return;
    active++; record.state = 'loading';
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), record.priority < 10 ? 8000 : 45000);
    fetch(record.url, {signal: controller.signal, cache: 'force-cache'})
      .then(response => { if (!response.ok) throw Error(`Video HTTP ${response.status}`); return response.blob(); })
      .then(blob => {
        if (bytes + blob.size > limit) {
          record.state = 'streaming'; record.resolve(record.url); return;
        }
        record.blobUrl = URL.createObjectURL(blob); record.bytes = blob.size;
        bytes += blob.size; record.state = 'ready'; record.resolve(record.blobUrl);
      }).catch(error => {
        // A failed speculative fetch must not make a working native stream unusable.
        record.state = 'streaming'; record.resolve(record.url);
        console.warn('Using streamed video instead of a complete buffer:', record.url, error.message);
      }).finally(() => { clearTimeout(timer); active--; pump(); });
  }
  function get(url, priority = 20) {
    const key = new URL(url, document.baseURI).href;
    let record = records.get(key);
    if (!record) {
      record = {url: key, priority, state: 'queued', bytes: 0};
      record.promise = new Promise(resolve => { record.resolve = resolve; });
      records.set(key, record);
    }
    record.priority = Math.min(record.priority, priority); pump();
    return record.promise;
  }
  window.vidmapMedia = {
    get,
    afterIntroWarm(callback) { warm.then(callback); },
    markIntroWarm() { if (!introWarm) { introWarm = true; resolveWarm(); pump(); } },
    inspect() { return {bytes, limit, introWarm, autoplayBusy: autoplayBusy(),
      videos: [...records.values()].map(({url, priority, state, bytes}) => ({url, priority, state, bytes}))}; },
  };
  window.addEventListener('vidmap-intro-change', pump);
  document.addEventListener('visibilitychange', pump);
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) window.vidmapMedia.markIntroWarm();

  // Warm the synchronized city streams roughly one viewport before arrival.
  // Do not let the browser's generous iframe lazy-load margin compete with the intro.
  const city = document.getElementById('lin-comparison');
  let cityNear = false;
  function warmCity() {
    if (!cityNear || city.getAttribute('src') || (!introWarm && !cityVisible)) return;
    city.loading = 'eager'; city.src = city.dataset.src;
  }
  new IntersectionObserver(entries => { cityNear = entries[0].isIntersecting; warmCity(); },
    {rootMargin: '1200px'}).observe(city);
  new IntersectionObserver(entries => {
    cityVisible = entries[0].isIntersecting && entries[0].intersectionRatio >= .03;
    warmCity(); pump();
  }, {threshold: [0, .03]}).observe(city);
  warm.then(warmCity);

  // Only posters may load before a gallery card is activated.
  const gallery = document.getElementById('video-flythroughs');
  const posters = new IntersectionObserver(entries => entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    const video = entry.target.querySelector('video');
    if (video.dataset.poster) video.poster = video.dataset.poster;
    posters.unobserve(entry.target);
  }), {rootMargin: '700px'});
  gallery.querySelectorAll('.flythrough-stage').forEach(stage => posters.observe(stage));
  function activate(event) {
    if (event.type === 'keydown' && !['Enter', ' '].includes(event.key)) return;
    const stage = event.target.closest('.flythrough-stage');
    if (!stage) return;
    const video = stage.querySelector('video'), source = video.querySelector('source');
    if (video.dataset.poster) video.poster = video.dataset.poster;
    if (!source.hasAttribute('src')) {
      source.src = source.dataset.src; video.preload = 'auto'; video.load();
    }
  }
  gallery.addEventListener('click', activate, true);
  gallery.addEventListener('keydown', activate, true);
  const galleryState = () => {
    galleryActive = [...gallery.querySelectorAll('video')].some(video => !video.paused);
    pump();
  };
  gallery.addEventListener('play', galleryState, true);
  gallery.addEventListener('pause', galleryState, true);
})();
