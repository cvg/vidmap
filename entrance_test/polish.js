(() => {
  // Warm matched images as their comparison approaches, without blocking entry.
  const preload = new IntersectionObserver(entries => entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    entry.target.querySelectorAll('[data-src]').forEach(tab => {
      const image = new Image(); image.src = tab.dataset.src; image.decode().catch(() => {});
    });
    preload.unobserve(entry.target);
  }), { rootMargin: '600px' });
  document.querySelectorAll('[data-switcher]').forEach(node => preload.observe(node));

  const dialog = document.createElement('dialog');
  dialog.className = 'figure-dialog';
  dialog.setAttribute('aria-label', 'Inspect benchmark figure');
  dialog.innerHTML = '<div class="figure-tools"><button type="button" data-zoom="out" aria-label="Zoom out">−</button><span aria-live="polite">100%</span><button type="button" data-zoom="in" aria-label="Zoom in">+</button><button type="button" data-zoom="reset">Reset</button></div><div class="figure-sheet"><img alt=""></div>';
  document.body.append(dialog);
  const sheet = dialog.querySelector('.figure-sheet');
  const image = sheet.querySelector('img');
  let zoom = 1, trigger;
  function setZoom(next) {
    const old = zoom;
    zoom = Math.min(4, Math.max(1, next));
    const x = (sheet.scrollLeft + sheet.clientWidth / 2) / old;
    const y = (sheet.scrollTop + sheet.clientHeight / 2) / old;
    image.style.width = `${zoom * 100}%`;
    sheet.scrollLeft = x * zoom - sheet.clientWidth / 2;
    sheet.scrollTop = y * zoom - sheet.clientHeight / 2;
    dialog.querySelector('span').textContent = `${Math.round(zoom * 100)}%`;
    dialog.querySelector('[data-zoom=out]').disabled = zoom === 1;
    dialog.querySelector('[data-zoom=in]').disabled = zoom === 4;
  }
  dialog.querySelectorAll('[data-zoom]').forEach(button => button.addEventListener('click', () => {
    setZoom(button.dataset.zoom === 'reset' ? 1 : zoom + (button.dataset.zoom === 'in' ? .5 : -.5));
  }));
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  dialog.addEventListener('close', () => { document.body.classList.remove('figure-open'); trigger?.focus({ preventScroll: true }); });
  document.querySelectorAll('.plot-card > img').forEach(plot => {
    plot.tabIndex = 0;
    plot.setAttribute('role', 'button');
    plot.setAttribute('aria-haspopup', 'dialog');
    plot.setAttribute('aria-label', `Inspect ${plot.alt}`);
    const open = () => {
      trigger = plot; image.src = plot.src; image.alt = plot.alt;
      dialog.showModal(); document.body.classList.add('figure-open');
      zoom = 1; setZoom(1); sheet.scrollTo(0, 0);
    };
    plot.addEventListener('click', open);
    plot.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); }
    });
  });
  const lin = document.getElementById('lin-comparison');
  const cityTabs = [...document.querySelectorAll('[data-city-mode]')];
  function selectCityMode(mode) {
    cityTabs.forEach(tab => {
      const selected = tab.dataset.cityMode === mode;
      tab.classList.toggle('active', selected);
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    lin.setAttribute('aria-labelledby', `city-${mode}-tab`);
  }
  cityTabs.forEach((tab, index) => {
    tab.addEventListener('click', () => lin.contentWindow?.postMessage({ type: 'vidmap-city-mode', mode: tab.dataset.cityMode }, '*'));
    tab.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? 1 : 1 - index;
      cityTabs[next].focus(); cityTabs[next].click();
    });
  });
  let linVisibility = { visible: false, fraction: 0 };
  function notifyLin() {
    lin.contentWindow?.postMessage({ type: 'vidmap-lin-visibility', ...linVisibility,
      visible: linVisibility.visible && !document.hidden }, '*');
  }
  new IntersectionObserver(entries => {
    linVisibility = { visible: entries[0].isIntersecting, fraction: entries[0].intersectionRatio };
    notifyLin();
  }, { rootMargin: '-80px 0px 0px', threshold: Array.from({ length: 101 }, (_, i) => i / 100) }).observe(lin);
  lin.addEventListener('load', notifyLin);
  window.addEventListener('message', event => {
    if (event.source === lin.contentWindow && event.data?.type === 'vidmap-lin-ready') notifyLin();
    if (event.source === lin.contentWindow && event.data?.type === 'vidmap-city-mode-changed') selectCityMode(event.data.mode);
  });
  document.addEventListener('visibilitychange', notifyLin);
})();
