(() => {
  const hero = document.querySelector('.entrance-test');
  const stage = hero.querySelector('.entrance-stage');
  const title = hero.querySelector('.entrance-title');
  const visual = hero.querySelector('.entrance-visual');
  const cue = hero.querySelector('.entrance-cue');
  const nav = document.querySelector('.nav');
  const root = document.documentElement;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const intro = window.vidmapEntranceIntro;
  const clamp = value => Math.max(0, Math.min(1, value));
  const smooth = value => { const x = clamp(value); return x * x * (3 - 2 * x); };
  let pending = false;
  let loaded = false;
  let initialized = false;
  let state = 0;
  let transition = null;
  const transitionMs = 1800;

  function revealViewer() {
    if (loaded) return;
    loaded = true;
    window.dispatchEvent(new Event('vidmap-entrance-reveal'));
  }

  function paint(now = performance.now()) {
    pending = false;
    hero.classList.toggle('is-reduced-motion', reduced.matches);
    cue.hidden = reduced.matches;
    if (reduced.matches) {
      if (!intro.complete) intro.finish();
      initialized = false;
      transition = null;
      title.style.transform = 'none'; title.style.opacity = '1'; title.inert = false;
      visual.inert = false; nav.inert = false;
      visual.setAttribute('aria-hidden', 'false');
      root.style.setProperty('--entrance-nav', '1');
      root.style.setProperty('--entrance-controls', '1');
      hero.dataset.progress = '1';
      hero.dataset.state = 'static';
      revealViewer();
      return;
    }
    const start = hero.getBoundingClientRect().top + scrollY;
    const offset = scrollY - start;
    const enterThreshold = Math.max(64, Math.min(120, stage.clientHeight * .12));
    // Scrolling triggers a complete timed transition, never a scrubbed partial
    // frame. Separate enter/return thresholds prevent tiny scrolls toggling it.
    if (!initialized) {
      state = offset >= enterThreshold ? 1 : 0;
      initialized = true;
      if (state === 1) intro.finish();
      else intro.reset();
    }
    if (!transition) {
      const next = state === 0 && offset >= enterThreshold ? 1
        : state === 1 && offset <= 24 ? 0 : state;
      if (next !== state) {
        transition = { from: state, to: next, start: now };
        if (next === 1) { intro.prepare(); intro.start(); }
      }
    }
    let progress = state;
    let returnedToTitle = false;
    if (transition) {
      const elapsed = clamp((now - transition.start) / transitionMs);
      progress = transition.from + (transition.to - transition.from) * smooth(elapsed);
      if (elapsed === 1) {
        state = transition.to;
        transition = null;
        if (state === 1) intro.entered();
        else returnedToTitle = true;
      }
      // Finish even if scrolling stops. If the user scrolled back during the
      // animation, evaluate that request only after this transition completes.
      schedule();
    }
    const titleFade = smooth(progress / .42);
    const reveal = smooth((progress - .08) / .70);
    const controls = intro.complete ? smooth((progress - .72) / .10) : 0;
    const navFade = smooth((progress - .75) / .13);
    title.style.opacity = String(1 - titleFade);
    title.style.transform = `translateY(${-stage.clientHeight * .28 * smooth(progress / .50)}px) scale(${1 - .08 * titleFade})`;
    title.inert = progress > .36;
    cue.style.opacity = String(1 - smooth(progress / .13));
    cue.inert = progress > .12;
    visual.style.opacity = String(smooth((progress - .10) / .20));
    visual.style.transform = `translate(-50%, -50%) translateY(${(1 - reveal) * stage.clientHeight * .82}px) scale(${.62 + .38 * reveal})`;
    visual.inert = progress < .82 || !intro.complete;
    visual.setAttribute('aria-hidden', String(progress < .10));
    nav.inert = progress < .82;
    root.style.setProperty('--entrance-controls', String(controls));
    root.style.setProperty('--entrance-nav', String(navFade));
    hero.dataset.progress = progress.toFixed(4);
    hero.dataset.state = transition ? (transition.to ? 'entering' : 'returning')
      : state ? 'visual' : 'title';
    // Preserve the current map while it recedes. Reset only after the title is
    // restored and the visual is fully transparent, so no camera snap is seen.
    if (returnedToTitle) intro.reset();
    if (state === 1 || transition?.to === 1) revealViewer();
  }

  function schedule() {
    if (!pending) { pending = true; requestAnimationFrame(paint); }
  }
  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('vidmap-intro-change', schedule);
  window.addEventListener('resize', () => {
    // Recalculate geometry; do not intercept wheel, touch, or normal scrolling.
    schedule();
  }, { passive: true });
  reduced.addEventListener('change', schedule);
  document.querySelectorAll('[data-entrance-skip]').forEach(link => {
    link.addEventListener('click', event => {
      event.preventDefault();
      const start = hero.getBoundingClientRect().top + scrollY;
      const target = reduced.matches ? visual.getBoundingClientRect().top + scrollY - 90
        : start + (hero.offsetHeight - stage.offsetHeight) * .90;
      window.scrollTo({ top: target, behavior: reduced.matches ? 'instant' : 'smooth' });
    });
  });
  // Run before the first user scroll, so the initial frame is strictly title-only.
  paint();
})();
