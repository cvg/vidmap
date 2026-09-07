// Each video contains the synchronized RGB-to-inset transition itself.
const galleryBackdrop = document.createElement('div');
galleryBackdrop.className = 'flythrough-backdrop';
galleryBackdrop.hidden = true;
document.body.append(galleryBackdrop);
let spotlight = null;
const motionDuration = () => matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 260;
const bounds = rect => ({ left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px`, height: `${rect.height}px` });
const centeredBounds = () => {
  const width = Math.min(1200, innerWidth - 32, (innerHeight - 112) * 16 / 9);
  const height = width * 9 / 16;
  return { left: (innerWidth - width) / 2, top: (innerHeight - height) / 2, width, height };
};

function openSpotlight(card, stage) {
  if (spotlight) return;
  const original = card.getBoundingClientRect();
  const placeholder = document.createElement('div');
  placeholder.style.height = `${original.height}px`;
  placeholder.setAttribute('aria-hidden', 'true');
  card.before(placeholder);
  const saved = {
    card, stage, placeholder, closing: false,
    style: card.getAttribute('style'),
    overflow: document.body.style.overflow,
    padding: document.body.style.paddingRight,
    focus: document.activeElement,
    inert: [],
  };
  spotlight = saved;
  // Keep the existing card/video in place: no reload, reparenting or time reset.
  const scrollbar = innerWidth - document.documentElement.clientWidth;
  document.body.style.paddingRight = `${parseFloat(getComputedStyle(document.body).paddingRight) + scrollbar}px`;
  document.body.style.overflow = 'hidden';
  card.classList.add('is-spotlight');
  card.setAttribute('role', 'dialog');
  card.setAttribute('aria-modal', 'true');
  card.setAttribute('aria-label', stage.querySelector('h3').textContent);
  stage.setAttribute('aria-expanded', 'true');
  stage.dispatchEvent(new Event('focusmodechange'));
  Object.assign(card.style, bounds(centeredBounds()));
  for (let node = card; node.parentElement && node !== document.body; node = node.parentElement) {
    for (const sibling of node.parentElement.children) {
      if (sibling === node || sibling === galleryBackdrop || /^(SCRIPT|STYLE|LINK)$/.test(sibling.tagName)) continue;
      saved.inert.push([sibling, sibling.inert]);
      sibling.inert = true;
    }
  }
  galleryBackdrop.hidden = false;
  galleryBackdrop.animate([{ opacity: 0 }, { opacity: 1 }], { duration: motionDuration(), easing: 'ease-out' });
  card.animate([bounds(original), bounds(centeredBounds())], { duration: motionDuration(), easing: 'cubic-bezier(.2,.8,.2,1)' });
  stage.focus({ preventScroll: true });
}

async function closeSpotlight() {
  const saved = spotlight;
  if (!saved || saved.closing) return;
  saved.closing = true;
  saved.stage.querySelector("video").pause();
  saved.card.getAnimations().forEach(animation => animation.cancel());
  const animation = saved.card.animate(
    [bounds(saved.card.getBoundingClientRect()), bounds(saved.placeholder.getBoundingClientRect())],
    { duration: motionDuration(), easing: 'cubic-bezier(.2,.8,.2,1)' },
  );
  galleryBackdrop.getAnimations().forEach(item => item.cancel());
  galleryBackdrop.animate([{ opacity: 1 }, { opacity: 0 }], { duration: motionDuration(), fill: 'forwards' });
  try { await animation.finished; } catch (_) { /* Resizing can end the animation early. */ }
  saved.card.classList.remove('is-spotlight');
  if (saved.style === null) saved.card.removeAttribute('style');
  else saved.card.setAttribute('style', saved.style);
  ['role', 'aria-modal', 'aria-label'].forEach(name => saved.card.removeAttribute(name));
  saved.stage.setAttribute('aria-expanded', 'false');
  saved.stage.dispatchEvent(new Event('focusmodechange'));
  saved.inert.forEach(([element, value]) => { element.inert = value; });
  saved.placeholder.remove();
  galleryBackdrop.hidden = true;
  galleryBackdrop.getAnimations().forEach(item => item.cancel());
  document.body.style.overflow = saved.overflow;
  document.body.style.paddingRight = saved.padding;
  spotlight = null;
  const target = saved.focus instanceof HTMLElement && saved.focus !== document.body ? saved.focus : saved.stage;
  target.focus({ preventScroll: true });
}

galleryBackdrop.addEventListener('click', closeSpotlight);
document.addEventListener('keydown', event => {
  if (!spotlight) return;
  if (event.key === 'Escape') { event.preventDefault(); closeSpotlight(); }
  if (event.key === 'Tab') {
    event.preventDefault();
    spotlight.stage.focus({ preventScroll: true });
  }
});
window.addEventListener('resize', () => {
  if (!spotlight || spotlight.closing) return;
  spotlight.card.getAnimations().forEach(animation => animation.cancel());
  Object.assign(spotlight.card.style, bounds(centeredBounds()));
});

document.querySelectorAll('#video-flythroughs .flythrough-stage').forEach(stage => {
  const video = stage.querySelector('video');
  if (video.querySelector('source')?.getAttribute('src').includes('/drone.mp4')) {
    video.defaultPlaybackRate = 2;
    video.playbackRate = 2;
  }
  const card = stage.closest('article');
  const title = card.querySelector('h3').textContent;
  stage.setAttribute('role', 'button');
  stage.setAttribute('aria-haspopup', 'dialog');
  stage.setAttribute('aria-expanded', 'false');
  stage.tabIndex = 0;
  const update = () => {
    const action = stage.getAttribute('aria-expanded') === 'true'
      ? (video.paused ? 'Play' : 'Pause') : 'Open and play';
    stage.setAttribute('aria-label', `${action} ${title}`);
    stage.setAttribute('aria-pressed', String(!video.paused));
  };
  const activate = async () => {
    if (spotlight?.closing || (spotlight && spotlight.card !== card)) return;
    if (spotlight?.card === card) {
      if (!video.paused) { video.pause(); return; }
    } else {
      openSpotlight(card, stage);
    }
    try {
      await video.play();
      if (spotlight?.card !== card || spotlight.closing || document.hidden) video.pause();
    }
    catch (error) {
      stage.setAttribute('aria-label', `Retry ${title} playback`);
      console.error('Flythrough playback failed', error);
    }
  };
  // A single click opens and plays immediately; ignore the second click of an
  // accidental double-click so it does not immediately pause the video again.
  stage.addEventListener('click', event => {
    if (event.detail < 2) activate();
  });
  stage.addEventListener('dblclick', event => {
    event.preventDefault();
  });
  stage.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      activate();
    }
  });
  video.addEventListener('play', update);
  video.addEventListener('pause', update);
  stage.addEventListener('focusmodechange', update);
  video.addEventListener('ended', () => { video.currentTime = 0; update(); });
  update();
});

const galleryVisibility = new IntersectionObserver(entries => {
  entries.forEach(entry => { if (!entry.isIntersecting) entry.target.querySelector('video').pause(); });
});
document.querySelectorAll('.flythrough-stage').forEach(stage => galleryVisibility.observe(stage));
function pauseGallery() { document.querySelectorAll('.flythrough-stage video').forEach(video => video.pause()); }
document.addEventListener('visibilitychange', () => { if (document.hidden) pauseGallery(); });
window.addEventListener('pagehide', pauseGallery);
