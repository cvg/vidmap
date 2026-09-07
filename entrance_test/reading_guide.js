/* Responsive placement preserves the existing controls and their event state. */
(() => {
  const card = document.querySelector('.entrance-visual .hero-visual');
  const dock = document.createElement('div');
  dock.className = 'hero-mobile-dock';
  card.after(dock);
  const progress = card.querySelector('.entrance-optimization-progress');
  const legend = card.querySelector('.legend');
  const narrow = matchMedia('(max-width: 700px)');
  const placements = [progress, legend].map(node => {
    const marker = document.createComment('responsive control position');
    node.before(marker);
    return {node, marker};
  });
  function placeControls() {
    placements.forEach(({node, marker}) => {
      if (narrow.matches) dock.append(node);
      else marker.after(node);
    });
  }
  narrow.addEventListener('change', placeControls);
  placeControls();

  // Match the city tabs' keyboard behavior without changing pointer behavior.
  document.querySelectorAll('.entrance-visual .hero-modes, [data-ablation-player] .tabs').forEach((group, index) => {
    const tabs = [...group.querySelectorAll('button')];
    const panel = group.closest('[data-ablation-player]')?.querySelector('.ablation-frame') || card;
    panel.id ||= `comparison-panel-${index}`;
    panel.setAttribute('role', 'tabpanel');
    function sync() {
      tabs.forEach((tab, i) => {
        tab.id ||= `comparison-tab-${index}-${i}`;
        tab.setAttribute('role', 'tab');
        tab.setAttribute('aria-controls', panel.id);
        tab.tabIndex = tab.getAttribute('aria-selected') === 'true' ? 0 : -1;
        if (tab.tabIndex === 0) panel.setAttribute('aria-labelledby', tab.id);
      });
    }
    tabs.forEach((tab, i) => tab.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
        : (i + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length;
      tabs[next].focus(); tabs[next].click();
    }));
    new MutationObserver(sync).observe(group, {subtree: true, attributes: true, attributeFilter: ['aria-selected']});
    sync();
  });
  const cite = document.getElementById('cite-toggle');
  const citation = document.getElementById('citation-panel');
  cite.addEventListener('click', () => {
    citation.hidden = !citation.hidden;
    cite.setAttribute('aria-expanded', String(!citation.hidden));
  });
})();
