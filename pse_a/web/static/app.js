// Jedan SSE kanal (§7.2). Samo id-jevi i skalari; nista od rezultata ne zivi ovde.
(function(){
  const units = document.getElementById('units');
  const ecg = document.getElementById('ecg');
  const ctx = ecg && ecg.getContext('2d');
  const act = new Map();

  function draw(trace){
    if(!ctx || !trace || trace.length < 2) return;
    const w = ecg.width, h = ecg.height, lo = 20, hi = 40;
    ctx.clearRect(0,0,w,h);
    const band = y => h - (y-lo)/(hi-lo)*h;
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--na');
    ctx.fillRect(0, band(35), w, band(25)-band(35));          // zdrav opseg [25,35]
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent');
    ctx.lineWidth = 1.3; ctx.beginPath();
    trace.forEach((v,i) => { const x = i/(trace.length-1)*w, y = band(v);
                             i ? ctx.lineTo(x,y) : ctx.moveTo(x,y); });
    ctx.stroke();
  }
  function render(){
    if(!act.size){ units.textContent = 'nema aktivnih jedinica'; return; }
    units.textContent = [...act.values()].map(a =>
      `${a.kind === 'qdiff' ? 'poređenje' : 'spektar'} ${a.step}/${a.total}`).join(' · ');
  }
  const es = new EventSource('/stream');
  es.addEventListener('unit.progress', e => {
    const a = JSON.parse(e.data); a.seen = Date.now(); act.set(a.run_id, a);
    render(); draw(a.trace);
  });
  es.addEventListener('run.sealed', e => {
    const a = JSON.parse(e.data); act.delete(a.run_id); render();
    units.textContent += ' · novi rezultat je zapečaćen — osveži listu';
  });
  es.addEventListener('inbox.new', e => {
    const n = JSON.parse(e.data).unseen, b = document.getElementById('unseen');
    if (b) { b.textContent = n; b.hidden = !n; }
  });
  es.addEventListener('guardian.tier', e => {
    const g = JSON.parse(e.data), el = document.getElementById('shield');
    if (el) { el.className = 'shield ' + g.level; el.title = g.text; }
  });
  setInterval(() => { const t = Date.now();
    for (const [k,a] of act) if (t - a.seen > 20000) act.delete(k); render(); }, 5000);
})();

// Razgovor: odgovor stize rec po rec preko fetch strima (bez dodatne biblioteke).
document.addEventListener('submit', async ev => {
  const f = ev.target;
  if (f.id !== 'chatform') return;
  ev.preventDefault();
  const ta = f.querySelector('textarea'), q = ta.value.trim();
  if (!q) return;
  const msgs = document.getElementById('msgs'), think = document.getElementById('thinking');
  const u = document.createElement('div'); u.className = 'msg user'; u.textContent = q;
  const a = document.createElement('div'); a.className = 'msg assistant'; a.textContent = '…';
  msgs.append(u, a); msgs.scrollTop = 1e9; ta.value = ''; think.style.display = 'block';
  f.querySelector('button').disabled = true;
  try {
    const r = await fetch('/chat/stream', {method: 'POST', body: new URLSearchParams({text: q})});
    const rd = r.body.getReader(), dec = new TextDecoder(); let buf = '';
    for (;;) {
      const {value, done} = await rd.read(); if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, i); buf = buf.slice(i + 2);
        const ev_ = (block.match(/^event: (.*)$/m) || [])[1], dl = (block.match(/^data: (.*)$/m) || [])[1];
        if (!dl) continue;
        const d = JSON.parse(dl);
        if (ev_ === 'partial') { a.textContent = d.text; }
        if (ev_ === 'done') { const t = document.createElement('div'); t.innerHTML = d.html;
                              const n = t.firstElementChild; a.replaceWith(n); htmx.process(n); }
        msgs.scrollTop = 1e9;
      }
    }
  } catch (e) { a.textContent = 'Veza sa serverom je prekinuta.'; }
  think.style.display = 'none'; f.querySelector('button').disabled = false;
});
