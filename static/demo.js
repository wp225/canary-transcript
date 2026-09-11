const fileInput = document.getElementById('audio-file');
const startButton = document.getElementById('start-upload');
const status = document.getElementById('status');
const picker = document.getElementById('sample-picker');

/* Every URL is relative, so the page works both at the server root and under
   the /<repo>/ subpath GitHub Pages serves from. The Pages copy has no backend:
   build_site.py pre-bakes the samples to JSON files the page GETs instead. */
const STATIC_BUILD = Boolean(window.STATIC_BUILD);
const apiUrl = (path) => `api/${path}${STATIC_BUILD ? '.json' : ''}`;
const resultsSection = document.getElementById('results');
const summary = document.getElementById('summary');
const notice = document.getElementById('notice');
const player = document.getElementById('player');
const playButton = document.getElementById('play-btn');

const spectrogram = document.getElementById('spectrogram');
const specScroll = document.getElementById('spec-scroll');
const specTrack = document.getElementById('spec-track');
const segLayer = document.getElementById('seg-layer');
const labelLane = document.getElementById('label-lane');
const playhead = document.getElementById('playhead');
const ruler = document.getElementById('ruler');
const freqGutter = document.getElementById('freq-gutter');
const zoomRange = document.getElementById('zoom-range');
const zoomLabel = document.getElementById('zoom-label');

const LANE_ROWS = 2;
const ROW_HEIGHT = 19;
const LABEL_GAP = 4;
const MAX_PX_PER_S = 2000;
const RULER_STEPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60];

let data = null;          // last /api/transcribe payload
let pxPerSec = 100;
let minPxPerSec = 10;
let activeIndex = -1;
let objectUrl = null;
let stopAt = null;
let boxes = [];           // .seg-box per segment, index-aligned
let laneTokens = [];      // .lane-token per segment
let laneTicks = [];       // connector line per segment
const widthCache = new Map();

function showStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle('error', Boolean(isError));
}

/* ── geometry ─────────────────────────────────────────────────────────────── */
function trackWidth() {
  return data ? data.duration_s * pxPerSec : 0;
}

function timeToPx(t) {
  return t * pxPerSec;
}

function pxToTime(x) {
  return pxPerSec ? x / pxPerSec : 0;
}

function measureLabel(text) {
  if (!widthCache.has(text)) {
    const probe = document.createElement('span');
    probe.className = 'lane-token';
    probe.style.visibility = 'hidden';
    probe.style.left = '-9999px';
    probe.textContent = text;
    labelLane.appendChild(probe);
    widthCache.set(text, probe.offsetWidth);
    probe.remove();
  }
  return widthCache.get(text);
}

/* ── label placement ──────────────────────────────────────────────────────────
   Greedy left-to-right across LANE_ROWS staggered rows: a label is drawn only if
   it clears the last label already placed in that row. Zooming in widens the gaps,
   so more labels appear. The active syllable is always drawn, on top. */
function placeLabels() {
  if (!data) return;
  const rowEnds = new Array(LANE_ROWS).fill(-Infinity);

  data.segments.forEach((seg, i) => {
    const token = laneTokens[i];
    const tick = laneTicks[i];
    const centre = timeToPx((seg.start_s + seg.end_s) / 2);
    const width = measureLabel(seg.label);
    const left = centre - width / 2;

    let row = -1;
    for (let r = 0; r < LANE_ROWS; r += 1) {
      if (left >= rowEnds[r] + LABEL_GAP) { row = r; break; }
    }

    if (row === -1 && i === activeIndex) row = 0;  // never hide what is playing

    if (row === -1) {
      token.classList.add('hidden');
      tick.style.display = 'none';
      return;
    }

    if (i !== activeIndex || rowEnds[row] < left) rowEnds[row] = left + width;
    token.classList.remove('hidden');
    token.style.left = `${centre}px`;
    token.style.top = `${row * ROW_HEIGHT + 2}px`;
    tick.style.display = '';
    tick.style.left = `${centre}px`;
    tick.style.top = '0px';
    tick.style.height = `${row * ROW_HEIGHT + 2}px`;
  });
}

/* The gutter is a flex sibling of the scroll box, so its top edge is not guaranteed
   to line up with the image's. Measure the real offset instead of assuming zero. */
function drawFreqGutter() {
  if (!data) return;
  const spec = data.spectrogram;
  const offset = spectrogram.getBoundingClientRect().top - freqGutter.getBoundingClientRect().top;

  freqGutter.innerHTML = '';
  spec.freq_ticks.forEach((t) => {
    const label = document.createElement('span');
    label.textContent = t.label;
    // clamp so the topmost/bottommost tick is not half-clipped by the image edge
    const y = (1 - t.frac) * spec.height_px;
    label.style.top = `${offset + Math.min(Math.max(y, 9), spec.height_px - 9)}px`;
    freqGutter.appendChild(label);
  });
}

function drawRuler() {
  if (!data) return;
  const step = RULER_STEPS.find((s) => s * pxPerSec >= 70) ?? RULER_STEPS[RULER_STEPS.length - 1];
  ruler.innerHTML = '';
  for (let t = 0; t <= data.duration_s; t += step) {
    const x = timeToPx(t);
    const tick = document.createElement('i');
    tick.style.left = `${x}px`;
    ruler.appendChild(tick);
    const label = document.createElement('span');
    label.style.left = `${x}px`;
    label.textContent = step < 1 ? `${t.toFixed(2)}s` : `${t.toFixed(step < 1 ? 1 : 0)}s`;
    ruler.appendChild(label);
  }
}

function layout() {
  if (!data) return;
  specTrack.style.width = `${trackWidth()}px`;
  boxes.forEach((box, i) => {
    const seg = data.segments[i];
    box.style.left = `${timeToPx(seg.start_s)}px`;
    box.style.width = `${Math.max(timeToPx(seg.end_s - seg.start_s), 1)}px`;
  });
  drawRuler();
  drawFreqGutter();
  placeLabels();
  updatePlayhead();
  zoomLabel.textContent = `${Math.round(pxPerSec)} px/s`;
  zoomRange.value = String(zoomToSlider(pxPerSec));
}

/* ── zoom ─────────────────────────────────────────────────────────────────── */
function sliderToZoom(value) {
  const ratio = value / 1000;
  return minPxPerSec * Math.pow(MAX_PX_PER_S / minPxPerSec, ratio);
}

function zoomToSlider(px) {
  const ratio = Math.log(px / minPxPerSec) / Math.log(MAX_PX_PER_S / minPxPerSec);
  return Math.max(0, Math.min(1000, Math.round(ratio * 1000)));
}

function setZoom(next, anchorClientX) {
  const clamped = Math.max(minPxPerSec, Math.min(MAX_PX_PER_S, next));
  if (clamped === pxPerSec) return;

  const rect = specScroll.getBoundingClientRect();
  const anchorX = anchorClientX === undefined ? rect.width / 2 : anchorClientX - rect.left;
  const anchorTime = pxToTime(specScroll.scrollLeft + anchorX);

  pxPerSec = clamped;
  layout();
  specScroll.scrollLeft = timeToPx(anchorTime) - anchorX;
}

function fitZoom() {
  if (!data || !data.duration_s) return;
  minPxPerSec = Math.max(4, specScroll.clientWidth / data.duration_s);
  if (pxPerSec < minPxPerSec) pxPerSec = minPxPerSec;
}

document.getElementById('zoom-in').addEventListener('click', () => setZoom(pxPerSec * 1.6));
document.getElementById('zoom-out').addEventListener('click', () => setZoom(pxPerSec / 1.6));
document.getElementById('zoom-fit').addEventListener('click', () => {
  fitZoom();
  setZoom(minPxPerSec);
  specScroll.scrollLeft = 0;
  layout();
});
zoomRange.addEventListener('input', () => setZoom(sliderToZoom(Number(zoomRange.value))));

specScroll.addEventListener('wheel', (event) => {
  if (!data) return;
  if (event.ctrlKey || event.metaKey) {
    event.preventDefault();
    setZoom(pxPerSec * (event.deltaY < 0 ? 1.12 : 1 / 1.12), event.clientX);
  } else if (Math.abs(event.deltaX) < Math.abs(event.deltaY)) {
    event.preventDefault();
    specScroll.scrollLeft += event.deltaY;  // vertical wheel pans through time
  }
}, { passive: false });

/* ── drag to pan, click to seek ───────────────────────────────────────────── */
let dragging = false;
let dragMoved = false;
let dragStartX = 0;
let dragStartScroll = 0;

specScroll.addEventListener('mousedown', (event) => {
  dragging = true;
  dragMoved = false;
  dragStartX = event.clientX;
  dragStartScroll = specScroll.scrollLeft;
});

window.addEventListener('mousemove', (event) => {
  if (!dragging) return;
  const dx = event.clientX - dragStartX;
  if (Math.abs(dx) > 3) {
    dragMoved = true;
    specScroll.classList.add('dragging');
  }
  specScroll.scrollLeft = dragStartScroll - dx;
});

window.addEventListener('mouseup', () => {
  dragging = false;
  specScroll.classList.remove('dragging');
});

specScroll.addEventListener('click', (event) => {
  if (dragMoved || !data) return;
  const rect = specTrack.getBoundingClientRect();
  seek(pxToTime(event.clientX - rect.left));
});

/* ── playback ─────────────────────────────────────────────────────────────── */
function seek(time) {
  stopAt = null;
  player.currentTime = Math.max(0, Math.min(time, data.duration_s));
  player.play();
}

function playRange(start, end) {
  stopAt = end;
  player.currentTime = start;
  player.play();
}

function indexAt(time) {
  const segs = data.segments;
  let lo = 0;
  let hi = segs.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (time < segs[mid].start_s) hi = mid - 1;
    else if (time >= segs[mid].end_s) lo = mid + 1;
    else return mid;
  }
  return -1;
}

function setActive(index) {
  if (index === activeIndex) return;
  [[activeIndex, false], [index, true]].forEach(([i, on]) => {
    if (i < 0) return;
    boxes[i]?.classList.toggle('active', on);
    laneTokens[i]?.classList.toggle('active', on);
    laneTicks[i]?.classList.toggle('active', on);
  });
  activeIndex = index;
  placeLabels();
  drawScatter();
}

function updatePlayhead() {
  if (!data) return;
  const x = timeToPx(player.currentTime);
  playhead.style.left = `${x}px`;
  playhead.style.height = `${data.spectrogram.height_px}px`;
  playhead.style.display = player.currentTime > 0 ? 'block' : 'none';
}

function follow() {
  const x = timeToPx(player.currentTime);
  const left = specScroll.scrollLeft;
  const width = specScroll.clientWidth;
  if (x < left + 40 || x > left + width - 40) {
    specScroll.scrollLeft = x - width / 2;
  }
}

function tick() {
  if (!data) return;
  if (stopAt !== null && player.currentTime >= stopAt) {
    player.pause();
    stopAt = null;
  }
  updatePlayhead();
  setActive(indexAt(player.currentTime));
  if (!player.paused) {
    follow();
    requestAnimationFrame(tick);
  }
}

player.addEventListener('play', () => { playButton.textContent = '❚❚'; playButton.classList.add('playing'); requestAnimationFrame(tick); });
player.addEventListener('pause', () => { playButton.textContent = '▶'; playButton.classList.remove('playing'); });
player.addEventListener('seeked', () => { updatePlayhead(); setActive(indexAt(player.currentTime)); });

playButton.addEventListener('click', () => {
  if (player.paused) { stopAt = null; player.play(); } else { player.pause(); }
});

/* ── rendering ────────────────────────────────────────────────────────────── */
function buildSpectrogramLayer() {
  const spec = data.spectrogram;
  spectrogram.src = spec.url || 'data:image/png;base64,' + spec.png;
  spectrogram.style.height = `${spec.height_px}px`;
  segLayer.style.height = `${spec.height_px}px`;
  labelLane.style.height = `${LANE_ROWS * ROW_HEIGHT + 4}px`;

  drawFreqGutter();

  segLayer.innerHTML = '';
  labelLane.innerHTML = '';
  boxes = [];
  laneTokens = [];
  laneTicks = [];

  data.segments.forEach((seg, i) => {
    const box = document.createElement('div');
    box.className = 'seg-box';
    box.style.height = `${spec.height_px}px`;
    box.title = `${seg.label} · ${seg.duration_ms} ms`;
    box.addEventListener('click', (event) => {
      event.stopPropagation();
      if (!dragMoved) playRange(seg.start_s, seg.end_s);
    });
    segLayer.appendChild(box);
    boxes.push(box);

    const tick = document.createElement('div');
    tick.className = 'lane-tick';
    labelLane.appendChild(tick);
    laneTicks.push(tick);

    const token = document.createElement('span');
    token.className = 'lane-token';
    token.textContent = seg.label;
    token.addEventListener('click', (event) => {
      event.stopPropagation();
      playRange(seg.start_s, seg.end_s);
    });
    labelLane.appendChild(token);
    laneTokens.push(token);
  });
}

function render(payload) {
  data = payload;
  activeIndex = -1;
  widthCache.clear();

  summary.textContent =
    `${payload.segments.length} syllables · ${payload.duration_s.toFixed(2)} s · ${payload.segmentation_model}`;

  buildSpectrogramLayer();

  if (payload.segments.length) {
    notice.hidden = true;
  } else {
    const why =
      payload.peak_confidence < payload.threshold
        ? `the segmentor never crossed its ${payload.threshold} threshold (peak ${payload.peak_confidence})`
        : `the segmentor only crossed its ${payload.threshold} threshold in bursts shorter than the ` +
          `${payload.min_syllable_ms} ms minimum syllable length (peak ${payload.peak_confidence})`;
    notice.textContent =
      `No syllables found — ${why}. This recording is probably calls or cage noise rather than song; ` +
      `the model was trained on canary song syllables.`;
    notice.hidden = false;
  }

  resultsSection.hidden = false;
  fitZoom();
  pxPerSec = minPxPerSec;
  layout();
  initScatter();
}

window.addEventListener('resize', () => {
  if (!data) return;
  const wasFit = Math.abs(pxPerSec - minPxPerSec) < 0.5;
  fitZoom();
  if (wasFit) pxPerSec = minPxPerSec;
  layout();
  baseUnit = null;  // panel size changed, refit
  drawScatter();
});

/* The full view lives in the landing page; the picker just swaps what it shows. */
function showResults(payload, compact = false) {
  document.body.classList.toggle('has-results', compact);  // uploads shrink the masthead; samples don't
  resultsSection.hidden = false;
  render(payload);
}

async function loadPreloadedSamples() {
  try {
    const response = await fetch(apiUrl('samples'));
    if (!response.ok) throw new Error('Unable to load samples');
    const samples = await response.json();

    picker.innerHTML = '';
    samples.forEach((sample) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = `${sample.bird_name} · ${sample.filename}`;
      button.addEventListener('click', () => loadSamplePreview(sample, button));
      picker.appendChild(button);
    });
    if (samples.length) picker.firstElementChild.click();
  } catch (error) {
    // fetch() is blocked on file:// URLs, so double-clicking the built page fails
    // here rather than anywhere obvious. Say so instead of blaming the samples.
    picker.textContent = location.protocol === 'file:'
      ? 'Serve this page over HTTP — the browser blocks data loading on file:// URLs.'
      : 'Samples are unavailable right now.';
  }
}

async function loadSamplePreview(sample, button) {
  const sampleId = sample.id;
  picker.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
  showStatus('Loading sample view...');
  try {
    const path = `samples/${encodeURIComponent(sampleId)}`;
    const response = await fetch(apiUrl(path), STATIC_BUILD ? undefined : { method: 'POST' });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Request failed');
    if (sample.audio) player.src = sample.audio.replace(/\\/g, '/');
    showResults(payload);
    showStatus(`Loaded ${payload.source?.bird_name || sampleId}.`);
  } catch (error) {
    showStatus(error.message, true);
  }
}

loadPreloadedSamples();

/* ── 3-D PCA scatter ──────────────────────────────────────────────────────────
   Small hand-rolled projector: rotate, apply weak perspective, painter's-algorithm
   sort, draw. No 3-D library, so the page stays dependency-free and offline-safe. */
const scatter = document.getElementById('scatter');
const scatterCtx = scatter.getContext('2d');

// Default view keeps PC1 — the widest-variance axis — running across the panel's
// long dimension, so a filament-shaped cloud uses the width instead of the void.
const DEFAULT_YAW = 0.3;
const DEFAULT_PITCH = -0.22;
let yaw = DEFAULT_YAW;
let pitch = DEFAULT_PITCH;
let scatterZoom = 1;
let baseUnit = null;      // px per unit at zoom 1, fitted to the panel
let sceneCentre = [0, 0, 0];
let sceneRoll = 0;        // in-plane rotation aligning the cloud to the panel
let projected = [];   // {x, y, depth, index} for hit-testing, segments only

function clusterColour(clusterId, alpha = 1) {
  const hue = (clusterId * 137.5) % 360;
  return `hsla(${hue}, 68%, 62%, ${alpha})`;
}

function sizeScatter() {
  const dpr = window.devicePixelRatio || 1;
  const width = scatter.clientWidth;
  const height = scatter.clientHeight;
  scatter.width = Math.round(width * dpr);
  scatter.height = Math.round(height * dpr);
  scatterCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width, height };
}

function rotate(point) {
  const [px, py, pz] = point;
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const x = px * cy - pz * sy;
  const z = px * sy + pz * cy;
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  return [x, py * cp - z * sp, py * sp + z * cp];
}

/* rotate(), then roll the picture plane so the cloud's own major axis lies along
   the panel's long axis — a tall filament in a wide panel wastes most of it. */
function project3(point) {
  const [x, y, z] = rotate(point);
  const cr = Math.cos(sceneRoll);
  const sr = Math.sin(sceneRoll);
  return [x * cr - y * sr, x * sr + y * cr, z];
}


/* Emission-time ramp, cool → warm, all dark enough to read on cream paper. */
const RAMP = [
  [0.0, [40, 58, 100]],
  [0.35, [92, 66, 106]],
  [0.72, [160, 50, 50]],
  [1.0, [166, 112, 40]],
];

function rampColour(t, alpha = 1) {
  const clamped = Math.max(0, Math.min(1, t));
  let lo = RAMP[0];
  let hi = RAMP[RAMP.length - 1];
  for (let i = 0; i < RAMP.length - 1; i += 1) {
    if (clamped >= RAMP[i][0] && clamped <= RAMP[i + 1][0]) { lo = RAMP[i]; hi = RAMP[i + 1]; break; }
  }
  const span = hi[0] - lo[0] || 1;
  const k = (clamped - lo[0]) / span;
  const rgb = lo[1].map((v, i) => Math.round(v + (hi[1][i] - v) * k));
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
}

function segmentRamp(seg) {
  return data.duration_s ? (seg.start_s + seg.end_s) / 2 / data.duration_s : 0;
}

/* Centre on this recording's own cloud and scale it to the panel, fitting width and
   height separately so a long thin cloud fills the long axis. Percentiles rather
   than extremes, so one stray syllable cannot shrink everything else to a dot. */
function fitScatter(plotWidth, plotHeight) {
  const placed = data.segments.filter((s) => s.pca);
  if (!placed.length) {
    sceneCentre = [0, 0, 0];
    baseUnit = Math.min(plotWidth, plotHeight) * 0.4;
    return;
  }

  const pct = (sorted, q) =>
    sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(sorted.length * q)))];

  // Midrange of the 5th–95th percentiles, not the mean: a handful of outlying
  // syllables would otherwise drag the whole cloud off to one side.
  sceneCentre = [0, 1, 2].map((axis) => {
    const values = placed.map((s) => s.pca[axis]).sort((a, b) => a - b);
    return (pct(values, 0.05) + pct(values, 0.95)) / 2;
  });

  // Roll the picture plane so the cloud's major axis runs along the panel's long
  // axis: the principal direction of the projected 2-D covariance.
  sceneRoll = 0;
  const flat = placed.map((s) => rotate(s.pca.map((v, i) => v - sceneCentre[i])));
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  flat.forEach(([x, y]) => { sxx += x * x; syy += y * y; sxy += x * y; });
  const major = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  sceneRoll = plotWidth >= plotHeight ? -major : -major + Math.PI / 2;

  const xs = [];
  const ys = [];
  placed.forEach((s) => {
    const [x, y] = project3(s.pca.map((v, i) => v - sceneCentre[i]));
    xs.push(x);
    ys.push(y);
  });
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);

  // The 5–95 band fills ~60% of the panel: the remaining margin absorbs the
  // outlying syllables, their labels, and perspective magnification.
  const spanX = Math.max(pct(xs, 0.95) - pct(xs, 0.05), 0.05);
  const spanY = Math.max(pct(ys, 0.95) - pct(ys, 0.05), 0.05);
  baseUnit = Math.min((plotWidth * 0.6) / spanX, (plotHeight * 0.58) / spanY);
}

function drawScatter() {
  if (!data || !data.pca) return;
  const { width, height } = sizeScatter();
  const ctx = scatterCtx;
  const padLeft = 78;    // colour scale
  const padRight = 104;  // coefficient readout
  const cx = padLeft + (width - padLeft - padRight) / 2;
  const cy = height / 2 + 8;
  const depth = 3.4;

  ctx.fillStyle = '#f2ede4';
  ctx.fillRect(0, 0, width, height);

  if (baseUnit === null) fitScatter(width - padLeft - padRight, height);
  const unit = baseUnit * scatterZoom;

  const toScreen = (point) => {
    const [x, y, z] = project3(point.map((v, i) => v - sceneCentre[i]));
    const f = depth / (depth + z);
    return { x: cx + x * unit * f, y: cy - y * unit * f, z, f };
  };

  // reference corpus: dim haze behind everything
  ctx.fillStyle = 'rgba(42, 42, 50, 0.2)';
  data.pca.reference.forEach((p) => {
    const { x, y, f } = toScreen(p);
    ctx.fillRect(x, y, Math.max(1, 1.3 * f), Math.max(1, 1.3 * f));
  });

  const points = data.segments
    .map((seg, index) => (seg.pca ? { seg, index, p: toScreen(seg.pca) } : null))
    .filter(Boolean);

  // song order: faint links between consecutive syllables
  ctx.strokeStyle = 'rgba(42, 42, 50, 0.14)';
  ctx.lineWidth = 0.7;
  ctx.beginPath();
  points.forEach((pt, i) => {
    if (i === 0) ctx.moveTo(pt.p.x, pt.p.y);
    else ctx.lineTo(pt.p.x, pt.p.y);
  });
  ctx.stroke();

  // Decide which markers get a numeric readout before drawing anything: nearest
  // first, active always, rejecting any label whose box overlaps one already kept.
  const kept = new Set();
  const rects = [];
  const reserve = (item) => {
    const w = 12 + Math.max(34, item.seg.label.length * 4.6 + 30);
    const box = { l: item.p.x + 5, t: item.p.y - 11, r: item.p.x + 5 + w, b: item.p.y + 11 };
    if (rects.some((q) => !(box.r < q.l || box.l > q.r || box.b < q.t || box.t > q.b))) return false;
    rects.push(box);
    return true;
  };

  const activeItem = points.find((pt) => pt.index === activeIndex);
  if (activeItem) { reserve(activeItem); kept.add(activeItem.index); }
  [...points]
    .sort((a, b) => b.p.f - a.p.f)
    .forEach((item) => {
      if (kept.has(item.index) || item.p.f < 0.7) return;
      if (reserve(item)) kept.add(item.index);
    });

  // markers, far to near
  const ordered = [...points].sort((a, b) => b.p.z - a.p.z);

  projected = [];
  ordered.forEach((item) => {
    const { x, y, f } = item.p;
    const active = item.index === activeIndex;
    const colour = rampColour(segmentRamp(item.seg), 0.95);
    const size = (active ? 7 : 3.6) * f;

    ctx.shadowColor = 'rgba(160, 50, 50, 0.35)';
    ctx.shadowBlur = active ? 10 * f : 0;
    ctx.fillStyle = active ? 'rgba(160, 50, 50, 0.95)' : colour;
    ctx.fillRect(x - size / 2, y - size / 2, size, size);
    ctx.shadowBlur = 0;

    // selection box, as in the reference manifold
    if (active) {
      const box = 15 * f;
      ctx.strokeStyle = 'rgba(160, 50, 50, 0.8)';
      ctx.lineWidth = 1.1;
      ctx.strokeRect(x - box / 2, y - box / 2, box, box);
    }

    projected.push({ x, y, index: item.index });

    // per-marker numerics, only where a slot was reserved above
    if (kept.has(item.index)) {
      ctx.font = '8.5px "JetBrains Mono", ui-monospace, monospace';
      ctx.fillStyle = active ? 'rgba(160, 50, 50, 0.85)' : 'rgba(42, 42, 50, 0.5)';
      ctx.fillText((item.seg.duration_ms / 1000).toFixed(4), x + 7 * f, y - 2);
      ctx.font = '7px "JetBrains Mono", ui-monospace, monospace';
      ctx.fillStyle = active ? 'rgba(160, 50, 50, 0.6)' : 'rgba(42, 42, 50, 0.32)';
      ctx.fillText(`${item.seg.label} ${item.seg.start_s.toFixed(2)}s`, x + 7 * f, y + 7);
    }
  });

  drawScatterChrome(ctx, width, height, toScreen);
}

function drawScatterChrome(ctx, width, height, toScreen) {
  const spec = data.pca;

  // title block
  ctx.letterSpacing = '2.5px';  // ignored on browsers without canvas letterSpacing
  ctx.font = '500 10px "JetBrains Mono", "JetBrains Mono", ui-monospace, monospace';
  ctx.fillStyle = 'rgba(42, 42, 50, 0.42)';
  ctx.fillText('SPATIOTEMPORAL SYLLABLE MANIFOLD', 16, 24);
  ctx.font = '400 9px "JetBrains Mono", "JetBrains Mono", ui-monospace, monospace';
  ctx.fillStyle = 'rgba(42, 42, 50, 0.3)';
  ctx.fillText('(26 FEATURES → 3D PCA)', 16, 40);

  const variance = spec.explained_variance;
  const total = variance.reduce((a, b) => a + b, 0);
  ctx.font = '8.5px "JetBrains Mono", ui-monospace, monospace';
  ctx.fillStyle = 'rgba(42, 42, 50, 0.3)';
  ctx.fillText(
    `PC1-3  ${(total * 100).toFixed(1)}%  (${variance.map((v) => `${(v * 100).toFixed(1)}`).join(' / ')})`,
    16, 54,
  );
  ctx.letterSpacing = '0px';

  // vertical emission-time scale
  const barX = 20;
  const barTop = 78;
  const barH = Math.max(80, height - barTop - 42);
  for (let i = 0; i < barH; i += 1) {
    ctx.fillStyle = rampColour(1 - i / barH, 0.95);
    ctx.fillRect(barX, barTop + i, 7, 1);
  }
  ctx.font = '7.5px "JetBrains Mono", ui-monospace, monospace';
  ctx.fillStyle = 'rgba(42, 42, 50, 0.34)';
  ctx.fillText(`${data.duration_s.toFixed(1)}s`, barX + 11, barTop + 5);
  ctx.fillText('0.0s', barX + 11, barTop + barH);
  ctx.save();
  ctx.translate(barX - 6, barTop + barH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = 'rgba(42, 42, 50, 0.28)';
  ctx.fillText('EMISSION TIME', -34, 0);
  ctx.restore();

  // coefficient readout
  const listX = width - 96;
  ctx.font = '6.5px "JetBrains Mono", ui-monospace, monospace';
  const names = spec.feature_names || [];
  const rowH = Math.min(11, (height - 60) / Math.max(names.length, 1));
  names.forEach((name, i) => {
    const y = 30 + i * rowH;
    const load = spec.loadings[i] ?? 0;
    ctx.fillStyle = `rgba(42, 42, 50, ${0.22 + load * 0.5})`;
    ctx.fillText(name.toUpperCase().slice(0, 15), listX + 16, y);
    ctx.fillStyle = rampColour(load, 0.5 + load * 0.45);
    ctx.fillRect(listX, y - 4, 13 * load + 1, 3);
  });

  // orientation gizmo
  const gx = width - 150;
  const gy = 40;
  const axes = [[[1, 0, 0], 'PC1'], [[0, 1, 0], 'PC2'], [[0, 0, 1], 'PC3']];
  axes.forEach(([axis, name]) => {
    const [x, y] = project3(axis);
    ctx.strokeStyle = 'rgba(42, 42, 50, 0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(gx, gy);
    ctx.lineTo(gx + x * 22, gy - y * 22);
    ctx.stroke();
    ctx.font = '6.5px "JetBrains Mono", ui-monospace, monospace';
    ctx.fillStyle = 'rgba(42, 42, 50, 0.4)';
    ctx.fillText(name, gx + x * 32 - 6, gy - y * 32 + 2);
  });
  ctx.strokeStyle = 'rgba(160, 50, 50, 0.8)';
  ctx.lineWidth = 1;
  ctx.strokeRect(gx - 3, gy - 3, 6, 6);
}

function initScatter() {
  if (!data || !data.pca) return;
  yaw = DEFAULT_YAW;
  pitch = DEFAULT_PITCH;
  scatterZoom = 1;
  baseUnit = null;  // refit to the new recording
  drawScatter();
}

let scatterDragging = false;
let scatterMoved = false;
let lastPointer = null;

scatter.addEventListener('mousedown', (event) => {
  scatterDragging = true;
  scatterMoved = false;
  lastPointer = { x: event.clientX, y: event.clientY };
});

window.addEventListener('mousemove', (event) => {
  if (!scatterDragging) return;
  const dx = event.clientX - lastPointer.x;
  const dy = event.clientY - lastPointer.y;
  if (Math.abs(dx) + Math.abs(dy) > 3) {
    scatterMoved = true;
    scatter.classList.add('dragging');
  }
  lastPointer = { x: event.clientX, y: event.clientY };
  yaw += dx * 0.008;
  pitch = Math.max(-1.5, Math.min(1.5, pitch + dy * 0.008));
  drawScatter();
});

window.addEventListener('mouseup', () => {
  scatterDragging = false;
  scatter.classList.remove('dragging');
});

scatter.addEventListener('wheel', (event) => {
  event.preventDefault();
  scatterZoom = Math.max(0.4, Math.min(6, scatterZoom * (event.deltaY < 0 ? 1.12 : 1 / 1.12)));
  drawScatter();
}, { passive: false });

scatter.addEventListener('click', (event) => {
  if (scatterMoved || !data) return;
  const rect = scatter.getBoundingClientRect();
  const mx = event.clientX - rect.left;
  const my = event.clientY - rect.top;

  let best = null;
  let bestDist = 14;
  projected.forEach((point) => {
    const dist = Math.hypot(point.x - mx, point.y - my);
    if (dist < bestDist) { bestDist = dist; best = point; }
  });
  if (best) {
    const seg = data.segments[best.index];
    playRange(seg.start_s, seg.end_s);
  }
});

/* ── upload ───────────────────────────────────────────────────────────────── */
async function uploadAndTranscribe(file) {
  if (STATIC_BUILD) {
    showStatus('Upload needs the local server — see the README. Pick a sample instead.', true);
    fileInput.value = '';
    return;
  }
  showStatus('Segmenting and transcribing...');
  startButton.style.pointerEvents = 'none';

  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  player.src = objectUrl;

  const form = new FormData();
  form.append('file', file);

  try {
    const response = await fetch(apiUrl('transcribe'), { method: 'POST', body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Request failed');

    showResults(payload, true);
    showStatus(`Done — ${payload.segments.length} syllables detected.`);
  } catch (error) {
    showStatus(error.message, true);
  } finally {
    startButton.style.pointerEvents = '';
    fileInput.value = '';
  }
}

fileInput.addEventListener('change', () => {
  if (fileInput.files && fileInput.files.length) {
    uploadAndTranscribe(fileInput.files[0]);
  }
});
