'use strict';

// Class orders match training/common.py (XBAT+ 1.0). The X-ray model numbers its classes from 1; 0 is background.
const RGB_CLASSES = ['AlarmClock', 'ComputerMouse', 'E-Razor', 'E-Toothbrush', 'FireAlarm', 'FlashLight', 'GameController',
  'iPad', 'KidToy', 'MobilePhone', 'NewsRadio', 'NightTorch', 'PhotoCamera', 'RemoteControl', 'Thermometer'];
const XRAY_CLASSES = ['AlarmClock', 'Battery', 'ComputerMouse', 'E-Razor', 'E-Toothbrush', 'FireAlarm', 'FlashLight',
  'GameController', 'iPad', 'KidToy', 'MobilePhone', 'NewsRadio', 'NightTorch', 'NoBattery', 'PhotoCamera',
  'RemoteControl', 'Thermometer'];
const NAMES = {
  AlarmClock: 'alarm clock', ComputerMouse: 'computer mouse', 'E-Razor': 'electric razor',
  'E-Toothbrush': 'electric toothbrush', FireAlarm: 'smoke alarm', FlashLight: 'flashlight',
  GameController: 'game controller', iPad: 'tablet', KidToy: 'toy', MobilePhone: 'mobile phone', NewsRadio: 'radio',
  NightTorch: 'night light', PhotoCamera: 'camera', RemoteControl: 'remote control', Thermometer: 'thermometer',
};
let CLS_SIZE = 224; // replaced by the trained input size from models/metrics-classifier.json
const DET_SIZE = 640;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const CONFIDENT = 0.6; // below this the check asks a question instead of naming the device
const EPA = 'https://www.epa.gov/recycle/used-lithium-ion-batteries';

const $ = (sel) => document.querySelector(sel);

// Small DOM builder; every piece of text goes in as a text node.
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) el.append(c instanceof Node ? c : String(c));
  return el;
}

const pct = (x) => `${Math.round(x * 100)}%`;
const an = (word) => (/^[aeiou]/i.test(word) ? 'an' : 'a');

/* ---------- views ---------- */

function showView() {
  const view = (location.hash || '#check').slice(1);
  const known = ['check', 'xray', 'results'].includes(view) ? view : 'check';
  for (const id of ['check', 'xray', 'results']) $(`#view-${id}`).hidden = id !== known;
  for (const a of document.querySelectorAll('.tabs a')) {
    if (a.dataset.view === known) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  }
  if (known === 'xray') xrayEnter();
  if (known === 'results') renderResults();
}

/* ---------- models ---------- */

async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url} (${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(got / total);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

const sessions = {};
function loadModel(name, onProgress) {
  if (!sessions[name]) {
    sessions[name] = (async () => {
      ort.env.wasm.wasmPaths = new URL('vendor/ort/', location.href).href; // module imports need an absolute URL
      ort.env.wasm.numThreads = 1; // static hosting is not cross-origin isolated, so no threads
      const bytes = await fetchBytes(`models/${name}.onnx`, onProgress);
      return ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    })().catch((e) => { delete sessions[name]; throw e; });
  }
  return sessions[name];
}

async function bitmapFrom(source) {
  if (source instanceof Blob) return createImageBitmap(source, { imageOrientation: 'from-image' });
  const img = new Image();
  img.src = source;
  await img.decode();
  return createImageBitmap(img);
}

/* ---------- item check ---------- */

function centerSquare(bitmap, size) {
  const side = Math.min(bitmap.width, bitmap.height);
  const c = document.createElement('canvas');
  c.width = c.height = size;
  c.getContext('2d').drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side, 0, 0, size, size);
  return c;
}

function classifierInput(canvas) {
  const { data } = canvas.getContext('2d').getImageData(0, 0, CLS_SIZE, CLS_SIZE);
  const n = CLS_SIZE * CLS_SIZE;
  const x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let ch = 0; ch < 3; ch++) x[ch * n + i] = (data[i * 4 + ch] / 255 - MEAN[ch]) / STD[ch];
  }
  return new ort.Tensor('float32', x, [1, 3, CLS_SIZE, CLS_SIZE]);
}

function loadingCard(text) {
  const bar = h('span');
  const card = h('div', { class: 'card' }, h('div', { class: 'status' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), text));
  const progress = h('div', { class: 'progress', 'aria-hidden': 'true' }, bar);
  card.append(progress);
  return { card, set: (f) => { bar.style.width = pct(f); } };
}

function errorCard(message) {
  return h('div', { class: 'card' }, h('div', { class: 'verdict unsure' }, 'This check did not run.'),
    h('p', { class: 'why' }, message));
}

function batteryCard(thumb, name, confidence, onWrong) {
  return h('div', { class: 'card' },
    h('div', { class: 'verdict battery' }, 'Battery inside: keep it out of the trash and recycling bins.'),
    h('div', { class: 'card-photo' }, thumb,
      h('p', { class: 'what' }, name ? [`Looks like ${an(name)} `, h('strong', {}, name), ` (${pct(confidence)} sure). `] : 'You said it charges, lights up, makes a sound or has a screen. ',
        name ? h('button', { class: 'linkish', type: 'button', onclick: onWrong }, `Not ${an(name)} ${name}?`) : null)),
    h('ol', { class: 'steps' },
      h('li', {}, 'If the battery comes out easily, take it out. If it is built in, leave it in.'),
      h('li', {}, 'Tape the ends of a loose battery, or put it in its own plastic bag.'),
      h('li', {}, 'Take the device and the battery to a battery or electronics drop-off.')),
    h('p', { class: 'why' }, 'Batteries crushed in garbage trucks and sorting machines can catch fire. ',
      h('a', { href: EPA }, 'EPA guidance')));
}

function questionCard(thumb, lead, onAnswer) {
  return h('div', { class: 'card' },
    h('div', { class: 'verdict unsure' }, lead),
    h('div', { class: 'card-photo' }, thumb, h('p', { class: 'what' }, 'One question decides it.')),
    h('div', { class: 'question' },
      h('p', { id: 'q-text' }, 'Does it charge, light up, make a sound or have a screen?'),
      h('div', { class: 'answers', role: 'group', 'aria-labelledby': 'q-text' },
        h('button', { class: 'button primary', type: 'button', onclick: () => onAnswer(true) }, 'Yes'),
        h('button', { class: 'button', type: 'button', onclick: () => onAnswer(false) }, 'No'))));
}

function noBatteryCard(thumb) {
  return h('div', { class: 'card' },
    h('div', { class: 'verdict none' }, 'No battery device recognised.'),
    h('div', { class: 'card-photo' }, thumb,
      h('p', { class: 'what' }, 'If it has a plug, a cable or a battery compartment, it still belongs at an electronics drop-off, not in the bin.')));
}

async function checkImage(source, label) {
  const out = $('#check-result');
  const loading = loadingCard('Loading the item model (about 17 MB, once)…');
  out.replaceChildren(loading.card);
  let probs;
  let thumbUrl;
  try {
    const [session, meta] = await Promise.all([loadModel('classifier', loading.set), getClassifierMeta()]);
    CLS_SIZE = meta.settings?.size || CLS_SIZE;
    loading.card.querySelector('.status').lastChild.textContent = 'Looking at the photo…';
    const bitmap = await bitmapFrom(source);
    const square = centerSquare(bitmap, CLS_SIZE);
    thumbUrl = square.toDataURL('image/jpeg', 0.85);
    const result = await session.run({ image: classifierInput(square) });
    probs = Array.from(result.probs.data);
  } catch (e) {
    out.replaceChildren(errorCard(`${e.message || e}. Try another photo or reload the page.`));
    return;
  }
  const best = probs.indexOf(Math.max(...probs));
  const thumb = () => h('img', { src: thumbUrl, alt: label ? `Photo: ${label}` : 'Your photo' });
  const ask = (lead) => out.replaceChildren(questionCard(thumb(), lead, (yes) => {
    out.replaceChildren(yes ? batteryCard(thumb(), null, 0) : noBatteryCard(thumb()));
  }));
  if (probs[best] >= CONFIDENT) {
    const name = NAMES[RGB_CLASSES[best]];
    out.replaceChildren(batteryCard(thumb(), name, probs[best], () => ask(`Not ${an(name)} ${name}.`)));
  } else {
    ask('Not sure what this is.');
  }
}

function onFile(input) {
  input.addEventListener('change', () => {
    const file = input.files?.[0];
    if (file) checkImage(file, null);
    input.value = '';
  });
}

/* ---------- X-ray line ---------- */

let manifest = null;
async function getManifest() {
  if (!manifest) manifest = fetch('samples/manifest.json').then((r) => r.json());
  return manifest;
}

let classifierMeta = null;
async function getClassifierMeta() {
  if (!classifierMeta) classifierMeta = fetch('models/metrics-classifier.json').then((r) => r.json()).catch(() => ({}));
  return classifierMeta;
}

let detectorMeta = null;
async function getDetectorMeta() {
  if (!detectorMeta) detectorMeta = fetch('models/metrics-detector.json').then((r) => r.json()).catch(() => ({ threshold: 0.5 }));
  return detectorMeta;
}

const tally = { scanned: 0, pulled: 0, passed: 0 };
function bumpTally(pulled) {
  tally.scanned += 1;
  tally[pulled ? 'pulled' : 'passed'] += 1;
  $('#tally-scanned').textContent = tally.scanned;
  $('#tally-pulled').textContent = tally.pulled;
  $('#tally-passed').textContent = tally.passed;
}

function letterbox(bitmap) {
  const c = $('#xray-canvas');
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, DET_SIZE, DET_SIZE);
  const s = Math.min(DET_SIZE / bitmap.width, DET_SIZE / bitmap.height);
  const w = bitmap.width * s;
  const hh = bitmap.height * s;
  ctx.drawImage(bitmap, (DET_SIZE - w) / 2, (DET_SIZE - hh) / 2, w, hh);
  const { data } = ctx.getImageData(0, 0, DET_SIZE, DET_SIZE);
  const n = DET_SIZE * DET_SIZE;
  const x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let ch = 0; ch < 3; ch++) x[ch * n + i] = data[i * 4 + ch] / 255;
  return new ort.Tensor('float32', x, [3, DET_SIZE, DET_SIZE]);
}

function overlap(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const hgt = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * hgt;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
}

function suppress(dets, iou = 0.3) {
  const kept = [];
  for (const d of [...dets].sort((x, y) => y.score - x.score)) {
    if (kept.every((k) => overlap(k.box, d.box) < iou)) kept.push(d);
  }
  return kept;
}

function drawBoxes(dets) {
  const ctx = $('#xray-canvas').getContext('2d');
  ctx.font = '600 15px system-ui, sans-serif';
  ctx.textBaseline = 'top';
  for (const d of dets) {
    const [x0, y0, x1, y1] = d.box;
    const kind = d.name === 'Battery' ? 'battery' : d.name === 'NoBattery' ? 'empty' : 'device';
    const color = { battery: '#f2a900', empty: '#5cc8e0', device: '#9a9488' }[kind];
    ctx.save();
    ctx.lineWidth = kind === 'device' ? 1.5 : 3;
    ctx.strokeStyle = color;
    if (kind === 'empty') ctx.setLineDash([8, 6]);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.restore();
    const text = `${kind === 'battery' ? 'Battery' : kind === 'empty' ? 'Empty compartment' : NAMES[d.name]} ${pct(d.score)}`;
    const w = ctx.measureText(text).width + 10;
    const ty = y0 >= 22 ? y0 - 22 : y0 + 2;
    ctx.fillStyle = color;
    ctx.fillRect(x0, ty, w, 20);
    ctx.fillStyle = '#0d0d0c';
    ctx.fillText(text, x0 + 5, ty + 2);
  }
}

function decisionView(dets, threshold, truth) {
  const batteries = dets.filter((d) => d.name === 'Battery' && d.score >= threshold);
  const empty = dets.filter((d) => d.name === 'NoBattery' && d.score >= threshold);
  const device = dets.filter((d) => !['Battery', 'NoBattery'].includes(d.name) && d.score >= threshold).sort((a, b) => b.score - a.score)[0];
  const pulled = batteries.length > 0;
  const details = [];
  if (device) details.push(`Device: ${NAMES[device.name]} (${pct(device.score)})`);
  if (pulled) details.push(`${batteries.length === 1 ? 'One battery' : `${batteries.length} batteries`} found, highest ${pct(Math.max(...batteries.map((b) => b.score)))}`);
  if (empty.length) details.push('Empty battery compartment seen');
  if (truth !== undefined) details.push(`Label in the test set: ${truth ? 'battery inside' : 'no battery'}`);
  const node = h('div', {},
    h('div', { class: `verdict ${pulled ? 'battery' : 'none'}` }, pulled ? 'Pull for battery removal' : 'No battery seen: continue to shredding'),
    h('ul', {}, details.map((d) => h('li', {}, d))));
  return { node, pulled };
}

async function scanXray(source, opts = {}) {
  const box = $('#xray-result');
  const loading = loadingCard('Loading the X-ray model (about 75 MB, once)…');
  box.replaceChildren(loading.card);
  try {
    const [session, meta] = await Promise.all([loadModel('detector', loading.set), getDetectorMeta()]);
    loading.card.querySelector('.status').lastChild.textContent = 'Scanning…';
    const bitmap = await bitmapFrom(source);
    const input = letterbox(bitmap);
    const out = await session.run({ image: input });
    const labels = Array.from(out.labels.data, Number);
    const scores = Array.from(out.scores.data);
    const b = out.boxes.data;
    const all = labels.map((l, i) => ({ name: XRAY_CLASSES[l - 1], score: scores[i], box: [b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]] }))
      .filter((d) => d.name && d.score >= meta.threshold);
    // One device box (the most likely type), and batteries or empty compartments without near-duplicates.
    const device = all.filter((d) => !['Battery', 'NoBattery'].includes(d.name)).sort((x, y) => y.score - x.score).slice(0, 1);
    const dets = [...device, ...suppress(all.filter((d) => d.name === 'Battery')), ...suppress(all.filter((d) => d.name === 'NoBattery'))];
    drawBoxes(dets);
    const { node, pulled } = decisionView(dets, meta.threshold, opts.truth);
    box.replaceChildren(node);
    $('#xray-canvas').setAttribute('aria-label', `X-ray${opts.label ? ` of ${an(opts.label)} ${opts.label}` : ''}. ${pulled ? 'Battery found.' : 'No battery seen.'}`);
    bumpTally(pulled);
    return pulled;
  } catch (e) {
    box.replaceChildren(errorCard(`${e.message || e}. Reload the page and try again.`));
    throw e;
  }
}

let xrayReady = false;
async function xrayEnter() {
  if (xrayReady) return;
  xrayReady = true;
  const m = await getManifest();
  const first = await bitmapFrom(`samples/xray/${m.xray[0].file}`);
  letterbox(first);
  const list = $('#xray-samples');
  list.replaceChildren(...m.xray.map((s, i) => h('li', {},
    h('button', { class: 'sample', type: 'button', 'aria-pressed': 'false', 'aria-label': `Test X-ray ${i + 1}: ${NAMES[s.device]}`, 'data-i': i,
      onclick: (ev) => pickXray(i, ev.currentTarget) }, h('img', { src: `samples/xray/${s.file}`, alt: '' })))));
}

async function pickXray(i, button) {
  const m = await getManifest();
  for (const b of document.querySelectorAll('#xray-samples .sample')) b.setAttribute('aria-pressed', String(b === button));
  const s = m.xray[i];
  const pulled = await scanXray(`samples/xray/${s.file}`, { truth: s.battery, label: NAMES[s.device] });
  button.classList.toggle('flag', pulled);
  return pulled;
}

async function runAll() {
  const btn = $('#xray-run-all');
  btn.disabled = true;
  try {
    const m = await getManifest();
    const buttons = [...document.querySelectorAll('#xray-samples .sample')];
    const pulled = [];
    for (const [i, b] of buttons.entries()) {
      pulled.push(await pickXray(i, b));
      await new Promise((r) => setTimeout(r, 700)); // long enough to read each decision before the next X-ray
    }
    // An honest read of this small set against its labels; the Results view has the full test set.
    const withBattery = m.xray.map((s, i) => [s.battery, pulled[i]]);
    const yes = withBattery.filter(([t]) => t);
    const no = withBattery.filter(([t]) => !t);
    $('#xray-summary').textContent = `These ${m.xray.length} test X-rays: ${yes.filter(([, p]) => p).length} of ${yes.length} with a battery pulled, `
      + `${no.filter(([, p]) => p).length} of ${no.length} without one pulled by mistake.`;
  } finally {
    btn.disabled = false;
  }
}

/* ---------- results ---------- */

let resultsDone = false;
async function renderResults() {
  if (resultsDone) return;
  resultsDone = true;
  const body = $('#results-body');
  try {
    const [cls, det] = await Promise.all(['classifier', 'detector'].map((n) => fetch(`models/metrics-${n}.json`).then((r) => r.json())));
    const t = cls.test;
    const perClass = Object.entries(t.per_class).sort((a, b) => b[1].recall - a[1].recall);
    const vq = det.VQ_test.battery_images;
    const hq = det.HQ_test.battery_images;
    body.replaceChildren(
      h('div', { class: 'card' },
        h('h2', {}, 'X-ray line: batteries inside devices'),
        h('p', { class: 'sub' }, `${det.VQ_test.images} test X-rays of ${det.test_devices} devices the model never saw.`),
        h('div', { class: 'big-numbers' },
          h('div', {}, h('div', { class: 'n' }, pct(vq.recall)), h('div', { class: 'l' }, `of X-rays with a battery flagged (${vq.found} of ${vq.with_battery})`)),
          h('div', {}, h('div', { class: 'n' }, String(vq.missed)), h('div', { class: 'l' }, 'X-rays with a battery missed')),
          h('div', {}, h('div', { class: 'n' }, String(vq.false_alarms)), h('div', { class: 'l' }, `of ${vq.images - vq.with_battery} without a battery flagged`))),
        h('p', { class: 'note' }, `Score threshold ${det.threshold}, chosen on validation devices for at least 90% precision. Battery boxes, AP50: ${det.VQ_test.battery_ap50.toFixed(2)} on all test X-rays, ${det.HQ_test.battery_ap50.toFixed(2)} on the ${hq.images} high-quality ones.`)),
      h('div', { class: 'card' },
        h('h2', {}, 'Item check: naming the device'),
        h('p', { class: 'sub' }, `${t.n} test photos of ${t.n} devices the model never saw, 15 device types.`),
        h('div', { class: 'big-numbers' },
          h('div', {}, h('div', { class: 'n' }, pct(t.accuracy)), h('div', { class: 'l' }, 'named correctly')),
          h('div', {}, h('div', { class: 'n' }, pct(t.top2)), h('div', { class: 'l' }, 'right in the top two'))),
        cls.app_rule ? h('p', { class: 'note lead' }, `With the app's rule (name the device at ${pct(cls.app_rule.threshold)} or more, otherwise ask): `
          + `named ${cls.app_rule.named} of ${cls.app_rule.photos}, ${cls.app_rule.named_wrong} of them with a wrong name, and asked the question on ${cls.app_rule.asked}.`) : null,
        h('ul', { class: 'bars', 'aria-label': 'Share named correctly, by device type' },
          perClass.map(([k, v]) => h('li', {}, h('span', {}, `${NAMES[k]} (${v.n})`),
            h('span', { class: 'bar', 'aria-hidden': 'true' }, h('span', { 'data-w': v.recall })),
            h('span', { class: 'v' }, pct(v.recall))))),
        h('p', { class: 'note' }, 'Settings chosen by 5-fold cross-validation on the training devices. Some device types have only two to four test photos, so their shares move a lot with one photo. Every type in the set has a battery, so a wrong name still gives the right advice.')));
    for (const s of body.querySelectorAll('.bar span[data-w]')) s.style.width = pct(Number(s.dataset.w));
  } catch (e) {
    body.replaceChildren(errorCard(`Could not load the measurements (${e.message || e}).`));
  }
}

/* ---------- start ---------- */

async function start() {
  onFile($('#photo-camera'));
  onFile($('#photo-file'));
  $('#xray-run-all').addEventListener('click', runAll);
  $('#xray-file').addEventListener('change', (ev) => {
    const file = ev.target.files?.[0];
    if (file) scanXray(file).catch(() => {});
    ev.target.value = '';
  });
  window.addEventListener('hashchange', showView);
  showView();
  const m = await getManifest();
  $('#check-samples').replaceChildren(...m.check.map((s) => h('li', {},
    h('button', { class: 'sample', type: 'button', 'aria-label': `Test photo: ${NAMES[s.device]}`,
      onclick: () => checkImage(`samples/check/${s.file}`, NAMES[s.device]) }, h('img', { src: `samples/check/${s.file}`, alt: '' })))));
}

start();
